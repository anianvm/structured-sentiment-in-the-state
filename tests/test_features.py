"""Unit tests for the state-building blocks in src/features/.

Every test uses small synthetic inputs with hand-computed expectations, so
the suite runs without network access and without the gitignored market
data. The one integration test that reads the committed sentiment scores is
skipped when that file is absent.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.market_context import _tbill_from_close, _zscore_from_close
from src.features.price_tensor import normalize_window
from src.features.sentiment_grid import (
    _aggregate_finbert,
    _aggregate_llm,
    _lag,
    arm_grid,
    build_llm_grid,
)

AAPL = config.DOW30.index("AAPL")
MSFT = config.DOW30.index("MSFT")

# Ten trading days: Mon 2021-01-04 .. Fri 2021-01-15 (indices 0..9).
DATES10 = pd.bdate_range("2021-01-04", periods=10)


# ---------------------------------------------------------------------------
# price_tensor.normalize_window
# ---------------------------------------------------------------------------

def _toy_ohlc():
    """(5, 3, 3) tensor with close/high/low = close, close+1, close-0.5."""
    close = np.array([
        [10.0, 4.0, 1.0],
        [20.0, 4.0, 2.0],
        [30.0, 4.0, 4.0],
        [40.0, 4.0, 8.0],
        [50.0, 4.0, 16.0],
    ])
    return np.stack([close, close + 1.0, close - 0.5], axis=2)


def test_normalize_window_hand_computed():
    out = normalize_window(_toy_ohlc(), t=4, k=3)
    assert out.shape == (3, 3, 3)
    assert out.dtype == np.float32
    # Latest close of every asset is normalized to exactly 1.
    np.testing.assert_allclose(out[2, :, 0], 1.0)
    # Close paths divided by close[t, i]: asset 0 by 50, asset 2 by 16.
    np.testing.assert_allclose(out[:, 0, 0], [0.6, 0.8, 1.0], rtol=1e-6)
    np.testing.assert_allclose(out[:, 2, 0], [0.25, 0.5, 1.0], rtol=1e-6)
    # High and low are divided by the same latest CLOSE, not by high/low.
    assert out[0, 0, 1] == pytest.approx(31.0 / 50.0, rel=1e-6)   # high
    assert out[2, 1, 2] == pytest.approx(3.5 / 4.0, rel=1e-6)     # low
    # Constant-price asset 1: whole close channel is 1.
    np.testing.assert_allclose(out[:, 1, 0], 1.0)


def test_normalize_window_needs_enough_history():
    with pytest.raises(AssertionError):
        normalize_window(_toy_ohlc(), t=1, k=3)   # only 2 days available


# ---------------------------------------------------------------------------
# market_context: expanding VIX z-score and T-bill alignment
# ---------------------------------------------------------------------------

def test_vix_zscore_expanding_hand_computed():
    vix_days = pd.bdate_range("2020-01-06", periods=4)     # Mon..Thu
    vix = pd.Series([10.0, 12.0, 11.0, 13.0], index=vix_days)
    # Request Tue..Thu plus Fri, which the VIX series lacks (ffill case).
    dates = pd.DatetimeIndex(list(vix_days[1:]) + [pd.Timestamp("2020-01-10")])
    z = _zscore_from_close(vix, dates)
    # Day 2: mean(10,12)=11, std ddof=1 = sqrt(2)  -> (12-11)/sqrt(2)
    # Day 3: mean=11, std=1                        -> 0
    # Day 4: mean=11.5, std=sqrt(5/3)              -> 1.5/sqrt(5/3)
    expected = [1.0 / np.sqrt(2.0), 0.0, 1.5 / np.sqrt(5.0 / 3.0)]
    np.testing.assert_allclose(z[:3], expected, rtol=1e-12)
    assert z[3] == z[2]   # missing Friday forward-filled from Thursday


def test_tbill_percent_to_decimal_and_fill():
    irx_days = pd.DatetimeIndex(["2020-01-06", "2020-01-08"])  # Mon, Wed
    irx = pd.Series([0.25, 0.30], index=irx_days)
    dates = pd.DatetimeIndex(["2020-01-03",   # before first obs -> bfill
                              "2020-01-06",
                              "2020-01-07",   # missing Tue -> ffill from Mon
                              "2020-01-08"])
    out = _tbill_from_close(irx, dates)
    np.testing.assert_allclose(out, [0.0025, 0.0025, 0.0025, 0.0030], rtol=1e-12)


# ---------------------------------------------------------------------------
# sentiment_grid: aggregation (calendar mapping, merge rule, normalisation)
# ---------------------------------------------------------------------------

def _llm_df(rows):
    return pd.DataFrame(rows, columns=[
        "date", "ticker", "direction", "magnitude", "horizon", "confidence"])


def test_weekend_headline_maps_to_next_trading_day():
    sat = pd.Timestamp("2021-01-09")           # Saturday -> Mon idx 5
    df = _llm_df([[sat, "AAPL", 1.0, 3.0, "days", 5.0]])
    grid = _aggregate_llm(df, DATES10, halflife=0.0)
    assert grid[5, AAPL, 4] == 1.0             # lands on Monday
    assert grid[:, :, 4].sum() == 1.0          # and nowhere else
    assert grid[5, AAPL, 0] == pytest.approx(1.0)          # direction
    assert grid[5, AAPL, 1] == pytest.approx((3 - 1) / 4)  # magnitude 3 -> 0.5
    assert grid[5, AAPL, 2] == pytest.approx((2 - 1) / 3)  # "days" -> 2 -> 1/3
    assert grid[5, AAPL, 3] == pytest.approx((5 - 1) / 4)  # confidence 5 -> 1


def test_same_day_merge_weighted_direction_max_magnitude_horizon():
    mon = DATES10[0]
    df = _llm_df([
        # story A: confident, high impact       w = 4*4 = 16
        [mon, "MSFT", 1.0, 4.0, "weeks", 4.0],
        # story B: weak, low confidence         w = 2*2 = 4
        [mon, "MSFT", -1.0, 2.0, "intraday", 2.0],
    ])
    grid = _aggregate_llm(df, DATES10, halflife=0.0)
    assert grid[0, MSFT, 0] == pytest.approx((16 - 4) / 20)     # dir = 0.6
    assert grid[0, MSFT, 1] == pytest.approx((4 - 1) / 4)       # mag = max = 4
    assert grid[0, MSFT, 2] == pytest.approx((3 - 1) / 3)       # A's "weeks"
    assert grid[0, MSFT, 3] == pytest.approx((3 - 1) / 4)       # conf mean = 3
    assert grid[0, MSFT, 4] == 1.0


def test_merge_dominant_horizon_independent_of_row_order():
    wed = DATES10[2]
    df = _llm_df([
        # low-magnitude story listed FIRST, with the "wrong" horizon
        [wed, "MSFT", 0.0, 1.0, "quarter_plus", 3.0],   # w = 3
        [wed, "MSFT", -1.0, 5.0, "intraday", 1.0],      # w = 5, dominant
    ])
    grid = _aggregate_llm(df, DATES10, halflife=0.0)
    assert grid[2, MSFT, 0] == pytest.approx(-5.0 / 8.0)
    assert grid[2, MSFT, 1] == pytest.approx(1.0)               # mag 5 -> bound
    assert grid[2, MSFT, 2] == pytest.approx(0.0)               # "intraday"
    assert grid[2, MSFT, 3] == pytest.approx((2 - 1) / 4)       # conf mean = 2


def test_normalisation_bounds_and_non_dow_tickers_dropped():
    df = _llm_df([
        [DATES10[0], "MSFT", 1.0, 5.0, "quarter_plus", 5.0],   # all upper bounds
        [DATES10[1], "AAPL", -1.0, 1.0, "intraday", 1.0],      # all lower bounds
        [DATES10[1], "ZZZT", 1.0, 5.0, "weeks", 5.0],          # not in Dow 30
    ])
    grid = _aggregate_llm(df, DATES10, halflife=0.0)
    assert grid[:, :, 0].min() >= -1 and grid[:, :, 0].max() <= 1
    for ch in (1, 2, 3):
        assert grid[:, :, ch].min() >= 0 and grid[:, :, ch].max() <= 1
    np.testing.assert_array_equal(np.unique(grid[:, :, 4]), [0.0, 1.0])
    assert grid[1, AAPL, 1:4] == pytest.approx(0.0)            # lower bounds
    assert grid[0, MSFT, 1:4] == pytest.approx(1.0)            # upper bounds
    assert grid[:, :, 4].sum() == 2.0                          # ZZZT dropped


# ---------------------------------------------------------------------------
# sentiment_grid: EWMA decay and the conservative lag
# ---------------------------------------------------------------------------

def test_ewma_impulse_halves_within_halflife_days_and_flag_stays_raw():
    dates = pd.bdate_range("2021-01-04", periods=60)
    d = 30                                       # impulse day
    df = _llm_df([[dates[d], "AAPL", 1.0, 5.0, "weeks", 5.0]])
    grid = _aggregate_llm(df, dates, halflife=5.0)
    for ch in range(4):                          # all four score channels decay
        ratio = grid[d + 5, AAPL, ch] / grid[d, AAPL, ch]
        assert ratio == pytest.approx(0.5, abs=0.01)
    # The news flag is NOT smoothed: 1 on the impulse day only.
    assert grid[d, AAPL, 4] == 1.0
    assert grid[d - 1, AAPL, 4] == 0.0 and grid[d + 1, AAPL, 4] == 0.0


def test_lag_shifts_whole_grid_and_zeroes_first_rows():
    grid = np.zeros((6, 2, 3), dtype=np.float32)
    grid[2, 1, 0] = 7.0
    grid[0, 0, 2] = 3.0
    lagged = _lag(grid, 1)
    assert lagged[3, 1, 0] == 7.0 and lagged[1, 0, 2] == 3.0   # moved by 1
    assert not lagged[2, 1, 0] and not lagged[0].any()         # first row zero
    lagged2 = _lag(grid, 2)
    assert lagged2[4, 1, 0] == 7.0
    assert not lagged2[:2].any()                               # first 2 rows zero
    np.testing.assert_array_equal(_lag(grid, 0), grid)         # lag 0 = no-op


def test_build_llm_grid_applies_lag_after_aggregation(tmp_path):
    csv = tmp_path / "scores.csv"
    _llm_df([[DATES10[1], "AAPL", 1.0, 5.0, "weeks", 5.0]]).to_csv(csv, index=False)
    grid = build_llm_grid(DATES10, halflife=0.0, lag=1, scores_csv=csv)
    assert grid[2, AAPL, 4] == 1.0     # Tuesday's news visible on Wednesday
    assert grid[2, AAPL, 0] == 1.0
    assert not grid[1].any()           # not on Tuesday itself
    assert not grid[0].any()           # first `lag` rows zero


# ---------------------------------------------------------------------------
# sentiment_grid: FinBERT variant and arm slicing
# ---------------------------------------------------------------------------

def test_finbert_aggregation_is_plain_mean():
    df = pd.DataFrame({
        "date": [DATES10[0], DATES10[0], DATES10[1]],
        "ticker": ["MSFT", "MSFT", "AAPL"],
        "finbert_score": [0.8, -0.2, 0.5],
    })
    grid = _aggregate_finbert(df, DATES10, halflife=0.0)
    assert grid.shape == (10, 30, 2)
    assert grid[0, MSFT, 0] == pytest.approx(0.3)   # mean of 0.8 and -0.2
    assert grid[0, MSFT, 1] == 1.0
    assert grid[1, AAPL, 0] == pytest.approx(0.5)
    assert grid[:, :, 1].sum() == 2.0


@pytest.fixture
def synthetic_score_csvs(tmp_path):
    """Small LLM + FinBERT score files covering a few stock-days."""
    llm = tmp_path / "llm.csv"
    _llm_df([
        [DATES10[0], "AAPL", 1.0, 4.0, "weeks", 4.0],
        [DATES10[2], "MSFT", -1.0, 3.0, "days", 5.0],
        [DATES10[5], "AAPL", -1.0, 2.0, "intraday", 2.0],
    ]).to_csv(llm, index=False)
    finbert = tmp_path / "finbert.csv"
    pd.DataFrame({
        "date": [DATES10[0], DATES10[2]],
        "ticker": ["AAPL", "MSFT"],
        "finbert_score": [0.9, -0.7],
    }).to_csv(finbert, index=False)
    return llm, finbert


def test_arm_grid_channel_counts_and_identity(synthetic_score_csvs):
    llm_csv, finbert_csv = synthetic_score_csvs
    grids = {arm: arm_grid(arm, DATES10, llm_csv=llm_csv, finbert_csv=finbert_csv)
             for arm in config.ARMS}

    assert grids["M1"] is None                       # price-only control
    for arm, n_channels in [("M2", 2), ("M3", 2), ("M4", 3), ("M5", 4), ("M6", 5)]:
        assert grids[arm].shape == (10, 30, n_channels), arm
        assert grids[arm].dtype == np.float32

    # M6 carries the full LLM grid in config.LLM_CHANNELS order.
    full = build_llm_grid(DATES10, scores_csv=llm_csv)
    np.testing.assert_array_equal(grids["M6"], full)
    # Nested arms slice the SAME channels: identical values, fewer of them.
    np.testing.assert_array_equal(grids["M3"][:, :, 0], grids["M6"][:, :, 0])  # direction
    np.testing.assert_array_equal(grids["M4"][:, :, 1], grids["M6"][:, :, 1])  # magnitude
    np.testing.assert_array_equal(grids["M5"][:, :, 2], grids["M6"][:, :, 2])  # horizon
    for arm in ("M3", "M4", "M5", "M6"):
        np.testing.assert_array_equal(grids[arm][:, :, -1], full[:, :, 4])     # news_flag
    # M2 comes from the FinBERT grid, not the LLM grid.
    from src.features.sentiment_grid import build_finbert_grid
    finbert_full = build_finbert_grid(DATES10, scores_csv=finbert_csv)
    np.testing.assert_array_equal(grids["M2"], finbert_full)


# ---------------------------------------------------------------------------
# Integration: the real frozen LLM scores (skipped if the file is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not config.LLM_SCORES_FILE.exists(),
                    reason="frozen sentiment scores not available")
def test_real_llm_grid_has_coverage():
    dates = pd.bdate_range("2015-02-02", "2016-12-30")
    grid = build_llm_grid(dates)
    assert grid.shape == (len(dates), 30, 5)
    coverage = (grid[:, :, 4] > 0).mean()
    assert coverage > 0.0
    assert np.isfinite(grid).all()
    assert grid[:, :, 0].min() >= -1.0 and grid[:, :, 0].max() <= 1.0
