"""Tests for src/market_data (adjusted OHLC loading).

All unit tests run on synthetic CSVs written into tmp_path in the exact
format download.py produces (``Date, Adj Close, Close, High, Low, Open,
Volume``) — no network and no real data needed. One integration test at the
bottom loads two real tickers and is skipped when the gitignored snapshot is
not on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make "src" importable no matter how pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.market_data.download import _select_tickers
from src.market_data.load import load_adj_close, load_ohlc, load_raw_close

# ---------------------------------------------------------------------------
# Synthetic CSV helper
# ---------------------------------------------------------------------------

CSV_HEADER = "Date,Adj Close,Close,High,Low,Open,Volume"


def write_csv(directory: Path, name: str, rows) -> Path:
    """Write a snapshot-format CSV.

    ``rows`` is a list of (date, adj_close, close, high, low) tuples; Open and
    Volume are filled with placeholders because the loader never reads them.
    """
    lines = [CSV_HEADER]
    for date, adj, close, high, low in rows:
        lines.append(f"{date},{adj},{close},{high},{low},{close},1000")
    path = directory / f"{name}.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Adjustment math
# ---------------------------------------------------------------------------

def test_adjustment_ratio_hand_computed(tmp_path):
    # Day 1: AdjClose/Close = 50/100 = 0.5, so high 110 -> 55, low 90 -> 45.
    # Day 2: factor = 1 (no dividends/splits since), levels pass through.
    write_csv(
        tmp_path,
        "AAA",
        [
            ("2020-01-02", 50.0, 100.0, 110.0, 90.0),
            ("2020-01-03", 102.0, 102.0, 104.0, 101.0),
        ],
    )
    dates, ohlc = load_ohlc(
        ["AAA"], start="2020-01-01", end="2020-12-31", market_dir=tmp_path
    )

    assert ohlc.shape == (2, 1, 3)
    assert ohlc.dtype == np.float64
    np.testing.assert_allclose(ohlc[0, 0, :], [50.0, 55.0, 45.0])
    np.testing.assert_allclose(ohlc[1, 0, :], [102.0, 104.0, 101.0])
    # The adjusted close must equal Yahoo's Adj Close exactly.
    np.testing.assert_allclose(ohlc[:, 0, 0], [50.0, 102.0])


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------

def test_calendar_is_intersection_of_tickers(tmp_path):
    # AAA trades on Jan 2, 3, 6; BBB on Jan 2, 6, 7. Shared calendar: 2 and 6.
    write_csv(
        tmp_path,
        "AAA",
        [
            ("2020-01-02", 10.0, 10.0, 11.0, 9.0),
            ("2020-01-03", 12.0, 12.0, 13.0, 11.0),
            ("2020-01-06", 14.0, 14.0, 15.0, 13.0),
        ],
    )
    write_csv(
        tmp_path,
        "BBB",
        [
            ("2020-01-02", 20.0, 20.0, 21.0, 19.0),
            ("2020-01-06", 22.0, 22.0, 23.0, 21.0),
            ("2020-01-07", 24.0, 24.0, 25.0, 23.0),
        ],
    )
    dates, ohlc = load_ohlc(
        ["AAA", "BBB"], start="2020-01-01", end="2020-12-31", market_dir=tmp_path
    )

    assert list(dates) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert ohlc.shape == (2, 2, 3)
    # Values line up per ticker and per date after the intersection.
    np.testing.assert_allclose(ohlc[:, 0, 0], [10.0, 14.0])  # AAA closes
    np.testing.assert_allclose(ohlc[:, 1, 0], [20.0, 22.0])  # BBB closes


def test_nan_row_drops_day_from_calendar(tmp_path):
    # A row with a missing close counts as "no data" for that ticker-day.
    lines = [
        CSV_HEADER,
        "2020-01-02,10.0,10.0,11.0,9.0,10.0,1000",
        "2020-01-03,,,,,10.0,1000",
        "2020-01-06,12.0,12.0,13.0,11.0,12.0,1000",
    ]
    (tmp_path / "AAA.csv").write_text("\n".join(lines) + "\n")
    dates, ohlc = load_ohlc(
        ["AAA"], start="2020-01-01", end="2020-12-31", market_dir=tmp_path
    )
    assert list(dates) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert not np.isnan(ohlc).any()


def test_start_end_clipping_is_inclusive(tmp_path):
    rows = [
        (f"2020-01-{day:02d}", float(day), float(day), float(day) + 1, float(day) - 1)
        for day in (2, 3, 6, 7, 8, 9, 10)
    ]
    write_csv(tmp_path, "AAA", rows)
    dates, ohlc = load_ohlc(
        ["AAA"], start="2020-01-06", end="2020-01-08", market_dir=tmp_path
    )
    assert list(dates) == [
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-07"),
        pd.Timestamp("2020-01-08"),
    ]
    np.testing.assert_allclose(ohlc[:, 0, 0], [6.0, 7.0, 8.0])


# ---------------------------------------------------------------------------
# Error on missing file
# ---------------------------------------------------------------------------

def test_missing_file_error_names_file_and_download_script(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_ohlc(["ZZZ"], start="2020-01-01", end="2020-12-31", market_dir=tmp_path)
    message = str(excinfo.value)
    assert "ZZZ.csv" in message
    assert "01_download_market_data" in message


# ---------------------------------------------------------------------------
# Single-series loaders
# ---------------------------------------------------------------------------

def test_load_adj_close_and_raw_close_differ(tmp_path):
    write_csv(
        tmp_path,
        "DIA",
        [
            ("2020-01-02", 50.0, 100.0, 110.0, 90.0),
            ("2020-01-03", 51.0, 102.0, 104.0, 101.0),
        ],
    )
    adj = load_adj_close("DIA", market_dir=tmp_path)
    raw = load_raw_close("DIA", market_dir=tmp_path)
    np.testing.assert_allclose(adj.to_numpy(), [50.0, 51.0])
    np.testing.assert_allclose(raw.to_numpy(), [100.0, 102.0])
    assert adj.name == "DIA"


def test_caret_is_stripped_from_ticker_names(tmp_path):
    # Files are saved caret-less (VIX.csv), but "^VIX" must load too.
    write_csv(tmp_path, "VIX", [("2020-01-02", 14.0, 14.0, 15.0, 13.0)])
    with_caret = load_raw_close("^VIX", market_dir=tmp_path)
    without = load_raw_close("VIX", market_dir=tmp_path)
    np.testing.assert_allclose(with_caret.to_numpy(), without.to_numpy())


def test_series_clipping(tmp_path):
    rows = [
        (f"2020-01-{day:02d}", float(day), float(day), float(day) + 1, float(day) - 1)
        for day in (2, 3, 6)
    ]
    write_csv(tmp_path, "AAA", rows)
    series = load_adj_close("AAA", start="2020-01-03", end="2020-01-06", market_dir=tmp_path)
    assert list(series.index) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]


# ---------------------------------------------------------------------------
# download.py ticker selection (no network involved)
# ---------------------------------------------------------------------------

def test_select_tickers_only_filter():
    assert _select_tickers(None) == config.DOW30 + config.AUX_TICKERS
    assert _select_tickers(["IRX"]) == ["^IRX"]        # caret restored
    assert _select_tickers(["aapl", "^VIX"]) == ["AAPL", "^VIX"]
    with pytest.raises(ValueError):
        _select_tickers(["NOPE"])


# ---------------------------------------------------------------------------
# Integration test on the real snapshot (skipped when data is absent)
# ---------------------------------------------------------------------------

_REAL_FILES = [config.MARKET_DIR / f"{t}.csv" for t in ("AAPL", "MSFT")]


@pytest.mark.skipif(
    not all(p.exists() for p in _REAL_FILES),
    reason="real market snapshot not on disk (data/market/ is gitignored)",
)
def test_real_snapshot_two_tickers():
    dates, ohlc = load_ohlc(["AAPL", "MSFT"])
    assert ohlc.shape == (len(dates), 2, 3)
    assert len(dates) > 2000  # ~11 years of trading days
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates[0] >= pd.Timestamp(config.SAMPLE_START)
    assert dates[-1] <= pd.Timestamp(config.SAMPLE_END)
    assert not np.isnan(ohlc).any()
    assert (ohlc > 0).all()
    # Adjusted high must stay above adjusted low (positive factor preserves it).
    assert (ohlc[:, :, 1] >= ohlc[:, :, 2] - 1e-9).all()
