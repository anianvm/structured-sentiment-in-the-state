"""Price tensor X_t of the agent state.

Implements the price block of the thesis methodology chapter, section "The Portfolio Markov
Decision Process -- State" (sec:state-meth): X_t contains each asset's close,
high, and low prices over the previous k trading days, and every series is
divided by that asset's latest close. The result is a window of relative
prices that is invariant to the asset's price level, following Jiang et
al. (2017). By construction the last row of the close channel is exactly 1.

The raw (T, N, 3) tensor of adjusted price levels comes from
src.market_data.load; this module only cuts and normalizes windows from it.
"""

import numpy as np

from src import config


def load_ohlc_tensor(start=config.SAMPLE_START, end=config.SAMPLE_END):
    """Load the full (dates, ohlc) pair for the Dow 30 sample.

    Thin delegation to src.market_data.load.load_ohlc so that state-building
    code only ever imports from src.features. Returns (dates, ohlc) where
    dates is a pd.DatetimeIndex of length T and ohlc is a float64 array of
    shape (T, N, 3) holding adjusted [close, high, low] price levels.
    """
    # Imported inside the function so this module (and the unit tests, which
    # only exercise normalize_window on synthetic arrays) can be imported
    # without the market-data loader or its gitignored CSV files present.
    from src.market_data.load import load_ohlc

    return load_ohlc(tickers=config.DOW30, start=start, end=end)


def normalize_window(ohlc: np.ndarray, t: int, k: int) -> np.ndarray:
    """Return the normalized k-day price window ending at day t.

    Takes rows t-k+1 .. t of the (T, N, 3) tensor of adjusted price levels
    and divides every field (close, high, low) of asset i by close[t, i],
    the latest close of that asset. Output is float32 with shape (k, N, 3),
    and out[k-1, :, 0] == 1 for every asset.
    """
    assert t >= k - 1, f"day t={t} has fewer than k={k} days of history"
    assert t < ohlc.shape[0], f"day t={t} outside tensor of length {ohlc.shape[0]}"
    window = ohlc[t - k + 1 : t + 1]              # (k, N, 3) price levels
    latest_close = ohlc[t, :, 0]                  # (N,)
    return (window / latest_close[None, :, None]).astype(np.float32)
