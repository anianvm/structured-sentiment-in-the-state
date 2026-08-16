"""Market-context vector v_t of the agent state.

Implements the v_t block of the thesis methodology chapter, section "The Portfolio Markov
Decision Process -- State" (sec:state-meth). v_t has exactly two components:

  1. The CBOE VIX as a z-score. The z-score uses the EXPANDING mean and
     standard deviation of the VIX through day t inclusive, so the moments at
     day t are estimated from history only -- a fixed full-sample mean/std
     would leak future information into early observations. The expansion is
     anchored at config.CONTEXT_ANCHOR_START (2005), a decade before the
     sample starts, so that even the first sample days get moments estimated
     on thousands of observations rather than a handful.

  2. The annualized 13-week Treasury bill rate (^IRX) as a decimal. This is
     the rate the cash asset accrues in the environment, so the agent
     observes the current reward of holding cash.

Only these two series enter v_t on purpose (the spec's parsimony argument):
short-term realized volatility is already reflected in the return series and
high-low ranges of the price tensor, while the VIX adds forward-looking,
option-implied information; further predictors would enlarge the state and
may hinder learning (Velay et al. 2023).
"""

import numpy as np
import pandas as pd

from src import config


def _zscore_from_close(vix_close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Expanding z-score of a VIX close series, aligned to `dates`.

    For each day t in the series, z_t = (vix_t - mean_t) / std_t where mean_t
    and std_t (sample std, ddof=1) are computed over the series from its
    start THROUGH day t inclusive -- no future observation is used. The
    z-series is then aligned to the requested dates, forward-filling any
    requested day the VIX history lacks (e.g. calendar mismatches).
    """
    mean = vix_close.expanding().mean()
    std = vix_close.expanding().std(ddof=1)   # NaN on the very first day only
    z = (vix_close - mean) / std
    return z.reindex(dates, method="ffill").to_numpy(dtype=np.float64)


def _tbill_from_close(irx_close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """^IRX close (annualized discount yield in PERCENT) -> decimal, aligned.

    Divides by 100, forward-fills onto the requested dates, and back-fills
    any leading days that predate the first IRX observation.
    """
    decimal = irx_close / 100.0
    aligned = decimal.reindex(dates, method="ffill").bfill()
    return aligned.to_numpy(dtype=np.float64)


def vix_zscore(dates: pd.DatetimeIndex) -> np.ndarray:
    """(T,) expanding VIX z-score for the requested trading days."""
    # Imported inside the function so this module (and the unit tests, which
    # exercise the pure helpers on synthetic series) can be imported without
    # the market-data loader or its gitignored CSV files present.
    from src.market_data.load import load_raw_close

    vix = load_raw_close("VIX", start=config.CONTEXT_ANCHOR_START)
    return _zscore_from_close(vix, dates)


def tbill_rate(dates: pd.DatetimeIndex) -> np.ndarray:
    """(T,) annualized DECIMAL 13-week T-bill rate for the requested days."""
    from src.market_data.load import load_raw_close  # lazy; see vix_zscore

    irx = load_raw_close("IRX")
    return _tbill_from_close(irx, dates)


def build_market_context(dates: pd.DatetimeIndex) -> np.ndarray:
    """(T, 2) float32 context matrix, columns [vix_zscore, tbill_rate]."""
    return np.stack([vix_zscore(dates), tbill_rate(dates)], axis=1).astype(np.float32)
