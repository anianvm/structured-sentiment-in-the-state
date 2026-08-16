"""Passive benchmark strategies: UCRP and DIA buy-and-hold.

Implements the benchmarks of the thesis methodology chapter, section "Benchmarks and
Metrics" (sec:metrics-meth). The analysis compares every agent against

* the uniform constant-rebalanced portfolio (UCRP), rebalanced daily to
  equal weights over the 30 stocks and charged the same linear transaction
  cost as the agents (optimized strategies often fail to beat 1/N out of
  sample, DeMiguel et al. 2009), and
* a buy-and-hold position in the DIA fund, the investable passive benchmark.

The third benchmark, the price-only control arm M1, is an agent and comes out
of src/experiments/walkforward.py, not out of this module.
"""

import numpy as np
import pandas as pd

from src import config


def ucrp_log_returns(
    dates: pd.DatetimeIndex, ohlc: np.ndarray, cost: float = config.COST_C
) -> pd.Series:
    """Daily log returns of the uniform constant-rebalanced portfolio.

    The UCRP holds the 30 stocks only (no cash) and rebalances to equal
    weights every day. It is charged the same linear cost as the environment:
    on each decision day the turnover is tau = sum_i |target_i - drifted_i|
    over the stocks, and the net gross return of the day is
    (1 - cost * tau) * (target . rel), exactly mirroring the environment's
    rho = (1 - c * tau) * (w . y) - 1.

    Convention (mirror of the environment's all-cash start): the portfolio is
    bought in from nothing on the first decision day, so day one carries the
    full deployment turnover tau = sum_i |1/N - 0| = 1 and pays (1 - c) once —
    exactly what an agent pays when it moves from the initial 100%-cash state
    into the stocks ("incurs the same transaction costs as the agents",
    sec:metrics-meth). Each log return is indexed at the DECISION day t and
    realizes the price move from t to t+1, again mirroring the environment, so
    the returned series is indexed by dates[:-1] and has len(dates) - 1
    entries.

    Parameters
    ----------
    dates : trading calendar, aligned with the first axis of ``ohlc``
        (contract of src.market_data.load.load_ohlc).
    ohlc  : (T, N, 3) adjusted price levels; channel 0 is the close.
    cost  : proportional transaction cost c.
    """
    closes = ohlc[:, :, 0]
    n_assets = closes.shape[1]
    target = np.full(n_assets, 1.0 / n_assets)

    drifted = np.zeros(n_assets)  # bought in from nothing: day-one tau = 1
    log_returns = np.empty(len(dates) - 1)
    for t in range(len(dates) - 1):
        turnover = np.abs(target - drifted).sum()
        rel = closes[t + 1] / closes[t]           # gross return of each stock
        gross = float(target @ rel)               # portfolio gross return
        log_returns[t] = np.log((1.0 - cost * turnover) * gross)
        # Weights drift with the realized returns until tomorrow's rebalance.
        drifted = target * rel / gross

    return pd.Series(log_returns, index=dates[:-1], name="ucrp_log_return")


def dia_buy_hold_log_returns(start, end) -> pd.Series:
    """Daily log returns of buying DIA once at ``start`` and holding to ``end``.

    DIA (the SPDR Dow Jones Industrial Average ETF) is the investable passive
    benchmark of the study (sec:metrics-meth). Buy-and-hold means there is no
    rebalancing and therefore no transaction cost after the initial purchase;
    the daily log return is simply log(p_t / p_{t-1}) of the adjusted close,
    indexed at the day t on which it is realized.
    """
    # Imported lazily so this module can be imported (and UCRP unit-tested)
    # without the gitignored market data on disk.
    from src.market_data.load import load_adj_close

    prices = load_adj_close("DIA", start, end)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "dia_log_return"
    return log_returns
