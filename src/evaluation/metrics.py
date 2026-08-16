"""Performance and behaviour metrics for the walk-forward evaluation.

Implements the measures of the thesis methodology chapter, section "Benchmarks and
Metrics" (sec:metrics-meth):

* headline performance, following Rezaei (2025): cumulative return,
  annualized return, annualized volatility, annualized Sharpe ratio over the
  13-week T-bill, and maximum drawdown;
* behaviour diagnostics: one-way turnover, portfolio entropy, average largest
  position, average cash share, and the two downside measures reported
  alongside them, maximum drawdown and conditional value at risk.

Every performance function takes a pd.Series of DAILY LOG returns indexed by
date. Log returns add across days (their sum is log terminal wealth, matching
the reward of the environment), but risk statistics are conventionally
computed on simple returns, so the functions convert where needed and say so.
"""

import numpy as np
import pandas as pd

# Annualization factor: trading days per year. Not in config.py (it is a
# market convention, not a design choice), so it is defined here.
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Headline performance (five measures, Rezaei 2025 reporting convention)
# ---------------------------------------------------------------------------

def cumulative_return(log_returns: pd.Series) -> float:
    """Total simple return over the period: exp(sum of log returns) - 1."""
    return float(np.expm1(log_returns.sum()))


def annualized_return(log_returns: pd.Series) -> float:
    """Geometric annualized return: (1 + cumulative)^(252 / n_days) - 1."""
    n_days = len(log_returns)
    return float((1.0 + cumulative_return(log_returns)) ** (TRADING_DAYS / n_days) - 1.0)


def annualized_volatility(log_returns: pd.Series) -> float:
    """Sample std (ddof=1) of the daily SIMPLE returns, scaled by sqrt(252)."""
    simple = np.expm1(log_returns)
    return float(simple.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(log_returns: pd.Series, tbill_annual) -> float:
    """Annualized Sharpe ratio of daily excess returns over the 13-week T-bill.

    Convention (sec:metrics-meth): daily log returns are converted to simple
    returns; the annualized T-bill rate is divided by 252, aligned to the
    return dates and forward-filled (then back-filled for any leading gap),
    and subtracted; Sharpe = mean(excess) / std(excess, ddof=1) * sqrt(252).
    The same T-bill series is what the environment's cash asset earns, so a
    fully-cash policy has an excess return of ~0 by construction.

    ``tbill_annual`` may be a pd.Series of annualized decimal rates indexed by
    date, or a plain float (convenient for tests). Returns NaN when the excess
    returns are constant (std = 0), where the ratio is undefined.
    """
    simple = np.expm1(log_returns)
    if isinstance(tbill_annual, pd.Series):
        daily_tbill = tbill_annual.reindex(simple.index).ffill().bfill() / TRADING_DAYS
    else:
        daily_tbill = float(tbill_annual) / TRADING_DAYS
    excess = simple - daily_tbill
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(log_returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the cumulated wealth curve.

    Returned as a POSITIVE fraction, e.g. 0.20 means a 20% drawdown.
    """
    wealth = np.exp(log_returns.cumsum())
    peak = wealth.cummax()
    return float(((peak - wealth) / peak).max())


def performance_summary(log_returns: pd.Series, tbill_annual) -> dict:
    """The five headline measures of sec:metrics-meth as one dict."""
    return {
        "cumulative_return": cumulative_return(log_returns),
        "annualized_return": annualized_return(log_returns),
        "annualized_volatility": annualized_volatility(log_returns),
        "sharpe_ratio": sharpe_ratio(log_returns, tbill_annual),
        "max_drawdown": max_drawdown(log_returns),
    }


# ---------------------------------------------------------------------------
# Behaviour diagnostics
# ---------------------------------------------------------------------------
# ``weights`` is the panel written by the walk-forward runner: one row per
# test day, columns [CASH] + the 30 tickers, each row summing to 1.
# ``turnover`` is the environment's two-way turnover tau_t in [0, 2].

def avg_oneway_turnover(turnover: pd.Series) -> float:
    """Average daily ONE-WAY turnover, mean(tau_t / 2).

    The environment's tau sums |target - drifted| over the risky assets and
    therefore counts every reallocation twice (once sold, once bought);
    sec:metrics-meth reports one-way turnover tau/2, following Rezaei (2025).
    """
    return float((turnover / 2.0).mean())


def portfolio_entropy(weights: pd.DataFrame) -> float:
    """Average Shannon entropy (natural log) of the full N+1 weight vector.

    -sum_i w_i ln(w_i) per day, averaged over days, with 0 * ln(0) := 0.
    The uniform allocation over all 31 positions scores ln(31) ~ 3.43; lower
    values mean more concentrated (more selective) allocations.
    """
    w = weights.to_numpy(dtype=float)
    plogp = np.where(w > 0, w * np.log(np.where(w > 0, w, 1.0)), 0.0)
    return float((-plogp.sum(axis=1)).mean())


def avg_largest_position(weights: pd.DataFrame) -> float:
    """Average over days of the largest single weight (cash included)."""
    return float(weights.max(axis=1).mean())


def avg_cash_share(weights: pd.DataFrame) -> float:
    """Average weight of the cash asset (the [CASH] column, index 0)."""
    cash_col = "CASH" if "CASH" in weights.columns else weights.columns[0]
    return float(weights[cash_col].mean())


def risky_rescaled(weights: pd.DataFrame) -> pd.DataFrame:
    """Risky columns rescaled to sum to one per day (cash column dropped).

    Days with an all-cash book (risky sum ~ 0) are dropped rather than
    divided by zero; a composition cannot be defined for an empty book.

    Concentration measured on this panel answers a different question from
    the same measure on the full vector: how selective the STOCK PICKING is,
    with the cash decision divided out. The two are reported side by side
    because a policy can look diversified only because it holds cash.
    """
    cash_cols = [c for c in weights.columns if c.upper().endswith("CASH")]
    risky = weights.drop(columns=cash_cols)
    total = risky.sum(axis=1)
    keep = total > 1e-9
    return risky.loc[keep].div(total[keep], axis=0)


def behaviour_summary(weights: pd.DataFrame, turnover: pd.Series) -> dict:
    """The behaviour diagnostics of sec:metrics-meth as one dict.

    Entropy and the largest position are reported twice: on the full N+1
    weight vector, and on the risky composition alone (see risky_rescaled).
    """
    risky = risky_rescaled(weights)
    return {
        "avg_oneway_turnover": avg_oneway_turnover(turnover),
        "portfolio_entropy": portfolio_entropy(weights),
        "avg_largest_position": avg_largest_position(weights),
        "avg_cash_share": avg_cash_share(weights),
        "risky_entropy": portfolio_entropy(risky) if len(risky)
        else float("nan"),
        "risky_largest_position": avg_largest_position(risky) if len(risky)
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Downside measures reported with the behaviour diagnostics
# ---------------------------------------------------------------------------
# max_drawdown lives with the headline measures above because
# sec:metrics-meth reports it there as well; it is repeated in
# behaviour_tail_summary so the two downside views sit side by side.

CVAR_ALPHA = 0.05


def cvar(log_returns: pd.Series, alpha: float = CVAR_ALPHA) -> float:
    """Conditional value at risk: the mean of the worst ``alpha`` tail of days.

    The textbook estimator of E[R | R <= VaR_alpha]: the mean of every
    observation at or below the empirical alpha-quantile of the daily SIMPLE
    returns (simple, like volatility and the Sharpe ratio). Returned as a
    NEGATIVE fraction: -0.021 means the average of the worst 5% of trading
    days is a 2.1% loss.

    The quantile is taken with numpy's default linear interpolation, so it
    need not be an observed return; the tail is then the observations weakly
    below it, and always contains at least the minimum, which keeps the
    measure defined for short series without a special case.

    Where the maximum drawdown records a single worst PATH — a peak-to-trough
    decline that can accumulate over months — CVaR averages the worst
    individual DAYS and so describes the return distribution rather than its
    ordering. The two answer different questions and are reported together.
    """
    simple = np.expm1(log_returns).to_numpy(dtype=float)
    if simple.size == 0:
        return float("nan")
    var = np.quantile(simple, alpha)
    return float(simple[simple <= var].mean())


def behaviour_tail_summary(log_returns: pd.Series) -> dict:
    """The two downside measures of sec:metrics-meth as one dict."""
    return {
        "max_drawdown": max_drawdown(log_returns),
        "cvar_5": cvar(log_returns),
    }
