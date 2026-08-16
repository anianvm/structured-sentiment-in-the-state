"""Allocation measures for the risk-aversion hypotheses (§ Benchmarks and
Metrics; definitions in App. Evaluation Measures).

These are the three measures H1b and H4 are read on, as opposed to the
headline performance of metrics.py:

  * the risk-free share (metrics.avg_cash_share, already defined there);
  * the COMPOSITION DISTANCE between two average risky allocations, which
    tests the separation prediction that risk aversion moves the cash split
    and not the composition of the risky part;
  * NEWS RESPONSIVENESS, the coefficient of the cash share on the aggregate
    news state, which is what H4 predicts rises with the richness of the
    representation.

Everything here is computed from the per-step ledgers the walk-forward runner
writes plus the committed sentiment scores, so no retraining is ever needed
to add or revise a measure.
"""

import numpy as np
import pandas as pd

from src import config
from src.evaluation.inference import newey_west_lags


# ---------------------------------------------------------------------------
# Composition distance (H1b)
# ---------------------------------------------------------------------------

def average_composition(weights: pd.DataFrame) -> pd.Series:
    """Time-average of the risky composition, i.e. of the stock weights
    rescaled to sum to one.

    `weights` is a panel of daily portfolio weights with CASH in the first
    column and the risky assets after it. Dividing each row's risky part by
    its own sum removes the cash decision, so what remains describes only how
    the invested part of the portfolio is allocated. Days that are entirely
    in cash carry no composition and are skipped rather than counted as zero.
    """
    risky = weights.iloc[:, 1:]
    invested = risky.sum(axis=1)
    held = invested > 0            # a fully-cash day has no composition
    normalized = risky[held].div(invested[held], axis=0)
    return normalized.mean(axis=0)


def composition_distance(a: pd.Series, b: pd.Series) -> float:
    """One-way turnover between two average risky compositions, in [0, 1].

    Half the sum of absolute weight differences: the fraction of the
    portfolio that would have to change hands to convert composition `a` into
    composition `b`. Zero denotes identical allocations, one denotes
    completely disjoint ones. The measure is deliberately on the same scale
    as the daily one-way turnover diagnostic, so the two can be read against
    each other.
    """
    a, b = a.align(b, fill_value=0.0)
    return float(0.5 * np.abs(a.to_numpy() - b.to_numpy()).sum())


# ---------------------------------------------------------------------------
# News responsiveness (H4)
# ---------------------------------------------------------------------------

def news_index(dates: pd.DatetimeIndex) -> pd.Series:
    """The aggregate news state g_t: cross-sectional mean of the smoothed
    directional signal across the thirty constituents.

    This is exactly the series the sentiment arms observe (lagged and
    EWMA-smoothed by src/features/sentiment_grid.py), averaged over assets,
    so the regression asks whether the agent's cash decision responds to the
    news state it was given.
    """
    from src.features.sentiment_grid import build_llm_grid  # lazy: heavy read

    grid = build_llm_grid(dates)
    direction = grid[:, :, config.LLM_CHANNELS.index("direction")]
    return pd.Series(direction.mean(axis=1), index=dates, name="news_index")


def _hac_sandwich(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients.

    The Bartlett kernel weights lag l by 1 - l/(lags+1), which keeps the
    estimated covariance positive semi-definite (Newey and West 1987). The
    sandwich is (X'X)^-1 S (X'X)^-1 with S the weighted sum of
    autocovariances of the score X_t * u_t.
    """
    scores = X * resid[:, None]
    S = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = scores[lag:].T @ scores[:-lag]
        S += weight * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


def news_responsiveness(cash_share: pd.Series, news: pd.Series,
                        vix_z: pd.Series, lags: int | None = None) -> dict:
    """Regress the daily cash share on the news state, controlling for the VIX.

        w_{t,0} = alpha + beta * g_t + gamma * vix_z_t + e_t

    beta is the responsiveness H4 is about. Conditioning on the volatility
    component of the market context matters because every arm observes the
    VIX: without it, beta would partly capture a reaction to volatility that
    has nothing to do with news. Standard errors are heteroskedasticity and
    autocorrelation consistent, since both the cash share and the smoothed
    news signal are strongly serially correlated by construction.

    Returns beta, its HAC standard error and t statistic, the VIX
    coefficient, the intercept, R^2, and the number of observations.
    """
    frame = pd.concat({"cash": cash_share, "news": news, "vix": vix_z},
                      axis=1).dropna()
    if len(frame) < 10:
        return {"beta": np.nan, "se": np.nan, "t": np.nan,
                "gamma": np.nan, "alpha": np.nan, "r2": np.nan,
                "n_obs": len(frame)}

    y = frame["cash"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(frame)),
                         frame["news"].to_numpy(dtype=float),
                         frame["vix"].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    cov = _hac_sandwich(X, resid, newey_west_lags(len(frame))
                        if lags is None else lags)
    se = np.sqrt(np.diag(cov))
    total = float(((y - y.mean()) ** 2).sum())
    return {
        "beta": float(coef[1]),
        "se": float(se[1]),
        "t": float(coef[1] / se[1]) if se[1] > 0 else np.nan,
        "gamma": float(coef[2]),
        "alpha": float(coef[0]),
        "r2": float(1.0 - (resid ** 2).sum() / total) if total > 0 else np.nan,
        "n_obs": int(len(frame)),
    }
