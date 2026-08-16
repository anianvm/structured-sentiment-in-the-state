"""Statistical inference for the hypotheses (§ Statistical Inference).

Two tools, matching what the chapter pre-registers:

  * SHARPE_DIFFERENCE_TEST — the studentized time-series bootstrap of Ledoit
    and Wolf (2008) for the difference between two annualized Sharpe ratios.
    The conventional test of Jobson and Korkie (1981), corrected by Memmel
    (2003) and used by DeMiguel et al. (2009), assumes returns that are
    neither fat-tailed nor serially dependent; daily equity returns are both,
    which is precisely the case Ledoit and Wolf address.

  * HOLM_ADJUST — the step-down multiple-testing correction of Holm (1979),
    applied within the primary family of arm contrasts at eta = 1. Harvey and
    Liu (2015) recommend exactly this family of procedures when many strategy
    variants are compared on one sample.

Both operate on the per-step ledgers written by the walk-forward runner, so
the whole inference stage is post-hoc: nothing here requires retraining, and
a revised decision rule can be applied to finished runs.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import TRADING_DAYS

DEFAULT_BLOCK = 5        # bootstrap block length in trading days (one week)
DEFAULT_DRAWS = 4999     # bootstrap replications


# ---------------------------------------------------------------------------
# Sharpe-ratio difference (H1, H3, H4)
# ---------------------------------------------------------------------------

def _excess_simple(log_returns: pd.Series, tbill_annual) -> pd.Series:
    """Daily SIMPLE returns in excess of the 13-week T-bill.

    Mirrors metrics.sharpe_ratio so that the test and the reported Sharpe
    ratio are computed on the identical series.
    """
    simple = np.expm1(log_returns)
    if isinstance(tbill_annual, pd.Series):
        daily = (tbill_annual.reindex(simple.index, method="ffill").bfill()
                 / TRADING_DAYS)
    else:
        daily = float(tbill_annual) / TRADING_DAYS
    return simple - daily


def newey_west_lags(n_obs: int) -> int:
    """Automatic bandwidth floor(4 (n/100)^(2/9)) (Newey and West 1994).

    The single definition for the whole evaluation stage: the Sharpe test's
    studentization and the responsiveness regressions of `allocation` and
    `hypotheses` all import it from here, so the rule cannot drift between
    the places the write-up claims are using the same one.
    """
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def _moments_and_grad(r1: np.ndarray, r2: np.ndarray):
    """Sharpe difference, its delta-method gradient, and the moment vector.

    With v = (mu1, mu2, gamma1, gamma2) and gamma the second RAW moment,
        f(v) = mu1 / sqrt(gamma1 - mu1^2) - mu2 / sqrt(gamma2 - mu2^2),
    which is Ledoit and Wolf (2008) Eq. (2); `grad` is their Eq. (4). A
    constant return series has no Sharpe ratio at all, so the difference is
    undefined rather than merely imprecise and (nan, None, None) comes back.
    """
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    var1, var2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    if var1 <= 0 or var2 <= 0:
        return np.nan, None, None
    diff = mu1 / np.sqrt(var1) - mu2 / np.sqrt(var2)
    grad = np.array([
        g1 / var1 ** 1.5,            # d/d mu1
        -g2 / var2 ** 1.5,           # d/d mu2
        -mu1 / (2 * var1 ** 1.5),    # d/d gamma1
        mu2 / (2 * var2 ** 1.5),     # d/d gamma2
    ])
    return float(diff), grad, (mu1, mu2, g1, g2)


def _residuals(r1: np.ndarray, r2: np.ndarray, moments) -> np.ndarray:
    """Moment residuals y_t of Ledoit and Wolf Section 3.1, centred on `moments`."""
    mu1, mu2, g1, g2 = moments
    return np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])


def _hac_moment_cov(V: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West covariance of the moment residuals (Bartlett kernel).

    Carries Ledoit and Wolf's T/(T-4) small-sample adjustment, which offsets
    the estimation of the 4-vector v. Bartlett rather than their prewhitened
    QS kernel: it is the positive-semi-definite choice, and Remark 4.1 of the
    paper reports that substituting kernels leaves their results virtually
    identical. Used for the ORIGINAL sample only -- see `_natural_moment_cov`
    for what the bootstrap world requires instead.
    """
    n = len(V)
    S = V.T @ V / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = V[lag:].T @ V[:-lag] / n
        S += weight * (G + G.T)
    return S * (n / (n - 4.0))


def _natural_moment_cov(V: np.ndarray, block: int):
    """The 'natural' covariance for circular-block bootstrap data (LW 3.2.2).

    A resample is a concatenation of independent blocks, so its dependence
    structure is not unknown -- it is something we built. Psi* therefore reads
    straight off the block sums instead of being re-estimated by a kernel,
    which is the whole point: inside the bootstrap world the answer is known.

    Normalized by the block count l, NOT by T as the paper's formula prints.
    Eq. (5) requires Psi*/T = Var*(v-hat*) = Var*(S)/(l b^2), hence
    Psi* = Var*(S)/b = (1/l) sum_j zeta_j zeta_j'. Dividing by T instead
    understates Psi* by exactly b and inflates the studentized draws by
    sqrt(b); their footnote-9 check cannot catch it because b = 1 makes l = T.
    The diagnostic is that the draws must have unit standard deviation.
    """
    n_blocks = len(V) // block
    if n_blocks < 2:
        return None
    zeta = (V[:n_blocks * block].reshape(n_blocks, block, V.shape[1])
            .sum(axis=1) / np.sqrt(block))
    return zeta.T @ zeta / n_blocks


def _se_from_cov(grad: np.ndarray, cov: np.ndarray, n: int) -> float:
    """se = sqrt(grad' Psi grad / T), Ledoit and Wolf Eqs. (5) and (8).

    A zero variance means the two series move together exactly (the limiting
    case being A against itself), so the difference is not an estimate and has
    no standard error.
    """
    variance = float(grad @ cov @ grad / n)
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def _circular_blocks(n: int, block: int, rng) -> np.ndarray:
    """Index vector for one circular block-bootstrap resample.

    Blocks wrap around the end of the sample, so every observation has the
    same chance of being drawn, and resampling whole blocks preserves the
    short-range serial dependence that invalidates the iid tests.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()
    return idx[:n]


def sharpe_difference_test(log_returns_a: pd.Series, log_returns_b: pd.Series,
                           tbill_annual, block: int = DEFAULT_BLOCK,
                           draws: int = DEFAULT_DRAWS, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """Test whether strategy A's Sharpe ratio differs from strategy B's.

    Implements the studentized circular block bootstrap of Ledoit and Wolf
    (2008): the studentized statistic is resampled, recentred on the observed
    difference, and the two-sided p-value is the share of resamples whose
    studentized statistic is at least as extreme as the observed one. The
    confidence interval inverts the same distribution, so it is consistent
    with the p-value by construction.

    The two standard errors come from different estimators, as the paper
    requires: the original sample gets the kernel HAC of `_hac_moment_cov`,
    each resample gets the natural block-based one of `_natural_moment_cov`.
    Every resample is studentized by its OWN standard error, which is the
    step Remark 3.3 of the paper identifies as the one earlier bootstrap
    treatments of the Sharpe ratio got wrong.

    Returns the ANNUALIZED Sharpe difference and interval (the reported
    scale), plus the p-value and the two individual Sharpe ratios.
    """
    a, b = log_returns_a.align(log_returns_b, join="inner")
    r1 = _excess_simple(a, tbill_annual).to_numpy(dtype=float)
    r2 = _excess_simple(b, tbill_annual).to_numpy(dtype=float)
    n = len(r1)
    scale = np.sqrt(TRADING_DAYS)          # daily -> annualized Sharpe
    lags = newey_west_lags(n)

    sharpe_a = float(r1.mean() / r1.std(ddof=1) * scale) if r1.std(ddof=1) > 0 \
        else np.nan
    sharpe_b = float(r2.mean() / r2.std(ddof=1) * scale) if r2.std(ddof=1) > 0 \
        else np.nan

    diff, grad, moments = _moments_and_grad(r1, r2)
    se = np.nan if grad is None else _se_from_cov(
        grad, _hac_moment_cov(_residuals(r1, r2, moments), lags), n)
    if not np.isfinite(diff):                       # Sharpe itself undefined
        return {"sharpe_a": sharpe_a, "sharpe_b": sharpe_b,
                "difference": np.nan, "p_value": np.nan, "p_one_sided": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_obs": n, "draws": 0}
    if not np.isfinite(se) or se == 0:
        # Perfectly co-moving series: the difference is exact rather than
        # estimated, so there is nothing to bootstrap. A zero difference is
        # then certainly zero (p = 1); a non-zero one is certainly non-zero.
        return {"sharpe_a": sharpe_a, "sharpe_b": sharpe_b,
                "difference": float(diff * scale),
                "p_value": 1.0 if diff == 0 else 0.0,
                "p_one_sided": 1.0 if diff <= 0 else 0.0,
                "ci_low": float(diff * scale), "ci_high": float(diff * scale),
                "n_obs": n, "draws": 0}

    rng = np.random.default_rng(seed)
    studentized = []
    for _ in range(draws):
        idx = _circular_blocks(n, block, rng)
        x1, x2 = r1[idx], r2[idx]
        d_b, grad_b, _ = _moments_and_grad(x1, x2)
        if grad_b is None:
            continue
        # The resample's residuals are centred on the ORIGINAL-sample moments:
        # under the circular block bootstrap those are the bootstrap world's
        # true values. The gradient is evaluated at the resample's own moments
        # (Ledoit and Wolf Eq. 8).
        cov_b = _natural_moment_cov(_residuals(x1, x2, moments), block)
        if cov_b is None:
            continue
        se_b = _se_from_cov(grad_b, cov_b, n)
        if np.isfinite(se_b) and se_b > 0:
            studentized.append((d_b - diff) / se_b)
    studentized = np.asarray(studentized)          # SIGNED draws
    if studentized.size == 0:
        return {"sharpe_a": np.nan, "sharpe_b": np.nan, "difference": np.nan,
                "p_value": np.nan, "p_one_sided": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n_obs": n, "draws": 0}

    t_obs = diff / se
    abs_draws = np.abs(studentized)
    # The +1 in numerator and denominator keeps the p-value strictly positive:
    # with B draws it can never be 0. This is Eq. (9) of Ledoit and Wolf, who
    # give it for the two-sided statistic; `p_one` below is its directional
    # analogue, which the paper does not state.
    p_value = (1.0 + np.sum(abs_draws >= abs(t_obs))) / (1.0 + studentized.size)
    # One-sided test of H0: difference <= 0 against the directional H1 the
    # hypotheses are stated in. The upper tail of the SIGNED studentized
    # distribution is compared against the signed observed statistic, so a
    # correct-sign effect roughly halves its p-value and a wrong-sign effect
    # can never look significant -- the sign condition is subsumed here.
    p_one = (1.0 + np.sum(studentized >= t_obs)) / (1.0 + studentized.size)
    critical = float(np.quantile(abs_draws, 1.0 - alpha))
    return {
        "sharpe_a": sharpe_a,
        "sharpe_b": sharpe_b,
        "difference": float(diff * scale),
        "p_value": float(p_value),
        "p_one_sided": float(p_one),
        "ci_low": float((diff - critical * se) * scale),
        "ci_high": float((diff + critical * se) * scale),
        "n_obs": n,
        "draws": int(studentized.size),
    }


# ---------------------------------------------------------------------------
# Multiplicity (the primary family of arm contrasts at eta = 1)
# ---------------------------------------------------------------------------

def holm_adjust(p_values) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, in the input order.

    Sort ascending, multiply the k-th smallest of m by (m - k), and enforce
    monotonicity by carrying the running maximum forward, so an adjusted
    p-value can never fall below one that precedes it. Comparing the adjusted
    values to the nominal level controls the family-wise error rate, and Holm
    dominates Bonferroni: it rejects whenever Bonferroni does, and sometimes
    more.
    """
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


# ---------------------------------------------------------------------------
# The two remaining criteria of the decision rule (§ Statistical Inference)
# ---------------------------------------------------------------------------

def seed_band(seeds_a, seeds_b) -> float:
    """Spread a contrast must clear to count as more than training noise.

    Pass the per-seed values of the SAME statistic the contrast is measured
    in — annualized Sharpe for the performance hypotheses — for the two arms
    being compared. Each arm's seeds estimate how much its performance moves
    when only the random seed changes; treating the two as independent, the
    standard deviation of a seed-driven difference is the root of the summed
    variances. A contrast smaller than this cannot be attributed to the
    treatment, whatever its p-value \\parencite{Agarwal2021DeepPrecipice}.

    Returns NaN when either arm has fewer than two seeds, since no spread can
    be estimated from one run.
    """
    a = np.asarray(list(seeds_a), dtype=float)
    b = np.asarray(list(seeds_b), dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(np.sqrt(a.var(ddof=1) + b.var(ddof=1)))


def leave_one_window_out(by_year_a: dict, by_year_b: dict, tbill_annual,
                         block: int = DEFAULT_BLOCK,
                         draws: int = DEFAULT_DRAWS, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """Re-run the Sharpe contrast six times, each omitting one test year.

    The pooled out-of-sample record concatenates the annual test periods, so
    an aggregate difference could in principle come from a single unusual
    market regime. Dropping each year in turn and repeating the test answers
    that directly: if the conclusion depends on one year, one of the six
    refits will disagree.

    `by_year_a` and `by_year_b` map test year to that year's daily log
    returns. Years present in only one of the two are ignored, since a
    contrast needs both arms.

    Returns the full-sample result, one result per omitted year, and
    `survives`: True when every refit keeps the sign of the full-sample
    difference and excludes zero from its interval.
    """
    years = sorted(set(by_year_a) & set(by_year_b))
    if len(years) < 2:
        return {"survives": False, "full": None, "by_omitted_year": {},
                "reason": "need at least two test years"}

    def pooled(mapping, keep):
        return pd.concat([mapping[y] for y in keep]).sort_index()

    full = sharpe_difference_test(pooled(by_year_a, years),
                                  pooled(by_year_b, years), tbill_annual,
                                  block=block, draws=draws, seed=seed,
                                  alpha=alpha)
    results = {}
    survives = sign_stable = np.isfinite(full["difference"])
    for omitted in years:
        keep = [y for y in years if y != omitted]
        r = sharpe_difference_test(pooled(by_year_a, keep),
                                   pooled(by_year_b, keep), tbill_annual,
                                   block=block, draws=draws, seed=seed,
                                   alpha=alpha)
        results[omitted] = r
        same_sign = np.sign(r["difference"]) == np.sign(full["difference"])
        excludes_zero = (r["ci_low"] > 0) or (r["ci_high"] < 0)
        sign_stable = sign_stable and bool(same_sign)
        survives = survives and bool(same_sign and excludes_zero)

    # `survives` is the strict reading of the chapter's "it remains when any
    # single test year is removed": every refit keeps the sign AND its
    # interval excludes zero. `sign_stable` is the weak reading -- the sign
    # alone never flips. The strict reading embeds a significance requirement
    # in every 5/6 subsample, so in an under-powered sample it fails whenever
    # criterion 2 fails; reporting both lets the write-up separate "the
    # effect depends on one year" from "the sample cannot certify it".
    return {"survives": bool(survives), "sign_stable": bool(sign_stable),
            "full": full, "by_omitted_year": results, "reason": ""}


def contrast_verdict(difference: float, adjusted_p: float, band: float,
                     survives_loo: bool, predicted_positive: bool = True,
                     alpha: float = 0.05) -> dict:
    """Apply the decision rule to one contrast (amended 2026-08-08).

    A hypothesis is supported when the ONE-SIDED (H0: difference <= 0),
    Holm-adjusted bootstrap test rejects at `alpha` and the effect exceeds
    the seed-dispersion band. The sign condition of the original rule is
    subsumed by the one-sided test -- a wrong-sign effect cannot reject H0 --
    and the leave-one-year-out influence check is reported as a diagnostic
    rather than gating support: it answers whether a supported effect
    generalises beyond one regime, which is a caveat on interpretation, not
    on existence. Sign and influence flags are still returned so tables can
    show them.
    """
    checks = {
        "significant": bool(np.isfinite(adjusted_p) and adjusted_p < alpha),
        "exceeds_seed_band": bool(np.isfinite(band) and abs(difference) > band),
    }
    reported = {
        "correct_sign": (difference > 0) if predicted_positive else (difference < 0),
        "regime_robust": bool(survives_loo),
    }
    return {"supported": all(checks.values()), **checks, **reported}
