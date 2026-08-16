"""Allocation-based hypothesis evaluation: H2 and H5 (§ Statistical Inference).

H1/H3/H4 are return contrasts and live in contrasts.py. The two remaining
hypotheses are read from the ALLOCATIONS the policies produced:

  H2  — as risk aversion rises, the policy allocates a larger share to the
        risk-free asset. Supported when the cash share increases
        monotonically in eta -- the hypothesis claims the cash share and
        nothing else. Separation's further implications (lower volatility,
        smaller drawdowns, a risky composition that stays put relative to
        across-seed variation) are computed and reported as supporting
        evidence, not as conditions: the chapter examines them as
        implications rather than separate tests.

        The row also carries the ACROSS-SEED DISPERSION of the same cash
        share the verdict is read on: its standard deviation at each eta,
        the eta1->eta7 rise measured against a seed band built exactly like
        the return contrasts' (inference.seed_band), and how many individual
        seeds reproduce the monotone ordering. Every other hypothesis in the
        study is judged against a dispersion yardstick; a monotonicity rule
        over three point estimates has none, so the yardstick is reported
        beside it. These columns are DIAGNOSTIC ONLY -- `supported` remains
        the pre-registered monotonicity of the selected-seed sequence and no
        dispersion column enters it.

  H5  — news responsiveness: the coefficient on average smoothed directional
        sentiment in a regression of the daily cash share that controls for
        the VIX — the chapter's definition verbatim. Estimated per arm and
        eta with Newey–West HAC standard errors. The hypothesis predicts the
        responsiveness rises as the representation is enriched (M1 -> M6)
        and rises with eta.

Both operate on finished artifacts only: the selected seed's daily series
and weights panel for the headline numbers, the full test ledgers for the
across-seed reference. Nothing here retrains anything.
"""

import numpy as np
import pandas as pd

from src.evaluation import inference, metrics

H2_COLUMNS = [
    "algo", "arm", "cash_eta1", "cash_eta3", "cash_eta7",
    "vol_eta1", "vol_eta3", "vol_eta7", "mdd_eta1", "mdd_eta3", "mdd_eta7",
    "cash_monotone_up", "vol_monotone_down", "mdd_monotone_down",
    "cash_sd_eta1", "cash_sd_eta3", "cash_sd_eta7",
    "cash_rise_eta1_eta7", "cash_rise_band", "cash_rise_exceeds_band",
    "cash_rise_seed_mean", "cash_rise_seed_sd",
    "n_seeds", "n_seeds_cash_monotone", "n_seeds_cash_rising",
    "comp_dist_eta1_eta3", "comp_dist_eta3_eta7", "comp_dist_eta1_eta7",
    "seed_ref_eta1", "seed_ref_eta3", "seed_ref_eta7",
    "bench_eta1_eta3", "bench_eta3_eta7", "bench_eta1_eta7",
    "composition_stable", "supported",
]

H5_COLUMNS = [
    "algo", "arm", "eta", "news_index", "responsiveness_R", "t_R",
    "beta_sentiment", "se_sentiment", "t_sentiment", "beta_vix", "t_vix",
    "n_days", "hac_lags",
]


# ---------------------------------------------------------------------------
# Newey-West OLS (no statsmodels dependency)
# ---------------------------------------------------------------------------

def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS with Bartlett-kernel HAC standard errors (Newey and West 1987).

    `X` should NOT include a constant; one is prepended. Returns coefficient,
    standard error and t-statistic arrays ordered [const, x1, x2, ...], plus
    the lag count actually used (the same 4*(T/100)^(2/9) rule as the
    bootstrap's studentization).
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    n, k = X.shape
    if lags is None:
        lags = inference.newey_west_lags(n)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Z = X * u[:, None]
    S = Z.T @ Z
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Z[lag:].T @ Z[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": t, "lags": lags, "n": n}


# ---------------------------------------------------------------------------
# Composition distance (§ Benchmarks and Metrics)
# ---------------------------------------------------------------------------

# The rescaling lives in metrics.py, where the behaviour diagnostics also
# use it for the risky-only entropy and largest position; kept aliased here
# so the composition distance below reads the same as before.
_risky_rescaled = metrics.risky_rescaled


def composition_distance(weights_a: pd.DataFrame, weights_b: pd.DataFrame) -> float:
    """Distance between two AVERAGE risky compositions (eq:composition-distance).

    Each panel's risky weights are rescaled to sum to one, averaged over the
    evaluation period, and the distance is half the L1 difference of those
    two average compositions -- the one-way turnover needed to move one
    average allocation into the other, zero iff identical.

    The order of operations is the definition, not a detail. Averaging the
    daily distances instead gives a systematically larger number by Jensen's
    inequality (0.41 against 0.26 on PPO M1 here), because day-to-day
    rebalancing noise never cancels. Two policies with the same average
    composition but independent daily jitter are identical for this
    hypothesis and must score zero.
    """
    a, b = _risky_rescaled(weights_a), _risky_rescaled(weights_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    cols = a.columns.intersection(b.columns)
    return float((a[cols].mean() - b[cols].mean()).abs().sum() / 2.0)


def seed_composition_reference(ledgers_by_year: dict) -> float:
    """Across-seed composition distance at a FIXED eta: the noise yardstick.

    For each pair of seeds, build each seed's daily weight panel from the
    ledgers (w_* columns, the post-rebalance weights -- the same object the
    between-eta distances are computed from), pool the test years, and
    measure the composition distance; return the mean over pairs. H2's
    stability condition asks that the distance BETWEEN eta levels not
    exceed this within-eta spread.
    """
    if not ledgers_by_year:
        return float("nan")
    frames = pd.concat(ledgers_by_year.values())
    w_cols = [c for c in frames.columns if c.startswith("w_")]
    panels = {}
    for seed, g in frames.groupby("seed"):
        p = g.set_index("date")[w_cols]
        p.columns = [c[2:] for c in w_cols]
        panels[seed] = p
    seeds = sorted(panels)
    dists = [composition_distance(panels[a], panels[b])
             for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    dists = [d for d in dists if np.isfinite(d)]
    return float(np.mean(dists)) if dists else float("nan")


# ---------------------------------------------------------------------------
# Across-seed dispersion of the cash share (§ Robustness)
# ---------------------------------------------------------------------------

def per_seed_cash_share(ledgers_by_year: dict) -> dict[int, float]:
    """Pooled average cash share of every seed at one eta.

    Mirrors `contrasts.per_seed_sharpe`: concatenate the test years, then
    take one mean per seed, so a seed's number is formed exactly like the
    headline cash share and the two are directly comparable. The headline
    stitches the validation-selected seed window by window; a row here holds
    one seed fixed across all six, which is what a dispersion band asks.

    Read from the ledger's `w_CASH`, the post-trade weight that
    `test_daily.csv` reports as `cash_weight` -- verified identical on the
    selected seed. The drifted `wd_CASH` is a different quantity (it carries
    the intraday return) and is deliberately not used here.
    """
    if not ledgers_by_year:
        return {}
    frames = pd.concat(ledgers_by_year.values())
    if "seed" not in frames.columns or "w_CASH" not in frames.columns:
        return {}
    out = {}
    for seed, g in frames.groupby("seed"):
        value = float(g["w_CASH"].mean())
        if np.isfinite(value):
            out[int(seed)] = value
    return out


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------

def _monotone(values, increasing: bool) -> bool:
    v = [x for x in values if np.isfinite(x)]
    if len(v) < 2:
        return False
    pairs = zip(v, v[1:])
    return all(b >= a for a, b in pairs) if increasing \
        else all(b <= a for a, b in pairs)


def h2_rows(algo: str, by_arm: dict) -> list[dict]:
    """One row per arm: the separation predictions across eta.

    `by_arm[(arm, eta)]` must carry:
      "daily"   pooled daily frame with log_return and cash_weight columns,
      "weights" pooled selected-seed weights panel,
      "ledgers" {year: ledger frame} for the seed reference.
    Only etas 1, 3, 7 (the study's levels) are read.
    """
    etas = (1.0, 3.0, 7.0)
    arms = sorted({a for a, _ in by_arm})
    rows = []
    for arm in arms:
        have = [e for e in etas if (arm, e) in by_arm]
        if len(have) < 2:
            continue
        cash, vol, mdd, weights = {}, {}, {}, {}
        for e in have:
            d = by_arm[(arm, e)]["daily"]
            cash[e] = float(d["cash_weight"].mean())
            vol[e] = metrics.annualized_volatility(d["log_return"])
            mdd[e] = metrics.max_drawdown(d["log_return"])
            weights[e] = by_arm[(arm, e)]["weights"]

        def dist(e1, e2):
            if e1 in weights and e2 in weights and weights[e1] is not None \
                    and weights[e2] is not None:
                return composition_distance(weights[e1], weights[e2])
            return float("nan")

        # One reference per eta (eq:seed-composition); a pair is read against
        # the mean of its two levels, not against a single pooled scalar.
        ref = {e: seed_composition_reference(by_arm[(arm, e)].get("ledgers", {}))
               for e in have}

        def bench(e1, e2):
            r1, r2 = ref.get(e1, np.nan), ref.get(e2, np.nan)
            return float((r1 + r2) / 2.0) if np.isfinite(r1) and np.isfinite(r2) \
                else float("nan")

        d13, d37, d17 = dist(1.0, 3.0), dist(3.0, 7.0), dist(1.0, 7.0)
        b13, b37, b17 = bench(1.0, 3.0), bench(3.0, 7.0), bench(1.0, 7.0)
        pairs = [(d, b) for d, b in ((d13, b13), (d37, b37), (d17, b17))
                 if np.isfinite(d) and np.isfinite(b)]
        stable = bool(pairs and all(d <= b for d, b in pairs))

        cash_up = _monotone([cash.get(e, np.nan) for e in etas], increasing=True)
        vol_down = _monotone([vol.get(e, np.nan) for e in etas], increasing=False)
        mdd_down = _monotone([mdd.get(e, np.nan) for e in etas], increasing=False)

        # --- Across-seed dispersion of the cash share ----------------------
        # Diagnostic beside the verdict, never part of it. The rise is read
        # against the same band construction the return contrasts use, so
        # "exceeds the seed band" means the same thing in both places.
        seed_cash = {e: per_seed_cash_share(by_arm[(arm, e)].get("ledgers", {}))
                     for e in have}
        cash_sd = {}
        for e in have:
            values = np.asarray(list(seed_cash[e].values()), dtype=float)
            cash_sd[e] = float(values.std(ddof=1)) if values.size > 1 else np.nan
        rise = cash.get(7.0, np.nan) - cash.get(1.0, np.nan)
        rise_band = inference.seed_band(seed_cash.get(1.0, {}).values(),
                                        seed_cash.get(7.0, {}).values())
        rise_exceeds = bool(np.isfinite(rise) and np.isfinite(rise_band)
                            and rise > rise_band)

        # Seed index is a label shared across eta levels: the runs differ in
        # their reward but start from the same initialization, so a per-seed
        # difference is the paired counterpart of the band above and needs no
        # independence assumption. Its MEAN is the change an average seed
        # produced, against which the selected seed's rise can be read.
        shared = sorted(set.intersection(*[set(seed_cash[e]) for e in have])
                        if all(seed_cash[e] for e in have) else set())
        n_mono = sum(_monotone([seed_cash[e][s] for e in have], increasing=True)
                     for s in shared) if len(have) == len(etas) else 0
        paired = np.asarray(
            [seed_cash[7.0][s] - seed_cash[1.0][s] for s in shared]
            if {1.0, 7.0} <= set(have) else [], dtype=float)
        rise_seed_mean = float(paired.mean()) if paired.size else np.nan
        rise_seed_sd = float(paired.std(ddof=1)) if paired.size > 1 else np.nan
        n_rising = int((paired > 0).sum()) if paired.size else 0
        rows.append({
            "algo": algo.upper(), "arm": arm,
            "cash_eta1": cash.get(1.0, np.nan), "cash_eta3": cash.get(3.0, np.nan),
            "cash_eta7": cash.get(7.0, np.nan),
            "vol_eta1": vol.get(1.0, np.nan), "vol_eta3": vol.get(3.0, np.nan),
            "vol_eta7": vol.get(7.0, np.nan),
            "mdd_eta1": mdd.get(1.0, np.nan), "mdd_eta3": mdd.get(3.0, np.nan),
            "mdd_eta7": mdd.get(7.0, np.nan),
            "cash_monotone_up": cash_up, "vol_monotone_down": vol_down,
            "mdd_monotone_down": mdd_down,
            "cash_sd_eta1": cash_sd.get(1.0, np.nan),
            "cash_sd_eta3": cash_sd.get(3.0, np.nan),
            "cash_sd_eta7": cash_sd.get(7.0, np.nan),
            "cash_rise_eta1_eta7": rise, "cash_rise_band": rise_band,
            "cash_rise_exceeds_band": rise_exceeds,
            "cash_rise_seed_mean": rise_seed_mean,
            "cash_rise_seed_sd": rise_seed_sd,
            "n_seeds": len(shared), "n_seeds_cash_monotone": int(n_mono),
            "n_seeds_cash_rising": n_rising,
            "comp_dist_eta1_eta3": d13, "comp_dist_eta3_eta7": d37,
            "comp_dist_eta1_eta7": d17,
            "seed_ref_eta1": ref.get(1.0, np.nan),
            "seed_ref_eta3": ref.get(3.0, np.nan),
            "seed_ref_eta7": ref.get(7.0, np.nan),
            "bench_eta1_eta3": b13, "bench_eta3_eta7": b37,
            "bench_eta1_eta7": b17,
            "composition_stable": stable,
            # The hypothesis claims the cash share; everything else in this
            # row is separation's implications, reported as evidence.
            "supported": bool(cash_up),
        })
    return rows


# ---------------------------------------------------------------------------
# H5
# ---------------------------------------------------------------------------

def h5_rows(algo: str, by_arm: dict, sentiment: pd.Series, vix: pd.Series,
            sentiment_by_arm: dict | None = None,
            default_index_name: str = "llm") -> list[dict]:
    """News responsiveness per (arm, eta): cash share on sentiment + VIX.

    `sentiment` is the common regressor -- the average smoothed directional
    LLM signal -- used for every arm that shares the LLM extraction. Because
    M1 through M6 then face the same regressor, differences in the
    coefficient measure differences in RESPONSE rather than in information
    sets, which is what the enrichment ladder of H5 compares.

    M2 is the exception and is why `sentiment_by_arm` exists. It observes the
    FinBERT channel and never sees the LLM signal, so regressing its cash
    share on the LLM index would measure a response to information the policy
    was not given. Pass `{"M2": finbert_index}` to estimate it against its own
    news index instead; the chapter does this and consequently excludes M2
    from every Delta R comparison, since a coefficient on a different
    regressor is not comparable with the rest of the ladder.

    Each row records the index it used in `news_index`, so a reader can tell
    the two estimations apart in the artifact rather than inferring it.
    """
    overrides = sentiment_by_arm or {}
    rows = []
    for (arm, eta), data in sorted(by_arm.items()):
        d = data["daily"]
        series = overrides.get(arm, sentiment)
        frame = pd.DataFrame({
            "cash": d["cash_weight"],
            "sent": series.reindex(d.index),
            "vix": vix.reindex(d.index),
        }).dropna()
        if len(frame) < 60:
            continue
        fit = newey_west_ols(frame["cash"].to_numpy(),
                             frame[["sent", "vix"]].to_numpy())
        rows.append({
            "algo": algo.upper(), "arm": arm, "eta": eta,
            "news_index": (series.name if arm in overrides and series.name
                           else "finbert" if arm in overrides
                           else default_index_name),
            # The chapter reports R = -beta_1 (eq:news-response): a policy
            # that moves INTO cash on bad news has negative beta and positive
            # responsiveness. Both are stored so no reader has to infer the
            # flip from the sign of a coefficient.
            "responsiveness_R": -float(fit["beta"][1]),
            "t_R": -float(fit["t"][1]),
            "beta_sentiment": float(fit["beta"][1]),
            "se_sentiment": float(fit["se"][1]),
            "t_sentiment": float(fit["t"][1]),
            "beta_vix": float(fit["beta"][2]), "t_vix": float(fit["t"][2]),
            "n_days": int(fit["n"]), "hac_lags": int(fit["lags"]),
        })
    return rows


H5_DIFF_COLUMNS = [
    "algo", "kind", "step", "eta", "arm",
    "delta_R", "se", "t", "n_days", "hac_lags",
]

_LADDER_STEPS = [("M4", "M3"), ("M5", "M4"), ("M6", "M5")]
_ETA_STEPS = [(1.0, 3.0), (3.0, 7.0)]


def h5_difference_rows(algo: str, by_arm: dict, sentiment: pd.Series,
                       vix: pd.Series) -> list[dict]:
    """Differences in news responsiveness (eq:responsiveness-difference).

    Where two policies are regressed on the same news index and control,
    the difference in their responsiveness follows from one regression of
    the DIFFERENCE in their cash shares on those same regressors: the slope
    is delta_1 = beta^A_1 - beta^B_1, so Delta R = R_A - R_B = -delta_1,
    and the Newey-West standard error of delta_1 accounts for the covariance
    between the two policies' residuals, which treating the coefficients as
    independent would ignore. Applied to the representation-ladder steps at
    each eta (upper block of tab:res-h5-dr) and to the eta steps within each
    LLM arm (lower block). M2 is excluded throughout because it is estimated
    against a different news index (see h5_rows).
    """
    def diff_fit(key_hi, key_lo):
        hi = by_arm.get(key_hi, {}).get("daily")
        lo = by_arm.get(key_lo, {}).get("daily")
        if hi is None or lo is None:
            return None
        frame = pd.DataFrame({
            "dcash": hi["cash_weight"] - lo["cash_weight"],
            "sent": sentiment,
            "vix": vix,
        }).dropna()
        if len(frame) < 60:
            return None
        return newey_west_ols(frame["dcash"].to_numpy(),
                              frame[["sent", "vix"]].to_numpy())

    rows = []
    for eta in (1.0, 3.0, 7.0):
        for arm_hi, arm_lo in _LADDER_STEPS:
            fit = diff_fit((arm_hi, eta), (arm_lo, eta))
            if fit is None:
                continue
            rows.append({
                "algo": algo.upper(), "kind": "ladder",
                "step": f"{arm_hi}-{arm_lo}", "eta": eta, "arm": "",
                "delta_R": -float(fit["beta"][1]),
                "se": float(fit["se"][1]), "t": -float(fit["t"][1]),
                "n_days": int(fit["n"]), "hac_lags": int(fit["lags"]),
            })
    for arm in ("M3", "M4", "M5", "M6"):
        for eta_lo, eta_hi in _ETA_STEPS:
            fit = diff_fit((arm, eta_hi), (arm, eta_lo))
            if fit is None:
                continue
            rows.append({
                "algo": algo.upper(), "kind": "eta",
                "step": f"eta{eta_lo:g}-{eta_hi:g}", "eta": np.nan,
                "arm": arm,
                "delta_R": -float(fit["beta"][1]),
                "se": float(fit["se"][1]), "t": -float(fit["t"][1]),
                "n_days": int(fit["n"]), "hac_lags": int(fit["lags"]),
            })
    return rows
