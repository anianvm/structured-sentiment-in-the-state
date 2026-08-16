"""The pre-registered arm contrasts (§ Statistical Inference).

The performance tables describe what each arm did; this module decides
whether the differences between them mean anything. The pre-registered
contrast set covers the three return hypotheses:

  H1   every LLM-based arm against the price-only control (M3..M6 vs M1),
  H3   the LLM one-channel arm against the FinBERT arm (M3 vs M2),
  H4   each enrichment step of the structured representation
       (M4 vs M3, M5 vs M4, M6 vs M5),
  SUPP the FinBERT arm against the control (M2 vs M1) -- not a hypothesis:
       M2's role in the design is H3's extraction comparator, so its
       comparison against M1 is computed and reported as a supplementary
       descriptive contrast, outside every family.

nine contrasts per risk-aversion level, all predicted positive. The decision
rule (amended by the authors on 2026-08-08 from the chapter's original four
conditions) counts a contrast as support when

  1. the ONE-SIDED (H0: difference <= 0) Holm-adjusted bootstrap p-value is
     below alpha -- the directional H0 subsumes the original sign condition,
  2. the difference exceeds the cross-seed band.

The leave-one-year-out influence check is computed and reported (strict
`regime_robust` and weak `loo_sign_stable`) as a diagnostic on
generalisability, no longer as a gate: the block bootstrap cannot detect a
single year carrying the pooled effect (every resample contains that year),
so the diagnostic answers a question significance cannot -- but it is a
caveat on interpretation, not a condition for existence.

Multiplicity is handled per hypothesis, within each risk-aversion level:

  H1   is a genuine simultaneous family -- four contrasts, any of which
       would be claimed as support -- so Holm is applied over the four.
  H3   is a single pre-registered contrast; with m = 1 Holm is the identity,
       so the raw one-sided p is used.
  H4   is a MONOTONICITY claim: support requires every enrichment step to
       reject individually (an intersection-union test), which is
       conservative at level alpha WITHOUT any correction; correcting would
       be required only if a single rejecting step were claimed as support.
       Each step is therefore judged at its raw one-sided p, and the
       hypothesis-level verdict is the conjunction of the three rows.

eta = 1 is the primary level, where log rewards aggregate exactly to
terminal log wealth; eta = 3 and 7 are computed identically but flagged
exploratory.

The seed band uses each arm's five test-set Sharpe ratios, one per seed,
which exist because the runner rolls out every seed on the test set and not
only the selected one. Without those the band would have to be guessed.
Note the asymmetry this creates and keep it in mind when reading the output:
the CONTRAST is measured on the selected seed (the protocol's headline
strategy), while the BAND comes from all five. That is the intended
comparison -- how big is the treatment effect against how much the seed
alone moves performance -- but the two numbers come from different objects.
"""

import numpy as np
import pandas as pd

from src.evaluation import metrics
from src.evaluation.inference import (
    contrast_verdict, holm_adjust, leave_one_window_out, seed_band,
)

CONTRAST_COLUMNS = [
    "algo", "eta", "hypothesis", "arm", "baseline", "sharpe_arm",
    "sharpe_baseline", "difference", "ci_low", "ci_high", "p_value",
    "p_one_sided", "p_holm", "seed_band", "exceeds_seed_band", "regime_robust",
    "correct_sign", "significant", "supported", "loo_sign_stable",
    "n_days", "family", "exploratory",
]

# The same contrast set read on the two downside measures. Descriptive only:
# sec:inference-meth fixes the Sharpe difference as the test statistic, so
# these rows carry no p-value, no multiplicity adjustment and no verdict.
DOWNSIDE_COLUMNS = [
    "algo", "eta", "hypothesis", "arm", "baseline", "measure",
    "value_arm", "value_baseline", "difference",
    "seed_mean_arm", "seed_mean_baseline", "difference_seed_mean",
    "seed_sd_arm", "seed_sd_baseline", "seed_band", "exceeds_seed_band",
    "n_days", "exploratory",
]

DOWNSIDE_MEASURES = ("max_drawdown", "cvar_5")

# The contrast set: (arm, baseline, hypothesis). All predicted positive;
# SUPP is reported, never gated (see the module docstring).
CONTRAST_SET = [
    ("M3", "M1", "H1"), ("M4", "M1", "H1"),
    ("M5", "M1", "H1"), ("M6", "M1", "H1"),
    ("M3", "M2", "H3"),
    ("M4", "M3", "H4"), ("M5", "M4", "H4"), ("M6", "M5", "H4"),
    ("M2", "M1", "SUPP"),
]


def per_seed_sharpe(ledgers: dict, tbill_annual) -> list[float]:
    """Annualized test Sharpe for each seed, pooled across test years.

    `ledgers` maps test year to that year's ledger_test frame, which holds
    every seed's rollout. Seeds are pooled the same way the headline record
    is: concatenate the years, then compute one Sharpe per seed.
    """
    if not ledgers:
        return []
    frames = pd.concat(ledgers.values())
    out = []
    for _, g in frames.groupby("seed"):
        s = g.sort_values("date").set_index("date")["log_return"]
        out.append(metrics.sharpe_ratio(s, tbill_annual))
    return [v for v in out if np.isfinite(v)]


def per_seed_downside(ledgers: dict) -> dict:
    """Pooled maximum drawdown and CVaR for each seed.

    Built exactly like per_seed_sharpe -- concatenate the test years, then
    one value per seed -- so the resulting spreads feed the same seed band.
    Returns {measure: [value per seed]}, with non-finite values dropped.
    """
    out = {m: [] for m in DOWNSIDE_MEASURES}
    if not ledgers:
        return out
    frames = pd.concat(ledgers.values())
    for _, g in frames.groupby("seed"):
        s = g.sort_values("date").set_index("date")["log_return"]
        summary = metrics.behaviour_tail_summary(s)
        for measure in DOWNSIDE_MEASURES:
            value = summary[measure]
            if np.isfinite(value):
                out[measure].append(float(value))
    return out


def _pooled_downside(by_year: dict) -> dict:
    """The selected seed's pooled drawdown and CVaR, or NaN without data."""
    if not by_year:
        return {m: float("nan") for m in DOWNSIDE_MEASURES}
    series = pd.concat(by_year.values()).sort_index()
    return metrics.behaviour_tail_summary(series)


def downside_contrasts(algo: str, by_arm_daily: dict,
                       by_arm_ledger: dict) -> list[dict]:
    """The pre-registered contrast set read on drawdown and CVaR.

    Mirrors arm_contrasts in inputs and in how each quantity is formed: the
    contrast is the difference between the two arms' SELECTED-seed pooled
    values, and the band is built from all five seeds of each arm. Two
    differences from the Sharpe table matter when reading the output.

    First, no inference. The maximum drawdown is a functional of the wealth
    PATH rather than of the return distribution, so the studentized
    bootstrap of inference.py does not apply to it, and CVaR averages a
    5%-tail whose sampling error over this sample is large. Neither carries a
    p-value, a Holm adjustment or a verdict; the band is the only yardstick.

    Second, `exceeds_seed_band` compares the ABSOLUTE difference. The
    hypotheses predict a direction for the Sharpe ratio, not for these
    measures, and a contrast can move downside risk either way, so the
    question here is only whether the movement is larger than training noise.

    A seed-mean difference is reported beside the selected-seed one. Where
    they disagree in sign the selected-seed reading reflects which seeds
    validation happened to pick rather than a property of the arms, which is
    a distinction the Sharpe table cannot show.
    """
    etas = sorted({eta for _, eta in by_arm_daily})
    rows = []
    for eta in etas:
        # Each arm appears in several contrasts; compute its pooled values
        # and its per-seed spreads once per eta.
        pooled, seeds = {}, {}

        def load(key):
            if key not in pooled and key in by_arm_daily:
                pooled[key] = _pooled_downside(by_arm_daily[key])
                seeds[key] = per_seed_downside(by_arm_ledger.get(key, {}))
            return key in pooled

        for arm, baseline, hypothesis in CONTRAST_SET:
            key, base_key = (arm, eta), (baseline, eta)
            if not (load(key) and load(base_key)):
                continue
            n_days = sum(len(s) for s in by_arm_daily[key].values())
            for measure in DOWNSIDE_MEASURES:
                a, b = pooled[key][measure], pooled[base_key][measure]
                sa = np.asarray(seeds[key][measure], dtype=float)
                sb = np.asarray(seeds[base_key][measure], dtype=float)
                band = seed_band(sa, sb)
                difference = float(a - b)
                rows.append({
                    "algo": algo.upper(), "eta": eta,
                    "hypothesis": hypothesis, "arm": arm, "baseline": baseline,
                    "measure": measure,
                    "value_arm": float(a), "value_baseline": float(b),
                    "difference": difference,
                    "seed_mean_arm": float(sa.mean()) if sa.size else float("nan"),
                    "seed_mean_baseline": float(sb.mean()) if sb.size else float("nan"),
                    "difference_seed_mean": float(sa.mean() - sb.mean())
                    if sa.size and sb.size else float("nan"),
                    "seed_sd_arm": float(sa.std(ddof=1)) if sa.size > 1 else float("nan"),
                    "seed_sd_baseline": float(sb.std(ddof=1)) if sb.size > 1 else float("nan"),
                    "seed_band": band,
                    "exceeds_seed_band": bool(np.isfinite(band)
                                              and abs(difference) > band),
                    "n_days": n_days,
                    "exploratory": bool(eta != 1.0),
                })
    return rows


def arm_contrasts(algo: str, by_arm_daily: dict, by_arm_ledger: dict,
                  tbill_annual, alpha: float = 0.05,
                  draws: int | None = None) -> list[dict]:
    """Every sentiment arm against price-only, at each risk-aversion level.

    `by_arm_daily[(arm, eta)]` maps test year -> daily log returns of the
    selected seed; `by_arm_ledger[(arm, eta)]` maps test year -> that year's
    full ledger. Returns one row per contrast, Holm-adjusted within each eta
    family.
    """
    etas = sorted({eta for _, eta in by_arm_daily})
    rows = []
    for eta in etas:
        family = []
        for arm, baseline, hypothesis in CONTRAST_SET:
            key, base_key = (arm, eta), (baseline, eta)
            if key not in by_arm_daily or base_key not in by_arm_daily:
                continue
            loo_kwargs = {"alpha": alpha}
            if draws is not None:
                loo_kwargs["draws"] = draws
            loo = leave_one_window_out(by_arm_daily[key], by_arm_daily[base_key],
                                       tbill_annual, **loo_kwargs)
            full = loo["full"]
            if full is None:                      # fewer than two test years
                continue
            band = seed_band(
                per_seed_sharpe(by_arm_ledger.get(key, {}), tbill_annual),
                per_seed_sharpe(by_arm_ledger.get(base_key, {}), tbill_annual))
            family.append({
                "algo": algo.upper(), "eta": eta, "hypothesis": hypothesis,
                "arm": arm, "baseline": baseline,
                "sharpe_arm": full["sharpe_a"], "sharpe_baseline": full["sharpe_b"],
                "difference": full["difference"], "ci_low": full["ci_low"],
                "ci_high": full["ci_high"], "p_value": full["p_value"],
                "p_one_sided": full["p_one_sided"],
                "seed_band": band, "regime_robust": loo["survives"],
                "loo_sign_stable": loo["sign_stable"],
                "n_days": full.get("n_obs", np.nan),
                "family": f"eta={eta}",
                "exploratory": bool(eta != 1.0),
            })
        if not family:
            continue
        # Holm over the H1 subfamily alone; H3 and H4 rows carry their raw
        # one-sided p (see the module docstring for why that is correct for
        # each). p_holm is the per-row value the verdict gates on either way.
        h1 = [r for r in family if r["hypothesis"] == "H1"]
        adjusted = dict(zip((id(r) for r in h1),
                            holm_adjust([r["p_one_sided"] for r in h1])))
        for row in family:
            p_adj = adjusted.get(id(row), row["p_one_sided"])
            row["p_holm"] = float(p_adj)
            verdict = contrast_verdict(row["difference"], p_adj, row["seed_band"],
                                       row["regime_robust"], predicted_positive=True,
                                       alpha=alpha)
            row.update({k: verdict[k] for k in
                        ("supported", "correct_sign", "significant",
                         "exceeds_seed_band")})
        rows.extend(family)
    return rows
