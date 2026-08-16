"""Optimization diagnostics of the walk-forward training runs.

Reads the Stable Baselines3 progress logs written by step 5 (one
``sb3_log_seed{s}.csv`` per arm x eta x window x seed, 540 per algorithm) and
reduces them to two tables and one figure:

* ``{ALGO}_training_diagnostics.csv``  one row per run: the per-run scalar
  summaries (explained variance, action std, critic loss, temperature, ...).
* ``{ALGO}_training_summary.csv``      one row per metric: the distribution of
  those summaries over the 540 runs, plus the within-cell across-seed SD.
* ``fig_training_diagnostics``         PPO explained variance and SAC critic
  loss against training steps, median with an interquartile band over runs.

Why the two algorithms carry different metrics: PPO is on-policy, so empirical
return targets exist and SB3 logs ``train/explained_variance``, the
scale-free measure of how much of their variation the critic tracks. SAC is
off-policy and regresses on bootstrapped TD targets, so there is no empirical
return to explain and no explained variance to log; ``train/critic_loss`` is
the only critic-fit signal it exposes. See the thesis methodology chapter,
sec:gamma-meth.

Dispersion is reported three ways because they answer different questions.
The interquartile range over all 540 runs is total spread, pooling arms, risk
aversion levels, windows and seeds. The mean within-cell across-seed SD holds
arm, eta and window fixed and averages the five-seed SD over the 108 cells --
the same construction as the Sharpe seed band of sec:inference-meth, so a
diagnostic and the headline metric are read on the same yardstick. A wide IQR
with a narrow seed SD means the variation is between cells (regimes), not
between seeds.

Usage:  python scripts/08_training_diagnostics.py
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from src import config

ALGOS = ["ppo", "sac"]
WF_ROOT = config.RESULTS_DIR / "walkforward"
TABLES_DIR = config.RESULTS_DIR / "tables"
FIGURES_DIR = config.RESULTS_DIR / "figures"

# Number of points on the common grid the per-run trajectories are
# interpolated onto for the figure. SAC logs ~20 rows per run and PPO ~588, so
# the grid is capped by the coarser algorithm at plot time.
GRID_POINTS = 200

# Per-run scalar summaries: metric -> (sb3 column, reducer).
#
# "first"/"last" take the first and last logged row of the run. SB3 writes its
# first row after the first update, so "first" is the state after one batch,
# not at initialization; the initial action std is the configured 1.0
# (app:fixed-hyperparams).
_REDUCERS = {
    "first": lambda s: s.iloc[0],
    "last": lambda s: s.iloc[-1],
    "mean": lambda s: s.mean(),
}

PPO_METRICS = {
    "explained_variance_mean":  ("train/explained_variance", "mean"),
    "explained_variance_final": ("train/explained_variance", "last"),
    "approx_kl_mean":           ("train/approx_kl", "mean"),
    "clip_fraction_mean":       ("train/clip_fraction", "mean"),
    "action_std_start":         ("train/std", "first"),
    "action_std_end":           ("train/std", "last"),
    "value_loss_start":         ("train/value_loss", "first"),
    "value_loss_end":           ("train/value_loss", "last"),
}

SAC_METRICS = {
    "critic_loss_start":  ("train/critic_loss", "first"),
    "critic_loss_end":    ("train/critic_loss", "last"),
    "ent_coef_start":     ("train/ent_coef", "first"),
    "ent_coef_end":       ("train/ent_coef", "last"),
    "actor_loss_start":   ("train/actor_loss", "first"),
    "actor_loss_end":     ("train/actor_loss", "last"),
}

METRICS = {"ppo": PPO_METRICS, "sac": SAC_METRICS}

# The trajectory drawn in the figure: (column, axis label, smoothing window).
# PPO recomputes explained variance on each fresh rollout, so the raw series
# oscillates rollout to rollout; a centred rolling median over the grid makes
# the level readable and is declared in the caption. SAC logs ~20 points per
# run, which needs no smoothing and would not survive it.
TRAJECTORY = {
    "ppo": ("train/explained_variance", "Critic explained variance", 15),
    "sac": ("train/critic_loss", "Critic loss", 1),
}

_RUN_RE = re.compile(
    r"/(?P<arm>M\d)_eta(?P<eta>[\d.]+)/(?P<window>\d{4})/sb3_log_seed(?P<seed>\d)\.csv$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def run_paths(algo: str) -> list[str]:
    """Every SB3 progress log of one algorithm, in a stable order."""
    pattern = f"{WF_ROOT}/{algo}/*/*/sb3_log_seed*.csv"
    return sorted(glob.glob(str(pattern)))


def _run_key(path: str) -> dict:
    match = _RUN_RE.search(path.replace(os.sep, "/"))
    if match is None:
        raise ValueError(f"unrecognized run path: {path}")
    key = match.groupdict()
    return {"arm": key["arm"], "eta": float(key["eta"]),
            "window": int(key["window"]), "seed": int(key["seed"])}


def load_run(path: str) -> pd.DataFrame:
    """One run's progress log, rows with no recorded update dropped.

    SB3 writes a row per logging interval; rows before the first gradient
    update carry NaN in the train/ columns and would otherwise bias a "first"
    reduction.
    """
    frame = pd.read_csv(path)
    train_cols = [c for c in frame.columns if c.startswith("train/")]
    return frame.dropna(subset=train_cols, how="all").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-run summaries
# ---------------------------------------------------------------------------

def summarise_run(frame: pd.DataFrame, algo: str) -> dict:
    """Reduce one run's log to the scalar summaries of METRICS[algo]."""
    out = {}
    for name, (column, how) in METRICS[algo].items():
        if column not in frame.columns:
            out[name] = np.nan
            continue
        series = frame[column].dropna()
        out[name] = np.nan if series.empty else float(_REDUCERS[how](series))

    # NOTE on what is deliberately NOT derived here. SB3's SAC critic loss is
    # 0.5 * sum_i MSE(Q_i, y), an ABSOLUTE squared error whose scale depends on
    # the magnitude of the targets, so a rise in it does not by itself
    # establish a worse relative fit. Normalizing it would need the scale of
    # the targets, and SB3 does not log that. The actor loss is not a usable
    # substitute: it is (ent_coef * log_prob - min_i Q_i).mean(), and early in
    # training the entropy term dominates -- with ent_coef ~ 0.018 and a
    # 31-dimensional squashed Gaussian, the logged start value of about -122 is
    # far larger than any plausible undiscounted sum of daily log-utility
    # rewards. The actor loss is written to the per-run CSV as raw record only.
    # Making the scale-free version would require re-instrumenting the training
    # loop to log the target Q statistics.
    return out


def collect(algo: str) -> pd.DataFrame:
    """One row per run: identifying keys plus the scalar summaries."""
    rows = []
    for path in run_paths(algo):
        rows.append({**_run_key(path), **summarise_run(load_run(path), algo)})
    if not rows:
        raise FileNotFoundError(
            f"no SB3 logs under {WF_ROOT}/{algo}; run step 5 first")
    frame = pd.DataFrame(rows)
    return frame.sort_values(["arm", "eta", "window", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution across runs
# ---------------------------------------------------------------------------

def _seed_sd(frame: pd.DataFrame, metric: str) -> float:
    """Mean over cells of the within-cell across-seed SD.

    A cell is one (arm, eta, window); the SD is taken over its five seeds with
    ddof = 1, matching the seed band of sec:inference-meth, and averaged over
    the cells that have at least two seeds.
    """
    per_cell = (frame.groupby(["arm", "eta", "window"])[metric]
                .std(ddof=1))
    return float(per_cell.mean()) if per_cell.notna().any() else np.nan


def summarise(frame: pd.DataFrame, algo: str) -> pd.DataFrame:
    """Distribution of each per-run summary over the runs.

    Median and IQR rather than mean and SD as the headline pair: explained
    variance is unbounded below and the critic loss is positive and
    heavy-tailed, so a mean is dragged by a handful of runs. The mean is kept
    alongside so it stays visible how far the two differ.
    """
    metric_names = [c for c in frame.columns
                    if c not in ("arm", "eta", "window", "seed")]
    rows = []
    for metric in metric_names:
        values = frame[metric].dropna()
        if values.empty:
            continue
        p25, p75 = np.percentile(values, [25, 75])
        rows.append({
            "algorithm": algo,
            "metric": metric,
            "n_runs": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p25": float(p25),
            "p75": float(p75),
            "iqr": float(p75 - p25),
            "sd": float(values.std(ddof=1)),
            "seed_sd": _seed_sd(frame, metric),
            "min": float(values.min()),
            "max": float(values.max()),
            # Only interpretable where zero is a meaningful threshold, i.e. for
            # explained variance, where it separates runs whose critic beats
            # the mean of its own targets from runs whose critic does not.
            "frac_above_zero": float((values > 0).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def trajectory_band(algo: str, column: str,
                    grid_points: int = GRID_POINTS) -> pd.DataFrame:
    """Median and interquartile band of one logged series across all runs.

    Runs are interpolated onto a common grid of environment steps before the
    quantiles are taken, so windows with different training-block lengths can
    be pooled. The grid spans the steps every run covers, and is capped at the
    number of points the coarsest run actually logged so the curve is never
    smoother than the data.

    As in Gollart & Okhrin (2025, Fig. 6), a quantile path is a per-step
    combination of the runs and is not itself the path of any single run.
    """
    steps, series, lengths = [], [], []
    for path in run_paths(algo):
        frame = load_run(path)
        if column not in frame.columns:
            continue
        valid = frame[["time/total_timesteps", column]].dropna()
        if len(valid) < 2:
            continue
        steps.append(valid["time/total_timesteps"].to_numpy(dtype=float))
        series.append(valid[column].to_numpy(dtype=float))
        lengths.append(len(valid))

    if not series:
        raise FileNotFoundError(f"no {column} logged for {algo}")

    lo = max(s[0] for s in steps)
    hi = min(s[-1] for s in steps)
    grid = np.linspace(lo, hi, min(grid_points, min(lengths)))
    stack = np.vstack([np.interp(grid, s, v) for s, v in zip(steps, series)])

    p25, median, p75 = np.percentile(stack, [25, 50, 75], axis=0)
    return pd.DataFrame({"total_timesteps": grid, "p25": p25,
                         "median": median, "p75": p75,
                         "n_runs": stack.shape[0]})


def plot_training_diagnostics() -> list[str]:
    """Two panels: PPO explained variance and SAC critic loss over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, algo in zip(axes, ALGOS):
        column, label, window = TRAJECTORY[algo]
        band = trajectory_band(algo, column)
        steps = band["total_timesteps"] / 1e3

        if window > 1:
            # The raw median stays visible underneath so the smoothing cannot
            # hide how much the series actually moves between rollouts.
            ax.plot(steps, band["median"], color="tab:blue", lw=0.6, alpha=0.35)
            smooth = {q: band[q].rolling(window, center=True, min_periods=1).median()
                      for q in ("p25", "median", "p75")}
            median_label = f"median (rolling, {window} points)"
        else:
            smooth = {q: band[q] for q in ("p25", "median", "p75")}
            median_label = "median"

        ax.fill_between(steps, smooth["p25"], smooth["p75"],
                        color="tab:blue", alpha=0.22,
                        label="interquartile range")
        ax.plot(steps, smooth["median"], color="tab:blue", lw=1.6,
                label=median_label)
        if algo == "ppo":
            ax.axhline(0.0, color="grey", ls="--", lw=1.0,
                       label="predicting the target mean")
        ax.set_title(f"{algo.upper()}: {label.lower()} "
                     f"($n = {int(band['n_runs'].iloc[0])}$ runs)")
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_ylabel(label)
        ax.legend(fontsize="small", frameon=False)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = f"{FIGURES_DIR}/fig_training_diagnostics.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _write_readme() -> None:
    """Document the two CSVs.

    Written as its own file rather than appended to results/tables/README.md,
    which src.evaluation.report rewrites wholesale on every run of step 6.
    """
    (TABLES_DIR / "README_training_diagnostics.md").write_text("""\
# Training diagnostics

Built by `python scripts/08_training_diagnostics.py` from the Stable
Baselines3 progress logs (`sb3_log_seed{s}.csv`) written beside the
walk-forward artifacts of step 5. 540 runs per algorithm: 6 arms x 3 risk
aversion levels x 6 windows x 5 seeds.

The two algorithms carry different metrics because they expose different
signals. PPO is on-policy, so empirical return targets exist and SB3 logs
`train/explained_variance`. SAC is off-policy and regresses on bootstrapped
TD targets, so no empirical return exists to explain, no explained variance
is logged, and `train/critic_loss` is the only critic-fit signal available.

## `{ALGO}_training_diagnostics.csv`

One row per run. `arm`, `eta`, `window`, `seed` identify it; the remaining
columns are per-run reductions of the logged series. `_start` and `_end` are
the first and last logged row; SB3 writes its first row after the first
update, so `_start` is the state after one batch, not at initialization.

| Column | Meaning |
|---|---|
| `explained_variance_mean` | PPO. Mean over the run of `1 - Var(target - prediction) / Var(target)`, where the target is the GAE return `advantages + values` of the rollout just collected. 1 is perfect, 0 is no better than predicting the target mean, negative is worse than that. |
| `explained_variance_final` | PPO. The same quantity on the last rollout. |
| `approx_kl_mean`, `clip_fraction_mean` | PPO. Mean approximate KL per update and mean fraction of samples hitting the clip range. Disclosure only; no claim rests on them. |
| `action_std_start`, `action_std_end` | PPO. The state-independent `exp(log_sigma)` of the Gaussian policy, initialized at 1.0. |
| `value_loss_start`, `value_loss_end` | PPO. Raw MSE of the value head. Redundant with explained variance, which is its scale-free form; kept as record. |
| `critic_loss_start`, `critic_loss_end` | SAC. `0.5 * sum_i MSE(Q_i, y)` over the twin critics. An ABSOLUTE error: its scale depends on the magnitude of the targets, so a rise does not by itself establish a worse relative fit. |
| `ent_coef_start`, `ent_coef_end` | SAC. The learned entropy temperature (alpha). |
| `actor_loss_start`, `actor_loss_end` | SAC. `(ent_coef * log_prob - min_i Q_i).mean()`. Raw record only: early in training the entropy term dominates, so this is NOT a usable proxy for the scale of Q and is not used to normalize the critic loss. |

## `{ALGO}_training_summary.csv`

One row per metric: the distribution of the per-run values over the 540 runs.

| Column | Meaning |
|---|---|
| `n_runs` | Runs contributing a finite value. |
| `mean`, `median` | Median is the headline: explained variance is unbounded below and the critic loss is positive and heavy-tailed, so the mean is pulled by a few runs. Both are reported so the gap stays visible. |
| `p25`, `p75`, `iqr` | Total spread over all 540 runs, pooling arms, eta, windows and seeds. |
| `sd` | Sample SD (ddof=1) over the same 540 runs. |
| `seed_sd` | Mean over the 108 (arm, eta, window) cells of the within-cell SD across that cell's five seeds. Holds treatment, risk aversion and regime fixed, so it isolates seed noise -- the same construction as the Sharpe seed band of sec:inference-meth. A wide `iqr` with a narrow `seed_sd` means the variation is between cells, not between seeds. |
| `min`, `max` | Range, to show how far the tails reach. |
| `frac_above_zero` | Share of runs with a value above zero. Interpretable for explained variance only, where zero separates critics that beat the mean of their own targets from those that do not. |

## `fig_training_diagnostics`

PPO explained variance and SAC critic loss against environment steps, median
with an interquartile band across runs. Each run is interpolated onto a
common step grid before the quantiles are taken. As in Gollart & Okhrin
(2025, Fig. 6), a quantile path is a per-step combination of the runs and is
not the path of any single run. The PPO panel is drawn with a centred rolling
median over 15 grid points because the raw series is recomputed on each fresh
rollout and oscillates; the unsmoothed median is kept visible underneath.
""")


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    summaries = []
    for algo in ALGOS:
        per_run = collect(algo)
        per_run.to_csv(
            f"{TABLES_DIR}/{algo.upper()}_training_diagnostics.csv", index=False)
        summary = summarise(per_run, algo)
        summary.to_csv(
            f"{TABLES_DIR}/{algo.upper()}_training_summary.csv", index=False)
        summaries.append(summary)
        print(f"{algo.upper()}: {len(per_run)} runs -> "
              f"{TABLES_DIR}/{algo.upper()}_training_summary.csv")

    for path in plot_training_diagnostics():
        print(f"wrote {path}")
    _write_readme()
    print(f"wrote {TABLES_DIR}/README_training_diagnostics.md")

    combined = pd.concat(summaries, ignore_index=True)
    print()
    print(combined[["algorithm", "metric", "n_runs", "mean", "median",
                    "p25", "p75", "seed_sd"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
