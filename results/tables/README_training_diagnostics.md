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
