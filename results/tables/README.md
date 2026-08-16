# Result tables

Built by `python -m src.evaluation.report` from the walk-forward outputs in
`results/walkforward/`. One pair of tables per algorithm (PPO, SAC). All
measures follow the thesis methodology chapter, section "Benchmarks and Metrics".

## `{ALGO}_performance.csv`

| Column | Meaning |
|---|---|
| `strategy` | `{arm}_eta{eta}` for an agent (e.g. `M4_eta1.0`), or `UCRP` / `DIA` for the benchmarks. |
| `period` | A test year (`2021` ... `2026`) or the combined out-of-sample period (`2021-2026`). |
| `n_days` | Number of daily return observations in the period. |
| `cumulative_return` | Total simple return over the period, `exp(sum of log returns) - 1`. |
| `annualized_return` | Geometric annualization, `(1 + cumulative)^(252 / n_days) - 1`. |
| `annualized_volatility` | Std (ddof=1) of daily simple returns times `sqrt(252)`. |
| `sharpe_ratio` | Mean over std (ddof=1) of daily simple returns in excess of the 13-week T-bill (annual rate / 252), times `sqrt(252)`. The same T-bill is what the cash asset earns. |
| `max_drawdown` | Largest peak-to-trough loss of the wealth curve, as a positive fraction (0.20 = 20%). Repeated in `{ALGO}_behaviour.csv` beside `cvar_5`. |

Benchmark rows: `UCRP` is the uniform constant-rebalanced portfolio over the
30 stocks, rebalanced daily and charged the same linear cost as the agents —
including the full deployment cost on the first day of each test year, just
as an agent pays when it buys in from its all-cash start. `DIA` is
buy-and-hold in the Dow ETF, bought at each test year's start (costless by
convention). Both series are dated by realization day and cover exactly the
agents' test dates; combined periods concatenate the per-year series, like
the agents' combined record.

## `{ALGO}_seed_dispersion.csv`

One row per (strategy, test year, seed): `val_cum_log_return` is that seed's
cumulative validation log return, `selected` marks the seed chosen on the
highest cumulative validation utility (the summed per-period CRRA reward,
which is the log return at eta = 1), and `test_cum_log_return` is filled only
for the selected seed. Every seed IS rolled out on the test year and its
per-step ledger is persisted; those rollouts exist solely to report the
cross-seed dispersion, and nothing about the selection or the headline
consults a test number.
The spread across the five rows of a window is the training noise against
which differences between arms must be judged (Agarwal et al. 2021).

## `{ALGO}_behaviour.csv`

The four allocation diagnostics are agent-only: the benchmarks' behaviour is
fixed by construction (UCRP is uniform over stocks with zero cash; DIA never
trades), and no weights panel or turnover series is written for them, so
those columns are NaN in the `UCRP` and `DIA` rows. The two downside
measures are computed for every row, benchmarks included, because a passive
reference is what makes them readable.

| Column | Meaning |
|---|---|
| `strategy`, `period`, `n_days` | As above. |
| `avg_oneway_turnover` | Mean daily one-way turnover `tau/2` (Rezaei 2025 convention); the environment's `tau` counts each reallocation twice. |
| `portfolio_entropy` | Mean Shannon entropy (natural log) of the full 31-element weight vector (cash + 30 stocks); `ln(31) ~ 3.43` for uniform, lower = more concentrated. |
| `risky_entropy` | The same entropy on the risky composition alone — cash dropped, stock weights rescaled to sum to one; `ln(30) ~ 3.40` for uniform. Measures how selective the stock picking is with the cash decision divided out. |
| `avg_largest_position` | Mean over days of the single largest weight, cash included. |
| `risky_largest_position` | The same on the risky composition alone; `1/30 ~ 3.3%` for uniform. |
| `avg_cash_share` | Mean weight of the cash asset. |
| `max_drawdown` | As in `{ALGO}_performance.csv`, repeated here so the two downside measures sit together. |
| `cvar_5` | Conditional value at risk at 5%: mean of the daily simple returns at or below their empirical 5th percentile, as a NEGATIVE fraction (-0.021 = the worst 5% of days average a 2.1% loss). Describes the return distribution, where the drawdown describes the worst path. |

## `{ALGO}_downside_contrasts.csv`

The pre-registered contrast set of `{ALGO}_contrasts.csv` read on the two
downside measures instead of the Sharpe ratio: one row per (contrast,
measure, eta). DESCRIPTIVE ONLY — sec:inference-meth fixes the Sharpe
difference as the test statistic, so there is no p-value, no Holm column and
no `supported` column here. The seed band is the sole yardstick.

| Column | Meaning |
|---|---|
| `hypothesis`, `arm`, `baseline`, `eta`, `exploratory` | As in `{ALGO}_contrasts.csv`. |
| `measure` | `max_drawdown` or `cvar_5`. |
| `value_arm`, `value_baseline` | Each arm's pooled value on its validation-selected seed — the same series the headline record uses. |
| `difference` | `value_arm - value_baseline`. Lower is better for BOTH measures (drawdown is a positive fraction, CVaR a negative one), so a negative difference favours `arm` on the drawdown and a positive one favours it on CVaR. |
| `seed_mean_arm`, `seed_mean_baseline`, `difference_seed_mean` | The same difference formed from each arm's five-seed mean instead of its selected seed. A sign disagreement with `difference` means the selected-seed reading reflects which seeds validation picked, not a property of the arm. |
| `seed_sd_arm`, `seed_sd_baseline` | Sample SD (ddof=1) of the five per-seed values. |
| `seed_band` | `sqrt(sd_arm^2 + sd_baseline^2)`, the same construction as the Sharpe band. |
| `exceeds_seed_band` | Whether `abs(difference)` exceeds the band. Absolute, because the hypotheses predict a direction for the Sharpe ratio only. |
