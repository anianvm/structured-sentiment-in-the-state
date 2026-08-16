"""Assemble the result tables from the walk-forward runs.

Implements the reporting side of the thesis methodology chapter, section "Benchmarks
and Metrics" (sec:metrics-meth): for every completed (algorithm, arm, eta)
run under results/walkforward/, this script builds

* a performance table — the five headline measures, per test year and for the
  combined 2021–2026 out-of-sample period, with UCRP and DIA benchmark rows
  computed over the same periods — and
* a behaviour table — one-way turnover, portfolio entropy, largest position,
  cash share, and the two downside measures (maximum drawdown, CVaR at 5%) —
  per test year and combined, and
* a seed-dispersion table — every seed's validation number per window, so
  differences between arms can be judged relative to training noise
  (sec:walkforward-meth).

Output: results/tables/{PPO,SAC}_performance.csv, {PPO,SAC}_behaviour.csv,
{PPO,SAC}_seed_dispersion.csv and a README.md explaining every column. The
same pass also writes the inference and hypothesis tables built from the
finished ledgers ({ALGO}_contrasts.csv, {ALGO}_downside_contrasts.csv,
{ALGO}_h2_separation.csv, {ALGO}_h5_responsiveness.csv,
{ALGO}_h5_differences.csv) and the generated LaTeX row bodies under
results/tables/latex/ (see latex.py).
Missing or partial runs are skipped with a printed note, because students run
the arms incrementally.

Run as:  python -m src.evaluation.report
"""

import json

import pandas as pd

from src import config
from src.evaluation import contrasts, hypotheses, latex, metrics
from src.evaluation.benchmarks import dia_buy_hold_log_returns, ucrp_log_returns
from src.features.market_context import vix_zscore
from src.features.sentiment_grid import build_finbert_grid, build_llm_grid

PERF_COLUMNS = [
    "strategy", "period", "n_days",
    "cumulative_return", "annualized_return", "annualized_volatility",
    "sharpe_ratio", "max_drawdown",
]
BEHAVIOUR_COLUMNS = [
    "strategy", "period", "n_days",
    "avg_oneway_turnover", "portfolio_entropy", "risky_entropy",
    "avg_largest_position", "risky_largest_position", "avg_cash_share",
    "max_drawdown", "cvar_5",
]
DISPERSION_COLUMNS = [
    "strategy", "period", "seed",
    "val_cum_log_return", "selected", "test_cum_log_return",
]


# ---------------------------------------------------------------------------
# Discovering and reading run outputs
# ---------------------------------------------------------------------------

def _find_run_files(year_dir) -> dict | None:
    """Identify the three outputs of a walk-forward test year by content.

    The runner writes a daily log-returns CSV (columns date, log_return), a
    weights panel CSV (date plus one column per asset including CASH) and a
    summary.json. Files are recognized by their header rather than their name
    so this script does not depend on the runner's file naming.

    Columns are matched EXACTLY, not by substring. A year directory also
    holds curve_seed*.csv ("steps,val_log_return") and validation_seeds.csv
    ("seed,val_log_return,..."), both of which contain the substring
    "log_return"; with a substring test the last match in sorted order won
    and the daily file was read as validation_seeds.csv, which has no date
    column. Requiring a literal "date" column alongside is what separates the
    per-day series from the per-seed summaries.
    """
    daily_path, weights_path = None, None
    for path in sorted(year_dir.glob("*.csv")):
        with open(path) as fh:
            fields = [c.strip() for c in fh.readline().strip().split(",")]
        if "date" not in fields:
            continue
        if "log_return" in fields:
            daily_path = path
        elif "CASH" in fields:
            weights_path = path
    summary_path = year_dir / "summary.json"
    if daily_path is None:
        return None
    return {
        "daily": daily_path,
        "weights": weights_path,                       # may be None
        "summary": summary_path if summary_path.exists() else None,
    }


def _read_year(files: dict) -> dict:
    """Load one test year: daily returns/turnover, weights panel, summary."""
    daily = pd.read_csv(files["daily"], parse_dates=["date"], index_col="date")
    weights = None
    if files["weights"] is not None:
        weights = pd.read_csv(files["weights"], index_col=0, parse_dates=True)
    summary = None
    if files["summary"] is not None:
        with open(files["summary"]) as fh:
            summary = json.load(fh)
    return {"daily": daily, "weights": weights, "summary": summary}


def _dispersion_rows(strategy: str, year: str, summary: dict | None) -> list[dict]:
    """One row per seed: the cross-seed dispersion the protocol requires.

    The methodology (sec:walkforward-meth) demands that differences between
    arms be judged relative to training noise, so the report surfaces every
    seed's validation number next to the selected one, not just the winner.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("seeds"), list):
        return []
    selected = summary.get("selected_seed")
    rows = []
    for record in summary["seeds"]:
        if not isinstance(record, dict):
            continue
        is_selected = record.get("seed") == selected
        rows.append({
            "strategy": strategy, "period": year,
            "seed": record.get("seed"),
            "val_cum_log_return": record.get("val_log_return"),
            "selected": is_selected,
            # The test number exists only for the seed that was evaluated.
            "test_cum_log_return":
                summary.get("test_log_return") if is_selected else float("nan"),
        })
    return rows


def _discover_runs(wf_root) -> dict:
    """Map algo -> strategy label ("{arm}_eta{eta}") -> {year: loaded data}."""
    runs: dict = {}
    for algo_dir in sorted(p for p in wf_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            for year_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                files = _find_run_files(year_dir)
                if files is None:
                    print(f"note: skipping {year_dir} (no daily log-returns CSV)")
                    continue
                runs.setdefault(algo_dir.name, {}) \
                    .setdefault(run_dir.name, {})[year_dir.name] = _read_year(files)
    return runs


# ---------------------------------------------------------------------------
# Market data for the benchmark rows
# ---------------------------------------------------------------------------

def _load_market():
    """(dates, ohlc, tbill Series) for the benchmark rows and Sharpe ratios.

    The market data is a hard requirement: without it the tables would carry
    Sharpe ratios against a wrong (zero) risk-free rate and no benchmark rows,
    which is worse than no tables at all.
    """
    try:
        from src.features.price_tensor import load_ohlc_tensor
        from src.features.market_context import tbill_rate

        dates, ohlc = load_ohlc_tensor()
        tbill = pd.Series(tbill_rate(dates), index=dates, name="tbill")
        return dates, ohlc, tbill
    except Exception as exc:
        raise RuntimeError(
            f"market data unavailable ({exc}). The report needs prices for "
            "the UCRP/DIA benchmark rows and the 13-week T-bill for the "
            "Sharpe ratios — run scripts/01_download_market_data.py first."
        ) from exc


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------

def _perf_row(strategy, period, log_returns, tbill) -> dict:
    return {
        "strategy": strategy, "period": period, "n_days": len(log_returns),
        **metrics.performance_summary(log_returns, tbill),
    }


def _behaviour_row(strategy, period, log_returns, weights=None,
                   turnover=None) -> dict:
    """Behaviour diagnostics plus the two downside measures.

    The allocation diagnostics need a weights panel and the environment's
    turnover series, which only the agents produce; the benchmarks are
    carried with those fields NaN so that their drawdown and CVaR still
    appear as a reference in the same table.
    """
    allocation = metrics.behaviour_summary(weights, turnover) \
        if weights is not None and turnover is not None \
        else {k: float("nan") for k in ("avg_oneway_turnover",
                                        "portfolio_entropy",
                                        "risky_entropy",
                                        "avg_largest_position",
                                        "risky_largest_position",
                                        "avg_cash_share")}
    return {
        "strategy": strategy, "period": period, "n_days": len(log_returns),
        **allocation,
        **metrics.behaviour_tail_summary(log_returns),
    }


def _benchmark_rows(periods: dict, dates, ohlc, tbill) -> tuple[list, list]:
    """UCRP and DIA performance and behaviour rows over the agents' days.

    Returns (performance rows, behaviour rows). The behaviour rows carry only
    the drawdown and CVaR of sec:metrics-meth — a passive benchmark has no
    weights panel or turnover series in these artifacts — so that the two
    downside measures can be read against a passive reference in the same
    table as the agents'.

    Mirrors the walk-forward structure exactly: each TEST YEAR is entered
    fresh (the agent starts its test episode 100% in cash and buys in; UCRP
    pays the same deployment cost on day one), and both series are dated by
    the REALIZATION day, like the runner's test_daily.csv. The first return
    of a year is therefore realized on the year's first trading day, decided
    one trading day earlier — which is why the arrays are sliced from one row
    before the period start. Combined periods are the concatenation of the
    per-year series, just as the agents' combined record is.
    """
    year_periods = {p: se for p, se in periods.items() if p.isdigit()}
    ucrp_years, dia_years = {}, {}
    for year, (start, end) in sorted(year_periods.items()):
        first = int(dates.searchsorted(start))   # first realization day
        last = int(dates.searchsorted(end))      # last realization day
        j0 = max(first - 1, 0)                   # first DECISION day
        if last - j0 < 1:
            print(f"note: no market data inside {year}; benchmark rows skipped")
            continue
        ucrp = ucrp_log_returns(dates[j0:last + 1], ohlc[j0:last + 1])
        # ucrp_log_returns dates by decision day; re-date by realization day.
        ucrp.index = dates[j0 + 1:last + 1]
        ucrp_years[year] = ucrp
        try:
            dia_years[year] = dia_buy_hold_log_returns(
                str(dates[j0].date()), str(dates[last].date()))
        except Exception as exc:
            print(f"note: DIA series for {year} skipped ({exc})")

    perf, behaviour = [], []

    def add(name, period, series):
        perf.append(_perf_row(name, period, series, tbill))
        behaviour.append(_behaviour_row(name, period, series))

    for period in periods:
        if period in year_periods:
            if period in ucrp_years:
                add("UCRP", period, ucrp_years[period])
            if period in dia_years:
                add("DIA", period, dia_years[period])
        else:  # combined period: concatenate the per-year series
            if ucrp_years:
                add("UCRP", period,
                    pd.concat(ucrp_years.values()).sort_index())
            if dia_years:
                add("DIA", period, pd.concat(dia_years.values()).sort_index())
    return perf, behaviour


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wf_root = config.RESULTS_DIR / "walkforward"
    if not wf_root.exists():
        print(f"note: {wf_root} does not exist; nothing to report.")
        return
    runs = _discover_runs(wf_root)
    if not runs:
        print(f"note: no completed runs under {wf_root}; nothing to report.")
        return

    tables_dir = config.RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dates, ohlc, tbill = _load_market()

    for algo, strategies in runs.items():
        perf_rows, behaviour_rows, dispersion_rows = [], [], []
        periods: dict = {}  # period label -> (first date, last date)

        for strategy, years in sorted(strategies.items()):
            all_daily, all_weights = [], []
            for year, data in sorted(years.items()):
                daily = data["daily"]
                dispersion_rows += _dispersion_rows(strategy, year,
                                                    data["summary"])
                start, end = daily.index.min(), daily.index.max()
                prev = periods.get(year)
                periods[year] = (min(start, prev[0]), max(end, prev[1])) if prev \
                    else (start, end)

                perf_rows.append(
                    _perf_row(strategy, year, daily["log_return"], tbill))
                all_daily.append(daily)

                if data["weights"] is not None:
                    behaviour_rows.append(_behaviour_row(
                        strategy, year, daily["log_return"],
                        data["weights"], daily["turnover"]))
                    all_weights.append(data["weights"])
                else:
                    print(f"note: no weights panel for {algo}/{strategy}/{year}; "
                          "behaviour row skipped")

            # Combined out-of-sample record: the concatenated test years
            # (only meaningful once more than one year has been run).
            if len(years) > 1:
                combined = pd.concat(all_daily).sort_index()
                label = f"{min(years)}-{max(years)}"
                perf_rows.append(
                    _perf_row(strategy, label, combined["log_return"], tbill))
                if all_weights:
                    behaviour_rows.append(_behaviour_row(
                        strategy, label, combined["log_return"],
                        pd.concat(all_weights).sort_index(),
                        combined["turnover"]))

        # One combined period across everything this algo has run.
        if len(periods) > 1:
            starts, ends = zip(*periods.values())
            periods[f"{min(periods)}-{max(periods)}"] = (min(starts), max(ends))
        bench_perf, bench_behaviour = _benchmark_rows(
            periods, dates, ohlc, tbill)
        perf_rows += bench_perf
        behaviour_rows += bench_behaviour

        # --- Inference: the arm contrasts the hypotheses are stated in -----
        # The tables above describe; this decides. Runs on the finished
        # ledgers, so it can be re-run with a different rule without
        # retraining anything.
        by_arm_daily, by_arm_ledger = {}, {}
        for strategy, years in sorted(strategies.items()):
            if "_eta" not in strategy:
                continue
            arm, eta_text = strategy.split("_eta", 1)
            try:
                eta = float(eta_text)
            except ValueError:
                continue
            daily_by_year, ledger_by_year = {}, {}
            for year, data in years.items():
                daily_by_year[year] = data["daily"]["log_return"]
                ledger = (wf_root / algo / strategy / str(year)
                          / "ledger_test.parquet")
                if ledger.exists():
                    ledger_by_year[year] = pd.read_parquet(ledger)
            by_arm_daily[(arm, eta)] = daily_by_year
            by_arm_ledger[(arm, eta)] = ledger_by_year

        contrast_rows = contrasts.arm_contrasts(algo, by_arm_daily,
                                                by_arm_ledger, tbill)
        # The same contrasts on the two downside measures — descriptive, so
        # the drawdown and CVaR numbers quoted in the analysis chapter come
        # out of this pipeline rather than being computed by hand.
        downside_rows = contrasts.downside_contrasts(algo, by_arm_daily,
                                                     by_arm_ledger)

        # --- H1b / H4: the allocation-side hypotheses ----------------------
        # Pool each (arm, eta)'s test years into one daily frame and one
        # weights panel; H4 additionally needs the common sentiment regressor
        # and the VIX control on the pooled dates.
        by_arm_alloc = {}
        for strategy, years in sorted(strategies.items()):
            if "_eta" not in strategy:
                continue
            arm, eta_text = strategy.split("_eta", 1)
            try:
                eta = float(eta_text)
            except ValueError:
                continue
            daily = pd.concat([d["daily"] for d in years.values()]).sort_index()
            wpanels = [d["weights"] for d in years.values()
                       if d["weights"] is not None]
            weights = pd.concat(wpanels).sort_index() if wpanels else None
            key = (arm, eta)
            by_arm_alloc[key] = {
                "daily": daily, "weights": weights,
                "ledgers": by_arm_ledger.get(key, {}),
            }

        h2_rows = hypotheses.h2_rows(algo, by_arm_alloc)
        all_dates = sorted({d for data in by_arm_alloc.values()
                            for d in data["daily"].index})
        h5_rows, h5_diff_rows = [], []
        if all_dates:
            idx = pd.DatetimeIndex(all_dates)
            grid = build_llm_grid(dates)          # (T, 30, 5), lagged
            sent = pd.Series(grid[:, :, 0].mean(axis=1), index=dates)
            vix = pd.Series(vix_zscore(dates), index=dates)
            # M2 carries the FinBERT channel and never observes the LLM
            # signal, so it is estimated against its own news index; every
            # other arm shares the LLM one (see hypotheses.h5_rows).
            fin = build_finbert_grid(dates)       # (T, 30, 2), lagged
            finbert_index = pd.Series(fin[:, :, 0].mean(axis=1), index=dates,
                                      name="finbert")
            h5_rows = hypotheses.h5_rows(
                algo, by_arm_alloc, sent.reindex(idx), vix.reindex(idx),
                sentiment_by_arm={"M2": finbert_index.reindex(idx)})
            # tab:res-h5-dr: the differenced regression shares the LLM index
            # and control with the per-arm rows above, so M2 never appears.
            h5_diff_rows = hypotheses.h5_difference_rows(
                algo, by_arm_alloc, sent.reindex(idx), vix.reindex(idx))

        name = algo.upper()
        if contrast_rows:
            pd.DataFrame(contrast_rows, columns=contrasts.CONTRAST_COLUMNS).to_csv(
                tables_dir / f"{name}_contrasts.csv", index=False)
        if downside_rows:
            pd.DataFrame(downside_rows,
                         columns=contrasts.DOWNSIDE_COLUMNS).to_csv(
                tables_dir / f"{name}_downside_contrasts.csv", index=False)
        if h2_rows:
            pd.DataFrame(h2_rows, columns=hypotheses.H2_COLUMNS).to_csv(
                tables_dir / f"{name}_h2_separation.csv", index=False)
        if h5_rows:
            pd.DataFrame(h5_rows, columns=hypotheses.H5_COLUMNS).to_csv(
                tables_dir / f"{name}_h5_responsiveness.csv", index=False)
        if h5_diff_rows:
            pd.DataFrame(h5_diff_rows,
                         columns=hypotheses.H5_DIFF_COLUMNS).to_csv(
                tables_dir / f"{name}_h5_differences.csv", index=False)
        pd.DataFrame(perf_rows, columns=PERF_COLUMNS).to_csv(
            tables_dir / f"{name}_performance.csv", index=False)
        pd.DataFrame(behaviour_rows, columns=BEHAVIOUR_COLUMNS).to_csv(
            tables_dir / f"{name}_behaviour.csv", index=False)
        pd.DataFrame(dispersion_rows, columns=DISPERSION_COLUMNS).to_csv(
            tables_dir / f"{name}_seed_dispersion.csv", index=False)
        print(f"wrote {name}_performance.csv ({len(perf_rows)} rows), "
              f"{name}_behaviour.csv ({len(behaviour_rows)} rows) and "
              f"{name}_seed_dispersion.csv ({len(dispersion_rows)} rows)")

    _write_readme(tables_dir)
    _write_latex_bodies(tables_dir, wf_root, tbill)


def _write_latex_bodies(tables_dir, wf_root, tbill_annual) -> None:
    """Emit the chapter's table row bodies beside the CSVs they come from.

    Reads the CSVs back rather than threading every row list through the
    caller: the fragments must be derivable from the committed artifacts
    alone, and reading them here is the cheapest way to keep that honest.
    """
    read = lambda name: {a: pd.read_csv(tables_dir / f"{a}_{name}.csv")
                         for a in ("PPO", "SAC")}
    perf = read("performance")
    years = tuple(sorted({p for f in perf.values() for p in f["period"]
                          if p.isdigit()}))

    def _ledgers(algo, arm, eta):
        directory = wf_root / algo.lower() / f"{arm}_eta{eta}"
        if not directory.is_dir():
            return {}
        return {y.name: pd.read_parquet(y / "ledger_test.parquet")
                for y in sorted(directory.iterdir())
                if (y / "ledger_test.parquet").exists()}

    def sharpe_by_seed(algo, arm):
        ledgers = _ledgers(algo, arm, 1.0)
        values = {}
        for seed, group in pd.concat(ledgers.values()).groupby("seed"):
            series = group.sort_values("date").set_index("date")["log_return"]
            values[int(seed)] = metrics.sharpe_ratio(series, tbill_annual)
        # Selection happens per window, so no single seed is "the" selected
        # one for the pooled record; the per-window selections are shown in
        # the dispersion figure and the pooled table carries no marker.
        return values

    def cash_by_seed(algo, arm, eta):
        return hypotheses.per_seed_cash_share(_ledgers(algo, arm, eta))

    out_dir = latex.write_all(
        tables_dir, perf, read("behaviour"), read("contrasts"),
        read("h2_separation"), read("h5_responsiveness"),
        read("h5_differences"), years, sharpe_by_seed, cash_by_seed)
    print(f"wrote {len(list(out_dir.glob('*.tex')))} LaTeX table bodies to "
          f"{out_dir.relative_to(out_dir.parents[2])}")


def _write_readme(tables_dir) -> None:
    (tables_dir / "README.md").write_text("""\
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
""")


if __name__ == "__main__":
    main()
