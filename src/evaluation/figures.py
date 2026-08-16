"""Figures for the analysis chapter.

Two exhibits, both built from the finished walk-forward artifacts of step 5
(nothing is retrained and nothing is recomputed from a model):

* fig_wealth_curves    cumulative wealth of the six arms at eta = 1 against
                       the UCRP and DIA benchmarks, one panel per algorithm.
* fig_seed_dispersion  the five per-seed test Sharpe ratios of every arm in
                       every walk-forward window, with the validation-selected
                       seed marked -- the cross-seed dispersion the protocol
                       requires (sec:walkforward-meth, Agarwal et al. 2021).

House style follows the other figures in this repository (see
src/experiments/window_selection.py): Agg backend, matplotlib defaults, the
tab10 palette, grey dashed reference lines, a title on every axes, and
tight_layout. Each figure is written as both PNG and PDF at the same dpi as
fig_llm_finbert_confusion, so the thesis can include whichever it prefers.

Line styles pair a distinct hue with a distinct dash pattern so the eight
series stay separable in colour, in greyscale, and in print: no two curves
share both attributes.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from src import config
from src.evaluation import metrics
from src.evaluation.benchmarks import (
    dia_buy_hold_log_returns, ucrp_log_returns)
from src.features.market_context import tbill_rate
from src.features.price_tensor import load_ohlc_tensor

ARMS = [f"M{i}" for i in range(1, 7)]
ALGOS = ["ppo", "sac"]
ETA = 1.0
OUT_DIR = config.RESULTS_DIR / "figures"

# (colour, linestyle, linewidth). M1 is the control and carries the reference
# weight; benchmarks are grey so the arms read as one family against them.
STYLE = {
    "M1":   ("black",      "-",            2.0),
    "M2":   ("tab:orange", (0, (6, 2)),    1.4),
    "M3":   ("tab:blue",   "-",            1.4),
    "M4":   ("tab:green",  (0, (4, 1, 1, 1)), 1.4),
    "M5":   ("tab:red",    (0, (1, 1.2)),  1.6),
    "M6":   ("tab:purple", (0, (5, 1, 1, 1, 1, 1)), 1.4),
    "UCRP": ("grey",       "-",            1.0),
    "DIA":  ("dimgrey",    (0, (1, 2)),    1.2),
}


def _arm_daily(algo: str, arm: str, eta: float = ETA) -> pd.Series:
    """Selected seed's daily test log returns, test years stitched in order."""
    paths = sorted(glob.glob(
        f"{config.RESULTS_DIR}/walkforward/{algo}/{arm}_eta{eta}/*/test_daily.csv"))
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in paths]
    joined = pd.concat(frames).sort_values("date").set_index("date")
    return joined["log_return"]


def _benchmarks(index: pd.DatetimeIndex, dates, ohlc) -> pd.DataFrame:
    """UCRP and DIA over the same days, entered fresh each test year.

    Mirrors src.evaluation.report._benchmark_rows exactly: each test year is
    a separate deployment (the agents start their test episode in cash and
    buy in; UCRP pays the same day-one cost), and the years are concatenated
    the way the agents' record is.
    """
    ucrp_parts, dia_parts = [], []
    for year in sorted({d.year for d in index}):
        block = index[index.year == year]
        first = int(dates.searchsorted(block[0]))
        last = int(dates.searchsorted(block[-1]))
        j0 = max(first - 1, 0)
        series = ucrp_log_returns(dates[j0:last + 1], ohlc[j0:last + 1])
        series.index = dates[j0 + 1:last + 1]      # re-date by realization day
        ucrp_parts.append(series)
        dia_parts.append(dia_buy_hold_log_returns(
            str(dates[j0].date()), str(dates[last].date())))
    return pd.DataFrame({"UCRP": pd.concat(ucrp_parts),
                         "DIA": pd.concat(dia_parts)})


def _per_seed_sharpe(algo: str, arm: str, tbill, eta: float = ETA) -> pd.DataFrame:
    """Annualized test Sharpe of every seed in every window, selected flagged."""
    rows = []
    for path in sorted(glob.glob(
            f"{config.RESULTS_DIR}/walkforward/{algo}/{arm}_eta{eta}"
            f"/*/ledger_test.parquet")):
        ledger = pd.read_parquet(path, columns=["seed", "selected", "date",
                                                "log_return", "test_year"])
        ledger["date"] = pd.to_datetime(ledger["date"])
        year = int(ledger["test_year"].iloc[0])
        for seed, group in ledger.groupby("seed"):
            series = group.sort_values("date").set_index("date")["log_return"]
            rows.append({"year": year, "seed": int(seed),
                         "selected": bool(group["selected"].iloc[0]),
                         "sharpe": metrics.sharpe_ratio(series, tbill)})
    return pd.DataFrame(rows)


def _save(fig, stem: str) -> list[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for ext, dpi in (("png", 200), ("pdf", 200)):
        path = f"{OUT_DIR}/{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Figure 1: cumulative wealth
# ---------------------------------------------------------------------------

def plot_wealth_curves(dates, ohlc) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {a: {arm: _arm_daily(a, arm) for arm in ARMS} for a in ALGOS}
    index = curves["ppo"]["M1"].index
    bench = _benchmarks(index, dates, ohlc).reindex(index)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    for ax, algo in zip(axes, ALGOS):
        for name in ARMS + ["UCRP", "DIA"]:
            series = (curves[algo][name] if name in ARMS
                      else bench[name])
            wealth = np.exp(series.cumsum())
            colour, dash, width = STYLE[name]
            ax.plot(wealth.index, wealth.values, color=colour, ls=dash,
                    lw=width, label=name, zorder=3 if name == "M1" else 2)
        for year in sorted({d.year for d in index})[1:]:   # window boundaries
            ax.axvline(pd.Timestamp(f"{year}-01-01"), color="grey",
                       ls="--", lw=0.6, alpha=0.5, zorder=1)
        ax.set_yscale("log")
        ax.set_yticks([0.9, 1.0, 1.2, 1.5, 2.0, 2.5])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("cumulative wealth")
        ax.set_title(f"{algo.upper()} (eta = 1)", loc="left")
    axes[-1].set_xlabel("test year")
    axes[0].legend(ncol=4, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    written = _save(fig, "fig_wealth_curves")
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Figure 2: cross-seed dispersion
# ---------------------------------------------------------------------------

def plot_seed_dispersion(tbill) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {(a, arm): _per_seed_sharpe(a, arm, tbill)
            for a in ALGOS for arm in ARMS}
    years = sorted(data[("ppo", "M1")]["year"].unique())
    # deterministic per-seed offsets; no jitter, so reruns are byte-identical
    offset = {s: (s - 2) * 0.13 for s in range(config.N_SEEDS)}

    fig, axes = plt.subplots(2, len(ARMS), figsize=(11, 5.0),
                             sharex=True, sharey=True)
    for row, algo in enumerate(ALGOS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            frame = data[(algo, arm)]
            ax.axhline(0.0, color="grey", ls="--", lw=0.8, zorder=1)
            rest = frame[~frame["selected"]]
            ax.scatter([years.index(y) + offset[s]
                        for y, s in zip(rest["year"], rest["seed"])],
                       rest["sharpe"], s=14, color="lightsteelblue",
                       zorder=2, label="seed" if (row, col) == (0, 0) else None)
            sel = frame[frame["selected"]]
            ax.scatter([years.index(y) + offset[s]
                        for y, s in zip(sel["year"], sel["seed"])],
                       sel["sharpe"], s=22, color="tab:blue", zorder=3,
                       label="selected" if (row, col) == (0, 0) else None)
            if row == 0:
                ax.set_title(arm)
            if col == 0:
                ax.set_ylabel(f"{algo.upper()}\nannualized Sharpe")
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y)[-2:] for y in years], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("test year", fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    written = _save(fig, "fig_seed_dispersion")
    plt.close(fig)
    return written


def main() -> None:
    dates, ohlc = load_ohlc_tensor()
    tbill = pd.Series(tbill_rate(dates), index=dates)
    written = plot_wealth_curves(dates, ohlc) + plot_seed_dispersion(tbill)
    for path in written:
        print("wrote", path)
