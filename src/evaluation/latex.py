"""LaTeX row bodies for the thesis tables, written from the result CSVs.

Every table in the results chapter reports numbers that already exist in
`results/tables/*.csv`. Transcribing them by hand is where chapter and
artifact drift apart -- two such gaps were found by hand in August 2026, one
of them a whole arm estimated against the wrong regressor. The emitters here
close that path: `write_all` writes one `.tex` fragment per table into
`results/tables/latex/`, the chapter `\\input`s the fragment, and a printed
digit that disagrees with an artifact becomes impossible rather than merely
unlikely.

Only the ROW BODIES are generated. The table environment, column
specification, header rules and caption stay in the .tex, because they carry
typesetting decisions rather than numbers, and because a caption often says
something the CSV cannot know. Tables assembled from several sources or
written by hand (`tab:res-verdicts`, `tab:res-block`,
`tab:fixed-hyperparams`, `tab:sentiment-persistence`) are deliberately not
covered; `results/tables/latex/README.md` records that boundary so the
omission reads as a decision rather than an oversight.

Formatting conventions follow the chapter as it was written: negative
numbers are wrapped in math mode so the minus sign is typeset as one,
`\\textbf` marks the best value in a panel, and `---` marks a cell that does
not apply.
"""

from pathlib import Path

import pandas as pd

ARMS = ("M1", "M2", "M3", "M4", "M5", "M6")
ALGOS = (("PPO", "A"), ("SAC", "B"))
ETAS = (1.0, 3.0, 7.0)
POOLED = "2021-2026"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def num(value, dp: int = 2, bold: bool = False, math: bool = False) -> str:
    """One cell. Negatives go in math mode so LaTeX sets a real minus.

    `\\textbf` around `$-1.2$` would not embolden the digits, so a bold
    negative uses `{\\boldmath$-1.2$}` -- the same pair of spellings the
    hand-written tables use.

    `math` additionally wraps positives, which the contrast tables do so that
    every figure in a column is set from the same font; the descriptive
    tables leave positives in text mode. Both conventions are the chapter's.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "---"
    text = f"{value:.{dp}f}"
    if text.startswith("-"):
        text = "$-" + text[1:] + "$"
        return f"{{\\boldmath{text}}}" if bold else text
    if math:
        text = f"${text}$"
        return f"{{\\boldmath{text}}}" if bold else text
    return f"\\textbf{{{text}}}" if bold else text


def _panel(label: str, algo: str, span: int) -> str:
    return f"\\multicolumn{{{span}}}{{l}}{{\\emph{{Panel {label}: {algo}}}}} \\\\"


def _yesno(flag) -> str:
    return "yes" if bool(flag) else "no"


# ---------------------------------------------------------------------------
# Performance (tab:res-perf-pooled, tab:res-perf-window)
# ---------------------------------------------------------------------------

# column -> whether the best value in a panel is the largest one
_PERF_COLUMNS = {
    "cumulative_return": (True, 1, 100.0),
    "annualized_return": (True, 1, 100.0),
    "annualized_volatility": (False, 1, 100.0),
    "sharpe_ratio": (True, 2, 1.0),
    "max_drawdown": (False, 1, 100.0),
}


def perf_pooled(perf: dict) -> str:
    """`tab:res-perf-pooled`: pooled record per arm, benchmarks beneath.

    The best value of each column within each algorithm panel is emboldened,
    which for volatility and drawdown means the smallest.
    """
    out = []
    for algo, letter in ALGOS:
        frame = perf[algo]
        frame = frame[frame["period"] == POOLED].set_index("strategy")
        rows = {arm: frame.loc[f"{arm}_eta1.0"] for arm in ARMS
                if f"{arm}_eta1.0" in frame.index}
        best = {c: (max if hi else min)(r[c] for r in rows.values())
                for c, (hi, _, _) in _PERF_COLUMNS.items()}
        out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{Panel {letter}: {algo}}} \\\\")
        for arm, r in rows.items():
            cells = [num(r[c] * scale, dp, bold=r[c] == best[c])
                     for c, (_, dp, scale) in _PERF_COLUMNS.items()]
            out.append(f"{arm} & " + " & ".join(cells) + " \\\\")
        if algo == "PPO":
            out.append("\\addlinespace")
    out.append("\\midrule")
    bench = perf["PPO"]
    bench = bench[bench["period"] == POOLED].set_index("strategy")
    for name in ("UCRP", "DIA"):
        r = bench.loc[name]
        cells = [num(r[c] * scale, dp)
                 for c, (_, dp, scale) in _PERF_COLUMNS.items()]
        out.append(f"\\emph{{{name}}} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def perf_window(perf: dict, years: tuple) -> str:
    """`tab:res-perf-window`: Sharpe then annualized return, one column a year.

    Bold marks the best arm in each (panel, year) column, separately for the
    two blocks. Benchmarks are never emboldened; they are the comparison, not
    a competitor within the panel.
    """
    out = []
    for algo, letter in ALGOS:
        frame = perf[algo].set_index(["strategy", "period"])
        def value(arm, year, col):
            key = (f"{arm}_eta1.0", str(year))
            return frame.loc[key, col] if key in frame.index else None

        best = {}
        for col in ("sharpe_ratio", "annualized_return"):
            for year in years:
                vals = [v for v in (value(a, year, col) for a in ARMS)
                        if v is not None]
                best[(col, year)] = max(vals) if vals else None
        out.append(f"\\multicolumn{{13}}{{@{{}}l}}{{Panel {letter}: {algo}}} \\\\")
        for arm in ARMS:
            cells = []
            for col, dp, scale in (("sharpe_ratio", 2, 1.0),
                                   ("annualized_return", 1, 100.0)):
                for year in years:
                    v = value(arm, year, col)
                    cells.append(num(None if v is None else v * scale, dp,
                                     bold=v is not None
                                     and v == best[(col, year)]))
            out.append(f"{arm} & " + " & ".join(cells) + " \\\\")
        if algo == "PPO":
            out.append("\\addlinespace")
    out.append("\\midrule")
    frame = perf["PPO"].set_index(["strategy", "period"])
    for name in ("UCRP", "DIA"):
        cells = []
        for col, dp, scale in (("sharpe_ratio", 2, 1.0),
                               ("annualized_return", 1, 100.0)):
            for year in years:
                key = (name, str(year))
                cells.append(num(frame.loc[key, col] * scale
                                 if key in frame.index else None, dp))
        out.append(f"\\emph{{{name}}} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Return contrasts (tab:res-h1, tab:res-h3, tab:res-h4)
# ---------------------------------------------------------------------------

def _contrast_row(label: str, r) -> str:
    # The supplementary M2-M1 contrast sits outside every test family, so no
    # multiplicity applies to it and no verdict is drawn from it. The CSV
    # passes p_holm through unadjusted for that row; printing it would imply
    # a family membership the design explicitly denies.
    supplementary = r["hypothesis"] == "SUPP"
    holm = ("---" if supplementary or pd.isna(r["p_holm"])
            else num(r["p_holm"], 3, math=True))
    supported = "---" if supplementary else _yesno(r["supported"])
    return (f"{label} & {num(r['difference'], 2, math=True)} & "
            f"$[{r['ci_low']:.2f},\\,{r['ci_high']:.2f}]$ & "
            f"{num(r['p_one_sided'], 3, math=True)} & {holm} & "
            f"{num(r['seed_band'], 2, math=True)} & "
            f"{_yesno(r['exceeds_seed_band'])} & {supported} \\\\")


def contrasts_h1(contrasts: dict) -> str:
    """`tab:res-h1`: the four LLM arms against M1, with M2 as a supplement."""
    out = []
    for algo, letter in ALGOS:
        c = contrasts[algo]
        c = c[(c["eta"] == 1.0)]
        out.append(_panel(letter, algo, 8))
        for _, r in c[c["hypothesis"] == "H1"].iterrows():
            out.append(_contrast_row(r["arm"], r))
        supp = c[c["hypothesis"] == "SUPP"]
        if not supp.empty:
            out.append("\\addlinespace")
            out.append("\\multicolumn{8}{l}{\\emph{\\quad Supplementary "
                       "contrast (descriptive): M2 against M1}} \\\\")
            for _, r in supp.iterrows():
                out.append(_contrast_row(r["arm"], r))
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


def _contrast_table(contrasts: dict, hypothesis: str) -> str:
    out = []
    for algo, letter in ALGOS:
        c = contrasts[algo]
        c = c[(c["eta"] == 1.0) & (c["hypothesis"] == hypothesis)]
        out.append(_panel(letter, algo, 8))
        for _, r in c.iterrows():
            out.append(_contrast_row(f"{r['arm']}--{r['baseline']}", r))
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


def contrasts_h3(contrasts: dict) -> str:
    """`tab:res-h3`: the single extraction contrast, M3 against M2."""
    return _contrast_table(contrasts, "H3")


def contrasts_h4(contrasts: dict) -> str:
    """`tab:res-h4`: the three enrichment-ladder steps."""
    return _contrast_table(contrasts, "H4")


def regime(contrasts: dict) -> str:
    """`tab:res-regime`: the leave-one-year-out diagnostic.

    `Sign-stable` is the weak reading (the contrast never changes sign);
    `Strict` additionally demands every refit's interval exclude zero.
    """
    names = {"H1": "H1", "H3": "H3", "H4": "H4", "SUPP": "Suppl."}
    out = []
    for algo, letter in ALGOS:
        c = contrasts[algo]
        c = c[c["eta"] == 1.0]
        out.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{Panel {letter}: "
                   f"{algo}}}}} \\\\")
        previous = None
        for _, r in c.iterrows():
            head = names[r["hypothesis"]] if r["hypothesis"] != previous else ""
            previous = r["hypothesis"]
            out.append(f"{head:9} & {r['arm']}--{r['baseline']} & "
                       f"{'Yes' if r['loo_sign_stable'] else 'No'} & "
                       f"{'Yes' if r['regime_robust'] else 'No'} \\\\")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Allocation hypotheses (tab:res-h2, tab:res-h2-dist, tab:res-h5-r)
# ---------------------------------------------------------------------------

def h2_separation(h2: dict) -> str:
    """`tab:res-h2`: a value row per arm over a row of across-seed SDs."""
    out = []
    for algo, letter in ALGOS:
        out.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{Panel {letter}: "
                   f"{algo}}}}} \\\\")
        frame = h2[algo].set_index("arm")
        for i, arm in enumerate(ARMS):
            r = frame.loc[arm]
            cells = [f"{r[f'{m}_eta{int(e)}'] * 100:.1f}"
                     for m in ("cash", "vol", "mdd") for e in ETAS]
            flags = "/".join(_yesno(r[c]) for c in
                             ("cash_monotone_up", "vol_monotone_down",
                              "mdd_monotone_down"))
            out.append(f"{arm} & " + " & ".join(cells) + f" & {flags} \\\\")
            sds = " & ".join(
                f"{{\\scriptsize ({r[f'cash_sd_eta{int(e)}'] * 100:.1f})}}"
                for e in ETAS)
            out.append(f"   & {sds} & \\multicolumn{{6}}{{c}}{{}} & \\\\[2pt]")
            if i == 0:                  # M1 is tested, M2-M6 are robustness
                out.append("\\addlinespace")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


def h2_distance(h2: dict) -> str:
    """`tab:res-h2-dist`: composition distance against its across-seed bench."""
    pairs = (("eta1_eta3", "eta1_eta3"), ("eta3_eta7", "eta3_eta7"),
             ("eta1_eta7", "eta1_eta7"))
    out = []
    for algo, letter in ALGOS:
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{Panel {letter}: "
                   f"{algo}}}}} \\\\")
        frame = h2[algo].set_index("arm")
        for i, arm in enumerate(ARMS):
            r = frame.loc[arm]
            cells = []
            for dist, bench in pairs:
                cells += [f"{r[f'comp_dist_{dist}']:.3f}",
                          f"{r[f'bench_{bench}']:.3f}"]
            out.append(f"{arm} & " + " & ".join(cells) + " \\\\")
            if i == 0:
                out.append("\\addlinespace")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


_H5_LABELS = {"M1": "M1 (placebo)", "M2": "M2 (FinBERT)"}

_H5_DIFF_LADDER = [("M4", "M3"), ("M5", "M4"), ("M6", "M5")]
_H5_DIFF_ETA_STEPS = [("eta1-3", "1\\to3"), ("eta3-7", "3\\to7")]


def h5_difference(h5_diff: dict) -> str:
    """`tab:res-h5-dr`: Delta R = -delta_1 of the differenced regression.

    Upper block: representation-ladder steps at each eta. Lower block: the
    two eta steps within each LLM arm. PPO and SAC sit side by side as
    column groups rather than panels, following the hand-set original. M2
    never appears (different news index, see hypotheses.h5_difference_rows).
    """
    def cells(algo, kind, step, arm="", eta=None):
        f = h5_diff[algo]
        sel = f[(f["kind"] == kind) & (f["step"] == step)]
        if arm:
            sel = sel[sel["arm"] == arm]
        if eta is not None:
            sel = sel[sel["eta"] == eta]
        r = sel.iloc[0]
        return [num(r["delta_R"], 3), num(r["se"], 3), num(r["t"], 2)]

    out = ["\\multicolumn{7}{@{}l}{\\emph{Representation-ladder steps}} \\\\"]
    for eta in (1.0, 3.0, 7.0):
        for hi, lo in _H5_DIFF_LADDER:
            row = [f"$\\eta={eta:g}$: {hi}$-${lo}"]
            for algo, _ in ALGOS:
                row += cells(algo, "ladder", f"{hi}-{lo}", eta=eta)
            out.append(" & ".join(row) + " \\\\")
    out.append("\\midrule")
    out.append("\\multicolumn{7}{@{}l}{\\emph{Risk-aversion steps}} \\\\")
    for arm in ("M3", "M4", "M5", "M6"):
        for step, text in _H5_DIFF_ETA_STEPS:
            row = [f"{arm}: $\\eta\\,{text}$"]
            for algo, _ in ALGOS:
                row += cells(algo, "eta", step, arm=arm)
            out.append(" & ".join(row) + " \\\\")
    return "\n".join(out)


def h5_responsiveness(h5: dict) -> str:
    """`tab:res-h5-r`: R = -beta_1 with its standard error and t, per eta.

    M1 is labelled a placebo because it observes no sentiment, and M2 is
    labelled FinBERT because it is estimated against its own news index --
    the `news_index` column records which regressor each row used.
    """
    out = []
    for algo, letter in ALGOS:
        out.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{Panel {letter}: "
                   f"{algo}}}}} \\\\")
        frame = h5[algo].set_index(["arm", "eta"])
        for arm in ARMS:
            cells = []
            for eta in ETAS:
                r = frame.loc[(arm, eta)]
                cells += [num(r["responsiveness_R"], 3),
                          f"{r['se_sentiment']:.3f}", num(r["t_R"], 2)]
            out.append(f"{_H5_LABELS.get(arm, arm)} & "
                       + " & ".join(cells) + " \\\\")
            if arm == "M2":
                out.append("\\addlinespace")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Behaviour (tab:res-behaviour)
# ---------------------------------------------------------------------------

_BEHAVIOUR = (("avg_oneway_turnover", 2, 100.0), ("portfolio_entropy", 2, 1.0),
              ("risky_entropy", 2, 1.0), ("avg_largest_position", 1, 100.0),
              ("risky_largest_position", 1, 100.0),
              ("avg_cash_share", 1, 100.0), ("max_drawdown", 1, 100.0),
              ("cvar_5", 2, 100.0))


def behaviour(behaviour_tables: dict) -> str:
    """`tab:res-behaviour`: trading and concentration diagnostics at eta = 1.

    The benchmarks have no behaviour of their own to report -- UCRP is
    uniform by construction and DIA never trades -- so only their drawdown
    and CVaR are filled and the rest is `---`.
    """
    out = []
    for algo, letter in ALGOS:
        frame = behaviour_tables[algo]
        frame = frame[frame["period"] == POOLED].set_index("strategy")
        out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Panel {letter}: "
                   f"{algo}}}}} \\\\")
        for arm in ARMS:
            r = frame.loc[f"{arm}_eta1.0"]
            out.append(f"{arm} & " + " & ".join(
                num(r[c] * scale, dp) for c, dp, scale in _BEHAVIOUR) + " \\\\")
        out.append("\\midrule")
    out.append("\\multicolumn{9}{@{}l}{\\emph{Benchmarks}} \\\\")
    bench = behaviour_tables["PPO"]
    bench = bench[bench["period"] == POOLED].set_index("strategy")
    for name in ("UCRP", "DIA"):
        cells = ["---"] * 6
        for column, dp, scale in _BEHAVIOUR[6:]:
            cells.append(num(bench.loc[name, column] * scale, dp)
                         if name in bench.index else "---")
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Appendix (tab:resapp-seeds, tab:resapp-seed-cash)
# ---------------------------------------------------------------------------

def seed_sharpe(sharpe_by_seed) -> str:
    """`tab:resapp-seeds`: every seed's pooled test Sharpe at eta = 1.

    No seed is marked as selected: selection is per window, so the pooled
    record stitches a possibly different seed in each window and no single
    column corresponds to the headline result.
    """
    out = []
    for algo, letter in ALGOS:
        out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{Panel {letter}: "
                   f"{algo}}}}} \\\\[1pt]")
        for arm in ARMS:
            values = sharpe_by_seed(algo, arm)
            cells = [f"{v:.2f}" for _, v in sorted(values.items())]
            out.append(f"{arm} & " + " & ".join(cells) + " \\\\")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


def seed_cash(h2: dict, cash_by_seed) -> str:
    """`tab:resapp-seed-cash`: M1's five seeds at each eta, mean and headline.

    The headline column is the validation-selected series of `tab:res-h2`,
    which stitches the per-window selected seed and so matches no single
    seed column; printing them side by side is the point of the table.
    """
    out = []
    for algo, letter in ALGOS:
        out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textbf{{Panel {letter}: "
                   f"{algo}}}}} \\\\[1pt]")
        headline = h2[algo].set_index("arm").loc["M1"]
        for eta in ETAS:
            shares = cash_by_seed(algo, "M1", eta)
            cells = [f"{shares[s] * 100:.2f}" for s in sorted(shares)]
            mean = sum(shares.values()) / len(shares) * 100
            out.append(f"{int(eta)} & " + " & ".join(cells)
                       + f" & & {mean:.2f} & "
                       f"{headline[f'cash_eta{int(eta)}'] * 100:.2f} \\\\")
        if algo == "PPO":
            out.append("\\midrule")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

README = """# Generated table bodies

Written by `src/evaluation/latex.py` during `scripts/06_evaluate.py`. Each
file holds the ROW BODY of one thesis table -- everything between
`\\midrule` and `\\bottomrule` -- so the chapter can `\\input` it and no
printed digit is transcribed by hand.

Not generated, and transcribed deliberately: `tab:res-verdicts` and
`tab:res-block` (composed from several sources),
`tab:fixed-hyperparams` and `tab:sentiment-persistence` (configuration and
pipeline facts rather than results). Those four are the boundary of this
module; everything else in the results chapter and the results appendix is
written from the CSVs beside this directory.
"""


def write_all(tables_dir, perf: dict, behaviour_tables: dict,
              contrast_tables: dict, h2: dict, h5: dict, h5_diff: dict,
              years: tuple, sharpe_by_seed, cash_by_seed) -> Path:
    """Write every generated body into `tables_dir/latex/` and return it."""
    out_dir = Path(tables_dir) / "latex"
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies = {
        "res-perf-pooled": perf_pooled(perf),
        "res-perf-window": perf_window(perf, years),
        "res-h1": contrasts_h1(contrast_tables),
        "res-h3": contrasts_h3(contrast_tables),
        "res-h4": contrasts_h4(contrast_tables),
        "res-regime": regime(contrast_tables),
        "res-h2": h2_separation(h2),
        "res-h2-dist": h2_distance(h2),
        "res-h5-r": h5_responsiveness(h5),
        "res-h5-dr": h5_difference(h5_diff),
        "res-behaviour": behaviour(behaviour_tables),
        "resapp-seeds": seed_sharpe(sharpe_by_seed),
        "resapp-seed-cash": seed_cash(h2, cash_by_seed),
    }
    for name, body in bodies.items():
        (out_dir / f"{name}.tex").write_text(body + "\n")
    (out_dir / "README.md").write_text(README)
    return out_dir
