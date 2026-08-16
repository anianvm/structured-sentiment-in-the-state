"""Compare LLM (structured) vs FinBERT sentiment on the identical headlines.

Ported unchanged from master_thesis/src/sentiment/compare_scores.py, folded
together with master_thesis/data_exploration/llm_finbert_confusion.py
(methodology §Sentiment comparison); only paths, imports and module names were
adapted, and the helpers the two originals duplicated (Cohen's kappa, the
aligned merge, the confusion crosstab, the agreement report) appear once.

Both scorers saw the same anonymised, cleaned text, so any disagreement
reflects the scoring model, not the input. Produces:
  data/sentiment/sentiment_comparison.csv    per-headline side-by-side (LLM + FinBERT)
  results/llm_finbert_confusion.csv          counts of the 3x3 confusion matrix
  results/figures/fig_llm_finbert_confusion  row-normalised heatmap (.png and .pdf)
  printed report                             correlations, confusion matrix,
                                             agreement, Cohen's kappa

LLM signal      : direction in {-1,0,+1}; signed intensity = direction * magnitude.
FinBERT signal  : finbert_score = p_pos - p_neg in [-1,1]; label -> {+1,0,-1}.

Run from the project root:
  python -m src.sentiment_pipeline.compare_llm_finbert
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

LLM_CSV = config.LLM_SCORES_FILE
FB_CSV = config.FINBERT_SCORES_FILE
# Not in config: analysis outputs, regenerable from the two committed inputs.
OUT_PAIRS = config.SENTIMENT_DIR / "sentiment_comparison.csv"
OUT_CSV = config.RESULTS_DIR / "llm_finbert_confusion.csv"
OUT_FIG = config.RESULTS_DIR / "figures" / "fig_llm_finbert_confusion"

LABEL_TO_INT = {"positive": 1, "neutral": 0, "negative": -1}
ORDER = [1, 0, -1]                       # row/column order: positive, neutral, negative
NAME = {1: "positive", 0: "neutral", -1: "negative"}


def cohen_kappa(a: np.ndarray, b: np.ndarray, cats=(-1, 0, 1)) -> float:
    """Chance corrected agreement between two label sequences."""
    observed = np.mean(a == b)
    expected = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def load_aligned() -> pd.DataFrame:
    """Merge the LLM and FinBERT scores on the shared (date, ticker) key."""
    llm = pd.read_csv(LLM_CSV)
    fb = pd.read_csv(FB_CSV)
    m = llm.merge(fb, on=["date", "ticker"], suffixes=("_llm", "_fb"))
    before = len(m)
    m = m.dropna(subset=["direction", "magnitude", "finbert_score"]).copy()
    if len(m) < before:
        print(f"dropped {before - len(m)} rows with missing LLM scores")
    m["llm_direction"] = m["direction"].astype(int)
    m["llm_signed"] = m["direction"] * m["magnitude"]            # signed intensity
    m["fb_label_int"] = m["finbert_label"].map(LABEL_TO_INT).astype(int)
    return m


def confusion(m: pd.DataFrame) -> pd.DataFrame:
    """3x3 counts, rows = LLM direction, columns = FinBERT label."""
    return pd.crosstab(m["llm_direction"], m["fb_label_int"]).reindex(
        index=ORDER, columns=ORDER, fill_value=0)


def report(m: pd.DataFrame, ct: pd.DataFrame) -> None:
    print(f"aligned headlines: {len(m):,}\n")

    print("=== continuous correlation (FinBERT score vs LLM signal) ===")
    for label, col in (("LLM direction", "llm_direction"),
                       ("LLM signed intensity", "llm_signed")):
        p = pearsonr(m["finbert_score"], m[col])[0]
        s = spearmanr(m["finbert_score"], m[col])[0]
        print(f"  vs {label:22s}: Pearson {p:+.3f}   Spearman {s:+.3f}")

    print("\n=== confusion matrix (counts) ===")
    print("rows = LLM direction, columns = FinBERT label")
    print(ct.rename(index=NAME, columns=NAME).to_string())

    print("\n=== confusion matrix (row normalised, percent of each LLM class) ===")
    rown = ct.div(ct.sum(axis=1), axis=0) * 100
    print(rown.rename(index=NAME, columns=NAME).round(1).to_string())

    a, b = m["llm_direction"].values, m["fb_label_int"].values
    # where BOTH express a non-neutral opinion, do the signs match?
    both = m[(m["llm_direction"] != 0) & (m["fb_label_int"] != 0)]
    sign_agree = np.mean(np.sign(both["llm_direction"]) == np.sign(both["fb_label_int"]))
    print("\n=== agreement ===")
    print(f"  exact three class agreement : {np.mean(a == b):.1%}")
    print(f"  Cohen's kappa               : {cohen_kappa(a, b):.3f}")
    print(f"  both non neutral (n={len(both):,}): sign agreement {sign_agree:.1%}  "
          f"({len(both)/len(m):.0%} of all rows)")


def save_figure(ct: pd.DataFrame) -> None:
    rown = ct.div(ct.sum(axis=1), axis=0) * 100
    names = [NAME[c] for c in ORDER]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rown.values, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("FinBERT label")
    ax.set_ylabel("LLM direction")
    for i in range(3):
        for j in range(3):
            v = rown.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if v > 50 else "black",
                    fontweight="bold" if i == j else "normal", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("share of LLM class (%)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pairs_csv(m: pd.DataFrame) -> None:
    """Per-headline side-by-side table (the compare_scores.py output)."""
    cols = ["date", "ticker", "headline_llm", "llm_direction", "magnitude",
            "horizon", "confidence", "finbert_score", "finbert_label",
            "p_pos", "p_neg", "p_neu"]
    m[cols].to_csv(OUT_PAIRS, index=False)
    print(f"\nsaved pairs  -> {OUT_PAIRS}")


def main() -> None:
    m = load_aligned()
    ct = confusion(m)
    report(m, ct)
    save_pairs_csv(m)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    ct.rename(index=NAME, columns=NAME).to_csv(OUT_CSV)
    save_figure(ct)
    print(f"saved matrix -> {OUT_CSV.relative_to(config.ROOT)}")
    print(f"saved figure -> {OUT_FIG.relative_to(config.ROOT)}.png / .pdf")


if __name__ == "__main__":
    main()
