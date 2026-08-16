"""Effect of sector context on LLM sentiment: anonymised vs sector-informed.

Ported unchanged from master_thesis/data_exploration/sector_context_comparison.py
(methodology §LLM Sentiment extraction — anonymisation cost); only paths,
imports and module names were adapted.

Compares the original structured scores (fully anonymised headlines) with the
sector variant (identical input plus one line naming the company's GICS
sector). Both runs share the model, prompt core, parameters, and headlines, so
every difference measures the effect of the sector line alone. This is the
source of the numbers in the methodology chapter: direction changes on 11
percent of headlines, sign reversals on fewer than half a percent.

Outputs
  results/sector_context_flips.csv   per-sector flip and shift statistics
  printed report                     overall flip rate, transition matrix,
                                     dimension shifts, sector breakdown

Run from the project root:
  python -m src.sentiment_pipeline.compare_sector_context
"""

import numpy as np  # noqa: F401  (kept from the original import block)
import pandas as pd

from src import config

ORIG = config.LLM_SCORES_FILE
SECT = config.LLM_SECTOR_SCORES_FILE
# Not in config: analysis output, regenerable from the two committed inputs.
OUT = config.RESULTS_DIR / "sector_context_flips.csv"

HORIZON_ORD = {"intraday": 1, "days": 2, "weeks": 3, "quarter_plus": 4}


def main() -> None:
    o = pd.read_csv(ORIG)
    s = pd.read_csv(SECT)
    m = o.merge(s, on=["date", "ticker"], suffixes=("_o", "_s"))
    m = m.dropna(subset=["direction_o", "direction_s"]).copy()
    print(f"aligned headlines: {len(m):,}\n")

    # --- direction ---
    m["flip"] = m["direction_o"] != m["direction_s"]
    m["sign_flip"] = (m["direction_o"] * m["direction_s"]) < 0
    print("=== direction ===")
    print(f"changed at all : {m['flip'].mean():.2%}")
    print(f"sign reversal  : {m['sign_flip'].mean():.2%} "
          f"(positive <-> negative)")
    print("\ntransition matrix (rows = anonymised, cols = with sector, %):")
    ct = pd.crosstab(m["direction_o"], m["direction_s"], normalize=True) * 100
    print(ct.round(2).to_string())

    # --- other dimensions ---
    m["hor_o"] = m["horizon_o"].map(HORIZON_ORD)
    m["hor_s"] = m["horizon_s"].map(HORIZON_ORD)
    print("\n=== other dimensions (mean anonymised -> mean with sector) ===")
    for lab, a, b in [("magnitude", "magnitude_o", "magnitude_s"),
                      ("horizon (ordinal)", "hor_o", "hor_s"),
                      ("confidence", "confidence_o", "confidence_s")]:
        ch = (m[a] != m[b]).mean()
        print(f"{lab:18s}: {m[a].mean():.3f} -> {m[b].mean():.3f}   "
              f"changed on {ch:.1%} of headlines")

    # --- by sector ---
    sec = (m.groupby("sector")
           .agg(n=("flip", "size"), flip=("flip", "mean"),
                sign_flip=("sign_flip", "mean"),
                dir_o=("direction_o", "mean"), dir_s=("direction_s", "mean"))
           .sort_values("flip", ascending=False))
    print("\n=== by sector (share of headlines whose direction changed) ===")
    print(f"{'sector':26s} {'n':>7s} {'flip':>7s} {'signflip':>9s} "
          f"{'mean dir before':>16s} {'after':>7s}")
    for name, r in sec.iterrows():
        print(f"{name:26s} {int(r.n):>7,} {r.flip:>7.1%} {r.sign_flip:>9.2%} "
              f"{r.dir_o:>16.3f} {r.dir_s:>7.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sec.to_csv(OUT)

    # --- where flips happen: neutral boundary vs sign ---
    flips = m[m["flip"]]
    to_dir = (flips["direction_o"] == 0).mean()
    to_neu = (flips["direction_s"] == 0).mean()
    print(f"\nof all direction changes: {to_dir:.1%} gained a direction "
          f"(0 -> +/-), {to_neu:.1%} lost one (+/- -> 0), "
          f"{1 - to_dir - to_neu:.1%} reversed sign")
    print(f"\nsaved -> {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
