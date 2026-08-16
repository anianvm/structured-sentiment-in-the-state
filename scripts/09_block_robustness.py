"""Step 9 of the run order: block-length robustness of the eta=1 Sharpe
contrasts (tab:res-block).

Re-runs the studentized circular block bootstrap of src/evaluation/inference
for all eighteen eta=1 contrasts under the default block length b=5 and the
robustness value b=10, leaving the draw count and generator seed unchanged,
and writes results/tables/block_robustness.json. The Holm adjustment applies
within each algorithm's four-contrast H1 family; every other row carries its
raw one-sided p-value, exactly as in the thesis table.

The input is the selected-seed daily record (test_daily.csv) of each
walk-forward window, pooled across the six test years -- the same series the
headline contrasts of src/evaluation/contrasts.py are computed on.

Usage:  python scripts/09_block_robustness.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation.contrasts import CONTRAST_SET  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    holm_adjust, sharpe_difference_test,
)
from src.features.market_context import tbill_rate  # noqa: E402

BLOCKS = (5, 10)


def _pooled_selected(algo: str, arm: str) -> pd.Series:
    """Selected-seed daily log returns, concatenated over the test windows."""
    root = config.RESULTS_DIR / "walkforward" / algo.lower()
    parts = [
        pd.read_csv(d / "test_daily.csv", index_col=0,
                    parse_dates=True)["log_return"]
        for d in sorted((root / f"{arm}_eta1.0").iterdir()) if d.is_dir()
    ]
    return pd.concat(parts)


def main() -> None:
    out: dict[str, list[dict]] = {}
    for algo in ("sac", "ppo"):
        series = {arm: _pooled_selected(algo, arm)
                  for arm in ("M1", "M2", "M3", "M4", "M5", "M6")}
        tbill = pd.Series(tbill_rate(series["M1"].index),
                          index=series["M1"].index)
        rows = []
        for arm, base, hyp in CONTRAST_SET:
            row = {"algo": algo.upper(), "hyp": hyp, "arm": arm, "base": base}
            for block in BLOCKS:
                res = sharpe_difference_test(series[arm], series[base],
                                             tbill, block=block)
                row["d"] = res["difference"]
                row[f"p{block}"] = res["p_one_sided"]
            rows.append(row)
        # Holm within the four-contrast H1 family; other rows keep the raw p.
        for block in BLOCKS:
            h1 = [r for r in rows if r["hyp"] == "H1"]
            for r, adj in zip(h1, holm_adjust([r[f"p{block}"] for r in h1])):
                r[f"holm{block}"] = float(adj)
            for r in rows:
                r.setdefault(f"holm{block}", r[f"p{block}"])
        out[algo] = rows

    path = config.RESULTS_DIR / "tables" / "block_robustness.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    for rows in out.values():
        for r in rows:
            print("%s %-4s %s-%s  d=%+.3f  p5=%.4f p10=%.4f  "
                  "holm5=%.4f holm10=%.4f"
                  % (r["algo"], r["hyp"], r["arm"], r["base"], r["d"],
                     r["p5"], r["p10"], r["holm5"], r["holm10"]))


if __name__ == "__main__":
    main()
