"""Step 2 of the run order: select the lookback window k.

Run this ONCE, under PPO; the selected window is applied to SAC as well (see
src/experiments/window_selection.py for why). Update config.WINDOW_K to the
value it reports before running step 3, since k fixes the observation space
that every later stage is tuned and trained on.

Scores every candidate in config.K_CANDIDATES on the control arm M1 at eta = 1
(training split 2015-2018, scored on the 2019 tuning-validation year) and
writes the winner to results/window_selection/{algo}.json. Runs BEFORE the
hyperparameter search, because k fixes the observation space that every later
stage is tuned and trained on. Requires the market data of step 1.

Usage:  python scripts/02_select_window.py --algo ppo [--seeds 5]
"""

import argparse
import sys
from pathlib import Path

# The repo root must be importable when this file is run directly as
# "python scripts/02_select_window.py" from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.experiments.window_selection import select_window  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=config.ALGOS)
    parser.add_argument("--seeds", type=int, default=None,
                        help=f"seeds per candidate "
                             f"(default config.TUNE_RERUN_SEEDS = "
                             f"{config.TUNE_RERUN_SEEDS})")
    parser.add_argument("--candidates", type=int, nargs="+", default=None,
                        help=f"candidate windows "
                             f"(default config.K_CANDIDATES = "
                             f"{config.K_CANDIDATES})")
    args = parser.parse_args()
    select_window(args.algo, candidates=args.candidates, seeds=args.seeds)


if __name__ == "__main__":
    main()
