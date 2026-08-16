"""Step 5 of the run order: the walk-forward experiment itself.

Trains and evaluates ONE (algo, arm, eta) cell across all six windows and
writes the artifacts under results/walkforward/. The full study is the
cross product ppo/sac x M1..M6 x config.ETA_LEVELS — 36 invocations of
this script (each seeds x windows = 30 training runs). Requires steps 1-4;
without step 4 the budget falls back to config.TUNE_STEP_CAP.

Usage:  python scripts/05_walkforward.py --algo ppo --arm M4 --eta 1
        [--timesteps 500000] [--seeds 0,1,2,3,4]
"""

import argparse
import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/05_walkforward.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.experiments.walkforward import run_walkforward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=config.ALGOS)
    parser.add_argument("--arm", required=True, choices=list(config.ARMS))
    parser.add_argument("--eta", required=True, type=float,
                        help=f"CRRA risk aversion; the study runs "
                             f"{config.ETA_LEVELS}")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override the calibrated training budget")
    parser.add_argument("--seeds", default=None,
                        help=f"comma-separated seed list "
                             f"(default: {config.SEEDS})")
    parser.add_argument("--windows", default=None,
                        help="comma-separated test years to run (default: all "
                             "six). Each window is independent, so this only "
                             "slices the work: use it to spread one cell over "
                             "several processes, or to re-run a single failed "
                             "window.")
    args = parser.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else config.SEEDS)
    windows = ([int(w) for w in args.windows.split(",")]
               if args.windows else None)
    run_walkforward(args.algo, args.arm, args.eta,
                    seeds=seeds, timesteps=args.timesteps, windows=windows)


if __name__ == "__main__":
    main()
