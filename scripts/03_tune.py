"""Step 3 of the run order: hyperparameter search (once per algorithm).

Runs the Optuna TPE search of src/experiments/tune.py on the control arm M1
at eta = 1 and freezes the winner to results/tuning/{algo}_best.json.
Requires the market data of step 1. Safe to interrupt: the study lives in
sqlite and resumes. Run once with --algo ppo and once with --algo sac,
before steps 4 and 5.

Usage:  python scripts/03_tune.py --algo ppo
"""

import argparse
import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/03_tune.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.experiments.tune import run_tuning  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=config.ALGOS)
    parser.add_argument("--worker", type=int, default=0,
                        help="worker index for parallel search. Each\n"
                             "concurrent worker MUST get a distinct value: the\n"
                             "sampler seed is offset by it, and identical seeds\n"
                             "make every worker propose the same configuration.")
    args = parser.parse_args()
    run_tuning(args.algo, worker=args.worker)


if __name__ == "__main__":
    main()
