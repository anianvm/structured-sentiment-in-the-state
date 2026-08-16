"""Step 4 of the run order: calibrate the training budget (once per algorithm).

Retrains the tuned configuration of step 3 on M1 at eta = 1, traces the
validation-return curve, and freezes the plateau budget to
results/budget/{algo}.json — the number of steps every walk-forward run in
step 5 then trains for. Run once with --algo ppo and once with --algo sac.

Usage:  python scripts/04_calibrate_budget.py --algo ppo
"""

import argparse
import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/04_calibrate_budget.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.experiments.training_budget import (  # noqa: E402
    EVAL_EVERY_DEFAULT, MAX_STEPS_DEFAULT, calibrate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=config.ALGOS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT,
                        help="upper end of the calibration curve")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT,
                        help="validation-rollout spacing along the curve")
    args = parser.parse_args()
    calibrate(args.algo, max_steps=args.max_steps, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
