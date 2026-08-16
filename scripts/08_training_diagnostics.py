"""Step 8 of the run order: training diagnostics table and figure.

Reads the Stable Baselines3 progress logs written alongside the walk-forward
artifacts of step 5 and writes, to results/tables/ and results/figures/, the
per-run diagnostics, their distribution over the 540 runs per algorithm, and
the two-panel trajectory figure. Nothing is retrained.

Usage:  python scripts/08_training_diagnostics.py
"""

import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/08_training_diagnostics.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.training_diagnostics import main  # noqa: E402

if __name__ == "__main__":
    main()
