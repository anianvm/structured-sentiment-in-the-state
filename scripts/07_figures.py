"""Step 7 of the run order: analysis-chapter figures.

Reads the walk-forward artifacts of step 5 from results/walkforward/ and
writes the two exhibits of the analysis chapter to results/figures/ as PNG
and PDF. Nothing is retrained; run after (any subset of) step 5 has finished
and step 6 has been evaluated.

Usage:  python scripts/07_figures.py
"""

import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/07_figures.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.figures import main  # noqa: E402

if __name__ == "__main__":
    main()
