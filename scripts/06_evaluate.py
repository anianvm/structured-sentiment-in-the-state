"""Step 6 of the run order: evaluation report.

Reads the walk-forward artifacts of step 5 from results/walkforward/,
computes the benchmark strategies and all performance metrics
(§ Benchmarks and Metrics), and writes the thesis tables and figures.
Run after (any subset of) the step-5 cells have finished.

Usage:  python scripts/06_evaluate.py
"""

import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/06_evaluate.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.report import main  # noqa: E402

if __name__ == "__main__":
    main()
