"""Step 1 of the run order: download all market data from Yahoo Finance.

Fetches the Dow 30 plus the auxiliary series (DIA, ^DJI, ^VIX, ^IRX) into
data/market/. Everything downstream (features, environment, training) reads
these CSVs, so this script must run first — and it is the only step that
needs network access. Re-running refreshes the files in place.

Usage:  python scripts/01_download_market_data.py [--only AAPL,MSFT,^VIX]
"""

import argparse
import sys
from pathlib import Path

# The scripts/ directory is not a package; put the repo root on sys.path so
# "python scripts/01_download_market_data.py" finds the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market_data.download import download_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default=None,
        help="comma-separated subset of tickers to (re-)download "
             "(default: all Dow 30 + auxiliary series)")
    args = parser.parse_args()
    download_all(only=args.only.split(",") if args.only else None)


if __name__ == "__main__":
    main()
