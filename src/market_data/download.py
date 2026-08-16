"""Download the daily market-data snapshot from Yahoo Finance.

Implements the retrieval step of the thesis methodology chapter, section "Universe and
Market Data": daily OHLCV for the thirty Dow constituents (config.DOW30) plus
the auxiliary series (config.AUX_TICKERS: DIA, ^DJI, ^VIX, ^IRX), all pulled
from config.CONTEXT_ANCHOR_START (2005) so the market-context features have a
decade of pre-sample history.

Two details matter downstream:

* ``auto_adjust=False`` keeps BOTH the raw ``Close`` and the dividend/split
  adjusted ``Adj Close`` column. src/market_data/load.py later forms the
  adjustment factor AdjClose/Close per day and applies it to Close/High/Low,
  so the agent sees total-return price levels.
* One CSV per ticker, named with the caret stripped (``^VIX`` -> ``VIX.csv``),
  with columns ``Date, Adj Close, Close, High, Low, Open, Volume`` — the exact
  format load.py parses. Do not change this layout without changing load.py.

Usage (from the repo root):

    python -m src.market_data.download              # full snapshot
    python -m src.market_data.download --only IRX   # fetch one missing series
    python -m src.market_data.download --only VIX,IRX
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import config

# Column order of the on-disk snapshot (after the Date index). load.py selects
# columns by NAME, but keeping the order fixed makes the CSVs diff-able.
_CSV_COLUMNS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


def _select_tickers(only: list[str] | None) -> list[str]:
    """Resolve ``only`` names against the configured universe.

    Names are matched case-insensitively and with or without the leading "^"
    (so ``--only IRX`` finds ``^IRX``). Unknown names raise, because a typo
    would otherwise silently download a wrong series.
    """
    universe = config.DOW30 + config.AUX_TICKERS
    if only is None:
        return universe
    by_name = {t.replace("^", "").upper(): t for t in universe}
    selected = []
    for name in only:
        key = name.replace("^", "").upper()
        if key not in by_name:
            raise ValueError(
                f"Unknown ticker {name!r}. Valid names: {', '.join(universe)}"
            )
        selected.append(by_name[key])
    return selected


def _download_one(ticker: str, out_dir: Path) -> Path | None:
    """Fetch one ticker's full daily history and write <TICKER>.csv."""
    frame = yf.download(
        ticker,
        start=config.CONTEXT_ANCHOR_START,
        auto_adjust=False,
        progress=False,
    )
    if frame is None or frame.empty:
        print(f"WARNING: no data returned for {ticker}, nothing written")
        return None

    # Newer yfinance versions return a (field, ticker) MultiIndex even for a
    # single ticker; flatten it to plain field names.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame[_CSV_COLUMNS]

    # Make sure the index writes as plain YYYY-MM-DD dates.
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"

    path = out_dir / f"{ticker.replace('^', '')}.csv"
    frame.to_csv(path)
    print(
        f"{ticker}: {len(frame)} rows, "
        f"{frame.index[0].date()} -> {frame.index[-1].date()}  ({path.name})"
    )
    return path


def download_all(out_dir=config.MARKET_DIR, only: list[str] | None = None) -> list[Path]:
    """Download the snapshot (or, with ``only``, a subset) into ``out_dir``.

    ``only`` is a list of ticker names; use it to re-fetch a single missing
    series without touching the CSVs already on disk. Returns the list of
    paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ticker in _select_tickers(only):
        path = _download_one(ticker, out_dir)
        if path is not None:
            written.append(path)
    print(f"wrote {len(written)} file(s) to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        metavar="TICKER[,TICKER]",
        default=None,
        help="comma-separated subset to fetch (e.g. --only IRX); "
        "default: all of config.DOW30 + config.AUX_TICKERS",
    )
    args = parser.parse_args()
    only = None
    if args.only is not None:
        only = [name.strip() for name in args.only.split(",") if name.strip()]
    download_all(only=only)


if __name__ == "__main__":
    main()
