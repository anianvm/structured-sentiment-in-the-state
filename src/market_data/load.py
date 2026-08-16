"""Load the adjusted daily price data every other module builds on.

Implements the market-data half of the thesis methodology chapter, section "Universe
and Market Data": close, high, and low prices of the Dow 30 from Yahoo
Finance, adjusted for dividends and splits, on a common trading calendar.

Why adjusted prices. The agent's return in the environment is computed from
these levels, so they must reflect what a shareholder actually earns: price
changes PLUS dividends (and splits handled correctly). Unadjusted prices
would understate equity returns by the dividend yield (roughly 2% a year for
the Dow) and make holding cash look artificially attractive.

Why the ratio method. Yahoo provides an adjusted series ("Adj Close") only
for the close. We therefore form the per-day factor

    factor_t = AdjClose_t / Close_t

and multiply Close, High, and Low by it. This scales the whole trading range
of the day by the same dividend/split factor, which keeps high/low consistent
with the adjusted close (the adjusted close equals Adj Close exactly, and
high >= low is preserved because the factor is positive).

Survivorship bias. config.DOW30 is the index membership AFTER the November
2024 revision, held fixed over the whole 2015-2026 sample. Firms that were
removed along the way (e.g. INTC, DOW) are absent, so measured universe
returns are biased upward relative to a real-time investable Dow. The bias is
identical across all experimental arms and therefore cancels out of the
within-universe comparisons the thesis makes — see the methodology text.

The CSVs are one file per ticker under config.MARKET_DIR, written by
src/market_data/download.py with columns
``Date, Adj Close, Close, High, Low, Open, Volume`` and the caret stripped
from index tickers (``^VIX`` -> ``VIX.csv``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Rows missing any of these fields count as "no data" for that ticker-day and
# drop out of the shared trading calendar.
_REQUIRED_COLUMNS = ["Adj Close", "Close", "High", "Low"]


def _csv_path(ticker: str, market_dir=None) -> Path:
    """Path of the snapshot CSV for ``ticker`` ("^" is stripped in filenames)."""
    directory = config.MARKET_DIR if market_dir is None else Path(market_dir)
    return directory / f"{ticker.replace('^', '')}.csv"


def _read_ticker_csv(ticker: str, market_dir=None) -> pd.DataFrame:
    """Read one ticker's CSV, indexed by date, or raise a helpful error."""
    path = _csv_path(ticker, market_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Market data file for ticker {ticker!r} not found: {path}. "
            "Fetch the snapshot with scripts/01_download_market_data.py, or "
            f"just this series with: python -m src.market_data.download "
            f"--only {ticker.replace('^', '')}"
        )
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


def _adjusted_chl(frame: pd.DataFrame) -> pd.DataFrame:
    """Adjusted [close, high, low] levels via the AdjClose/Close ratio."""
    frame = frame.dropna(subset=_REQUIRED_COLUMNS)
    factor = frame["Adj Close"] / frame["Close"]
    return pd.DataFrame(
        {
            "close": frame["Close"] * factor,  # equals Adj Close exactly
            "high": frame["High"] * factor,
            "low": frame["Low"] * factor,
        },
        index=frame.index,
    )


def load_ohlc(tickers=None, start=None, end=None, market_dir=None):
    """Load the adjusted price tensor for a set of tickers.

    Parameters
    ----------
    tickers : list of str, default config.DOW30
    start, end : ISO date strings, default config.SAMPLE_START / SAMPLE_END
        Inclusive bounds on the trading calendar.
    market_dir : path, default config.MARKET_DIR
        Where the per-ticker CSVs live (tests point this at synthetic files).

    Returns
    -------
    dates : pd.DatetimeIndex, length T
        Trading calendar: days on which ALL requested tickers have data,
        restricted to [start, end].
    ohlc : np.ndarray, float64, shape (T, N, 3)
        ohlc[t, i, :] = adjusted [close, high, low] price LEVELS of ticker i
        on day t (field order = config.PRICE_FIELDS).
    """
    tickers = list(config.DOW30 if tickers is None else tickers)
    if not tickers:
        raise ValueError("load_ohlc needs at least one ticker")
    start = config.SAMPLE_START if start is None else start
    end = config.SAMPLE_END if end is None else end

    # Read and adjust every ticker; intersect the dates as we go, so the
    # calendar keeps only days on which every requested series exists.
    per_ticker: dict[str, pd.DataFrame] = {}
    calendar = None
    for ticker in tickers:
        adjusted = _adjusted_chl(_read_ticker_csv(ticker, market_dir))
        per_ticker[ticker] = adjusted
        calendar = (
            adjusted.index if calendar is None else calendar.intersection(adjusted.index)
        )

    calendar = calendar[
        (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
    ]
    dates = pd.DatetimeIndex(calendar, name="Date")

    ohlc = np.empty((len(dates), len(tickers), 3), dtype=np.float64)
    for i, ticker in enumerate(tickers):
        ohlc[:, i, :] = (
            per_ticker[ticker]
            .loc[dates, ["close", "high", "low"]]
            .to_numpy(dtype=np.float64)
        )
    return dates, ohlc


def _clip(series: pd.Series, start, end) -> pd.Series:
    """Restrict a date-indexed series to the inclusive [start, end] range."""
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]
    return series


def load_adj_close(ticker: str, start=None, end=None, market_dir=None) -> pd.Series:
    """Adjusted close of one ticker (e.g. "DIA", the passive benchmark).

    Uses Yahoo's "Adj Close" column directly, so the series is a total-return
    price level consistent with load_ohlc's close field.
    """
    frame = _read_ticker_csv(ticker, market_dir)
    series = frame["Adj Close"].dropna()
    series.name = ticker
    return _clip(series, start, end)


def load_raw_close(ticker: str, start=None, end=None, market_dir=None) -> pd.Series:
    """Plain (unadjusted) close of one ticker.

    For series where adjustment is meaningless: "VIX" (an index level) and
    "IRX" (an annualized yield in percent). Files are saved with the "^"
    stripped, and the caret is stripped from ``ticker`` too, so "VIX" and
    "^VIX" both work.
    """
    frame = _read_ticker_csv(ticker, market_dir)
    series = frame["Close"].dropna()
    series.name = ticker
    return _clip(series, start, end)
