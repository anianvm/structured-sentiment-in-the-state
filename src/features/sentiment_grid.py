"""Per-asset sentiment grid Z_t of the agent state.

Implements the representation half of the thesis methodology chapter, section "Sentiment
Extraction and Representation" (sec:sent-extraction-meth), and the lag rule
of section "State" (sec:state-meth). The frozen headline scores in
data/sentiment/ are turned into a (T, N, C) grid aligned to the trading days
of the price tensor, one row per trading day, one column per Dow 30 ticker
(config.DOW30 order).

LLM grid channels (config.LLM_CHANNELS order), normalized per the spec's
dimension table (tab:sentiment-dimensions) -- all bounds are fixed and known,
so the normalisation introduces no test-period information:

  0 direction   [-1, 1]  used unchanged
  1 magnitude   [0, 1]   integer 1..5 -> (x-1)/4
  2 horizon     [0, 1]   ordinal intraday/days/weeks/quarter_plus = 1..4 -> (x-1)/3
  3 confidence  [0, 1]   integer 1..5 -> (x-1)/4
  4 news_flag   {0, 1}   1 if any news that (trading-day, ticker) -- distinguishes
                         "no news" (all zeros) from "neutral news"

Three transformations, in order:

  1. Calendar alignment and merge. A headline dated on a non-trading day is
     assigned to the NEXT trading day, so a Monday can carry its own score
     plus weekend carry-overs. Multiple scores on one (trading-day, ticker)
     are combined into a single observation (spec, sec:LLM-Sent-extraction):
     direction is averaged with weights magnitude*confidence (a confident,
     high-impact article dominates), magnitude is the maximum, horizon is
     taken from the article of highest magnitude, confidence is averaged.

  2. Exponential decay. The daily signal is smoothed with an EWMA whose
     weight halves every config.SENTIMENT_HALFLIFE (~5) trading days,
     reflecting the gradual absorption of news and preventing abrupt
     day-to-day jumps that would induce excessive trading. Only the score
     channels are smoothed; the news flag stays raw (fresh-news marker).

  3. Lag. The whole grid (all channels, flag included) is shifted forward by
     config.SENTIMENT_LAG_DAYS trading days, so row t contains news dated
     through trading day t - lag. Per the spec: "Sentiment is lagged by one
     trading day, so the day-t decision uses headlines dated through t-1.
     This conservative lag prevents post-close news from entering the
     same-day state and creating lookahead bias." (sec:state-meth)

The FinBERT grid (config.FINBERT_CHANNELS = [finbert_score, news_flag]) is
built identically except that the per-day merge is a plain mean of the
scalar score p_pos - p_neg in [-1, 1] (sec:FinBERT-Sent-extraction), so the
M2 baseline differs from the LLM arms only in the scoring model.
"""

import numpy as np
import pandas as pd

from src import config

# Ordinal encoding of the LLM's horizon labels (tab:sentiment-dimensions).
HORIZON_ORD = {"intraday": 1, "days": 2, "weeks": 3, "quarter_plus": 4}


def _aggregate_llm(df: pd.DataFrame, dates: pd.DatetimeIndex,
                   halflife: float) -> np.ndarray:
    """Merge raw LLM scores into a normalized, EWMA-smoothed (T, N, 5) grid.

    `df` needs columns [date, ticker, direction, magnitude, horizon,
    confidence] with `date` already parsed to datetimes. Rows with missing
    direction or non-Dow tickers are dropped. NOT lagged yet -- row t here
    still contains news dated through day t itself; build_llm_grid applies
    _lag afterwards.
    """
    df = df[df["direction"].notna() & df["ticker"].isin(config.DOW30)].copy()

    # Map each headline's date to the next trading day in `dates`: side="left"
    # returns the first trading day >= the headline date, so trading-day news
    # maps to that same day and weekend/holiday news rolls forward. Headlines
    # dated after the last trading day are dropped.
    di = np.searchsorted(dates.values, df["date"].values, side="left")
    df = df[di < len(dates)].copy()
    df["di"] = di[di < len(dates)]
    df["ti"] = df["ticker"].map({t: i for i, t in enumerate(config.DOW30)})
    df["hor"] = df["horizon"].map(HORIZON_ORD)
    df["w"] = df["magnitude"] * df["confidence"]      # merge weight
    df["dw"] = df["direction"] * df["w"]

    # Vectorised merge per (trading-day, ticker), spec merge rule.
    g = df.groupby(["di", "ti"]).agg(
        dw=("dw", "sum"), w=("w", "sum"),
        mag=("magnitude", "max"), con=("confidence", "mean"),
    )
    g["dir"] = np.where(g["w"] > 0, g["dw"] / g["w"], 0.0)
    # Horizon of the dominant (max-magnitude) story of the day.
    dom = (df.sort_values(["di", "ti", "magnitude"])
           .drop_duplicates(["di", "ti"], keep="last")
           .set_index(["di", "ti"])["hor"])
    g["hor"] = dom
    g = g.reset_index()

    T, N = len(dates), len(config.DOW30)
    grid = np.zeros((T, N, 5), dtype=np.float32)
    rows, cols = g["di"].to_numpy(int), g["ti"].to_numpy(int)
    grid[rows, cols, 0] = g["dir"]                    # direction  [-1, 1]
    grid[rows, cols, 1] = (g["mag"] - 1) / 4          # magnitude  [0, 1]
    grid[rows, cols, 2] = (g["hor"] - 1) / 3          # horizon    [0, 1]
    grid[rows, cols, 3] = (g["con"] - 1) / 4          # confidence [0, 1]
    grid[rows, cols, 4] = 1.0                         # news flag

    if halflife and halflife > 0:                     # smooth the 4 score channels
        for ch in range(4):
            grid[:, :, ch] = (pd.DataFrame(grid[:, :, ch])
                              .ewm(halflife=halflife).mean().to_numpy())
    return grid


def _aggregate_finbert(df: pd.DataFrame, dates: pd.DatetimeIndex,
                       halflife: float) -> np.ndarray:
    """FinBERT counterpart of _aggregate_llm: (T, N, 2) [finbert_score, flag].

    Same calendar alignment; the per-day merge is the mean of finbert_score
    (p_pos - p_neg, already in [-1, 1]); same EWMA on the score channel with
    the flag left raw. NOT lagged yet.
    """
    df = df[df["finbert_score"].notna() & df["ticker"].isin(config.DOW30)].copy()
    di = np.searchsorted(dates.values, df["date"].values, side="left")
    df = df[di < len(dates)].copy()
    df["di"] = di[di < len(dates)]
    df["ti"] = df["ticker"].map({t: i for i, t in enumerate(config.DOW30)})
    g = df.groupby(["di", "ti"])["finbert_score"].mean().reset_index()

    grid = np.zeros((len(dates), len(config.DOW30), 2), dtype=np.float32)
    rows, cols = g["di"].to_numpy(int), g["ti"].to_numpy(int)
    grid[rows, cols, 0] = g["finbert_score"]
    grid[rows, cols, 1] = 1.0
    if halflife and halflife > 0:
        grid[:, :, 0] = (pd.DataFrame(grid[:, :, 0])
                         .ewm(halflife=halflife).mean().to_numpy())
    return grid


def _lag(grid: np.ndarray, lag: int) -> np.ndarray:
    """Shift the whole grid (all channels, flag included) forward by `lag` days.

    Row t of the result is row t - lag of the input; the first `lag` rows are
    zero (no news known yet). After this shift the day-t state only contains
    news dated through trading day t - lag -- the spec's "conservative lag"
    that prevents post-close news from entering the same-day state
    (sec:state-meth).
    """
    if lag <= 0:
        return grid
    lagged = np.zeros_like(grid)
    lagged[lag:] = grid[:-lag]
    return lagged


def build_llm_grid(dates: pd.DatetimeIndex,
                   halflife: float = config.SENTIMENT_HALFLIFE,
                   lag: int = config.SENTIMENT_LAG_DAYS,
                   scores_csv=config.LLM_SCORES_FILE) -> np.ndarray:
    """(T, 30, 5) float32 LLM grid, channels in config.LLM_CHANNELS order.

    The returned grid is ALREADY LAGGED: row t contains news dated through
    trading day t - lag.
    """
    df = pd.read_csv(scores_csv, parse_dates=["date"])
    return _lag(_aggregate_llm(df, dates, halflife), lag)


def build_finbert_grid(dates: pd.DatetimeIndex,
                       halflife: float = config.SENTIMENT_HALFLIFE,
                       lag: int = config.SENTIMENT_LAG_DAYS,
                       scores_csv=config.FINBERT_SCORES_FILE) -> np.ndarray:
    """(T, 30, 2) float32 FinBERT grid, config.FINBERT_CHANNELS order.

    ALREADY LAGGED, exactly like build_llm_grid.
    """
    df = pd.read_csv(scores_csv, parse_dates=["date"])
    return _lag(_aggregate_finbert(df, dates, halflife), lag)


def arm_grid(arm: str, dates: pd.DatetimeIndex,
             llm_csv=config.LLM_SCORES_FILE,
             finbert_csv=config.FINBERT_SCORES_FILE) -> np.ndarray | None:
    """Sentiment block for one experimental arm: (T, 30, C) or None for M1.

    Builds the source grid the arm draws from -- the FinBERT grid for the
    arm carrying finbert_score (M2), the LLM grid otherwise (M3-M6) -- and
    slices exactly the channels config.ARMS[arm] names, in that order. M1
    (price-only control) has no sentiment block and returns None.
    """
    channels = config.ARMS[arm]
    if not channels:
        return None                                   # M1: price-only control
    if "finbert_score" in channels:                   # M2
        grid = build_finbert_grid(dates, scores_csv=finbert_csv)
        names = config.FINBERT_CHANNELS
    else:                                             # M3-M6
        grid = build_llm_grid(dates, scores_csv=llm_csv)
        names = config.LLM_CHANNELS
    return grid[:, :, [names.index(c) for c in channels]]
