"""Select one headline per (ticker, trading-day) for sentiment scoring.

Ported unchanged from master_thesis/src/sentiment/select.py (methodology
§News headline cleaning and selection); only paths, imports and module names
were adapted.

Both the structured (LLM) and scalar (FinBERT) conditions must score the SAME
headline per stock-day, so the draw is made once here, with a fixed seed, and
reused by both arms. This is the fairness requirement from the design doc:
draw once, both conditions see identical text.

Input : data/news/dow_headlines.csv.gz  [date, ticker, headline, url, ...]
Output: data/sentiment/selected_headlines.csv  [date, ticker, headline, url]

Usage (from project root):
  python -m src.sentiment_pipeline.select_headlines [--seed 42]
"""

import argparse
import hashlib

import pandas as pd

from src import config

IN_FILE = config.HEADLINES_FILE
# Not in config: intermediate file, regenerated bit-for-bit by this script
# (deterministic hash draw, default seed 42).
OUT_FILE = config.SENTIMENT_DIR / "selected_headlines.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for the per-(ticker, day) draw")
    args = ap.parse_args()

    df = pd.read_csv(IN_FILE, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()

    # One headline per (ticker, day), chosen deterministically: each row gets a
    # hash of (ticker, date, headline, seed); within each (ticker, day) group we
    # keep the lowest hash. The pick depends only on that group's own rows, so
    # adding tickers later never changes existing picks — the (date, ticker)-keyed
    # score cache stays valid for an incremental re-run.
    df["_h"] = [
        int(hashlib.md5(f"{t}|{d.date()}|{h}|{args.seed}".encode()).hexdigest(), 16)
        for t, d, h in zip(df["ticker"], df["date"], df["headline"])
    ]
    sel = (
        df.sort_values(["ticker", "date", "_h"])
        .drop_duplicates(subset=["ticker", "date"], keep="first")
        .drop(columns="_h")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sel[["date", "ticker", "headline", "url"]].to_csv(OUT_FILE, index=False)

    print(f"{len(sel):,} selected (ticker, day) headlines -> {OUT_FILE}")
    print(f"tickers: {sel['ticker'].nunique()}, "
          f"{sel['date'].min().date()} -> {sel['date'].max().date()}")


if __name__ == "__main__":
    main()
