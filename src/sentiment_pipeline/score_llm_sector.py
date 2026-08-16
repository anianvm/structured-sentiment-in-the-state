"""Score ALL selected headlines with the LLM, adding coarse GICS sector context.

Ported unchanged from master_thesis/src/sentiment/llm_sector.py (methodology
§LLM Sentiment extraction — anonymisation cost); only paths, imports and
module names were adapted.

Sector-context variant of score_llm.py: the identical cleaned and
anonymised headline, the identical prompt and parameters, plus ONE added line
telling the model the company's GICS sector. Coarse sectors preserve the
anonymisation safeguard (a sector names thousands of firms, not one), while
restoring the context that signs macro news (rates -> banks, oil -> energy).

Writes to SEPARATE files — the original scores are never touched:
  cache : data/sentiment/scores_sector.jsonl      (resumable)
  output: data/sentiment/structured_scores_sector.csv

Run:  python -m src.sentiment_pipeline.score_llm_sector [--limit 200] [--export-only]
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src import config
from src.sentiment_pipeline.score_llm import (
    BASE_URL, DEFAULT_MODEL, PROMPT, SELECTED, anonymise, clean_headline,
    parse_scores,
)

CACHE = config.SENTIMENT_DIR / "scores_sector.jsonl"  # not in config: raw resumable API cache
OUT_CSV = config.LLM_SECTOR_SCORES_FILE

# GICS sectors (Visa under Financials per the 2023 reclassification)
SECTORS = {
    "AAPL": "Information Technology", "CRM": "Information Technology",
    "CSCO": "Information Technology", "IBM": "Information Technology",
    "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AXP": "Financials", "GS": "Financials", "JPM": "Financials",
    "TRV": "Financials", "V": "Financials",
    "AMGN": "Health Care", "JNJ": "Health Care", "MRK": "Health Care",
    "UNH": "Health Care",
    "BA": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    "MMM": "Industrials",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "KO": "Consumer Staples", "PG": "Consumer Staples",
    "WMT": "Consumer Staples",
    "DIS": "Communication Services", "VZ": "Communication Services",
    "CVX": "Energy", "SHW": "Materials",
}

# same prompt, one added sector line just before the news item
SECTOR_PROMPT = PROMPT.replace(
    "News item:",
    "The company operates in the {sector} sector.\n\nNews item:")


def score_one(client, model, date, ticker, headline):
    """One headline -> a cache record. Retries once on a bad/empty reply."""
    text = anonymise(clean_headline(headline), ticker)
    prompt = SECTOR_PROMPT.format(text=text, sector=SECTORS[ticker])
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=80,
            )
            scores = parse_scores(resp.choices[0].message.content)
            if scores is not None:
                return {"date": date, "ticker": ticker, **scores, "error": None}
        except Exception as e:  # transient API/network error -> retry once
            err = str(e)
    return {"date": date, "ticker": ticker, "direction": None,
            "magnitude": None, "horizon": None, "confidence": None,
            "error": locals().get("err", "invalid_json")}


def load_done() -> set:
    """(date, ticker) pairs already scored SUCCESSFULLY (errors get retried)."""
    done = set()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None and r.get("direction") is not None:
                    done.add((r["date"], r["ticker"]))
    return done


def export_csv() -> None:
    """Turn the JSONL cache into a tidy CSV (cleaned headlines + sector)."""
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE} — run scoring first")
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    scores = pd.DataFrame(rows)
    scores["_ok"] = scores["direction"].notna()
    scores = (scores.sort_values("_ok")
              .drop_duplicates(subset=["date", "ticker"], keep="last")
              .drop(columns="_ok"))
    sel = pd.read_csv(SELECTED)[["date", "ticker", "headline"]]
    sel["headline"] = sel["headline"].map(clean_headline)
    merged = sel.merge(scores, on=["date", "ticker"], how="inner")
    merged["sector"] = merged["ticker"].map(SECTORS)
    cols = ["date", "ticker", "sector", "headline", "direction", "magnitude",
            "horizon", "confidence"]
    merged = merged.sort_values(["ticker", "date"])[cols]
    merged.to_csv(OUT_CSV, index=False)
    n_err = scores["error"].notna().sum() if "error" in scores else 0
    print(f"{len(merged):,} scored rows -> {OUT_CSV}  ({n_err} errors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    if args.export_only:
        export_csv()
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY (e.g. in a .env file)")
    client = OpenAI(api_key=key, base_url=BASE_URL)

    df = pd.read_csv(SELECTED)
    done = load_done()
    todo = [(r.date, r.ticker, r.headline) for r in df.itertuples()
            if (r.date, r.ticker) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(df):,} selected, {len(done):,} already scored, "
          f"{len(todo):,} to score (sector variant)")

    lock = threading.Lock()
    n_ok = n_err = 0
    with open(CACHE, "a", encoding="utf-8") as cache, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, client, args.model, d, t, h)
                   for d, t, h in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                cache.write(json.dumps(rec) + "\n")
                if rec["error"] is None:
                    n_ok += 1
                else:
                    n_err += 1
                if i % 500 == 0:
                    cache.flush()
                    pct = 100 * i / len(todo)
                    print(f"  {i:>7,}/{len(todo):,}  ({pct:.1f}%)  "
                          f"ok {n_ok:,}  err {n_err:,}", flush=True)

    print(f"\ndone: ok {n_ok:,}  err {n_err:,}")
    export_csv()


if __name__ == "__main__":
    main()
