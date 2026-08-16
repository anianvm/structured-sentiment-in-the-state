"""Score selected headlines into structured 4-D sentiment with an LLM.

Ported unchanged from master_thesis/src/sentiment/llm_structured.py
(methodology §LLM Sentiment extraction and representation); only paths,
imports and module names were adapted.

Default backend: DeepSeek (OpenAI-compatible API). Each headline is anonymised
(company names stripped, per the lookahead-bias safeguard) and scored IN
ISOLATION into a strict JSON object:

    {"direction": -1|0|1, "magnitude": 1..5,
     "horizon": "intraday"|"days"|"weeks"|"quarter_plus", "confidence": 1..5}

Raw answers are cached to scores.jsonl (one JSON object per line, resumable —
re-running skips already-scored stock-days). The final, human-readable table is
exported to structured_scores.csv for inspection and the thesis.

Setup (one-time):
  pip install openai python-dotenv
  echo 'DEEPSEEK_API_KEY=sk-...' >> .env

Usage (from project root):
  pilot (cheap, ~200 rows):  python -m src.sentiment_pipeline.score_llm --limit 200
  full run:                  python -m src.sentiment_pipeline.score_llm
  rebuild CSV from cache:    python -m src.sentiment_pipeline.score_llm --export-only
"""

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src import config

SELECTED = config.SENTIMENT_DIR / "selected_headlines.csv"  # from select_headlines.py
CACHE = config.SENTIMENT_DIR / "scores.jsonl"  # not in config: raw resumable API cache
OUT_CSV = config.LLM_SCORES_FILE

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

VALID_HORIZON = {"intraday", "days", "weeks", "quarter_plus"}

# Company aliases stripped from the headline before scoring (anonymisation).
# Single overly-common words (e.g. "morgan", "coke", "visa") are deliberately
# omitted to avoid mangling unrelated text; the distinctive names are enough.
ALIASES = {
    "AAPL": ["apple"], "AMGN": ["amgen"], "AMZN": ["amazon"],
    "AXP": ["american express", "amex"], "BA": ["boeing"],
    "CAT": ["caterpillar"], "CRM": ["salesforce"], "CSCO": ["cisco"],
    "CVX": ["chevron"], "DIS": ["walt disney", "disney"],
    "GS": ["goldman sachs", "goldman"], "HD": ["home depot"],
    "HON": ["honeywell"], "IBM": ["international business machines", "ibm"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "johnson johnson"],
    "JPM": ["jpmorgan", "jp morgan"], "KO": ["coca-cola", "coca cola"],
    "MCD": ["mcdonald's", "mcdonalds", "mcdonald"], "MMM": ["3m"],
    "MRK": ["merck"], "MSFT": ["microsoft"], "NKE": ["nike"],
    "NVDA": ["nvidia"], "PG": ["procter & gamble", "procter and gamble",
                               "procter gamble"],
    "SHW": ["sherwin-williams", "sherwin williams"], "TRV": ["travelers"],
    "UNH": ["unitedhealth", "united health"], "V": ["visa inc", "visa"],
    "VZ": ["verizon"], "WMT": ["walmart", "wal-mart", "wal mart"],
}

PROMPT = """You are a financial analyst. You will be given a single news item \
about a company. Assess ONLY the information in this item. Do not use any \
outside knowledge of how events actually turned out.

Return your assessment as a single JSON object with exactly these four fields \
and nothing else:

{{
  "direction": <-1, 0, or 1>,
  "magnitude": <integer 1 to 5>,
  "horizon": <"intraday", "days", "weeks", or "quarter_plus">,
  "confidence": <integer 1 to 5>
}}

Definitions (these are INDEPENDENT of one another):

direction - likely effect on the company's stock price.
  -1 = bad news (price likely to fall)
   0 = neutral or irrelevant to the price
   1 = good news (price likely to rise)

magnitude - size of the expected price impact, IGNORING direction.
   1 = routine, no meaningful impact (minor mention, restated old information)
   2 = minor
   3 = clearly market-relevant (e.g. analyst rating change, normal earnings news)
   4 = major
   5 = company-defining (e.g. fraud, bankruptcy, merger, CEO removal)

horizon - period over which the effect is likely to play out.
  "intraday"     = the same trading day
  "days"         = the next few days
  "weeks"        = several weeks
  "quarter_plus" = a quarter or longer

confidence - how certain YOU are about the direction, INDEPENDENT of magnitude.
   1 = very unsure (ambiguous or conflicting signal)
   5 = very certain

Rules:
- magnitude = size of effect; confidence = your certainty. A small but obvious \
event is low magnitude, high confidence. A potentially huge but ambiguous event \
is high magnitude, low confidence.
- If the item is irrelevant to the price, set direction 0 and magnitude 1, and \
let confidence reflect how sure you are that it is irrelevant.
- Output ONLY the JSON object. No explanation, no markdown, no extra text.

News item:
{text}"""


def clean_headline(headline: str) -> str:
    """Strip slug artefacts so the model scores real text, not noise/dates.

    GDELT headlines are derived from URL slugs and carry junk: wire-service
    article IDs (idINL1N0VT07Y20150219, cm445887) and embedded dates
    (2015 02 23, 20150219). The embedded dates also matter for lookahead —
    stripping them is what makes removing the date field meaningful.
    """
    t = str(headline)
    # standalone dates: 20150219 / 2015 02 23 / 2015-02-23
    t = re.sub(r"\b20\d{2}[ \-]?\d{2}[ \-]?\d{2}\b", " ", t)
    # long alphanumeric code tokens (wire/article IDs): >=7 chars, >=3 digits
    t = " ".join(
        "" if (len(tok) >= 7 and sum(c.isdigit() for c in tok) >= 3
               and any(c.isalpha() for c in tok)) else tok
        for tok in t.split()
    )
    return re.sub(r"\s+", " ", t).strip()


def anonymise(headline: str, ticker: str) -> str:
    """Replace the company's names with 'the company' (case-insensitive)."""
    out = headline
    for alias in ALIASES.get(ticker, []):
        out = re.sub(re.escape(alias), "the company", out, flags=re.IGNORECASE)
    return out


def parse_scores(content: str) -> dict | None:
    """Validate the model's JSON; return None if malformed/out-of-range."""
    try:
        d = json.loads(content)
        direction = int(d["direction"])
        magnitude = int(d["magnitude"])
        horizon = str(d["horizon"])
        confidence = int(d["confidence"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    if direction not in (-1, 0, 1) or not (1 <= magnitude <= 5) \
            or horizon not in VALID_HORIZON or not (1 <= confidence <= 5):
        return None
    return {"direction": direction, "magnitude": magnitude,
            "horizon": horizon, "confidence": confidence}


def score_one(client, model, date, ticker, headline):
    """One headline -> a cache record. Retries once on a bad/empty reply."""
    text = anonymise(clean_headline(headline), ticker)
    prompt = PROMPT.format(text=text)
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
    """Turn the JSONL cache into a tidy CSV table (with original headlines)."""
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE} — run scoring first")
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    scores = pd.DataFrame(rows)
    # a retried pair can appear twice (error line + later success) — keep the
    # successful record per (date, ticker)
    scores["_ok"] = scores["direction"].notna()
    scores = (scores.sort_values("_ok")
              .drop_duplicates(subset=["date", "ticker"], keep="last")
              .drop(columns="_ok"))
    sel = pd.read_csv(SELECTED)[["date", "ticker", "headline"]]
    sel["headline"] = sel["headline"].map(clean_headline)  # de-junked for display
    merged = sel.merge(scores, on=["date", "ticker"], how="inner")
    cols = ["date", "ticker", "headline", "direction", "magnitude",
            "horizon", "confidence"]
    merged = merged.sort_values(["ticker", "date"])[cols]
    merged.to_csv(OUT_CSV, index=False)
    n_err = scores["error"].notna().sum() if "error" in scores else 0
    print(f"{len(merged):,} scored rows -> {OUT_CSV}  ({n_err} errors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most N new headlines (pilot mode)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--export-only", action="store_true",
                    help="skip scoring, just rebuild the CSV from the cache")
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

    if not SELECTED.exists():
        raise SystemExit(f"{SELECTED} missing — run src.sentiment_pipeline.select_headlines first")
    sel = pd.read_csv(SELECTED)
    done = load_done()
    todo = [r for r in sel.itertuples(index=False)
            if (str(r.date), r.ticker) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done):,} already cached, scoring {len(todo):,} new "
          f"(model={args.model}, workers={args.workers})")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n = 0
    with open(CACHE, "a", encoding="utf-8") as cache, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, client, args.model,
                               str(r.date), r.ticker, r.headline)
                   for r in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                cache.write(json.dumps(rec) + "\n")
                cache.flush()
                n += 1
                if n % 100 == 0:
                    print(f"  scored {n:,}/{len(todo):,}", flush=True)

    print(f"done: {n:,} newly scored")
    export_csv()


if __name__ == "__main__":
    main()
