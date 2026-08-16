"""Extract Dow 30 company news from GDELT's BigQuery mirror (gdelt-bq.gdeltv2.gkg_partitioned).

Ported unchanged from master_thesis/src/data/gdelt_bigquery.py (methodology
§News data); only paths, imports and module names were adapted.

Runs one query per calendar year (partition-pruned) and saves raw results to
data/news/gkg_raw/gkg_<year>.parquet. ALWAYS dry-runs first and prints the
bytes that WOULD be scanned; pass --execute to actually run and spend quota
(BigQuery sandbox free tier: 1 TB scanned / month).

Precision strategy: match company regexes against GKG's Organizations field,
restricted to financial-news domains. Headlines are derived later from URL
slugs by build_headlines.py.

Usage:
  python -m src.sentiment_pipeline.collect_gdelt --years 2017 2018 --project <PROJECT_ID>            # dry run
  python -m src.sentiment_pipeline.collect_gdelt --years 2017 2018 --project <PROJECT_ID> --execute
"""

import argparse
from pathlib import Path

from src import config

# ticker -> regex against the (lowercase) GKG Organizations field.
# High precision beats recall: matches are scoped to finance domains below.
ORG_PATTERNS = {
    "AAPL": r"apple inc|\bapple\b", "AMGN": r"amgen",
    "AMZN": r"amazon", "AXP": r"american express",
    "BA": r"\bboeing\b", "CAT": r"caterpillar", "CRM": r"salesforce",
    "CSCO": r"\bcisco\b", "CVX": r"\bchevron\b", "DIS": r"walt disney|\bdisney\b",
    "GS": r"goldman sachs", "HD": r"home depot", "HON": r"honeywell",
    "IBM": r"\bibm\b|international business machines",
    # GDELT strips '&' from org names: "johnson johnson", "procter gamble"
    "JNJ": r"johnson johnson",
    "JPM": r"jpmorgan|jp morgan", "KO": r"coca[- ]cola",
    "MCD": r"mcdonald", "MRK": r"\bmerck\b",
    "MSFT": r"microsoft", "NKE": r"\bnike\b", "NVDA": r"nvidia",
    "PG": r"procter gamble",
    "SHW": r"sherwin[- ]williams",
    "TRV": r"travelers (companies|cos|inc|insurance)",
    "UNH": r"unitedhealth", "V": r"\bvisa\b",
    "VZ": r"\bverizon\b", "WMT": r"walmart|wal[- ]mart",
}

FINANCE_DOMAINS = [
    "reuters.com", "cnbc.com", "marketwatch.com", "forbes.com", "fool.com",
    "seekingalpha.com", "benzinga.com", "barrons.com", "investorplace.com",
    "thestreet.com", "zacks.com", "businessinsider.com", "finance.yahoo.com",
    "investing.com", "streetinsider.com", "fortune.com", "wsj.com",
    "bloomberg.com", "ft.com", "nasdaq.com", "morningstar.com",
]

# GDELT's entity extractor never tags digit-led "3M"; match it via URL slug.
MMM_URL_PATTERN = r"[/.](3m)-"

ANY_COMPANY = "|".join(f"(?:{p})" for p in ORG_PATTERNS.values())
DOMAIN_LIST = ", ".join(f"'{d}'" for d in FINANCE_DOMAINS)

SQL_TEMPLATE = f"""
SELECT
  DATE,
  SourceCommonName,
  DocumentIdentifier,
  LOWER(Organizations) AS orgs,
  V2Tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{{y0}}-01-01')
  AND _PARTITIONTIME <  TIMESTAMP('{{y1}}-01-01')
  AND SourceCommonName IN ({DOMAIN_LIST})
  AND (REGEXP_CONTAINS(LOWER(Organizations), r"{ANY_COMPANY}")
       OR REGEXP_CONTAINS(LOWER(DocumentIdentifier), r"{MMM_URL_PATTERN}"))
"""


def main() -> None:
    from google.cloud import bigquery  # lazy: only needed to re-collect

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--execute", action="store_true", help="spend quota; default is dry-run")
    ap.add_argument("--out", default=str(config.NEWS_DIR / "gkg_raw"))
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tb = 0.0
    for year in args.years:
        sql = SQL_TEMPLATE.format(y0=year, y1=year + 1)
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
        tb = dry.total_bytes_processed / 1e12
        total_tb += tb
        print(f"{year}: would scan {tb:.3f} TB")
        if not args.execute:
            continue
        out_file = out_dir / f"gkg_{year}.parquet"
        if out_file.exists():
            print(f"{year}: {out_file} exists, skipping")
            continue
        df = client.query(sql).to_dataframe()
        df.to_parquet(out_file)
        print(f"{year}: {len(df):,} rows -> {out_file}")

    print(f"TOTAL scan {'(executed)' if args.execute else '(dry-run)'}: {total_tb:.3f} TB "
          f"(sandbox free tier: 1 TB/month)")


if __name__ == "__main__":
    main()
