"""Turn raw GDELT BigQuery extracts into per-ticker daily headline files.

Ported unchanged from master_thesis/src/data/gdelt_postprocess.py (methodology
§News data); only paths, imports and module names were adapted.

Input : data/news/gkg_raw/gkg_<year>.parquet  (from collect_gdelt.py)
Output: data/news/dow_headlines.csv.gz (config.HEADLINES_FILE) with columns
        [date, ticker, headline, domain, url, gdelt_tone]
        (gzipped CSV: small enough for plain git, opens as CSV when unzipped,
         read directly with pandas.read_csv(..., compression="gzip"))

Steps: map orgs -> tickers, derive headline from URL slug, drop junk slugs,
dedupe per (ticker, headline), parse GDELT tone (first field of V2Tone).

Usage:  python -m src.sentiment_pipeline.build_headlines
"""

import re

import pandas as pd

from src import config
from src.sentiment_pipeline.collect_gdelt import MMM_URL_PATTERN, ORG_PATTERNS


def slug_to_title(url: str) -> str:
    m = re.search(r"/([^/]+?)(?:-\d{6,})?/?(?:\.html|\.htm|\.php|\.aspx?)?$",
                  str(url).rstrip("/"))
    if not m:
        return ""
    slug = m.group(1)
    if "?" in slug or "=" in slug or "&" in slug:
        return ""
    title = re.sub(r"[-_]+", " ", slug).strip()
    if len(title.split()) < 4:          # ticker-only or junk slugs
        return ""
    return title


def main() -> None:
    in_dir = config.NEWS_DIR / "gkg_raw"
    files = sorted(in_dir.glob("gkg_*.parquet"))
    if not files:
        raise SystemExit("no gkg_<year>.parquet files found — run collect_gdelt.py first")

    compiled = {t: re.compile(p) for t, p in ORG_PATTERNS.items()}
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["DATE"].astype(str).str[:8], format="%Y%m%d")
        df["headline"] = df["DocumentIdentifier"].map(slug_to_title)
        df = df[df["headline"] != ""]
        df["gdelt_tone"] = (
            df["V2Tone"].astype(str).str.split(",").str[0].astype(float)
        )
        mmm_rx = re.compile(MMM_URL_PATTERN)
        for ticker in [*compiled, "MMM"]:
            if ticker == "MMM":  # 3M is matched on URL slug, not org tags
                hits = df[df["DocumentIdentifier"].str.lower().str.contains(mmm_rx)]
            else:
                hits = df[df["orgs"].str.contains(compiled[ticker], na=False)]
            if hits.empty:
                continue
            frames.append(pd.DataFrame({
                "date": hits["date"],
                "ticker": ticker,
                "headline": hits["headline"],
                "domain": hits["SourceCommonName"],
                "url": hits["DocumentIdentifier"],
                "gdelt_tone": hits["gdelt_tone"],
            }))
        print(f"{f.name}: {len(df):,} usable rows")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "headline"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    # gzipped CSV: ~72MB (fits plain git), opens as a normal CSV when unzipped
    out_file = config.HEADLINES_FILE
    out.to_csv(out_file, index=False, compression="gzip")

    print(f"\n{len(out):,} unique (ticker, headline) rows -> {out_file}")
    cov = out.groupby([out.ticker, out.date.dt.year]).size().unstack(fill_value=0)
    print(cov.to_string())


if __name__ == "__main__":
    main()
