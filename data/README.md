# Data directory

## Committed (do not delete, cannot be regenerated exactly)

| File | What it is | Produced by |
|---|---|---|
| `news/dow_headlines.csv.gz` | 1,205,231 headlines, 30 Dow tickers, Feb 2015 – Jun 2026, reconstructed from GDELT GKG article URLs. Columns: date, ticker, headline, domain, url, gdelt_tone | `src/sentiment_pipeline/collect_gdelt.py` + `build_headlines.py` (needs a Google BigQuery project) |
| `sentiment/structured_scores.csv` | 93,928 headlines scored by DeepSeek (`deepseek-v4-flash`, temperature 0): direction ∈ {−1,0,+1}, magnitude 1–5, horizon {intraday, days, weeks, quarter_plus}, confidence 1–5. Headlines were cleaned + **anonymized** (company name removed) before scoring. **This is the main score set used by arms M3–M6.** | `src/sentiment_pipeline/score_llm.py` |
| `sentiment/structured_scores_sector.csv` | Same headlines re-scored with the company's GICS sector named in the prompt. Robustness set only. | `src/sentiment_pipeline/score_llm_sector.py` |
| `sentiment/finbert_scores.csv` | Same (cleaned + anonymized) headlines scored by FinBERT: class probabilities and score = p_pos − p_neg. **Used by arm M2.** | `src/sentiment_pipeline/score_finbert.py` |

## Gitignored (regenerable)

| Directory | Contents | Regenerate with |
|---|---|---|
| `market/` | Daily OHLCV CSVs from yfinance: 30 constituents + DIA, DJI, VIX, IRX (13-week T-bill), from 2005 | `scripts/01_download_market_data.py` |
| `news/gkg_raw/` | Raw per-year GDELT GKG pulls (parquet), the input to `build_headlines.py` | `src/sentiment_pipeline/collect_gdelt.py` |

Note: yfinance data can change over time (adjustments, corrections). The CSVs
currently on disk are the snapshot the thesis results are based on — back them
up before re-downloading.
