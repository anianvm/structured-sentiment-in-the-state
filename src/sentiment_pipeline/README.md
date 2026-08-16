# Frozen sentiment pipeline

> **DO NOT RE-RUN THE SCORING SCRIPTS.** The LLM scoring alone made ~188,000
> DeepSeek API calls (93,928 headlines x 2 prompt variants) and the BigQuery
> collection consumes free-tier scan quota. Every output is committed under
> `data/news/` and `data/sentiment/` — the experiments only ever read those
> files. This package exists so the thesis can document *exactly* how they were
> produced, not so anyone runs it again.

This package is a verbatim port of the pipeline that produced the committed
news and sentiment data. Logic is identical to the original
`master_thesis` code; only file paths, imports, and module names were adapted
(each module's docstring names its original). The pipeline stages, in order:

```
collect_gdelt.py -> build_headlines.py -> select_headlines.py
                                              |
                        +---------------------+---------------------+
                        v                     v                     v
                   score_llm.py       score_llm_sector.py    score_finbert.py
                        |                     |                     |
                        +----------+----------+----------+---------+
                                   v                     v
                        compare_llm_finbert.py  compare_sector_context.py
```

## Provenance of the data files

| File | Produced by | Settings that pin it down |
|---|---|---|
| *(not committed — regenerable via BigQuery)* `data/news/gkg_raw/gkg_2015.parquet` … `gkg_2026.parquet` | `collect_gdelt.py --execute` | BigQuery table `gdelt-bq.gdeltv2.gkg_partitioned`, one partition-pruned query per calendar year 2015–2026; 21 financial-news domains; per-ticker regexes on the GKG Organizations field (3M matched via URL slug instead, since GDELT never tags digit-led "3M"). |
| `data/news/dow_headlines.csv.gz` | `build_headlines.py` | Headline = article URL slug, de-junked (slugs under 4 words or with query strings dropped); deduped per (ticker, headline); GDELT tone = first field of V2Tone. 1,205,231 rows, all 30 Dow tickers, Feb 2015 – Jun 2026. |
| *(not committed)* `data/sentiment/selected_headlines.csv` | `select_headlines.py` | One headline per (ticker, calendar day): deterministic draw keeping the lowest MD5 hash of `ticker\|date\|headline\|seed`, **seed 42**. 93,928 stock-days. Not committed because it is regenerated bit-for-bit from `dow_headlines.csv.gz`. |
| `data/sentiment/structured_scores.csv` | `score_llm.py` | DeepSeek API, model id `deepseek-chat` (served version **`deepseek-v4-flash`**, recorded per the methodology chapter), `temperature=0`, `max_tokens=80`, JSON-object response format, one retry on a bad reply. Input = cleaned (dates and wire-service ID tokens stripped) **and anonymised** headline (company aliases replaced by "the company"); each headline scored in isolation. 93,928 rows; **exactly one unrecoverable failure** (2026-04-08, AAPL — the row is present with empty score fields). Determinism was verified in the output, not assumed: 721 headline texts recurring across stock-days received the identical direction in 96.7% of cases. |
| `data/sentiment/structured_scores_sector.csv` | `score_llm_sector.py` | Identical model, prompt, parameters, and cleaned+anonymised input as above, plus one added line naming the company's coarse GICS sector. Robustness set only (sector context changes direction on ~11% of headlines but reverses its sign on <0.5%; see `compare_sector_context.py`). 93,928 rows, same single failure row. |
| `data/sentiment/finbert_scores.csv` | `score_finbert.py` | `ProsusAI/finbert`, batch 64, truncation at 128 tokens, softmax probabilities; `finbert_score = p_pos - p_neg`, `finbert_label` = argmax class. Input = the **identical cleaned + anonymised** text the LLM saw, so the LLM-vs-FinBERT comparison varies only the scoring model. 93,928 rows. |

The three analysis scripts (`compare_llm_finbert.py`, `compare_sector_context.py`,
`estimate_persistence.py`) are cheap and safe to re-run; the first two read only
the committed CSVs, persistence additionally needs the market snapshot from
script 01 for the trading calendar. They reproduce the agreement statistics,
the confusion-matrix figure, the sector-context flip numbers, and the
sentiment-persistence table quoted in the methodology chapter and appendix.

Intermediate JSONL caches (`scores.jsonl`, `scores_sector.jsonl` — the raw,
resumable API answer logs) are not committed; the exported CSVs supersede them.
