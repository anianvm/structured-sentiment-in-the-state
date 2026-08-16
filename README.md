# Structured Sentiment in the State

Master’s thesis code for a deep-RL portfolio allocator on the Dow 30 plus cash. 
The design compares six models (M1–M6) that are identical except for the sentiment 
information included in the state. Each model is trained with PPO and SAC at three 
CRRA risk-aversion levels, giving 36 configurations evaluated in a walk-forward setup.

The methodology chapter defines the design, and the code implements it section by section. 
All design constants are kept in one file, [src/config.py](src/config.py), with references 
to the relevant methodology sections.


## Repo map (methodology section → code)

| Methodology section | Code | What it does |
|---|---|---|
| Research design (arms M1–M6) | `src/config.py` (`ARMS`) | Which sentiment channels each arm observes |
| Data: universe & market data | `src/market_data/` | Download + load adjusted Dow 30 / DIA / ^DJI / ^VIX / ^IRX series |
| Data: news, cleaning, selection | `src/sentiment_pipeline/` | **FROZEN** — already run, outputs committed (see below) |
| Sentiment extraction (LLM + FinBERT) | `src/sentiment_pipeline/` | **FROZEN** — ditto |
| Sentiment representation | `src/features/sentiment_grid.py` | Scores → normalized per-asset daily grid (lag, merge, EWMA decay) |
| MDP: state | `src/features/` | Price tensor X_t, market context v_t, sentiment block Z_t |
| MDP: action, reward | `src/environment/portfolio_env.py` | Softmax weights, drift, turnover cost, CRRA reward |
| Learning algorithms | `src/agents/` | PPO / SAC construction (Stable-Baselines3), γ = 1 |
| Hyperparameter search | `src/experiments/tune.py` | Optuna TPE, 50 trials on M1 at η = 1, then frozen |
| Walk-forward protocol | `src/experiments/walkforward.py` | 5y train / 1y val / 1y test, 5 seeds, val-based selection |
| Benchmarks & metrics | `src/evaluation/` | UCRP, DIA buy-and-hold, Sharpe/MDD/turnover/entropy tables |
| Inference procedures | `src/evaluation/inference.py` | Studentized circular block bootstrap, Holm adjustment, seed band |

Beyond `src/`: `scripts/` are the numbered entry points, `tests/` the unit
suite, `data/` the frozen inputs, `results/` the committed run evidence and
generated tables.

## The frozen sentiment pipeline

The most computationally expensive part of the thesis has already been completed. 
This includes collecting 1.21 million GDELT headlines, selecting one headline per 
stock-day, and scoring 93,928 headlines with both DeepSeek (four structured sentiment 
dimensions) and FinBERT (a single sentiment score). The resulting outputs are committed:


- `data/news/dow_headlines.csv.gz` — 1.21M raw headlines, 30 tickers, 2015–2026
- `data/sentiment/structured_scores.csv` — LLM scores (anonymized; the main set)
- `data/sentiment/structured_scores_sector.csv` — sector-informed robustness set
- `data/sentiment/finbert_scores.csv` (+ `_named` variant) — FinBERT baseline

The code used to produce these outputs is retained in `src/sentiment_pipeline/` 
for transparency. Documented in `src/sentiment_pipeline/README.md`.


## Setup

Python 3.11.5 — the version all committed results were produced with;
`requirements.txt` pins the exact package set.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

## Run order

Scripts are numbered in execution order; each is a thin wrapper around one
module in `src/`:

```bash
.venv/bin/python scripts/01_download_market_data.py   # yfinance → data/market/
.venv/bin/python scripts/02_select_window.py --algo ppo  # lookback k (before tuning)
.venv/bin/python scripts/03_tune.py --algo ppo        # Optuna search (once per algo)
.venv/bin/python scripts/04_calibrate_budget.py       # training-steps plateau
.venv/bin/python scripts/05_walkforward.py --algo ppo --arm M1 --eta 1.0
.venv/bin/python scripts/06_evaluate.py               # tables → results/tables/
.venv/bin/python scripts/07_figures.py                # figures → results/figures/
.venv/bin/python scripts/08_training_diagnostics.py   # training-diagnostics table + figure
.venv/bin/python scripts/09_block_robustness.py       # bootstrap block-length check (b = 10)
```

Script 05 trains a single arm–algorithm–$\eta$ configuration. The full design 
consists of 36 configurations with five seeds each, that were run on rented 
cloud pods. There is no numbered sentiment script because that stage is already 
complete and frozen. The three appendix analyses — LLM–FinBERT agreement,
sector-context robustness, and sentiment-signal persistence — sit outside
the run order as modules of the frozen pipeline package
(`python -m src.sentiment_pipeline.compare_llm_finbert`, `…compare_sector_context`,
`…estimate_persistence`); their outputs back the corresponding appendix exhibits.


## Data policy

Committed to the repository are the sentiment scores, raw headlines, and all results 
except model checkpoints. The ledgers, summaries, and tables under `results/` support 
the numbers reported in the thesis. Market data, raw GKG files, and model weights 
are gitignored because they can be regenerated or are too large to store. See `data/README.md` 
for details.


## Results

`results/tables/README.md` documents the generated CSVs, and
`results/tables/latex/README.md` maps each generated LaTeX body to its
`tab:*` label in the thesis and names the four deliberately hand-set tables.
`results/tuning/README.md` inventories the search artifacts, including the
archived dead ends kept for provenance.
