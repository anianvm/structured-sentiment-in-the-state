"""Single source of truth for every design constant of the thesis.

Every number in this file is fixed by the methodology chapter
(the thesis methodology chapter).
"""

from pathlib import Path


# ---- Paths

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market"          # yfinance CSVs (regenerable, gitignored)
NEWS_DIR = DATA_DIR / "news"              # raw GDELT headlines (committed)
SENTIMENT_DIR = DATA_DIR / "sentiment"    # scored outputs (committed, FROZEN)
RESULTS_DIR = ROOT / "results"            # models, logs, tables (gitignored)

HEADLINES_FILE = NEWS_DIR / "dow_headlines.csv.gz"
LLM_SCORES_FILE = SENTIMENT_DIR / "structured_scores.csv"
LLM_SECTOR_SCORES_FILE = SENTIMENT_DIR / "structured_scores_sector.csv"  # robustness set
FINBERT_SCORES_FILE = SENTIMENT_DIR / "finbert_scores.csv"


# ---- Universe and sample period  (§ Data — Universe and Market Data)

# Dow Jones Industrial Average constituents after the November 2024 revision
DOW30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]
N_ASSETS = len(DOW30)                      # 30 risky assets. index 0 of the

# Passive benchmark, index level, implied vol, 13-week T-bill
AUX_TICKERS = ["DIA", "^DJI", "^VIX", "^IRX"]

# GKG v2.0 (the GDELT version with structured organisation fields) starts
# February 2015; that start date therefore binds the whole sample
SAMPLE_START = "2015-02-01"
SAMPLE_END = "2026-06-15"

# VIX and ^DJI history is pulled from 2005 so the expanding z-score in the
# market-context vector is estimated on a decade of pre-sample history
CONTEXT_ANCHOR_START = "2005-01-01"


# ----  State  (§ The Portfolio MDP — State)

# Lookback of the price tensor X_t, selected from K_CANDIDATES by the rule in
# src/experiments/window_selection.py (PPO, arm M1, eta=1, 5 seeds, scored on
# the 2019 tuning-validation year). It also matches the 14 days Aboussalah et al.
# (2022) obtain for PPO.
WINDOW_K = 15
K_CANDIDATES = [7, 15, 30, 60]
PRICE_FIELDS = ["close", "high", "low"]

SENTIMENT_LAG_DAYS = 1     # day-t decision sees headlines dated through t-1
SENTIMENT_HALFLIFE = 5.0   # EWMA half-life (trading days) smoothing the signal


# ----  Experimental arms  (§ Research Design)

# Each arm lists the sentiment channels appended to the state. Everything else
# (universe, window, costs, network, seeds, reward) is identical across arms.
# The news_flag channel distinguishes "no news" from "neutral news" and rides
# along in every sentiment-carrying arm.
ARMS = {
    "M1": [],                                                    # price-only control
    "M2": ["finbert_score", "news_flag"],                        # scalar FinBERT
    "M3": ["direction", "news_flag"],                            # LLM, 1 dimension
    "M4": ["direction", "magnitude", "news_flag"],
    "M5": ["direction", "magnitude", "horizon", "news_flag"],
    "M6": ["direction", "magnitude", "horizon", "confidence", "news_flag"],
}

LLM_CHANNELS = ["direction", "magnitude", "horizon", "confidence", "news_flag"]
FINBERT_CHANNELS = ["finbert_score", "news_flag"]


# ----  Reward  (§ The Portfolio MDP — Reward)

COST_C = 0.0025            # proportional transaction cost c (0.25%)
ETA_LEVELS = [1.0, 3.0, 7.0]    # CRRA risk aversion. eta = 1 is the log-utility
                                # anchor and also the mean estimate in economics
                                # contexts; 3 and 7 sit inside the 2-7 range
                                # estimated in finance contexts (Elminejad et
                                # al. 2025 meta-analysis), 7 at its top end.
GAMMA = 1.0 # discount factor (§ Discounting). Gamma = 1 makes the summed log rewards equal log terminal wealth.


# ---- Algorithms  (§ Learning Algorithms)

ALGOS = ["ppo", "sac"]


# ---- Hyperparameter tuning  (§ Implementation and Hyperparameter Search)

# Tuned once per algorithm on the control arm M1 at eta = 1, then frozen
TUNE_TRAIN_START, TUNE_TRAIN_END = "2015-02-01", "2018-12-31"
TUNE_VAL_START, TUNE_VAL_END = "2019-01-01", "2019-12-31"
TUNE_N_TRIALS = 50
# Per-trial training budget during the search. Bracketed by a budget pilot
# (2026-08: 500k-step validation curves under SB3 defaults on M1, eta=1):
# PPO plateaus at or below 50k steps, SAC peaks near 100k and then degrades.
# 200k sits at least 2x above both, so no trial is cut off while still
# learning; the deployed budget is re-calibrated with the tuned config
# afterwards (scripts/04_calibrate_budget.py).
TUNE_STEP_CAP = 200_000
TUNE_TOP_RERUN = 5              # best trials re-estimated with...
# ...this many seeds, ranked by IQM validation log return. Five, not three:
# the interquartile mean trims int(n/4) runs from each end, so at n = 3 it
# trims nothing and degenerates to the plain mean, and the robustness the
# Agarwal et al. (2021) aggregate is cited for does not materialize. At n = 5
# it drops the best and the worst run, which is the intended behaviour, and it
# matches the five seeds used per walk-forward window (N_SEEDS below).
TUNE_RERUN_SEEDS = 5


# ---- Walk-forward evaluation  (§ Evaluation Protocol)

TRAIN_YEARS = 5                 # each window: 5y train + 1y validation + 1y test
FIRST_TEST_YEAR = 2021          # first window: train 2015-19, val 2020, test 2021
LAST_TEST_YEAR = 2026           # last test period ends mid-2026 with the data
N_SEEDS = 5                     # per (arm, algo, eta, window); the seed with the
SEEDS = list(range(N_SEEDS))    # highest validation value of the trained
                                # objective (summed per-period CRRA reward;
                                # the log return at eta = 1) is tested.


# ----  Sanity guards

assert N_ASSETS == 30
assert len(ARMS) == 6 and list(ARMS) == ["M1", "M2", "M3", "M4", "M5", "M6"]
for _arm, _chans in ARMS.items():
    _valid = set(LLM_CHANNELS) | set(FINBERT_CHANNELS)
    assert set(_chans) <= _valid, f"{_arm} names unknown channels"
