"""Walk-forward training and evaluation (§ Evaluation Protocol,
Walk-Forward Design and Model Selection).

Annual sliding windows, adapted from Sood et al.: each window trains on five
calendar years, validates on the next year, and tests on the year after
that — the test year is never touched before the final rollout. The first
window trains 2015-2019, validates 2020, tests 2021; the window then slides
one year at a time until the data end mid-2026, giving six windows whose
test periods tile January 2021 -> June 2026 into one continuous
out-of-sample record.

Every window is trained FROM SCRATCH (no warm start — carrying weights
across windows would couple the window-level comparisons). Per window the
same frozen configuration is trained with config.N_SEEDS seeds; the seed
with the highest cumulative VALIDATION value of the trained objective — the
summed per-period CRRA reward sum_t u_eta(1 + rho_t), which is the cumulative
log return at eta = 1 — is selected, and that seed's test rollout provides
the headline result.

PRE-COMMITTED CONVENTION on the other seeds: every seed IS rolled out on the
test year, but those numbers exist for one purpose only — reporting the
cross-seed dispersion against which arm differences must be judged (Agarwal
et al. 2021). Model selection uses validation alone; nothing about the
selection or the headline ever looks at a test number. Recording the
non-selected seeds' test rollouts costs seconds and is what makes a proper
paired seed-by-window analysis possible without ever retraining.

The runner persists everything that would otherwise require retraining
(training is the only expensive step; every derived quantity is seconds):

    results/walkforward/{algo}/{arm}_eta{eta}/{test_year}/
        manifest.json          provenance: git state, config snapshot, tuned
                               params, library versions, input fingerprints,
                               wall-clock per seed
        model_seed{0..4}.zip   every trained policy (Tier 1 — the actual
                               "never train again" guarantee)
        ledger_val.parquet     per-step ledger, ALL seeds, validation year (local only, not committed)
        ledger_test.parquet    per-step ledger, ALL seeds, test year
        sb3_log_seed{s}.csv    SB3 training diagnostics (losses, entropy, ...)
        curve_seed{s}.csv      validation return at 25/50/75/100% of budget
        validation_seeds.csv   one row per seed: cumulative validation
                               utility (the selection criterion), cumulative
                               validation log return, and the dispersion-only
                               test log return
        test_daily.csv         SELECTED seed: date, log_return, turnover,
                               cash_weight   (headline artifact, read by
                               src/evaluation/report.py)
        test_weights.csv       SELECTED seed: date x [CASH]+DOW30 panel
        summary.json           all seed numbers, selected seed, headline
                               test numbers

Training runs sequentially with plain print() progress lines, which are the
run's live log on the training pod; the seed loop is independent, so per-seed
parallelism can be added later without changing any result.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config

CURVE_POINTS = 4  # validation evaluations per training run (25/50/75/100%)


# ---------------------------------------------------------------------------
# Window layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """One walk-forward window; all dates are ISO "YYYY-MM-DD" strings."""
    test_year: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def make_windows() -> list[Window]:
    """The six annual sliding windows of the evaluation protocol.

    For test year Y: train Y-6 .. Y-2 (five calendar years, ending Dec 31
    before the validation year), validate Y-1, test Y. The first window's
    nominal train start (2015-01-01) precedes the actual sample start
    (config.SAMPLE_START, Feb 2015 — bound by GDELT GKG v2.0 availability);
    the date-to-index mapping clamps it to the first trading day. The last
    test period ends with the data at config.SAMPLE_END (mid-2026).
    """
    windows = []
    for test_year in range(config.FIRST_TEST_YEAR, config.LAST_TEST_YEAR + 1):
        val_year = test_year - 1
        last = test_year == config.LAST_TEST_YEAR
        windows.append(Window(
            test_year=test_year,
            train_start=f"{val_year - config.TRAIN_YEARS}-01-01",
            train_end=f"{val_year - 1}-12-31",
            val_start=f"{val_year}-01-01",
            val_end=f"{val_year}-12-31",
            test_start=f"{test_year}-01-01",
            test_end=config.SAMPLE_END if last else f"{test_year}-12-31",
        ))
    return windows


def block_indices(dates, first_day: str, last_day: str,
                  k: int = config.WINDOW_K, eval_block: bool = False):
    """Map an ISO-date block to (start, end) decision indices for PortfolioEnv.

    The environment makes decisions on days t = start .. end-1, and the step
    at t consumes the return realized on day t+1. For a TRAINING block the
    first decision falls on the first trading day inside the block, clamped
    to k-1 (the earliest day with a full k-day price window). For an
    EVALUATION block (eval_block=True) the first decision moves one day
    earlier, onto the last trading day BEFORE the block, so the first
    consumed return is the one realized ON the block's first trading day —
    consecutive test years then stitch into a gapless out-of-sample record.
    In both cases the last decision (end-1) consumes the return of the
    block's last trading day, so nothing beyond `last_day` is ever used.
    """
    start = int(dates.searchsorted(first_day, side="left"))
    if eval_block:
        start -= 1
    start = max(k - 1, start)
    end = int(dates.searchsorted(last_day, side="right")) - 1
    return start, end


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def rollout(model, env) -> dict:
    """Deterministic rollout of a trained policy over one environment episode.

    Captures the environment's full per-step info ledger (see
    PortfolioEnv.step). Evaluation always reads "log_returns", NOT "rewards":
    the reward is CRRA utility (eta-dependent), while the protocol compares
    all agents on realized log returns; the rewards are recorded anyway so
    the objective the agent actually saw stays reconstructible.
    """
    obs, _ = env.reset()
    fields = {"log_returns": "log_return", "turnover": "turnover",
              "cost_paid": "cost_paid", "rho": "rho",
              "cash_gross": "cash_gross", "portfolio_gross": "portfolio_gross",
              "weights": "weights", "drifted_weights": "drifted_weights",
              "actions": "action", "t": "t"}
    out: dict = {name: [] for name in fields}
    out["rewards"] = []
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _truncated, info = env.step(action)
        for name, key in fields.items():
            out[name].append(info[key])
        out["rewards"].append(reward)
    return {name: np.asarray(values) for name, values in out.items()}


# ---------------------------------------------------------------------------
# Persistence helpers (Tiers 2-4)
# ---------------------------------------------------------------------------
def _ledger_frame(algo, arm, eta, test_year, block, seed, selected,
                  dates_str, roll) -> pd.DataFrame:
    """One seed's rollout as long-format ledger rows (one row per day).

    Column groups: run keys | post-rebalance weights w_* | pre-trade drifted
    weights wd_* | raw action scores a_* | per-day outcomes. Market context
    and the sentiment grid are NOT duplicated here — they are deterministic
    functions of the date (fingerprinted in manifest.json) and can be joined
    at analysis time.
    """
    assets = ["CASH"] + config.DOW30
    frame = {
        "algo": algo, "arm": arm, "eta": eta, "test_year": test_year,
        "block": block, "seed": seed, "selected": selected,
        "date": dates_str, "t": roll["t"],
    }
    for j, name in enumerate(assets):
        frame[f"w_{name}"] = roll["weights"][:, j]
    for j, name in enumerate(assets):
        frame[f"wd_{name}"] = roll["drifted_weights"][:, j]
    for j, name in enumerate(assets):
        frame[f"a_{name}"] = roll["actions"][:, j]
    frame.update({
        "turnover": roll["turnover"], "cost_paid": roll["cost_paid"],
        "rho": roll["rho"], "log_return": roll["log_returns"],
        "reward": roll["rewards"], "cash_gross": roll["cash_gross"],
        "portfolio_gross": roll["portfolio_gross"],
    })
    return pd.DataFrame(frame)


def _fingerprint(array: np.ndarray) -> str:
    """Short SHA-256 of an array's exact bytes.

    Lets a later reconstruction prove it ran on identical inputs — relevant
    because yfinance revises history (data/README.md) and the sentiment grid
    depends on config constants that could drift between runs.
    """
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return f"sha256:{digest[:16]}:shape{tuple(array.shape)}"


def _git_state() -> dict:
    """Commit SHA and dirty flag of the repo the run was launched from."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return {"sha": sha or None, "dirty": bool(dirty)}
    except OSError:
        return {"sha": None, "dirty": None}


def _run_manifest(algo, arm, eta, timesteps, budget_source, params,
                  fingerprints) -> dict:
    """Everything needed to trust and reproduce this run years later."""
    import gymnasium
    import stable_baselines3
    import torch

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "git": _git_state(),
        "run": {"algo": algo, "arm": arm, "eta": eta,
                "timesteps": timesteps, "budget_source": budget_source,
                "tuned_params": params, "k": config.WINDOW_K},
        # Full snapshot of the design constants the run was launched under.
        "config": {name: repr(value) for name, value in vars(config).items()
                   if name.isupper()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "input_fingerprints": fingerprints,
    }


def _train_with_curve(model, timesteps, make_val_env, curve_path):
    """Train in CURVE_POINTS chunks with a validation rollout after each.

    The intermediate rollouts give a coarse learning curve (divergence shows
    up without dense evaluation); the FINAL rollout doubles as the validation
    evaluation used for seed selection, so nothing is evaluated twice.
    reset_num_timesteps=False keeps the step counter, optimizer state, and
    replay buffer across the model.learn calls, so the curve is the
    trajectory of one continuous run. PPO rounds each chunk up to whole
    rollout buffers, so the realized total can exceed `timesteps` by up to
    (n_steps - 1) per chunk; model.num_timesteps records what actually ran.
    """
    chunk = timesteps // CURVE_POINTS
    steps, values, final_roll = [], [], None
    for _point in range(CURVE_POINTS):
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        final_roll = rollout(model, make_val_env())
        steps.append(int(model.num_timesteps))
        values.append(float(final_roll["log_returns"].sum()))
    pd.DataFrame({"steps": steps, "val_log_return": values}).to_csv(
        curve_path, index=False)
    return final_roll  # the 100%-of-budget validation rollout


# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------
def resolve_timesteps(algo: str, timesteps: int | None):
    """The per-run training budget and where it came from.

    Priority: explicit caller value > calibrated budget from
    results/budget/{algo}.json (scripts/04_calibrate_budget.py) > the
    provisional tuning cap config.TUNE_STEP_CAP.
    """
    if timesteps is not None:
        return int(timesteps), "explicit --timesteps"
    budget_file = config.RESULTS_DIR / "budget" / f"{algo}.json"
    if budget_file.exists():
        return int(json.loads(budget_file.read_text())["timesteps"]), str(budget_file)
    return config.TUNE_STEP_CAP, "config.TUNE_STEP_CAP (no calibrated budget found)"


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def run_walkforward(algo: str, arm: str, eta: float,
                    seeds=config.SEEDS, timesteps: int | None = None,
                    windows: list[int] | None = None) -> None:
    """Train and evaluate one (algo, arm, eta) cell.

    `windows` restricts the run to the given test years; None runs all six.
    Each (cell, window) is independent — it trains its own seeds, performs its
    own validation-based selection, and writes its own directory — so slicing
    by window changes nothing about the protocol or the results. It exists for
    two practical reasons: it turns the study's 18 cells into 108 independent
    tasks, which is what lets a machine with more cores than cells be used
    fully, and it allows a single failed window to be re-run on its own
    instead of repeating the whole cell.
    """
    # Imported lazily so make_windows() and its tests work without the
    # feature/env modules or the downloaded market data.
    from stable_baselines3.common.logger import configure as sb3_configure

    from src.agents.hyperparams import load_tuned
    from src.agents.make_agent import make_agent
    from src.environment.portfolio_env import PortfolioEnv
    from src.features.market_context import build_market_context, tbill_rate
    from src.features.price_tensor import load_ohlc_tensor
    from src.features.sentiment_grid import arm_grid

    eta = float(eta)  # normalizes the artifact path, e.g. eta=1 -> "eta1.0"

    # Build the full-sample arrays ONCE per process; every window's
    # environment receives the same arrays plus start/end indices and slices
    # internally, so there is exactly one copy of the data in memory.
    dates, ohlc = load_ohlc_tensor()
    tbill = tbill_rate(dates)
    context = build_market_context(dates)
    sentiment = arm_grid(arm, dates)  # None for the price-only control M1

    timesteps, budget_source = resolve_timesteps(algo, timesteps)
    manifest_base = _run_manifest(
        algo, arm, eta, timesteps, budget_source, load_tuned(algo),
        fingerprints={
            "dates": "sha256:" + hashlib.sha256(
                ",".join(d.isoformat() for d in dates).encode()
            ).hexdigest()[:16],
            "ohlc": _fingerprint(ohlc),
            "tbill": _fingerprint(tbill),
            "context": _fingerprint(context),
            "sentiment": None if sentiment is None else _fingerprint(sentiment),
        })
    print(f"walk-forward: algo={algo} arm={arm} eta={eta} seeds={list(seeds)}")
    print(f"sample: {dates[0].date()} -> {dates[-1].date()} "
          f"({len(dates)} trading days, {ohlc.shape[1]} risky assets)")
    print(f"training budget: {timesteps:,} steps per seed ({budget_source})\n")

    def env_for(start: int, end: int) -> PortfolioEnv:
        return PortfolioEnv(ohlc, tbill, context, sentiment=sentiment,
                            k=config.WINDOW_K, cost=config.COST_C, eta=eta,
                            start=start, end=end)

    chosen = make_windows()
    if windows is not None:
        wanted = set(int(y) for y in windows)
        chosen = [w for w in chosen if w.test_year in wanted]
        missing = wanted - {w.test_year for w in chosen}
        if missing:
            raise ValueError(f"no such test year(s): {sorted(missing)}; "
                             f"available: {[w.test_year for w in make_windows()]}")
        print(f"restricted to windows {[w.test_year for w in chosen]}\n")

    all_summaries = []
    for window in chosen:
        tr = block_indices(dates, window.train_start, window.train_end)
        va = block_indices(dates, window.val_start, window.val_end, eval_block=True)
        te = block_indices(dates, window.test_start, window.test_end, eval_block=True)
        out_dir = (config.RESULTS_DIR / "walkforward" / algo
                   / f"{arm}_eta{eta}" / str(window.test_year))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"window {window.test_year}: train {window.train_start}"
              f"..{window.train_end}, val {window.val_start[:4]}, "
              f"test {window.test_start[:4]}")

        # Rollout rows are dated by REALIZATION day: row j is the decision at
        # start+j whose return lands on dates[start+j+1]; the six test years
        # then tile without gaps and all artifacts line up on the same dates.
        val_dates = dates[va[0] + 1: va[1] + 1].strftime("%Y-%m-%d")
        test_dates = dates[te[0] + 1: te[1] + 1].strftime("%Y-%m-%d")

        # --- train every seed from scratch; persist everything (Tiers 1-3)
        seed_records, val_rolls, test_rolls, wall = [], {}, {}, {}
        for seed in seeds:
            started = time.perf_counter()
            model = make_agent(algo, env_for(*tr), seed=seed)
            # SB3's own diagnostics (losses, entropy, ...) as CSV per seed.
            log_dir = out_dir / f"sb3_seed{seed}"
            model.set_logger(sb3_configure(str(log_dir), ["csv"]))
            val = _train_with_curve(model, timesteps, lambda: env_for(*va),
                                    out_dir / f"curve_seed{seed}.csv")
            model.save(out_dir / f"model_seed{seed}")   # Tier 1: every policy
            # Dispersion-only test rollout (pre-commitment in the module
            # docstring): recorded now, never consulted for selection.
            test = rollout(model, env_for(*te))
            del model
            progress = log_dir / "progress.csv"
            if progress.exists():
                progress.rename(out_dir / f"sb3_log_seed{seed}.csv")
                log_dir.rmdir()
            wall[seed] = round(time.perf_counter() - started, 1)
            val_rolls[seed], test_rolls[seed] = val, test
            seed_records.append({
                "seed": int(seed),
                # Selection criterion: the validation value of the objective
                # the agent was trained on. Identical to val_log_return at
                # eta = 1, where the CRRA reward is the log return.
                "val_utility": float(val["rewards"].sum()),
                "val_log_return": float(val["log_returns"].sum()),
                "val_mean_turnover": float(val["turnover"].mean()),
                "val_days": int(len(val["log_returns"])),
                "test_log_return_dispersion_only":
                    float(test["log_returns"].sum()),
                "test_days": int(len(test["log_returns"])),
            })
            print(f"  seed {seed}: cumulative val utility "
                  f"{seed_records[-1]['val_utility']:+.4f} "
                  f"({wall[seed]:.0f}s)", flush=True)
        pd.DataFrame(seed_records).to_csv(
            out_dir / "validation_seeds.csv", index=False)

        # --- validation-based selection (never consults a test number)
        selected_seed = max(seed_records,
                            key=lambda r: r["val_utility"])["seed"]
        test = test_rolls[selected_seed]

        # --- Tier 2: per-step ledgers, all seeds, both evaluation blocks
        for block, rolls, block_dates in (("val", val_rolls, val_dates),
                                          ("test", test_rolls, test_dates)):
            frames = [_ledger_frame(algo, arm, eta, window.test_year, block,
                                    seed, seed == selected_seed,
                                    block_dates, roll)
                      for seed, roll in rolls.items()]
            pd.concat(frames, ignore_index=True).to_parquet(
                out_dir / f"ledger_{block}.parquet", index=False)

        # --- headline artifacts for the selected seed (report.py contract)
        pd.DataFrame({
            "date": test_dates,
            "log_return": test["log_returns"],
            "turnover": test["turnover"],
            "cash_weight": test["weights"][:, 0],
        }).to_csv(out_dir / "test_daily.csv", index=False)
        pd.DataFrame(test["weights"], index=test_dates,
                     columns=["CASH"] + config.DOW30,
                     ).to_csv(out_dir / "test_weights.csv", index_label="date")

        # --- Tier 4: provenance
        (out_dir / "manifest.json").write_text(json.dumps(
            {**manifest_base, "window": asdict(window),
             "wall_seconds_per_seed": wall}, indent=2))

        summary = {
            "algo": algo, "arm": arm, "eta": eta,
            "test_year": window.test_year,
            "window": asdict(window),
            "timesteps": timesteps,
            "budget_source": budget_source,
            "seeds": seed_records,
            "selected_seed": selected_seed,
            "test_log_return": float(test["log_returns"].sum()),
            "test_mean_turnover": float(test["turnover"].mean()),
            "test_days": int(len(test["log_returns"])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)
        print(f"  selected seed {selected_seed}: cumulative test log return "
              f"{summary['test_log_return']:+.4f}\n", flush=True)

    # Only meaningful as a stitched out-of-sample record when every window
    # ran; a windowed invocation reports its own subset instead.
    total = sum(s["test_log_return"] for s in all_summaries)
    years = [s["test_year"] for s in all_summaries]
    span = (f"stitched {config.FIRST_TEST_YEAR}-{config.LAST_TEST_YEAR}"
            if len(all_summaries) == len(make_windows()) else f"windows {years}")
    print(f"done: {span} out-of-sample log return {total:+.4f} "
          f"({len(all_summaries)} windows)")
