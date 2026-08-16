"""Hyperparameter search (§ Implementation and Hyperparameter Search).

Each algorithm is tuned EXACTLY ONCE, on the price-only control arm M1 at
eta = 1, and the winning configuration is then frozen for all arms, all
risk-aversion levels, and all walk-forward windows. This protocol cannot
interact with the experimental manipulation: the configuration never sees a
sentiment channel, so it cannot be adapted to one — which is conservative
for the sentiment hypotheses (if anything, the sentiment arms run with a
configuration optimized for their competitor). Tuning at eta = 1 anchors the
search on the log-utility objective, where the summed rewards equal log
terminal wealth. Both algorithms get the same protocol (trial budget, step
cap, data split). PPO's fifty trials ran sequentially; SAC's eight parallel
workers retained the seven trials still in flight when the fiftieth
completed, giving 57 -- the asymmetry the methodology chapter footnotes when
comparing the two algorithms.

Protocol per algorithm:
  1. Optuna TPE (seeded), config.TUNE_N_TRIALS trials over the table spaces
     in src/agents/hyperparams.py. Each trial trains on 2015-2018
     (config.TUNE_TRAIN_*) for config.TUNE_STEP_CAP steps and is scored by
     cumulative log return on the 2019 tuning-validation year — the 2020
     seed-selection year and all test years stay untouched.
  2. A single run is a noisy performance estimate (Henderson et al. 2017),
     so the config.TUNE_TOP_RERUN best configurations are re-estimated with
     config.TUNE_RERUN_SEEDS seeds each and ranked by interquartile-mean
     validation log return (Agarwal et al. 2021).
  3. The winner's kwargs are frozen to results/tuning/{algo}_best.json,
     which src.agents.hyperparams.load_tuned serves to every later run.

The Optuna study lives in sqlite (results/tuning/{algo}.db), so an
interrupted search resumes where it stopped.
"""

import json
import os
import time

import numpy as np

from src import config
from src.agents.hyperparams import sample_ppo, sample_sac, tuning_dir
from src.experiments.walkforward import block_indices, rollout

# Phase 2 coordination: how long a worker waits for the search to finish
# before giving up, and how often it re-checks.
TPE_BASE_SEED = 0        # sampler seed; each worker adds its index
PHASE2_WAIT_S = 12 * 3600
PHASE2_POLL_S = 30


def iqm(values) -> float:
    """Interquartile mean: drop the bottom and top 25% of runs, average the rest.

    The cross-seed aggregate recommended by Agarwal et al. (2021, "Deep
    Reinforcement Learning at the Edge of the Statistical Precipice"):
    unlike the mean it is robust to unusually strong or diverged runs,
    unlike the median it still uses half the data. The 25% cut is truncated
    toward zero: with the config.TUNE_RERUN_SEEDS = 5 runs used here it drops
    the single best and worst run and averages the middle three.
    """
    x = np.sort(np.asarray(values, dtype=float))
    cut = int(len(x) * 0.25)
    return float(x[cut: len(x) - cut].mean())


def _val_log_return(algo: str, params: dict, seed: int, make_env) -> float:
    """Train one configuration and score it on the tuning-validation year.

    make_env(block) -> PortfolioEnv for block in {"train", "val"}; passed in
    so the (expensive) data arrays are built once by run_tuning.
    """
    from src.agents.make_agent import make_agent  # lazy: keeps import light

    model = make_agent(algo, make_env("train"), seed=seed, params=params)
    model.learn(total_timesteps=config.TUNE_STEP_CAP)
    val = rollout(model, make_env("val"))
    return float(val["log_returns"].sum())


def run_tuning(algo: str, worker: int = 0) -> dict:
    """Run (or resume) the full search for `algo`; returns the winning kwargs."""
    import optuna  # lazy: only the tuning script needs it

    # Imported lazily so this module imports without the feature/env modules
    # or the downloaded market data (mirrors walkforward.run_walkforward).
    from src.environment.portfolio_env import PortfolioEnv
    from src.features.market_context import build_market_context, tbill_rate
    from src.features.price_tensor import load_ohlc_tensor

    # Data plumbing, once per process. Arm M1: sentiment=None, eta=1.
    dates, ohlc = load_ohlc_tensor()
    tbill = tbill_rate(dates)
    context = build_market_context(dates)
    blocks = {
        "train": block_indices(dates, config.TUNE_TRAIN_START,
                               config.TUNE_TRAIN_END),
        "val": block_indices(dates, config.TUNE_VAL_START,
                             config.TUNE_VAL_END, eval_block=True),
    }

    def make_env(block: str) -> PortfolioEnv:
        start, end = blocks[block]
        return PortfolioEnv(ohlc, tbill, context, sentiment=None,
                            k=config.WINDOW_K, cost=config.COST_C, eta=1.0,
                            start=start, end=end)

    sample = {"ppo": sample_ppo, "sac": sample_sac}[algo]

    def objective(trial) -> float:
        params = sample(trial)
        # trial.params is NOT usable as SB3 kwargs: a "log-uniform-or-zero"
        # entry draws {name}_on and {name}_value and collapses them into one
        # kwarg, so replaying trial.params would pass ent_coef_on=True to the
        # PPO constructor. Stash the materialised kwargs for the rerun phase.
        trial.set_user_attr("sb3_params", params)
        score = _val_log_return(algo, params, seed=0, make_env=make_env)
        print(f"trial {trial.number:3d}: val log return {score:+.4f}  "
              f"{params}", flush=True)
        return score

    # --- Phase 1: 50 TPE trials, resumable via sqlite -----------------------
    out = tuning_dir()
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=algo,
        storage=f"sqlite:///{out / f'{algo}.db'}",
        direction="maximize",
        # The sampler seed MUST differ per worker. Each worker builds its own
        # sampler, so an identical seed makes every worker draw the identical
        # proposal from the identical study state: eight workers then evaluate
        # one configuration eight times instead of eight configurations. This
        # is not hypothetical -- it is what an earlier run did, producing eight
        # completed trials with the same parameters and the same objective
        # value to six decimals. Offsetting by the worker index keeps the run
        # reproducible while making the proposals distinct.
        sampler=optuna.samplers.TPESampler(seed=TPE_BASE_SEED + worker),
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = config.TUNE_N_TRIALS - done
    print(f"tuning {algo}: {done} trials already in {out / f'{algo}.db'}, "
          f"{max(remaining, 0)} to run")

    def stop_at_budget(study_, _trial):
        """Stop this worker once the STUDY reaches config.TUNE_N_TRIALS.

        The trial budget belongs to the study, not to the process. Several
        workers may attach to the same sqlite study to run the search in
        parallel, and each of them computes `remaining` from the trial count
        it sees at start-up, which is zero for all of them if they launch
        together. Without this callback eight workers would therefore run
        fifty trials each. Re-counting completed trials after every trial and
        stopping at the budget keeps "fifty trials per algorithm" true however
        many workers are attached.
        """
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE
                        for t in study_.get_trials(deepcopy=False))
        if completed >= config.TUNE_N_TRIALS:
            study_.stop()

    if remaining > 0:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[stop_at_budget])

    # --- Phase 2: re-estimate the best configs with multiple seeds ----------
    # Phase 2 must run EXACTLY ONCE, and only after the whole search is in.
    # With several workers attached, each would otherwise re-estimate the top
    # five itself (eight workers x 25 runs = 200 runs for 25 runs of work),
    # and a worker that finished early would rank an incomplete study. So:
    # first wait for the study to reach its trial budget, then let exactly one
    # worker win an atomic lock and do the re-estimation while the rest exit.
    def _completed_count():
        return sum(t.state == optuna.trial.TrialState.COMPLETE
                   for t in study.get_trials(deepcopy=False))

    waited = 0
    while _completed_count() < config.TUNE_N_TRIALS and waited < PHASE2_WAIT_S:
        time.sleep(PHASE2_POLL_S)
        waited += PHASE2_POLL_S
    if _completed_count() < config.TUNE_N_TRIALS:
        raise RuntimeError(
            f"gave up waiting for the search: {_completed_count()} of "
            f"{config.TUNE_N_TRIALS} trials complete after {waited}s. "
            f"Re-run this script to resume; the study is on disk.")

    # Already finished on an earlier invocation: nothing to redo.
    best_path = out / f"{algo}_best.json"
    if best_path.exists():
        print(f"{best_path} already exists; re-estimation already done.")
        return json.loads(best_path.read_text())

    lock = out / f"{algo}_rerun.lock"
    try:
        # O_CREAT | O_EXCL is atomic: exactly one worker creates the file.
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(f"another worker is running the re-estimation "
              f"({lock} exists); exiting. If no worker is alive and "
              f"{best_path.name} is missing, this lock is stale: delete it "
              f"and re-run to resume.")
        return {}

    try:
        return _reestimate_and_freeze(algo, study, out, make_env, optuna)
    finally:
        # Release even on failure, so a crashed re-estimation does not lock
        # every later attempt out of a study that is already on disk.
        lock.unlink(missing_ok=True)


def _reestimate_and_freeze(algo, study, out, make_env, optuna) -> dict:
    """Phase 2 proper: re-estimate the top configurations and freeze a winner.

    Split out of run_tuning so the lock that guarantees it runs once can wrap
    it in a try/finally without indenting the whole body.
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value, reverse=True)
    top = top[:config.TUNE_TOP_RERUN]
    print(f"\nre-running top {len(top)} configurations with "
          f"{config.TUNE_RERUN_SEEDS} seeds each")
    candidates = []
    for trial in top:
        # SB3 kwargs, not trial.params -- see the note in objective(). Trials
        # from a study predating that user attribute fall back to trial.params,
        # which is correct for spaces without a "log-uniform-or-zero" entry.
        params = trial.user_attrs.get("sb3_params", trial.params)
        seed_values = [
            _val_log_return(algo, params, seed=s, make_env=make_env)
            for s in range(config.TUNE_RERUN_SEEDS)
        ]
        candidates.append({
            "trial_number": trial.number,
            "params": params,
            "search_value": trial.value,
            "seed_values": seed_values,
            "iqm_val_log_return": iqm(seed_values),
        })
        print(f"trial {trial.number:3d}: seeds {np.round(seed_values, 4)} "
              f"-> IQM {candidates[-1]['iqm_val_log_return']:+.4f}", flush=True)

    winner = max(candidates, key=lambda c: c["iqm_val_log_return"])
    # {algo}_best.json holds ONLY the constructor kwargs (what load_tuned
    # feeds into make_agent); the ranking evidence goes to {algo}_rerun.json.
    (out / f"{algo}_best.json").write_text(json.dumps(winner["params"], indent=2))
    (out / f"{algo}_rerun.json").write_text(json.dumps(candidates, indent=2))
    print(f"\nwinner: trial {winner['trial_number']} "
          f"(IQM {winner['iqm_val_log_return']:+.4f}) "
          f"-> {out / f'{algo}_best.json'}")
    return winner["params"]
