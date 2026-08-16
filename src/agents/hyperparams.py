"""Hyperparameter search spaces for PPO and SAC.

Implements the search table of § Implementation and Hyperparameter Search
(tab:algo-params in the thesis methodology chapter). The table is encoded once, as
data, in PPO_SPACE and SAC_SPACE; the Optuna samplers below read those dicts,
so the table exists in exactly one place in the code.

Each space is centered on the Stable Baselines3 default and widened to
bracket the values used in comparable studies (see the table caption for the
anchor citations). Everything NOT listed here stays at the SB3 defaults.

Optuna is deliberately not imported at module level: the samplers only call
methods on the `trial` object they receive, so this module (and load_tuned,
which make_agent needs on every training run) imports fine on machines
without Optuna installed.
"""

import json

from src import config

# Each entry is ("log-uniform", low, high) for a continuous log-scaled range,
# ("log-uniform-or-zero", low, high) for the same range with zero reachable, or
# ("choice", [candidates]) for a categorical set — verbatim from the table.
PPO_SPACE = {
    "learning_rate": ("log-uniform", 1e-5, 1e-3),
    "n_steps": ("choice", [512, 1024, 2048, 4096]),        # rollout length
    "batch_size": ("choice", [64, 128, 256]),              # minibatch size
    "n_epochs": ("choice", [5, 10, 16]),                   # surrogate epochs K
    "ent_coef": ("log-uniform-or-zero", 1e-4, 5e-2),       # c_2; 0 reachable
    "clip_range": ("choice", [0.1, 0.2, 0.3]),             # epsilon
    "gae_lambda": ("choice", [0.90, 0.95, 0.98]),
}
# Note: every n_steps candidate is a multiple of every batch_size candidate,
# so no sampled combination triggers SB3's truncated-minibatch warning.

SAC_SPACE = {
    "learning_rate": ("log-uniform", 1e-5, 1e-3),
    "batch_size": ("choice", [128, 256, 512]),
    "tau": ("log-uniform", 1e-3, 2e-2),                    # target smoothing nu
    # Replay capacity as a fraction of the training budget: quarter, half, and
    # full retention. A buffer at least as large as the number of steps trained
    # never evicts, so any candidate above the budget behaves EXACTLY like the
    # budget itself — the search would spend trials distinguishing configurations
    # that cannot differ — while still paying for the memory (a dictionary
    # transition here is ~12.4 KiB, so full retention is ~2.5 GB per run).
    # Changing TUNE_STEP_CAP changes these candidates, which an existing Optuna
    # study cannot absorb: start a fresh study if the cap moves.
    "buffer_size": ("choice", [config.TUNE_STEP_CAP // 4,
                               config.TUNE_STEP_CAP // 2,
                               config.TUNE_STEP_CAP]),
}
# gradient_steps is FIXED at the Stable Baselines3 default of one update per
# environment step, and is deliberately not searched. Two reasons.
#
# The protocol equalises ENVIRONMENT steps: every trial trains for
# TUNE_STEP_CAP steps, and both algorithms receive the same cap so that
# neither is better optimised than the other. But gradient_steps multiplies
# how much learning happens per environment step, so a trial at four would
# perform four times the gradient computation on the same nominal budget --
# the cap would no longer equalise what it is meant to equalise. At 2e5 steps
# that is 8e5 updates over a training period of roughly 1,200 transitions,
# well past the point where the budget pilot found SAC already degrading.
#
# It also made the search cost unpredictable: measured on the rented box, a
# trial at gradient_steps=4 with batch 512 was on track for ~15 h against
# ~114 min at gradient_steps=1, an eightfold spread that put the total search
# time anywhere between half a day and several days depending on what the
# sampler happened to draw.

# SAC's entropy temperature alpha is NOT searched: it is tuned automatically
# against the fixed target entropy -(N+1) (§ Soft Actor-Critic), which
# make_agent sets explicitly.


def _sample(trial, space: dict) -> dict:
    """Draw one configuration from `space` using an Optuna trial.

    The returned dict is the SB3 constructor kwargs, which is NOT the same as
    trial.params once a "log-uniform-or-zero" entry is present: that spec draws
    two Optuna parameters and collapses them into one kwarg. Callers that need
    the kwargs of a finished trial must therefore read them back from the
    trial's user attributes rather than from trial.params (see tune.py).
    """
    params = {}
    for name, spec in space.items():
        if spec[0] == "log-uniform":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif spec[0] == "log-uniform-or-zero":
            # A log-uniform range cannot contain zero, so regularisation is
            # switched on or off by its own categorical and the coefficient is
            # drawn only when it is on. This lets the search select "no entropy
            # bonus" -- the Stable Baselines3 default -- rather than assuming
            # that some entropy is always required.
            if trial.suggest_categorical(f"{name}_on", [True, False]):
                params[name] = trial.suggest_float(
                    f"{name}_value", spec[1], spec[2], log=True)
            else:
                params[name] = 0.0
        else:  # "choice"
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def sample_ppo(trial) -> dict:
    """Draw one PPO configuration (kwargs for stable_baselines3.PPO)."""
    return _sample(trial, PPO_SPACE)


def sample_sac(trial) -> dict:
    """Draw one SAC configuration (kwargs for stable_baselines3.SAC)."""
    return _sample(trial, SAC_SPACE)


def tuning_dir():
    """Directory holding the Optuna storage and the frozen winners.

    A function rather than a module constant (and a path not in config.py,
    which owns only methodology-fixed values) so tests can monkeypatch
    config.RESULTS_DIR.
    """
    return config.RESULTS_DIR / "tuning"


def load_tuned(algo: str) -> dict | None:
    """Return the frozen tuned configuration for `algo`, or None if absent.

    Reads results/tuning/{algo}_best.json, written once by
    src/experiments/tune.py. The JSON holds exactly the constructor kwargs
    selected by the search; every hyperparameter not listed in it stays at
    the Stable Baselines3 default. None means "not tuned yet" and make_agent
    then falls back to pure SB3 defaults.
    """
    path = tuning_dir() / f"{algo}_best.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
