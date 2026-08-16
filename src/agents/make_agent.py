"""Construction of the two learners (§ Learning Algorithms).

Both agents use the identical "MultiInputPolicy" network over the dictionary
observation of the environment, so PPO and SAC differ in the learning rule
and in nothing else.

Why gamma = config.GAMMA = 1 (§ Discounting): with log-return rewards the
UNDISCOUNTED sum of per-step rewards equals the log of terminal wealth, so
gamma = 1 trains the agent on the growth-optimal objective exactly. Any
gamma < 1 would overweight early rewards and define a different objective.
For PPO this is unproblematic (finite episodes, GAE lambda bounds the
critic's horizon); for SAC it is nonstandard — the soft Bellman operator is
only guaranteed to contract for gamma < 1 — and is accepted as a documented
methodological risk, visible in the per-window seed dispersion.
"""

from stable_baselines3 import PPO, SAC

from src import config
from src.agents.hyperparams import load_tuned


def make_agent(algo: str, env, seed: int, params: dict | None = None):
    """Build a Stable Baselines3 PPO or SAC model on `env`.

    params=None loads the frozen tuned configuration from
    results/tuning/{algo}_best.json (or falls back to SB3 defaults when no
    tuning has been run yet); a dict overrides that, which is how the Optuna
    search injects trial configurations. Everything not in `params` stays at
    the SB3 default, per the hyperparameter table.
    """
    if params is None:
        params = load_tuned(algo) or {}

    if algo == "ppo":
        return PPO(
            "MultiInputPolicy", env,
            gamma=config.GAMMA, seed=seed, verbose=0,
            **params,
        )
    if algo == "sac":
        return SAC(
            "MultiInputPolicy", env,
            gamma=config.GAMMA, seed=seed, verbose=0,
            # Automatic temperature tuning against the fixed target entropy
            # -(N+1), the negative dimension of the action space
            # (§ Soft Actor-Critic). SB3's "auto" target would resolve to the
            # same number, but the methodology pins it, so it is set
            # explicitly rather than left to a library default.
            ent_coef="auto",
            target_entropy=float(-(config.N_ASSETS + 1)),
            **params,
        )
    raise ValueError(f"unknown algo {algo!r}; expected one of {config.ALGOS}")
