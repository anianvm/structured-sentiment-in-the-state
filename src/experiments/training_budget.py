"""Training-budget calibration (§ Implementation and Hyperparameter Search;
App. training-budget).

The training budget — environment steps per run — is calibrated AFTER
hyperparameter tuning, because how much is learned per step depends on each
algorithm's update settings (rollout length, epochs, gradient steps, ...).
The tuned configuration is retrained on the control arm M1 at eta = 1 and
the cumulative validation log return is measured every `eval_every` steps.
The budget is then set "where validation log return begins to plateau",
operationalized by a simple plateau rule:

    chosen = the SMALLEST evaluated step count whose validation log return
             is within 2% of the maximum over the whole curve
             (v >= v_max - 0.02 * |v_max|).

The rule is deliberately crude: the curve is a single-seed estimate and any
finer rule would fit its noise. The budget is chosen separately for PPO and
SAC (results/budget/{algo}.json) and then held fixed across all arms, eta
levels, and walk-forward windows, so every configuration trains with equal
intensity. One episode covers the five-year training period (~1,260 daily
steps); a larger budget means more passes over the same data, not new
experience.
"""

import json

import pandas as pd

from src import config
from src.agents.hyperparams import load_tuned
from src.experiments.walkforward import block_indices, rollout

EVAL_EVERY_DEFAULT = 100_000    # curve resolution; not a methodology constant
# Curve length cap; also not a methodology constant. 500k, matching the range
# the budget pilot actually covered: PPO was flat
# across the whole 500k, and SAC peaked at 100k and fell to 40% of that peak
# by 500k. The plateau rule takes the smallest point within 2% of the curve
# max, so a longer curve can only change the chosen budget if a later point
# beats the early peak -- which would mean SAC recovering from 0.118 back
# above 0.297. Tuning also selected FASTER learning than the defaults the
# pilot ran -- top-5 median lr 7.9e-4 against 3.0e-4, and 9.9e-4 in the
# second-ranked trial -- which moves the peak earlier, not later. Four of the
# five took a 100k replay buffer rather than the smallest 50k on offer, so
# the memory argument cuts mildly the other way; the 2.6x learning rate is
# the load-bearing one.
MAX_STEPS_DEFAULT = 500_000
PLATEAU_TOLERANCE = 0.02        # "within 2% of the curve max" (module docstring)


def choose_budget(steps: list[int], val_returns: list[float]) -> int:
    """The plateau rule of the module docstring, as a pure function."""
    v_max = max(val_returns)
    threshold = v_max - PLATEAU_TOLERANCE * abs(v_max)
    for s, v in zip(steps, val_returns):
        if v >= threshold:
            return s
    return steps[-1]  # unreachable: v_max itself always passes


def calibrate(algo: str, max_steps: int = MAX_STEPS_DEFAULT,
              eval_every: int = EVAL_EVERY_DEFAULT) -> int:
    """Trace the validation curve for `algo` and freeze the chosen budget.

    Writes under results/budget/: {algo}_curve.csv, {algo}_curve.png, and
    {algo}.json {"timesteps": chosen} — which run_walkforward reads when no
    explicit budget is given. Returns the chosen budget.
    """
    # Lazy imports: keep this module importable without the data modules.
    from src.agents.make_agent import make_agent
    from src.environment.portfolio_env import PortfolioEnv
    from src.features.market_context import build_market_context, tbill_rate
    from src.features.price_tensor import load_ohlc_tensor

    params = load_tuned(algo)
    if params is None:
        print(f"WARNING: no tuned configuration for {algo} "
              f"(run scripts/03_tune.py first); calibrating on SB3 defaults")
        params = {}

    # Same split as tuning: train 2015-2018, validate on 2019; the 2020
    # seed-selection year and all test years stay untouched.
    dates, ohlc = load_ohlc_tensor()
    tbill = tbill_rate(dates)
    context = build_market_context(dates)
    tr = block_indices(dates, config.TUNE_TRAIN_START, config.TUNE_TRAIN_END)
    va = block_indices(dates, config.TUNE_VAL_START, config.TUNE_VAL_END,
                       eval_block=True)

    def make_env(start, end):
        return PortfolioEnv(ohlc, tbill, context, sentiment=None,
                            k=config.WINDOW_K, cost=config.COST_C, eta=1.0,
                            start=start, end=end)

    # One continuous training run, paused every eval_every steps for a
    # deterministic validation rollout (reset_num_timesteps=False keeps the
    # step counter, optimizer state, and replay buffer across model.learn
    # calls, so the curve is the trajectory of a single run).
    print(f"calibrating {algo}: up to {max_steps:,} steps, "
          f"evaluating every {eval_every:,}")
    model = make_agent(algo, make_env(*tr), seed=0, params=params)
    steps, val_returns = [], []
    for step in range(eval_every, max_steps + 1, eval_every):
        model.learn(total_timesteps=eval_every, reset_num_timesteps=False)
        val = rollout(model, make_env(*va))
        v = float(val["log_returns"].sum())
        steps.append(step)
        val_returns.append(v)
        print(f"  {step:>9,d} steps: cumulative val log return {v:+.4f}",
              flush=True)

    chosen = choose_budget(steps, val_returns)

    out = config.RESULTS_DIR / "budget"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"steps": steps, "val_log_return": val_returns}).to_csv(
        out / f"{algo}_curve.csv", index=False)
    _plot_curve(algo, steps, val_returns, chosen, out / f"{algo}_curve.png")
    (out / f"{algo}.json").write_text(json.dumps({
        "timesteps": chosen,
        "rule": f"smallest step count within {PLATEAU_TOLERANCE:.0%} "
                f"of the curve max",
        "curve_max": float(max(val_returns)),
        "max_steps": max_steps,
        "eval_every": eval_every,
    }, indent=2))
    print(f"chosen budget: {chosen:,} steps -> {out / f'{algo}.json'}")
    return chosen


def _plot_curve(algo, steps, val_returns, chosen, path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless: write the file, never open a window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, val_returns, marker="o", ms=3)
    ax.axvline(chosen, ls="--", color="grey",
               label=f"chosen budget = {chosen:,}")
    ax.set_xlabel("training steps")
    ax.set_ylabel("cumulative validation log return")
    ax.set_title(f"{algo.upper()} training-budget calibration "
                 f"(M1, eta=1, seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
