"""Lookback-window selection (§ MDP — State).

The price tensor X_t carries the previous k trading days. Published choices
range from 7 (Aboussalah et al.) to 60 (Sood et al.), so k is selected here
rather than assumed, and then applied unchanged to both algorithms, all six
arms, all risk-aversion levels, and all walk-forward windows.

The selection is run ONCE, under PPO, and its result is used for SAC as well.
The lookback fixes the observation space rather than the learning rule, so a
common value keeps the state identical across the two learners and leaves the
PPO/SAC comparison a comparison of learning rules alone. Repeating the search
under SAC is also disproportionate: SAC trains roughly thirty times slower per
environment step, so a second sweep would spend about a third of the study's
compute budget re-deciding a single design constant. The --algo flag is kept
so the choice can be checked under SAC if compute allows.

The rule mirrors the hyperparameter protocol of src/experiments/tune.py so
that the two selections cannot be played off against each other:

  * arm M1 (price-only control) at eta = 1, so the choice never sees a
    sentiment channel and cannot be adapted to the treatment;
  * trained on the tuning split (config.TUNE_TRAIN_*, 2015-2018) for
    config.TUNE_STEP_CAP steps and scored by cumulative log return on the
    2019 tuning-validation year, leaving the 2020 seed-selection year and all
    test years untouched;
  * repeated over config.TUNE_RERUN_SEEDS seeds and aggregated by the
    interquartile mean (Agarwal et al. 2021), because a single run is a noisy
    estimate of a configuration's quality.

Two deliberate choices deserve comment. First, the candidates are compared
under the Stable Baselines3 DEFAULTS, not under a tuned configuration: k fixes
the observation space, so it has to be settled before hyperparameters can be
tuned on that space. This is the same one-pass resolution used for the
training budget, and the subsequent tuning runs on the selected k confirm
whether the choice was reasonable. Second, every candidate trains on the
IDENTICAL set of decision days: the first decision index is clamped to
max(K_CANDIDATES) - 1 for all k, so a longer window does not silently cost
training days and the comparison isolates the window length itself.

Writes results/window_selection/{algo}.json (the selected k plus the full
per-seed evidence), {algo}_scores.csv and {algo}_scores.png.

Run as:  python scripts/02_select_window.py --algo ppo
"""

import json

import numpy as np
import pandas as pd

from src import config
from src.experiments.tune import iqm
from src.experiments.walkforward import block_indices, rollout


def select_window(algo: str, candidates=None, seeds: int | None = None) -> int:
    """Score every candidate k and return the selected one."""
    # Lazy imports: keep this module importable without the data modules.
    from src.agents.make_agent import make_agent
    from src.environment.portfolio_env import PortfolioEnv
    from src.features.market_context import build_market_context, tbill_rate
    from src.features.price_tensor import load_ohlc_tensor

    candidates = list(candidates or config.K_CANDIDATES)
    n_seeds = config.TUNE_RERUN_SEEDS if seeds is None else int(seeds)

    dates, ohlc = load_ohlc_tensor()
    tbill = tbill_rate(dates)
    context = build_market_context(dates)

    # Identical training days for every candidate: the longest window needs the
    # most history, so every k starts where the longest one can start.
    warmup = max(candidates) - 1
    tr_start, tr_end = block_indices(dates, config.TUNE_TRAIN_START,
                                     config.TUNE_TRAIN_END, k=1)
    tr_start = max(tr_start, warmup)
    va_start, va_end = block_indices(dates, config.TUNE_VAL_START,
                                     config.TUNE_VAL_END, k=1, eval_block=True)
    va_start = max(va_start, warmup)

    print(f"window selection: algo={algo} candidates={candidates} "
          f"seeds={n_seeds} steps={config.TUNE_STEP_CAP:,}")
    print(f"train days {tr_end - tr_start}, validation days {va_end - va_start} "
          f"(identical for every candidate)\n")

    def env_for(k: int, start: int, end: int) -> PortfolioEnv:
        # Arm M1: sentiment=None. eta=1: the log-utility anchor.
        return PortfolioEnv(ohlc, tbill, context, sentiment=None, k=k,
                            cost=config.COST_C, eta=1.0, start=start, end=end)

    records = []
    for k in candidates:
        seed_values = []
        for seed in range(n_seeds):
            model = make_agent(algo, env_for(k, tr_start, tr_end), seed=seed)
            model.learn(total_timesteps=config.TUNE_STEP_CAP)
            val = rollout(model, env_for(k, va_start, va_end))
            seed_values.append(float(val["log_returns"].sum()))
            del model
        records.append({
            "k": int(k),
            "seed_values": seed_values,
            "iqm_val_log_return": iqm(seed_values),
            "mean_val_log_return": float(np.mean(seed_values)),
            "std_val_log_return": float(np.std(seed_values, ddof=1)),
        })
        print(f"k={k:>3}: seeds {np.round(seed_values, 4)} "
              f"-> IQM {records[-1]['iqm_val_log_return']:+.4f}", flush=True)

    best = max(records, key=lambda r: r["iqm_val_log_return"])
    selected = best["k"]

    out = config.RESULTS_DIR / "window_selection"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "seed_values"}
                  for r in records]).to_csv(out / f"{algo}_scores.csv", index=False)
    _plot(algo, records, selected, out / f"{algo}_scores.png")
    (out / f"{algo}.json").write_text(json.dumps({
        "selected_k": selected,
        "rule": f"max interquartile-mean validation log return over "
                f"{n_seeds} seeds, arm M1, eta=1, SB3 defaults",
        "candidates": candidates,
        "seeds": n_seeds,
        "timesteps": config.TUNE_STEP_CAP,
        "train": [config.TUNE_TRAIN_START, config.TUNE_TRAIN_END],
        "validation": [config.TUNE_VAL_START, config.TUNE_VAL_END],
        "records": records,
    }, indent=2))
    print(f"\nselected k = {selected} -> {out / f'{algo}.json'}")
    print("Set config.WINDOW_K to this value before tuning "
          "(it fixes the observation space every later stage depends on).")
    return selected


def _plot(algo, records, selected, path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless: write the file, never open a window
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in records]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in records:  # individual seeds behind the aggregate
        ax.scatter([r["k"]] * len(r["seed_values"]), r["seed_values"],
                   s=14, color="lightsteelblue", zorder=2)
    ax.plot(ks, [r["iqm_val_log_return"] for r in records],
            marker="o", color="tab:blue", zorder=3, label="IQM over seeds")
    ax.axvline(selected, ls="--", color="grey", label=f"selected k = {selected}")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("lookback window k (trading days)")
    ax.set_ylabel("cumulative validation log return (2019)")
    ax.set_title(f"{algo.upper()} lookback-window selection (M1, eta=1)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
