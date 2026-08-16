# Tuning artifacts

- `ppo.db`, `sac.db` — the Optuna studies behind the tuned configuration
  (PPO: 50 sequential trials; SAC: 57 — eight parallel workers retained the
  seven in-flight trials when the fiftieth completed, as disclosed in the
  methodology chapter).
- `ppo_best.json`, `sac_best.json` — the frozen winning configuration per
  algorithm, read by the walk-forward runner.
- `ppo_rerun.json`, `sac_rerun.json` — phase-2 five-seed reruns of the top
  five candidates (IQM-ranked selection).
