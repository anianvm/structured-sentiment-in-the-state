"""The hyperparameter search table (§ Implementation and Hyperparameter Search).

Checks that the Optuna samplers only ever draw values inside the table
domains encoded in src/agents/hyperparams.py, and that load_tuned degrades
to None (-> SB3 defaults) when no tuning result exists. No market data, no
network; the sampler tests skip cleanly when Optuna is not installed.
"""

import json

import pytest

from src import config
from src.agents import hyperparams

optuna = pytest.importorskip("optuna")

N_TRIALS = 50


def _random_draws(sample_fn, n=N_TRIALS) -> list[dict]:
    """Sample n configurations with an (unseeded-model-free) random sampler."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    draws = []

    def objective(trial):
        draws.append(sample_fn(trial))
        return 0.0

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=n)
    return draws


def test_ppo_sampler_respects_table_domains():
    for params in _random_draws(hyperparams.sample_ppo):
        assert set(params) == set(hyperparams.PPO_SPACE)
        assert 1e-5 <= params["learning_rate"] <= 1e-3
        assert params["n_steps"] in {512, 1024, 2048, 4096}
        assert params["batch_size"] in {64, 128, 256}
        # SB3 splits each rollout into minibatches; the table's choices make
        # every batch_size divide every n_steps, so no draw can violate this.
        assert params["n_steps"] % params["batch_size"] == 0
        assert params["n_epochs"] in {5, 10, 16}
        # c_2 is drawn from the table's log-uniform range OR set to exactly
        # zero, so that the search can select "no entropy bonus" (the SB3
        # default) instead of assuming some regularisation is always needed.
        assert params["ent_coef"] == 0.0 or 1e-4 <= params["ent_coef"] <= 5e-2
        assert params["clip_range"] in {0.1, 0.2, 0.3}
        assert params["gae_lambda"] in {0.90, 0.95, 0.98}


def test_ppo_sampler_can_select_zero_entropy():
    """Both branches of the entropy switch must be reachable.

    Guards the collapse case: if the categorical were dropped, every draw
    would silently carry a positive coefficient again and the search would
    re-encode the assumption it exists to test.
    """
    drawn = [p["ent_coef"] for p in _random_draws(hyperparams.sample_ppo)]
    assert any(c == 0.0 for c in drawn), "search can never switch entropy off"
    assert any(c > 0.0 for c in drawn), "search can never switch entropy on"


def test_sac_sampler_respects_table_domains():
    for params in _random_draws(hyperparams.sample_sac):
        assert set(params) == set(hyperparams.SAC_SPACE)
        assert 1e-5 <= params["learning_rate"] <= 1e-3
        assert params["batch_size"] in {128, 256, 512}
        assert 1e-3 <= params["tau"] <= 2e-2
        # Replay capacity is quarter/half/full of the training budget: a buffer
        # at or above the budget never evicts, so larger candidates would be
        # behaviourally identical to full retention (see SAC_SPACE).
        assert params["buffer_size"] in {config.TUNE_STEP_CAP // 4,
                                         config.TUNE_STEP_CAP // 2,
                                         config.TUNE_STEP_CAP}
        assert params["buffer_size"] <= config.TUNE_STEP_CAP
        # gradient_steps is fixed at the library default rather than searched
        # (see SAC_SPACE), so a sampled configuration must not set it.
        assert "gradient_steps" not in params


def test_load_tuned_returns_none_when_no_result_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert hyperparams.load_tuned("ppo") is None
    assert hyperparams.load_tuned("sac") is None


def test_load_tuned_round_trips_a_frozen_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    frozen = {"learning_rate": 3e-4, "n_steps": 2048}
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning" / "ppo_best.json").write_text(json.dumps(frozen))
    assert hyperparams.load_tuned("ppo") == frozen


# ---------------------------------------------------------------------------
# Parallel search: workers must not propose the same configuration
# ---------------------------------------------------------------------------

def test_sac_space_fixes_gradient_steps():
    """gradient_steps is no longer searched: it would break the step-cap
    fairness (four updates per environment step is four times the learning on
    the same nominal budget) and made trial cost vary eightfold."""
    assert "gradient_steps" not in hyperparams.SAC_SPACE
    discrete = [v[1] for v in hyperparams.SAC_SPACE.values() if v[0] == "choice"]
    combos = 1
    for d in discrete:
        combos *= len(d)
    assert combos == 9          # batch (3) x buffer (3)


def test_parallel_workers_propose_distinct_configurations():
    """The failure this guards against actually happened: eight workers each
    built TPESampler(seed=0), drew the identical proposal, and produced eight
    completed trials with the same parameters and the same objective value."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    import json, tempfile, os
    from src.experiments.tune import TPE_BASE_SEED

    db = os.path.join(tempfile.mkdtemp(), "t.db")
    storage = f"sqlite:///{db}"
    optuna.create_study(study_name="x", storage=storage, direction="maximize")

    def draw(seed):
        study = optuna.create_study(
            study_name="x", storage=storage, direction="maximize",
            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed))
        return json.dumps(hyperparams.sample_sac(study.ask()), sort_keys=True)

    same = {draw(TPE_BASE_SEED) for _ in range(8)}
    offset = {draw(TPE_BASE_SEED + w) for w in range(8)}
    assert len(same) == 1, "a shared seed should collapse to one proposal"
    assert len(offset) == 8, "per-worker seeds must give distinct proposals"
