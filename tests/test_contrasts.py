"""The arm-contrast layer (src/evaluation/contrasts.py).

These pin the decision rule's plumbing, not the statistics -- inference.py
has its own tests. What matters here: the Holm family is per risk-aversion
level, every arm is compared against M1 and never against itself, and a
contrast is only "supported" when all four pre-registered conditions hold.
Fast by construction: the bootstrap draw count is turned right down, since
these assert structure rather than p-values.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import contrasts


def _series(n, mean, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(rng.normal(mean, 0.01, n), index=idx)


def _ledger(n, mean, seed, seeds=5):
    frames = []
    idx = pd.bdate_range("2021-01-04", periods=n)
    for s in range(seeds):
        rng = np.random.default_rng(seed * 100 + s)
        frames.append(pd.DataFrame({
            "seed": s, "date": idx,
            "log_return": rng.normal(mean, 0.01, n)}))
    return pd.concat(frames, ignore_index=True)


def _fixture(arms=("M1", "M2", "M3"), etas=(1.0, 3.0), years=("2021", "2022")):
    daily, ledger = {}, {}
    for i, arm in enumerate(arms):
        for eta in etas:
            daily[(arm, eta)] = {y: _series(60, 0.0004 * (i + 1), i * 10 + j)
                                 for j, y in enumerate(years)}
            ledger[(arm, eta)] = {y: _ledger(60, 0.0004 * (i + 1), i * 10 + j)
                                  for j, y in enumerate(years)}
    return daily, ledger


def test_the_preregistered_contrast_set_is_instantiated():
    daily, ledger = _fixture()      # arms M1..M3 -> H1a x2 + H2 x1 per eta
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    got = {(r["arm"], r["baseline"], r["hypothesis"], r["eta"]) for r in rows}
    for eta in (1.0, 3.0):
        assert ("M3", "M1", "H1", eta) in got
        assert ("M3", "M2", "H3", eta) in got
        # M2 vs M1 is supplementary: computed and reported, not a hypothesis
        assert ("M2", "M1", "SUPP", eta) in got
    assert len(rows) == 6                                  # 3 contrasts x 2 etas
    # eta = 1 is the primary family; other levels are exploratory
    assert all(not r["exploratory"] for r in rows if r["eta"] == 1.0)
    assert all(r["exploratory"] for r in rows if r["eta"] != 1.0)


def test_full_arm_set_yields_nine_contrasts_per_eta():
    daily, ledger = _fixture(arms=("M1", "M2", "M3", "M4", "M5", "M6"),
                             etas=(1.0,))
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    assert len(rows) == 9
    assert sum(r["hypothesis"] == "H1" for r in rows) == 4
    assert sum(r["hypothesis"] == "H3" for r in rows) == 1
    assert sum(r["hypothesis"] == "H4" for r in rows) == 3
    assert sum(r["hypothesis"] == "SUPP" for r in rows) == 1


def test_holm_only_on_the_h1a_family_h2_h3_carry_raw_one_sided_p():
    daily, ledger = _fixture(arms=("M1", "M2", "M3", "M4", "M5", "M6"),
                             etas=(1.0,))
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    h1 = [r for r in rows if r["hypothesis"] == "H1"]
    assert len(h1) == 4
    for r in h1:      # Holm over m = 4: bounded by [p_one, 4 * p_one]
        assert r["p_holm"] >= r["p_one_sided"] - 1e-12
        assert r["p_holm"] <= min(1.0, 4 * r["p_one_sided"]) + 1e-12
    for r in rows:     # H2/H3 judged individually; SUPP outside every family
        if r["hypothesis"] in ("H3", "H4", "SUPP"):
            assert r["p_holm"] == pytest.approx(r["p_one_sided"])


def test_support_is_significance_and_band_under_the_amended_rule():
    daily, ledger = _fixture()
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    for r in rows:
        assert r["supported"] == (r["significant"] and r["exceeds_seed_band"])
        # sign and both influence readings are still reported per row
        for col in ("correct_sign", "regime_robust", "loo_sign_stable"):
            assert col in r


def test_seed_band_uses_all_five_seeds_not_the_selected_one():
    _, ledger = _fixture()
    band_input = contrasts.per_seed_sharpe(ledger[("M2", 1.0)], 0.02)
    assert len(band_input) == 5
    assert all(np.isfinite(v) for v in band_input)


def test_missing_control_drops_h1_but_keeps_h3():
    """Without M1 there is no control to compare against, but the H3
    contrast is M3 vs M2 and needs no control at all."""
    daily, ledger = _fixture(arms=("M2", "M3"))
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    assert {r["hypothesis"] for r in rows} == {"H3"}
    assert all((r["arm"], r["baseline"]) == ("M3", "M2") for r in rows)


def test_single_test_year_cannot_be_regime_checked():
    daily, ledger = _fixture(years=("2021",))
    rows = contrasts.arm_contrasts("sac", daily, ledger, 0.02, draws=49)
    assert rows == []            # leave-one-window-out needs >= 2 years


# ---------------------------------------------------------------------------
# Downside contrasts (drawdown and CVaR) — descriptive, no inference
# ---------------------------------------------------------------------------

def test_downside_contrasts_cover_every_contrast_and_measure():
    daily, ledger = _fixture()          # 3 arms -> 3 contrasts, 2 etas
    rows = contrasts.downside_contrasts("sac", daily, ledger)
    assert len(rows) == 3 * 2 * len(contrasts.DOWNSIDE_MEASURES)
    got = {(r["arm"], r["baseline"], r["hypothesis"], r["eta"], r["measure"])
           for r in rows}
    for eta in (1.0, 3.0):
        for measure in contrasts.DOWNSIDE_MEASURES:
            assert ("M3", "M1", "H1", eta, measure) in got
            assert ("M3", "M2", "H3", eta, measure) in got
            assert ("M2", "M1", "SUPP", eta, measure) in got
    assert all(set(r) == set(contrasts.DOWNSIDE_COLUMNS) for r in rows)


def test_downside_contrasts_carry_no_verdict_or_p_value():
    """sec:inference-meth tests the Sharpe difference; these only describe."""
    daily, ledger = _fixture()
    rows = contrasts.downside_contrasts("sac", daily, ledger)
    forbidden = {"p_value", "p_one_sided", "p_holm", "supported",
                 "significant", "regime_robust"}
    assert not forbidden & set(contrasts.DOWNSIDE_COLUMNS)
    assert all(not forbidden & set(r) for r in rows)


def test_downside_difference_is_arm_minus_baseline():
    daily, ledger = _fixture()
    for row in contrasts.downside_contrasts("sac", daily, ledger):
        assert np.isclose(row["difference"],
                          row["value_arm"] - row["value_baseline"])
        assert np.isclose(row["difference_seed_mean"],
                          row["seed_mean_arm"] - row["seed_mean_baseline"])


def test_downside_band_is_the_root_of_summed_seed_variances():
    daily, ledger = _fixture()
    for row in contrasts.downside_contrasts("sac", daily, ledger):
        assert np.isclose(row["seed_band"],
                          np.hypot(row["seed_sd_arm"], row["seed_sd_baseline"]))
        assert row["exceeds_seed_band"] == (abs(row["difference"])
                                            > row["seed_band"])


def test_downside_exceeds_band_is_direction_agnostic():
    """A large NEGATIVE difference must register, unlike the Sharpe rule."""
    daily, ledger = _fixture()
    rows = contrasts.downside_contrasts("sac", daily, ledger)
    flipped = [dict(r, difference=-abs(r["difference"]) * 100)
               for r in rows if np.isfinite(r["seed_band"])]
    assert flipped, "fixture produced no finite bands"
    for row in flipped:
        assert abs(row["difference"]) > row["seed_band"]


def test_per_seed_downside_returns_one_value_per_seed():
    ledgers = {"2021": _ledger(80, 0.0004, 1, seeds=5),
               "2022": _ledger(80, 0.0004, 2, seeds=5)}
    per_seed = contrasts.per_seed_downside(ledgers)
    assert set(per_seed) == set(contrasts.DOWNSIDE_MEASURES)
    for measure, values in per_seed.items():
        assert len(values) == 5
    # Drawdowns are positive fractions; CVaR at 5% is a loss.
    assert all(v > 0 for v in per_seed["max_drawdown"])
    assert all(v < 0 for v in per_seed["cvar_5"])


def test_per_seed_downside_without_ledgers_is_empty_not_an_error():
    per_seed = contrasts.per_seed_downside({})
    assert per_seed == {m: [] for m in contrasts.DOWNSIDE_MEASURES}


def test_downside_contrasts_survive_a_missing_ledger():
    """Bands go NaN, values still computed — a partial run must not crash."""
    daily, ledger = _fixture()
    ledger.pop(("M1", 1.0))
    rows = contrasts.downside_contrasts("sac", daily, ledger)
    hit = [r for r in rows if r["baseline"] == "M1" and r["eta"] == 1.0]
    assert hit
    for row in hit:
        assert np.isnan(row["seed_band"])
        assert row["exceeds_seed_band"] is False
        assert np.isfinite(row["value_arm"])
