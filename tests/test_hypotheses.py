"""The allocation-side hypotheses (src/evaluation/hypotheses.py).

Pins the H2/H5 machinery: the Newey-West estimator recovers a known
coefficient, the composition distance is the one-way turnover between
rescaled risky books, monotonicity is judged across the three eta levels,
and H2 support is the conjunction of all four separation predictions.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import hypotheses


def test_newey_west_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = 1.0 + 2.5 * x + rng.normal(scale=0.1, size=n)
    fit = hypotheses.newey_west_ols(y, x[:, None])
    assert fit["beta"][1] == pytest.approx(2.5, abs=0.05)
    assert abs(fit["t"][1]) > 10          # unmistakably significant


def test_newey_west_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    e = np.zeros(n)
    for i in range(1, n):                  # AR(1) errors, rho = 0.8
        e[i] = 0.8 * e[i - 1] + rng.normal(scale=0.1)
    y = 0.5 * x + e
    hac = hypotheses.newey_west_ols(y, x[:, None])
    iid = hypotheses.newey_west_ols(y, x[:, None], lags=0)
    assert hac["se"][0] > iid["se"][0]     # intercept absorbs the AR noise


def _panel(rows, cash, spread):
    idx = pd.bdate_range("2021-01-04", periods=rows)
    a = (1 - cash) * (0.5 + spread)
    b = (1 - cash) * (0.5 - spread)
    return pd.DataFrame({"CASH": cash, "AAA": a, "BBB": b}, index=idx)


def test_composition_distance_is_oneway_turnover_of_rescaled_books():
    # Same risky composition at different cash levels -> distance 0.
    assert hypotheses.composition_distance(
        _panel(50, 0.1, 0.1), _panel(50, 0.6, 0.1)) == pytest.approx(0.0)
    # 60/40 vs 40/60 risky split -> one-way turnover 0.2 regardless of cash.
    assert hypotheses.composition_distance(
        _panel(50, 0.2, 0.1), _panel(50, 0.5, -0.1)) == pytest.approx(0.2)


def test_composition_distance_compares_averages_not_daily_pairs():
    """eq:composition-distance averages the compositions FIRST. Two books
    with the same average composition but opposite daily jitter are the same
    allocation for this hypothesis; averaging daily distances would score
    them as maximally different."""
    idx = pd.bdate_range("2021-01-04", periods=50)
    flip = np.where(np.arange(50) % 2 == 0, 0.8, 0.2)
    a = pd.DataFrame({"CASH": 0.0, "AAA": flip, "BBB": 1 - flip}, index=idx)
    b = pd.DataFrame({"CASH": 0.0, "AAA": 1 - flip, "BBB": flip}, index=idx)
    assert hypotheses.composition_distance(a, b) == pytest.approx(0.0, abs=1e-12)


def _arm_data(cash, vol_scale, seed, spread=0.1, rows=120, seed_cash=None):
    """One (arm, eta) cell. `seed_cash` gives the five seeds' own cash shares;
    by default every seed sits on the headline value, so the dispersion
    columns are exactly zero unless a test asks for spread."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=rows)
    daily = pd.DataFrame({
        "log_return": rng.normal(0.0003, 0.01 * vol_scale, rows),
        "cash_weight": cash}, index=idx)
    per_seed = [cash] * 5 if seed_cash is None else list(seed_cash)
    ledgers = {"2021": pd.concat([
        pd.DataFrame({"seed": s, "date": idx,
                      "w_CASH": per_seed[s],
                      "wd_CASH": cash,
                      "wd_AAA": (1 - cash) * (0.5 + spread),
                      "wd_BBB": (1 - cash) * (0.5 - spread)})
        for s in range(5)])}
    return {"daily": daily, "weights": _panel(rows, cash, spread),
            "ledgers": ledgers}


def test_h2_supported_by_cash_monotonicity_alone():
    """The hypothesis claims the cash share; separation's implications are
    reported, not gated."""
    by_arm = {("M1", 1.0): _arm_data(0.02, 1.6, 0),
              ("M1", 3.0): _arm_data(0.10, 1.2, 0),
              ("M1", 7.0): _arm_data(0.30, 0.8, 0)}
    rows = hypotheses.h2_rows("sac", by_arm)
    assert len(rows) == 1
    r = rows[0]
    assert r["cash_monotone_up"] and r["supported"]
    # the implications are still computed and reported
    assert r["vol_monotone_down"]
    assert r["comp_dist_eta1_eta7"] == pytest.approx(0.0)


def test_h2_support_ignores_a_failing_implication():
    # cash rises monotonically but volatility does NOT fall: still supported,
    # with the implication flag False for the discussion to report.
    by_arm = {("M1", 1.0): _arm_data(0.02, 0.8, 0),
              ("M1", 3.0): _arm_data(0.10, 1.2, 0),
              ("M1", 7.0): _arm_data(0.30, 1.6, 0)}
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert r["supported"] and not r["vol_monotone_down"]


def test_h2_not_supported_when_cash_falls_with_eta():
    by_arm = {("M1", 1.0): _arm_data(0.30, 1.6, 0),
              ("M1", 3.0): _arm_data(0.10, 1.2, 0),
              ("M1", 7.0): _arm_data(0.02, 0.8, 0)}
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert not r["cash_monotone_up"]
    assert not r["supported"]


def test_per_seed_cash_share_pools_years_like_the_headline():
    """One mean per seed over the concatenated years, read from w_CASH."""
    idx = pd.bdate_range("2021-01-04", periods=10)
    ledgers = {
        "2021": pd.DataFrame({"seed": [0] * 10 + [1] * 10,
                              "date": list(idx) * 2,
                              "w_CASH": [0.10] * 10 + [0.30] * 10}),
        "2022": pd.DataFrame({"seed": [0] * 10 + [1] * 10,
                              "date": list(idx) * 2,
                              "w_CASH": [0.20] * 10 + [0.50] * 10}),
    }
    shares = hypotheses.per_seed_cash_share(ledgers)
    assert shares == pytest.approx({0: 0.15, 1: 0.40})


def test_h2_reports_seed_dispersion_without_changing_the_verdict():
    """A rise inside across-seed noise stays supported: the pre-registered
    rule is monotonicity, and the dispersion columns only qualify it."""
    by_arm = {
        ("M1", 1.0): _arm_data(0.049, 1.0, 0,
                               seed_cash=[0.01, 0.02, 0.05, 0.08, 0.09]),
        ("M1", 3.0): _arm_data(0.053, 1.0, 0,
                               seed_cash=[0.02, 0.01, 0.06, 0.07, 0.10]),
        ("M1", 7.0): _arm_data(0.062, 1.0, 0,
                               seed_cash=[0.01, 0.03, 0.04, 0.09, 0.12]),
    }
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert r["supported"] and r["cash_monotone_up"]
    # the 1.3pp rise is dwarfed by a ~3.5pp across-seed spread
    assert r["cash_rise_eta1_eta7"] == pytest.approx(0.013)
    assert r["cash_sd_eta1"] > r["cash_rise_eta1_eta7"]
    assert r["cash_rise_band"] > r["cash_rise_eta1_eta7"]
    assert not r["cash_rise_exceeds_band"]


def test_h2_counts_the_seeds_that_reproduce_the_ordering():
    # seeds 0 and 3 rise monotonically; 1 falls, 2 dips, 4 is flat-then-down
    by_arm = {
        ("M1", 1.0): _arm_data(0.02, 1.0, 0,
                               seed_cash=[0.01, 0.09, 0.05, 0.02, 0.05]),
        ("M1", 3.0): _arm_data(0.05, 1.0, 0,
                               seed_cash=[0.04, 0.05, 0.01, 0.06, 0.05]),
        ("M1", 7.0): _arm_data(0.08, 1.0, 0,
                               seed_cash=[0.07, 0.02, 0.09, 0.11, 0.03]),
    }
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert r["n_seeds"] == 5
    assert r["n_seeds_cash_monotone"] == 2


def test_h2_seed_average_can_oppose_the_selected_seed():
    """The headline rise is one seed's; the paired per-seed mean is what the
    average seed did, and the two can disagree in sign."""
    by_arm = {
        ("M1", 1.0): _arm_data(0.04, 1.0, 0,
                               seed_cash=[0.06, 0.07, 0.07, 0.05, 0.00]),
        ("M1", 3.0): _arm_data(0.05, 1.0, 0,
                               seed_cash=[0.06, 0.05, 0.05, 0.04, 0.00]),
        ("M1", 7.0): _arm_data(0.06, 1.0, 0,
                               seed_cash=[0.03, 0.02, 0.06, 0.07, 0.01]),
    }
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert r["supported"]                          # verdict untouched
    assert r["cash_rise_eta1_eta7"] > 0            # selected seed rises
    assert r["cash_rise_seed_mean"] < 0            # the average seed falls
    assert r["n_seeds_cash_rising"] == 2
    # paired dispersion needs no independence assumption, unlike the band
    assert r["cash_rise_seed_sd"] < r["cash_rise_band"]


def test_h2_dispersion_columns_are_nan_without_per_seed_cash():
    """Ledgers predating the w_CASH column degrade to NaN, not to a crash,
    and the verdict is unaffected."""
    def strip(cell):
        for frame in cell["ledgers"].values():
            frame.drop(columns=["w_CASH"], inplace=True)
        return cell

    by_arm = {("M1", 1.0): strip(_arm_data(0.02, 1.0, 0)),
              ("M1", 3.0): strip(_arm_data(0.05, 1.0, 0)),
              ("M1", 7.0): strip(_arm_data(0.08, 1.0, 0))}
    r = hypotheses.h2_rows("sac", by_arm)[0]
    assert r["supported"]
    assert np.isnan(r["cash_sd_eta1"]) and np.isnan(r["cash_rise_band"])
    assert r["n_seeds"] == 0 and r["n_seeds_cash_monotone"] == 0


def test_h5_regression_finds_a_planted_response():
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.bdate_range("2021-01-04", periods=n)
    sent = pd.Series(rng.normal(size=n), index=idx)
    vix = pd.Series(rng.normal(size=n), index=idx)
    daily = pd.DataFrame({
        "log_return": rng.normal(0, 0.01, n),
        "cash_weight": 0.1 - 0.05 * sent + 0.02 * vix
        + rng.normal(0, 0.005, n)}, index=idx)
    rows = hypotheses.h5_rows(
        "sac", {("M2", 1.0): {"daily": daily}}, sent, vix)
    assert len(rows) == 1
    assert rows[0]["beta_sentiment"] == pytest.approx(-0.05, abs=0.01)
    assert rows[0]["beta_vix"] == pytest.approx(0.02, abs=0.01)
    assert abs(rows[0]["t_sentiment"]) > 5
    assert rows[0]["news_index"] == "llm"


def test_h5_estimates_m2_against_its_own_news_index():
    """M2 observes FinBERT, not the LLM signal. Given an override it must be
    regressed on that index, and the row must say which one it used."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2021-01-04", periods=n)
    llm = pd.Series(rng.normal(0, 1, n), index=idx)
    finbert = pd.Series(rng.normal(0, 1, n), index=idx, name="finbert")
    vix = pd.Series(rng.normal(0, 1, n), index=idx)
    # the cash share responds to FinBERT only; the LLM index is noise to it
    daily = pd.DataFrame({
        "log_return": rng.normal(0, 0.01, n),
        "cash_weight": 0.1 - 0.05 * finbert + rng.normal(0, 0.005, n),
    }, index=idx)
    by_arm = {("M2", 1.0): {"daily": daily}, ("M3", 1.0): {"daily": daily}}

    rows = {r["arm"]: r for r in hypotheses.h5_rows(
        "sac", by_arm, llm, vix, sentiment_by_arm={"M2": finbert})}

    assert rows["M2"]["news_index"] == "finbert"
    assert rows["M2"]["beta_sentiment"] == pytest.approx(-0.05, abs=0.01)
    # M3 keeps the common LLM regressor, which this series does not respond to
    assert rows["M3"]["news_index"] == "llm"
    assert abs(rows["M3"]["beta_sentiment"]) < 0.01
