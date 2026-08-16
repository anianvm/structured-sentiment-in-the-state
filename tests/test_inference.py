"""Hand-computed tests for the inference and allocation measures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.allocation import (
    average_composition, composition_distance, news_responsiveness,
)
from src.evaluation.inference import holm_adjust, sharpe_difference_test


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------

def test_holm_hand_computed():
    # p = [0.01, 0.04, 0.03], m = 3. Ascending order: 0.01, 0.03, 0.04.
    #   rank 0: 3 * 0.01 = 0.03
    #   rank 1: 2 * 0.03 = 0.06
    #   rank 2: 1 * 0.04 = 0.04 -> raised to 0.06 by the running maximum
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(20)
    adj = holm_adjust(p)
    assert (adj <= 1.0).all() and (adj >= p - 1e-12).all()  # never shrinks a p
    # Sorting by raw p must leave the adjusted values non-decreasing.
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_holm_single_test_is_unchanged():
    assert np.allclose(holm_adjust([0.023]), [0.023])


# ---------------------------------------------------------------------------
# Composition distance
# ---------------------------------------------------------------------------

def _weights(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def test_composition_distance_ignores_the_cash_decision():
    # Same 50/50 split of the risky part, very different cash weights.
    cols = ["CASH", "A", "B"]
    a = average_composition(_weights([[0.5, 0.25, 0.25]], cols))
    b = average_composition(_weights([[0.0, 0.50, 0.50]], cols))
    assert np.isclose(composition_distance(a, b), 0.0)


def test_composition_distance_hand_computed():
    # Compositions [0.7, 0.3] and [0.4, 0.6]:
    #   0.5 * (|0.7-0.4| + |0.3-0.6|) = 0.5 * 0.6 = 0.3
    cols = ["CASH", "A", "B"]
    a = average_composition(_weights([[0.0, 0.7, 0.3]], cols))
    b = average_composition(_weights([[0.0, 0.4, 0.6]], cols))
    assert np.isclose(composition_distance(a, b), 0.3)


def test_composition_distance_disjoint_portfolios_is_one():
    cols = ["CASH", "A", "B"]
    a = average_composition(_weights([[0.2, 0.8, 0.0]], cols))
    b = average_composition(_weights([[0.9, 0.0, 0.1]], cols))
    assert np.isclose(composition_distance(a, b), 1.0)


def test_average_composition_skips_fully_cash_days():
    cols = ["CASH", "A", "B"]
    w = _weights([[1.0, 0.0, 0.0], [0.0, 0.75, 0.25]], cols)
    # The all-cash day carries no composition, so the mean is the second row.
    assert np.allclose(average_composition(w).to_numpy(), [0.75, 0.25])


# ---------------------------------------------------------------------------
# News responsiveness
# ---------------------------------------------------------------------------

def test_news_responsiveness_recovers_a_known_coefficient():
    n = 400
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-04", periods=n)
    news = pd.Series(rng.normal(size=n), index=idx)
    vix = pd.Series(rng.normal(size=n), index=idx)
    # Construct the cash share exactly: 0.30 + 0.25*news - 0.10*vix.
    cash = 0.30 + 0.25 * news - 0.10 * vix

    out = news_responsiveness(cash, news, vix)
    assert np.isclose(out["beta"], 0.25, atol=1e-8)
    assert np.isclose(out["gamma"], -0.10, atol=1e-8)
    assert np.isclose(out["alpha"], 0.30, atol=1e-8)
    assert out["r2"] > 0.99 and out["n_obs"] == n


def test_news_responsiveness_is_zero_when_cash_ignores_news():
    n = 300
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-04", periods=n)
    news = pd.Series(rng.normal(size=n), index=idx)
    vix = pd.Series(rng.normal(size=n), index=idx)
    cash = pd.Series(0.2 + 0.05 * rng.normal(size=n), index=idx)  # news-blind

    out = news_responsiveness(cash, news, vix)
    assert abs(out["t"]) < 3.0          # indistinguishable from zero
    assert out["n_obs"] == n


def test_news_responsiveness_handles_too_few_observations():
    idx = pd.bdate_range("2021-01-04", periods=4)
    s = pd.Series(np.zeros(4), index=idx)
    assert np.isnan(news_responsiveness(s, s, s)["beta"])


# ---------------------------------------------------------------------------
# Ledoit-Wolf Sharpe difference test
# ---------------------------------------------------------------------------

def _log_returns(values, start="2021-01-04"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_identical_series_give_zero_difference_and_large_p():
    rng = np.random.default_rng(3)
    r = _log_returns(rng.normal(0.0004, 0.01, 500))
    out = sharpe_difference_test(r, r.copy(), 0.0, draws=299, seed=0)
    # A series against itself moves together exactly, so the difference is not
    # an estimate: it is zero with no standard error, and p is exactly 1.
    assert np.isclose(out["difference"], 0.0, atol=1e-12)
    assert np.isclose(out["sharpe_a"], out["sharpe_b"])
    assert out["p_value"] == 1.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_clearly_better_series_is_detected():
    n = 1000
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 0.01, n)
    good = _log_returns(0.0010 + noise)   # same noise, much higher mean
    poor = _log_returns(0.0000 + noise)
    out = sharpe_difference_test(good, poor, 0.0, draws=299, seed=0)
    assert out["difference"] > 0
    assert out["p_value"] < 0.05
    assert out["ci_low"] > 0             # interval excludes zero


def test_difference_is_antisymmetric_and_ci_brackets_it():
    rng = np.random.default_rng(5)
    a = _log_returns(rng.normal(0.0006, 0.011, 400))
    b = _log_returns(rng.normal(0.0002, 0.009, 400))
    ab = sharpe_difference_test(a, b, 0.0, draws=199, seed=1)
    ba = sharpe_difference_test(b, a, 0.0, draws=199, seed=1)
    assert np.isclose(ab["difference"], -ba["difference"])
    assert ab["ci_low"] < ab["difference"] < ab["ci_high"]


def test_p_value_is_never_exactly_zero():
    # The (1 + count) / (1 + draws) convention keeps p strictly positive.
    n = 800
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 0.005, n)
    out = sharpe_difference_test(_log_returns(0.01 + noise),
                                 _log_returns(noise), 0.0, draws=99, seed=0)
    assert 0.0 < out["p_value"] <= 1.0


def test_excess_returns_use_the_tbill():
    # A constant-return series has zero volatility in excess of a matching
    # constant rate, so the statistic is undefined rather than infinite.
    flat = _log_returns(np.full(200, np.log1p(0.02 / 252)))
    out = sharpe_difference_test(flat, flat.copy(), 0.02, draws=99, seed=0)
    assert np.isnan(out["difference"])


# ---------------------------------------------------------------------------
# Seed-dispersion band
# ---------------------------------------------------------------------------

def test_seed_band_is_root_of_summed_variances():
    from src.evaluation.inference import seed_band
    # sd([1,2,3]) = 1 for both arms, so the band is sqrt(1 + 1).
    assert np.isclose(seed_band([1, 2, 3], [1, 2, 3]), np.sqrt(2.0))


def test_seed_band_grows_with_noisier_arms():
    from src.evaluation.inference import seed_band
    tight = seed_band([1.0, 1.0, 1.1], [1.0, 1.0, 1.1])
    loose = seed_band([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert loose > tight


def test_seed_band_undefined_for_a_single_seed():
    from src.evaluation.inference import seed_band
    assert np.isnan(seed_band([1.0], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# Leave-one-window-out
# ---------------------------------------------------------------------------

def _year_series(year, values):
    idx = pd.bdate_range(f"{year}-01-04", periods=len(values))
    return pd.Series(values, index=idx)


def test_leave_one_out_flags_a_result_driven_by_one_year():
    """A contrast created entirely by 2022 must not survive dropping 2022."""
    from src.evaluation.inference import leave_one_window_out
    rng = np.random.default_rng(0)
    a, b = {}, {}
    for year in (2021, 2022, 2023):
        noise = rng.normal(0.0, 0.01, 250)
        edge = 0.0025 if year == 2022 else 0.0   # only 2022 differs
        a[year] = _year_series(year, edge + noise)
        b[year] = _year_series(year, noise)
    out = leave_one_window_out(a, b, 0.0, draws=199, seed=0)
    assert out["full"]["difference"] > 0          # pooled looks positive
    assert not out["survives"]                    # but it hinges on 2022
    dropped = out["by_omitted_year"][2022]
    assert abs(dropped["difference"]) < abs(out["full"]["difference"])


def test_leave_one_out_survives_a_consistent_effect():
    """An edge present in every year should survive every omission."""
    from src.evaluation.inference import leave_one_window_out
    rng = np.random.default_rng(1)
    a, b = {}, {}
    for year in (2021, 2022, 2023):
        noise = rng.normal(0.0, 0.008, 250)
        a[year] = _year_series(year, 0.0012 + noise)
        b[year] = _year_series(year, noise)
    out = leave_one_window_out(a, b, 0.0, draws=199, seed=0)
    assert out["survives"]
    assert len(out["by_omitted_year"]) == 3


def test_leave_one_out_needs_two_years():
    from src.evaluation.inference import leave_one_window_out
    one = {2021: _year_series(2021, np.zeros(50))}
    out = leave_one_window_out(one, one, 0.0, draws=49)
    assert not out["survives"] and "two test years" in out["reason"]


# ---------------------------------------------------------------------------
# The combined decision rule
# ---------------------------------------------------------------------------

def test_verdict_amended_rule_gates_on_significance_and_band_only():
    """The amended rule (2026-08-08): support = one-sided Holm significance
    AND seed band. Sign is subsumed by the one-sided test upstream; the
    leave-one-year influence check is reported, not gated."""
    from src.evaluation.inference import contrast_verdict
    ok = contrast_verdict(difference=0.30, adjusted_p=0.01, band=0.10,
                          survives_loo=True)
    assert ok["supported"] and ok["significant"] and ok["exceeds_seed_band"]
    # The two gating conditions each withhold support alone.
    assert not contrast_verdict(0.30, 0.20, 0.10, True)["supported"]    # p
    assert not contrast_verdict(0.30, 0.01, 0.50, True)["supported"]    # noise
    # Influence and sign are REPORTED but no longer gate support.
    v = contrast_verdict(0.30, 0.01, 0.10, False)
    assert v["supported"] and not v["regime_robust"]
    # A wrong-sign effect cannot reach a small one-sided p in practice; the
    # verdict function still reports the sign flag for the tables.
    assert contrast_verdict(-0.30, 0.01, 0.10, True)["correct_sign"] is False


def test_verdict_reports_which_condition_failed():
    from src.evaluation.inference import contrast_verdict
    v = contrast_verdict(0.05, 0.01, 0.40, True)
    assert not v["supported"] and not v["exceeds_seed_band"]
    assert v["significant"] and v["correct_sign"]   # the others still hold


def test_verdict_handles_a_negative_prediction():
    from src.evaluation.inference import contrast_verdict
    v = contrast_verdict(-0.30, 0.01, 0.10, True, predicted_positive=False)
    assert v["supported"]
