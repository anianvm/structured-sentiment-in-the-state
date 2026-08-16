"""Hand-computed synthetic tests for src/evaluation/ (metrics, benchmarks, report).

Every expected value is derived by hand in the test body, so a failure points
directly at a formula. No network, no real data; the only file I/O uses
pytest's tmp_path.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics, report
from src.evaluation.benchmarks import ucrp_log_returns


def _series(values, start="2021-01-04"):
    """Daily log-return series on consecutive business days."""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


# ---------------------------------------------------------------------------
# Headline performance measures
# ---------------------------------------------------------------------------

def test_cumulative_return_two_days():
    # Wealth: 1.0 -> 1.1 -> 1.1 * 0.9 = 0.99, so cumulative return = -1%.
    s = _series([np.log(1.1), np.log(0.9)])
    assert np.isclose(metrics.cumulative_return(s), 0.99 - 1.0)


def test_annualized_return_three_days():
    # Three days of +1% compound to 1.01^3; annualized over 252/3 periods
    # per year: (1.01^3)^(252/3) - 1 = 1.01^252 - 1.
    s = _series([np.log(1.01)] * 3)
    assert np.isclose(metrics.annualized_return(s), 1.01 ** 252 - 1.0)


def test_annualized_volatility_hand_computed():
    # Simple returns [0.02, -0.01]: mean 0.005, sample std (ddof=1)
    # = sqrt((0.015^2 + 0.015^2) / 1) = 0.015 * sqrt(2).
    s = _series([np.log(1.02), np.log(0.99)])
    expected = 0.015 * np.sqrt(2) * np.sqrt(252)
    assert np.isclose(metrics.annualized_volatility(s), expected)


def test_sharpe_ratio_hand_computed_zero_tbill():
    # Same series as above with a zero T-bill: excess = simple returns,
    # Sharpe = 0.005 / (0.015 * sqrt(2)) * sqrt(252).
    s = _series([np.log(1.02), np.log(0.99)])
    tbill = pd.Series(0.0, index=s.index)
    expected = 0.005 / (0.015 * np.sqrt(2)) * np.sqrt(252)
    assert np.isclose(metrics.sharpe_ratio(s, tbill), expected)


def test_sharpe_ratio_constant_series_is_nan():
    # A constant return series has zero standard deviation; the ratio is
    # undefined and the implementation returns NaN rather than dividing by 0.
    s = _series([np.log(1.001)] * 5)
    assert np.isnan(metrics.sharpe_ratio(s, 0.0))


def test_max_drawdown_is_20_percent():
    # Wealth path 1.0 -> 1.25 -> 1.00 -> 1.25: the drop from the 1.25 peak
    # to 1.00 is a 20% drawdown, and the recovery must not reduce it.
    s = _series([np.log(1.25), np.log(0.80), np.log(1.25)])
    assert np.isclose(metrics.max_drawdown(s), 0.20)


def test_performance_summary_keys():
    s = _series([np.log(1.02), np.log(0.99), np.log(1.01)])
    out = metrics.performance_summary(s, 0.0)
    assert set(out) == {"cumulative_return", "annualized_return",
                        "annualized_volatility", "sharpe_ratio", "max_drawdown"}


# ---------------------------------------------------------------------------
# UCRP benchmark
# ---------------------------------------------------------------------------

def test_ucrp_two_assets_three_days_hand_computed():
    # Closes: day0 [100, 100], day1 [110, 90], day2 [121, 99]. Cost 1%.
    #
    # Day 0 (bought in from nothing): tau = |0.5-0| + |0.5-0| = 1,
    #   rel = [1.1, 0.9], gross = 0.5*1.1 + 0.5*0.9 = 1.0
    #   ->  log return = log((1 - 0.01*1) * 1.0) = log(0.99).
    #   Drifted weights afterwards: [0.5*1.1, 0.5*0.9] / 1.0 = [0.55, 0.45].
    # Day 1: rebalance tau = |0.5-0.55| + |0.5-0.45| = 0.1, rel = [1.1, 1.1],
    #   gross = 1.1  ->  log((1 - 0.01*0.1) * 1.1) = log(1.0989).
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    closes = np.array([[100.0, 100.0], [110.0, 90.0], [121.0, 99.0]])
    ohlc = np.stack([closes, closes, closes], axis=2)  # high/low unused

    out = ucrp_log_returns(pd.DatetimeIndex(dates), ohlc, cost=0.01)

    assert list(out.index) == list(dates[:2])  # indexed at the decision day
    assert np.isclose(out.iloc[0], np.log(0.99))
    assert np.isclose(out.iloc[1], np.log(1.0989))


def test_ucrp_zero_cost_matches_plain_rebalanced_mean():
    # With cost = 0 the UCRP log return is just log(mean of asset relatives).
    dates = pd.bdate_range("2021-01-04", periods=3)
    closes = np.array([[100.0, 200.0], [105.0, 190.0], [110.25, 200.0]])
    ohlc = np.stack([closes, closes, closes], axis=2)
    out = ucrp_log_returns(dates, ohlc, cost=0.0)
    assert np.isclose(out.iloc[0], np.log((1.05 + 0.95) / 2))
    assert np.isclose(out.iloc[1], np.log((1.05 + 200 / 190) / 2))


# ---------------------------------------------------------------------------
# Behaviour diagnostics
# ---------------------------------------------------------------------------

def test_entropy_of_uniform_weights_is_ln_n():
    w = pd.DataFrame(np.full((5, 4), 0.25), columns=["CASH", "A", "B", "C"])
    assert np.isclose(metrics.portfolio_entropy(w), np.log(4))


def test_entropy_handles_zero_weights():
    # A fully concentrated portfolio has entropy 0 (0 * ln 0 := 0).
    w = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["CASH", "A", "B"])
    assert metrics.portfolio_entropy(w) == 0.0


def test_behaviour_summary_hand_computed():
    w = pd.DataFrame(
        [[0.5, 0.3, 0.2],
         [0.1, 0.6, 0.3]],
        columns=["CASH", "A", "B"],
        index=pd.bdate_range("2021-01-04", periods=2),
    )
    tau = pd.Series([0.0, 0.4], index=w.index)  # two-way turnover from the env
    out = metrics.behaviour_summary(w, tau)
    assert np.isclose(out["avg_oneway_turnover"], (0.0 + 0.2) / 2)  # mean tau/2
    assert np.isclose(out["avg_largest_position"], (0.5 + 0.6) / 2)
    assert np.isclose(out["avg_cash_share"], (0.5 + 0.1) / 2)


# ---------------------------------------------------------------------------
# Conditional value at risk
# ---------------------------------------------------------------------------

def test_cvar_averages_the_worst_alpha_tail():
    # 100 days, alpha = 0.05 -> the 5 smallest simple returns.
    simple = np.concatenate([np.full(5, -0.10), np.full(95, 0.01)])
    log_returns = pd.Series(np.log1p(simple))
    assert np.isclose(metrics.cvar(log_returns, 0.05), -0.10)


def test_cvar_stays_defined_on_a_series_shorter_than_the_tail():
    # 0.05 * 10 < 1: the tail is still non-empty, holding the worst day.
    log_returns = pd.Series(np.log1p([-0.07] + [0.01] * 9))
    assert np.isclose(metrics.cvar(log_returns, 0.05), -0.07)


def test_cvar_is_negative_for_a_loss_making_tail_and_undefined_when_empty():
    rng = np.random.default_rng(0)
    log_returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    value = metrics.cvar(log_returns)
    assert value < 0
    # The tail mean can never exceed the overall mean.
    assert value < np.expm1(log_returns).mean()
    assert np.isnan(metrics.cvar(pd.Series(dtype=float)))


def test_behaviour_tail_summary_reports_both_downside_measures():
    log_returns = pd.Series(np.log1p([0.01, -0.05, 0.02, -0.03]))
    summary = metrics.behaviour_tail_summary(log_returns)
    assert set(summary) == {"max_drawdown", "cvar_5"}
    assert summary["max_drawdown"] > 0      # positive fraction
    assert summary["cvar_5"] < 0            # negative fraction


# ---------------------------------------------------------------------------
# Report helpers (file discovery and summary parsing, tmp_path only)
# ---------------------------------------------------------------------------

def test_find_run_files_identifies_outputs_by_header(tmp_path):
    (tmp_path / "test_returns.csv").write_text(
        "date,log_return,turnover,cash_weight\n2021-01-04,0.001,0.1,0.2\n")
    (tmp_path / "weights.csv").write_text(
        "date,CASH,AAPL\n2021-01-04,0.5,0.5\n")
    (tmp_path / "summary.json").write_text("{}")

    files = report._find_run_files(tmp_path)
    assert files["daily"].name == "test_returns.csv"
    assert files["weights"].name == "weights.csv"
    assert files["summary"].name == "summary.json"


def test_find_run_files_returns_none_without_daily_csv(tmp_path):
    (tmp_path / "weights.csv").write_text("date,CASH,AAPL\n2021-01-04,1,0\n")
    assert report._find_run_files(tmp_path) is None


def test_behaviour_row_without_weights_keeps_the_downside_measures():
    """Benchmark rows carry drawdown and CVaR; allocation columns are NaN."""
    log_returns = pd.Series(np.log1p([0.01, -0.04, 0.02]),
                            index=pd.date_range("2021-01-04", periods=3))
    row = report._behaviour_row("UCRP", "2021", log_returns)
    assert np.isfinite(row["max_drawdown"]) and np.isfinite(row["cvar_5"])
    assert all(np.isnan(row[c]) for c in ("avg_oneway_turnover",
                                          "portfolio_entropy",
                                          "avg_largest_position",
                                          "avg_cash_share"))
    assert set(row) == set(report.BEHAVIOUR_COLUMNS)


def test_behaviour_summary_reports_full_and_risky_concentration():
    """The chapter's table pairs each concentration measure with a risky-only
    twin; both must come out of the pipeline, not a side calculation."""
    idx = pd.date_range("2021-01-04", periods=2)
    weights = pd.DataFrame({"CASH": [0.5, 0.0], "AAPL": [0.3, 0.6],
                            "MSFT": [0.2, 0.4]}, index=idx)
    turnover = pd.Series([0.02, 0.04], index=idx)
    summary = metrics.behaviour_summary(weights, turnover)
    assert set(summary) == {"avg_oneway_turnover", "portfolio_entropy",
                            "risky_entropy", "avg_largest_position",
                            "risky_largest_position", "avg_cash_share"}
    # Day 1 is half cash; rescaling makes it 60/40, the same split as day 2.
    # Largest position: full = mean(0.5 cash, 0.6) ; risky = mean(0.6, 0.6).
    assert np.isclose(summary["avg_largest_position"], (0.5 + 0.6) / 2)
    assert np.isclose(summary["risky_largest_position"], 0.6)
    # Entropy: day 1 loses the cash term, day 2 is unchanged.
    full_day1 = -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    risky = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
    assert np.isclose(summary["portfolio_entropy"], (full_day1 + risky) / 2)
    assert np.isclose(summary["risky_entropy"], risky)
    # Dropping cash concentrates the book on both measures.
    assert summary["risky_entropy"] < summary["portfolio_entropy"]
    assert summary["risky_largest_position"] > summary["avg_largest_position"]
    assert np.isclose(summary["avg_cash_share"], 0.25)


def test_risky_rescaled_drops_all_cash_days_and_normalizes():
    idx = pd.date_range("2021-01-04", periods=3)
    weights = pd.DataFrame({"CASH": [1.0, 0.5, 0.0], "AAPL": [0.0, 0.3, 0.7],
                            "MSFT": [0.0, 0.2, 0.3]}, index=idx)
    risky = metrics.risky_rescaled(weights)
    assert len(risky) == 2                       # the all-cash day is dropped
    assert "CASH" not in risky.columns
    assert np.allclose(risky.sum(axis=1), 1.0)
    assert np.isclose(risky.loc[idx[1], "AAPL"], 0.6)
