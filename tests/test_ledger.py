"""Tests for the walk-forward persistence layer (ledger + env info dict)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.environment.portfolio_env import PortfolioEnv
from src.experiments.walkforward import _ledger_frame


def _tiny_env():
    """2 risky assets + cash, constant prices except asset 1 (+10%/day)."""
    T, N = 6, 2
    closes = np.ones((T, N))
    closes[:, 0] = 1.1 ** np.arange(T)
    ohlc = np.stack([closes, closes, closes], axis=2)
    tbill = np.full(T, 0.0252)          # 2.52% annual -> 0.0001/day
    context = np.zeros((T, 2))
    return PortfolioEnv(ohlc, tbill, context, sentiment=None, k=2,
                        cost=0.01, eta=1.0, start=1, end=4)


def test_info_dict_carries_the_full_ledger():
    env = _tiny_env()
    env.reset(seed=0)
    action = np.array([0.0, 3.0, 0.0])          # tilt toward asset 1
    _obs, reward, _term, _trunc, info = env.step(action)

    # Pre-trade holdings: episode starts 100% in cash.
    assert np.allclose(info["drifted_weights"], [1.0, 0.0, 0.0])
    # The raw action the softmax received is recorded verbatim.
    assert np.allclose(info["action"], action)
    # cost_paid is exactly c * tau.
    assert np.isclose(info["cost_paid"], 0.01 * info["turnover"])
    # Cash accrues the day-t rate as a simple daily accrual.
    assert np.isclose(info["cash_gross"], 1 + 0.0252 / 252)
    # portfolio_gross = w . y and rho = (1 - c*tau) * (w.y) - 1 tie out.
    w, y_cash = info["weights"], info["cash_gross"]
    y = np.array([y_cash, 1.1, 1.0])             # asset 1 gains 10%
    assert np.isclose(info["portfolio_gross"], float(w @ y))
    assert np.isclose(
        info["rho"], (1 - 0.01 * info["turnover"]) * float(w @ y) - 1)
    # The reward returned by step is the CRRA utility of 1 + rho (eta=1: log).
    assert np.isclose(reward, np.log1p(info["rho"]))


def test_ledger_frame_layout_and_values():
    n_days, n_assets = 3, len(config.DOW30) + 1
    rng = np.random.default_rng(0)
    roll = {
        "weights": rng.random((n_days, n_assets)),
        "drifted_weights": rng.random((n_days, n_assets)),
        "actions": rng.random((n_days, n_assets)),
        "turnover": rng.random(n_days),
        "cost_paid": rng.random(n_days),
        "rho": rng.random(n_days),
        "log_returns": rng.random(n_days),
        "rewards": rng.random(n_days),
        "cash_gross": rng.random(n_days),
        "portfolio_gross": rng.random(n_days),
        "t": np.arange(n_days),
    }
    dates = pd.bdate_range("2021-01-04", periods=n_days).strftime("%Y-%m-%d")

    frame = _ledger_frame("ppo", "M4", 5.0, 2021, "test", seed=3,
                          selected=True, dates_str=dates, roll=roll)

    assert len(frame) == n_days
    # 9 key columns + 3 x 31 asset groups + 7 outcome columns.
    assert frame.shape[1] == 9 + 3 * n_assets + 7
    assert list(frame["date"]) == list(dates)
    assert (frame["arm"] == "M4").all() and (frame["seed"] == 3).all()
    assert bool(frame["selected"].all())
    # Weight columns land in [CASH] + DOW30 order with intact values.
    assert np.allclose(frame["w_CASH"], roll["weights"][:, 0])
    assert np.allclose(frame[f"w_{config.DOW30[-1]}"], roll["weights"][:, -1])
    assert np.allclose(frame["wd_CASH"], roll["drifted_weights"][:, 0])
    assert np.allclose(frame[f"a_{config.DOW30[0]}"], roll["actions"][:, 1])
    assert np.allclose(frame["reward"], roll["rewards"])


def test_ledger_frame_roundtrips_through_parquet(tmp_path):
    n_days, n_assets = 2, len(config.DOW30) + 1
    roll = {name: np.zeros((n_days, n_assets)) for name in
            ("weights", "drifted_weights", "actions")}
    roll.update({name: np.zeros(n_days) for name in
                 ("turnover", "cost_paid", "rho", "log_returns", "rewards",
                  "cash_gross", "portfolio_gross")})
    roll["t"] = np.arange(n_days)
    dates = pd.bdate_range("2021-01-04", periods=n_days).strftime("%Y-%m-%d")
    frame = _ledger_frame("sac", "M1", 1.0, 2021, "val", 0, False, dates, roll)

    path = tmp_path / "ledger.parquet"
    frame.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame, back)
