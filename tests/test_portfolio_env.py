"""Hand-computed correctness tests for the portfolio MDP environment.

Every expected number is derived in a comment next to the assertion, so a
reader can check the environment against the methodology equations
(drift eq:drift-meth, net return eq:netreturn-meth, reward eq:reward-meth)
with pencil and paper. All fixtures are synthetic: two risky assets plus
cash, deterministic price paths, no data files, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.portfolio_env import (  # noqa: E402
    PortfolioEnv,
    crra_utility,
    softmax,
)

# ---------------------------------------------------------------------------
# Fixtures: two risky assets (A, B) plus cash. Deterministic closes:
#   day:      0      1      2      3
#   asset A:  100 -> 110 -> 99  -> 110   (+10%, -10%, +11.11%)
#   asset B:  100 -> 100 -> 120 -> 114   (flat, +20%, -5%)
# ---------------------------------------------------------------------------
CLOSES = [
    [100.0, 100.0],
    [110.0, 100.0],
    [99.0, 120.0],
    [110.0, 114.0],
]


def make_ohlc(closes):
    """(T, N) closes -> (T, N, 3) tensor with close = high = low.

    Only the close channel (index 0) enters returns, so flat high/low keep
    the arithmetic obvious.
    """
    arr = np.asarray(closes, dtype=np.float64)
    return np.repeat(arr[:, :, None], 3, axis=2)


def make_env(closes=CLOSES, annual_rate=0.0, sentiment=None, k=1,
             cost=0.0, eta=1.0, start=None, end=None, context=None):
    ohlc = make_ohlc(closes)
    T = ohlc.shape[0]
    tbill = np.full(T, annual_rate, dtype=np.float64)
    if context is None:
        context = np.zeros((T, 2))
    return PortfolioEnv(ohlc, tbill, context, sentiment=sentiment, k=k,
                        cost=cost, eta=eta, start=start, end=end)


def weights_action(w):
    """Invert the softmax: softmax(log(w)) == w when w sums to one."""
    return np.log(np.asarray(w, dtype=np.float64))


# ---------------------------------------------------------------------------
# (a) drift equation
# ---------------------------------------------------------------------------

class TestDrift:
    def test_drift_after_known_return(self):
        # Hold w = [0, 1/2, 1/2] through day 0 -> 1: y = [1, 1.10, 1.00].
        # eq:drift-meth: w' = (w * y) / (w . y)
        #   w . y = 0.5*1.10 + 0.5*1.00 = 1.05
        #   w'_A  = 0.55/1.05 = 0.5238095...,  w'_B = 0.50/1.05 = 0.4761904...
        env = make_env(cost=0.0)
        env.reset()
        obs, *_ = env.step(weights_action([1e-12, 0.5, 0.5]))
        drifted = obs["weights"]
        assert drifted[1] == pytest.approx(0.55 / 1.05, abs=1e-7)
        assert drifted[2] == pytest.approx(0.50 / 1.05, abs=1e-7)
        assert drifted.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# (b) turnover and linear cost, (c) eta=1 reward, (d) eta=2 reward
# ---------------------------------------------------------------------------

class TestRewardArithmetic:
    def test_turnover_risky_only_and_cost_linear(self):
        # From the initial 100%-cash state, rebalance to w = [0.5, 0.25, 0.25].
        # Turnover counts the RISKY legs only (cash is the settlement medium):
        #   tau = |0.25 - 0| + |0.25 - 0| = 0.5   (NOT 1.0)
        # y = [1, 1.10, 1.00], w . y = 0.5 + 0.275 + 0.25 = 1.025
        # rho = (1 - 0.0025*0.5) * 1.025 - 1 = 0.99875 * 1.025 - 1
        #     = 1.02371875 - 1 = 0.02371875
        c = 0.0025
        env = make_env(cost=c)
        env.reset()
        _, _, _, _, info = env.step(weights_action([0.5, 0.25, 0.25]))
        assert info["turnover"] == pytest.approx(0.5, abs=1e-12)
        assert info["rho"] == pytest.approx(0.02371875, abs=1e-9)

        # The cost enters rho linearly: rho(c=0) - rho(c) = c * tau * (w . y)
        #   = 0.0025 * 0.5 * 1.025 = 0.00128125
        env0 = make_env(cost=0.0)
        env0.reset()
        _, _, _, _, info0 = env0.step(weights_action([0.5, 0.25, 0.25]))
        assert info0["rho"] - info["rho"] == pytest.approx(0.00128125, abs=1e-9)

    def test_log_reward_eta_1(self):
        # Same step as above: reward = ln(1 + rho) = ln(1.02371875)
        env = make_env(cost=0.0025, eta=1.0)
        env.reset()
        _, reward, _, _, info = env.step(weights_action([0.5, 0.25, 0.25]))
        assert reward == pytest.approx(np.log(1.02371875), abs=1e-9)
        assert info["log_return"] == pytest.approx(reward, abs=1e-12)

    def test_crra_reward_eta_2(self):
        # Cost-free, w = [0, 1/2, 1/2], y = [1, 1.10, 1.00]:
        #   rho = 1.05 - 1 = 0.05
        # u_2(1.05) = (1.05^(1-2) - 1)/(1-2) = 1 - 1/1.05 = 0.0476190476...
        env = make_env(cost=0.0, eta=2.0)
        env.reset()
        _, reward, *_ = env.step(weights_action([1e-12, 0.5, 0.5]))
        assert reward == pytest.approx(1.0 - 1.0 / 1.05, abs=1e-9)


# ---------------------------------------------------------------------------
# (e) crra_utility function
# ---------------------------------------------------------------------------

class TestCrraUtility:
    @pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_zero_at_gross_one(self, eta):
        # u_eta(1) = (1 - 1)/(1 - eta) = 0 for eta != 1;  ln(1) = 0 for eta = 1
        assert crra_utility(1.0, eta) == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_monotone_increasing_in_gross(self, eta):
        # More wealth is always preferred, at every risk-aversion level.
        assert crra_utility(0.95, eta) < crra_utility(1.0, eta)
        assert crra_utility(1.0, eta) < crra_utility(1.10, eta)

    def test_eta_1_is_log(self):
        assert crra_utility(1.05, 1.0) == pytest.approx(np.log(1.05), abs=1e-15)


# ---------------------------------------------------------------------------
# (f) cash-only policy, (g) single-stock buy and hold
# ---------------------------------------------------------------------------

class TestPolicyMechanics:
    def test_cash_only_earns_tbill_with_zero_turnover(self):
        # Annual rate 5.04% -> daily accrual 0.0504/252 = 0.0002 exactly.
        # Staying (essentially) 100% cash from the initial 100%-cash state:
        # tau ~ 0 (risky legs stay ~0), so the cost charges nothing even
        # with c > 0, and rho = 1 + 0.0002 - 1 = 0.0002 each step.
        env = make_env(annual_rate=0.0504, cost=0.0025)
        env.reset()
        # softmax([10,-10,-10]) puts ~1 - 4e-9 on cash: near-one-hot within
        # the action box bounds.
        action = np.array([10.0, -10.0, -10.0])
        for _ in range(2):
            _, _, _, _, info = env.step(action)
            assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
            assert info["rho"] == pytest.approx(0.0002, abs=1e-8)

    def test_buy_and_hold_single_stock_zero_later_turnover(self):
        # First rebalance moves ~everything from cash into asset A:
        #   tau_1 ~ |1 - 0| + |0 - 0| = 1.
        # A is then the only holding, so however its price moves, the
        # drifted weight stays 1 on A and repeating the action trades ~0.
        env = make_env(cost=0.0025)
        env.reset()
        action = np.array([-10.0, 10.0, -10.0])  # near-pure asset A
        obs, _, _, _, info1 = env.step(action)
        assert info1["turnover"] == pytest.approx(1.0, abs=1e-6)
        assert obs["weights"][1] == pytest.approx(1.0, abs=1e-6)
        _, _, _, _, info2 = env.step(action)
        assert info2["turnover"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# (h) observation structure
# ---------------------------------------------------------------------------

class TestObservation:
    def test_shapes_normalization_weights_and_sentiment(self):
        # T=4, N=2, C=2 sentiment, k=2 -> default start = 1.
        sentiment = np.arange(4 * 2 * 2, dtype=np.float64).reshape(4, 2, 2)
        context = np.column_stack([np.arange(4.0), 10 + np.arange(4.0)])
        env = make_env(sentiment=sentiment, k=2, context=context)
        obs, _ = env.reset()

        assert set(obs) == {"prices", "context", "weights", "sentiment"}
        assert env.observation_space.contains(obs)

        # Prices window: (k, N, 3); Jiang normalization divides every field
        # by the day-t close, so the last close row is exactly [1, 1].
        assert obs["prices"].shape == (2, 2, 3)
        assert obs["prices"].dtype == np.float32
        np.testing.assert_allclose(obs["prices"][-1, :, 0], [1.0, 1.0])
        # First window row = day-0 closes over day-1 closes: [100/110, 100/100]
        np.testing.assert_allclose(
            obs["prices"][0, :, 0], [100 / 110, 1.0], rtol=1e-6)

        # Context and sentiment come straight from row t = start = 1.
        np.testing.assert_allclose(obs["context"], context[1])
        np.testing.assert_allclose(obs["sentiment"], sentiment[1])

        # Initial weights: 100% cash, cash first.
        np.testing.assert_allclose(obs["weights"], [1.0, 0.0, 0.0])

        # After a step, the weights key holds the DRIFTED allocation, not the
        # chosen one: w = [0, 1/2, 1/2], y = [1, 99/110, 120/100]
        #   -> w' = [0, 0.45, 0.60]/1.05 = [0, 3/7, 4/7]
        obs, *_ = env.step(weights_action([1e-12, 0.5, 0.5]))
        np.testing.assert_allclose(obs["weights"], [0.0, 3 / 7, 4 / 7],
                                   atol=1e-6)

    def test_sentiment_key_absent_for_price_only_arm(self):
        env = make_env(sentiment=None, k=2)
        obs, _ = env.reset()
        assert "sentiment" not in obs
        assert "sentiment" not in env.observation_space.spaces
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# (i) episode indexing
# ---------------------------------------------------------------------------

class TestEpisodeIndexing:
    def test_decisions_run_start_to_end_minus_1(self):
        # T=4, k=2 -> default start=1, end=3: decisions at t=1 and t=2,
        # terminated exactly when t reaches end=3.
        env = make_env(k=2)
        env.reset()
        _, _, term1, trunc1, info1 = env.step(np.zeros(3))
        assert info1["t"] == 1 and not term1 and not trunc1
        _, _, term2, trunc2, info2 = env.step(np.zeros(3))
        assert info2["t"] == 2 and term2 and not trunc2

    def test_explicit_start_end_single_step(self):
        env = make_env(k=1, start=2, end=3)
        env.reset()
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert info["t"] == 2 and terminated

    def test_constructor_asserts_window_and_bounds(self):
        with pytest.raises(AssertionError):
            make_env(k=2, start=0)         # start < k-1: window incomplete
        with pytest.raises(AssertionError):
            make_env(k=1, end=4)           # end > T-1 = 3: no returns row


# ---------------------------------------------------------------------------
# (j) log rewards sum to log terminal wealth
# ---------------------------------------------------------------------------

class TestLogWealthIdentity:
    def test_summed_log_rewards_equal_log_terminal_wealth(self):
        # eta=1, gamma=1: sum_t ln(1 + rho_t) = ln prod_t (1 + rho_t)
        # = ln(terminal wealth). 3-step episode (t = 0, 1, 2) with costs and
        # a nonzero cash rate, equal-score (uniform-weight) actions.
        env = make_env(annual_rate=0.0504, cost=0.0025, eta=1.0, k=1)
        env.reset()
        total_reward = 0.0
        wealth = 1.0
        terminated = False
        n_steps = 0
        while not terminated:
            _, reward, terminated, _, info = env.step(np.zeros(3))
            total_reward += reward
            wealth *= 1.0 + info["rho"]
            n_steps += 1
        assert n_steps == 3
        assert total_reward == pytest.approx(np.log(wealth), abs=1e-12)


# ---------------------------------------------------------------------------
# softmax helper
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_simplex_and_stability(self):
        w = softmax(np.array([3.0, -1.0, 0.5]))
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert (w > 0).all()
        # Shift invariance is what makes the max-subtraction trick exact.
        np.testing.assert_allclose(
            w, softmax(np.array([3.0, -1.0, 0.5]) + 100.0), atol=1e-12)
