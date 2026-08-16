"""Gymnasium environment for the portfolio MDP.

Implements the thesis methodology chapter, section "The Portfolio Markov Decision
Process": a daily-rebalanced, long-only portfolio over N risky assets plus an
interest-bearing cash account. One episode replays one contiguous slice of
history. The step logic follows the trading-timeline figure of that section
line by line: the previous allocation drifts with realized returns
(Equation eq:drift-meth), rebalancing consumes wealth proportional to turnover,
the market moves, and the CRRA utility of the resulting net return
(Equations eq:netreturn-meth and eq:reward-meth) is the reward.

All inputs are prepared by src/market_data and src/features and are assumed
clean: strictly positive closes, no missing values, all arrays aligned on the
same trading calendar.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import config
from src.features.price_tensor import normalize_window

# Day-count convention for the simple daily accrual of the cash leg
# (1 + annual rate / 252). Not in config because it is a market convention,
# not a design choice of the thesis.
TRADING_DAYS_PER_YEAR = 252


def softmax(scores: np.ndarray) -> np.ndarray:
    """Map raw scores onto the probability simplex.

    Subtracting the maximum before exponentiating keeps the exponentials in
    a safe range without changing the result (softmax is shift-invariant).
    """
    shifted = np.exp(scores - np.max(scores))
    return shifted / shifted.sum()


def crra_utility(gross: float, eta: float) -> float:
    """CRRA utility of a one-period gross return (Equation eq:reward-meth).

    u_eta(gross) = (gross^(1-eta) - 1) / (1 - eta)   for eta != 1
                 = ln(gross)                          for eta == 1

    eta is the coefficient of relative risk aversion. eta == 1 is the
    logarithmic limit: summed undiscounted over an episode, log rewards equal
    log terminal wealth, which makes it the growth-optimal (Kelly) anchor of
    the thesis. Requires gross > 0; a long-only, fully invested portfolio
    with positive prices and cost factor (1 - c*tau) >= 1 - 2c > 0
    guarantees this.
    """
    if eta == 1.0:
        return float(np.log(gross))
    return float((gross ** (1.0 - eta) - 1.0) / (1.0 - eta))


class PortfolioEnv(gym.Env):
    """Daily portfolio-choice MDP with transaction costs and CRRA rewards.

    State S_t = (X_t, v_t, w'_t, Z_t)  (methodology, Equation eq:state-meth)
    maps onto the observation dict as follows:

    ==============  ======  ===========================================
    observation     symbol  content
    ==============  ======  ===========================================
    "prices"        X_t     (k, N, 3) close/high/low window ending at t,
                            every field divided by the day-t close of its
                            asset (Jiang normalization; the last close row
                            is therefore all ones)
    "context"       v_t     (2,) market context [VIX z-score, T-bill rate]
    "weights"       w'_t    (N+1,) drift-adjusted allocation, CASH FIRST
                            (index 0); what the portfolio looks like before
                            today's rebalancing decision
    "sentiment"     Z_t     (N, C) per-asset sentiment block, already lagged
                            by the pipeline; key absent in the price-only
                            arm M1 (sentiment=None)
    ==============  ======  ===========================================

    Environment assumptions A1-A4 and where the code enforces them:

    A1 (zero market impact): the environment replays the fixed historical
       arrays passed to __init__; nothing the agent does ever alters prices,
       so transitions are exogenous and replayable.
    A2 (full liquidity): step() rebalances to exactly the softmax weights at
       the daily close, with no partial fills or slippage; residual frictions
       are the proportional cost c * tau.
    A3 (nonnegative holdings): the softmax squashes raw scores onto the
       simplex, so weights are in [0, 1] and sum to 1 -- long-only,
       unlevered, fully invested by construction.
    A4 (divisibility and scale independence): state, action, and dynamics are
       expressed entirely in wealth proportions (weights), never in dollars
       or share counts; wealth itself never appears, so W_0 = 1 is without
       loss of generality.

    Timing (trading-timeline figure): the agent decides on days
    t = start .. end-1; the step taken at decision day t consumes the market
    returns of row t+1. The episode terminates when t reaches ``end``.

    Parameters
    ----------
    ohlc : (T, N, 3) float array
        Adjusted [close, high, low] price levels (market_data contract).
    tbill : (T,) float array
        Annualized 13-week T-bill rate as a decimal.
    context : (T, 2) float array
        Market-context vector [VIX z-score, T-bill rate].
    sentiment : (T, N, C) float array or None
        Per-asset sentiment channels, already lagged. None for arm M1.
    k : int
        Lookback window of the price tensor.
    cost : float
        Proportional transaction cost c.
    eta : float
        CRRA risk-aversion coefficient of the reward.
    start, end : int or None
        Integer indices into the T axis bounding the episode. Defaults:
        start = k-1 (first day with a full price window), end = T-1
        (the step from day end-1 needs returns of row end).
    """

    def __init__(
        self,
        ohlc: np.ndarray,
        tbill: np.ndarray,
        context: np.ndarray,
        sentiment: np.ndarray | None = None,
        k: int = config.WINDOW_K,
        cost: float = config.COST_C,
        eta: float = 1.0,
        start: int | None = None,
        end: int | None = None,
    ):
        super().__init__()
        self.ohlc = np.asarray(ohlc, dtype=np.float64)
        assert self.ohlc.ndim == 3 and self.ohlc.shape[2] == 3, \
            "ohlc must be (T, N, 3) [close, high, low] levels"
        self.T, self.N, _ = self.ohlc.shape

        self.tbill = np.asarray(tbill, dtype=np.float64)
        assert self.tbill.shape == (self.T,), "tbill must align with ohlc days"
        self.context = np.asarray(context, dtype=np.float64)
        assert self.context.ndim == 2 and self.context.shape[0] == self.T, \
            "context must align with ohlc days"
        self.sentiment = (
            None if sentiment is None
            else np.asarray(sentiment, dtype=np.float64)
        )
        if self.sentiment is not None:
            assert self.sentiment.shape[:2] == (self.T, self.N), \
                "sentiment must align with ohlc days and assets"

        self.k = int(k)
        self.cost = float(cost)
        self.eta = float(eta)
        self.start = self.k - 1 if start is None else int(start)
        self.end = self.T - 1 if end is None else int(end)
        assert self.start >= self.k - 1, \
            "start needs a full k-day price window ending at the first decision day"
        assert self.end <= self.T - 1, \
            "the step from decision day end-1 consumes returns of row end"
        assert self.start < self.end, "episode needs at least one decision day"

        obs_spaces = {
            "prices": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.k, self.N, 3), dtype=np.float32),
            "context": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.context.shape[1],), dtype=np.float32),
            "weights": spaces.Box(
                low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32),
        }
        if self.sentiment is not None:
            obs_spaces["sentiment"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N, self.sentiment.shape[2]), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        # Raw scores; the bounded box matters for SAC, whose tanh-squashed
        # policy needs finite bounds to squash onto.
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.N + 1,), dtype=np.float32)

        self._t: int = self.start
        self._drifted = np.zeros(self.N + 1)
        self._drifted[0] = 1.0

    # ------------------------------------------------------------- helpers

    def _observation(self) -> dict:
        """State S_t at the current decision day (data through day t only)."""
        obs = {
            "prices": normalize_window(self.ohlc, self._t, self.k),
            "context": self.context[self._t].astype(np.float32),
            "weights": self._drifted.astype(np.float32),
        }
        if self.sentiment is not None:
            # Already lagged by the sentiment pipeline: row t holds news
            # dated through trading day t - lag.
            obs["sentiment"] = self.sentiment[self._t].astype(np.float32)
        return obs

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.start
        self._drifted = np.zeros(self.N + 1)
        self._drifted[0] = 1.0  # episode starts 100% in cash (w' = e_0)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        """One decision day, following the trading-timeline figure.

        The agent decides at the close of day t; the position is then held
        through day t+1, whose returns settle the step.
        """
        t = self._t  # decision day
        action = np.asarray(action, dtype=np.float64)
        drifted_before = self._drifted.copy()  # w'_t, the pre-trade holdings

        # (1) Softmax maps the raw scores to the chosen allocation w_t:
        #     long-only, unlevered, fully invested (assumption A3).
        w = softmax(action)

        # (2) Turnover against the drift-adjusted holdings, RISKY legs only
        #     (tau in [0, 2]). Cash is the settlement medium through which
        #     purchases and sales clear, not a traded security, so it is
        #     excluded from the cost sum (footnote to eq:netreturn-meth).
        tau = float(np.abs(w[1:] - self._drifted[1:]).sum())

        # (3) Gross return vector y_t over the holding period t -> t+1.
        #     The cash leg accrues the T-bill rate observed on decision day t
        #     -- known before the position is held, so no lookahead -- as a
        #     simple daily accrual 1 + annual rate / 252.
        cash_gross = 1.0 + self.tbill[t] / TRADING_DAYS_PER_YEAR
        risky_gross = self.ohlc[t + 1, :, 0] / self.ohlc[t, :, 0]
        y = np.concatenate([[cash_gross], risky_gross])

        # (4) Net portfolio return (eq:netreturn-meth):
        #     rho_t = (1 - c*tau) * (w . y) - 1.
        #     (1 - c*tau) is the first-order term in c of Jiang's iterative
        #     transaction-remainder factor mu_t -- a linear approximation
        #     instead of his fixed-point solution.
        portfolio_gross = float(w @ y)
        rho = (1.0 - self.cost * tau) * portfolio_gross - 1.0

        # (5) Reward is the CRRA utility of the net gross return
        #     (eq:reward-meth); eta = 1 gives the log-utility anchor.
        reward = crra_utility(1.0 + rho, self.eta)

        # (6) Weights drift with the realized returns to w'_{t+1}
        #     (eq:drift-meth). The cost factor scales total wealth, not the
        #     asset mix, so it cancels from the weight ratio.
        self._drifted = (w * y) / portfolio_gross

        self._t = t + 1
        terminated = self._t == self.end

        # The info dict is the environment's full per-step ledger: everything a
        # later analysis could need that is not cheaply reconstructible from
        # the inputs. src/experiments/walkforward.py persists it verbatim.
        info = {
            "rho": rho,
            "log_return": float(np.log1p(rho)),
            "turnover": tau,
            "cost_paid": self.cost * tau,          # wealth fraction lost to trading
            "weights": w.copy(),                   # w_t, the post-rebalance target
            "drifted_weights": drifted_before,     # w'_t, holdings before the trade
            "action": action.copy(),               # raw scores the softmax received
            "cash_gross": cash_gross,
            "portfolio_gross": portfolio_gross,
            "t": t,  # the decision day this step acted on
        }
        return self._observation(), float(reward), terminated, False, info
