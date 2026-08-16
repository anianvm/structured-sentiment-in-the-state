"""Persistence of the sentiment signal — the estimate the myopia argument needs.

The methodology chapter bounds the bias from the per-period CRRA reward by
arguing that the omitted intertemporal hedging demand is small, and that
argument rests on sentiment innovations being short-lived: hedging demand
grows with predictor persistence (Kim & Omberg 1996), so a fast-decaying
predictor generates little of it. This script produces the number.

What is measured: the RAW aggregated daily direction signal — after the
calendar alignment and multi-article merge of src/features/sentiment_grid.py,
but BEFORE the exponential smoothing and the one-day lag. Those two are design
choices of this thesis (a 5-day half-life is imposed by construction), so
smoothing the series first would measure our own filter rather than the news
process. The quantity of interest is how long a sentiment shock survives in
the world, not how long we make it survive in the state.

Two estimates are reported per ticker and pooled across the Dow 30:
  * the autocorrelation function of the daily direction series at lags 1..10;
  * an AR(1) coefficient phi and the implied half-life ln(0.5)/ln(phi) in
    trading days, i.e. how long until a shock has decayed to half its size.

Caveat worth stating in the appendix: days without news enter the series as
zero, which is how the aggregation encodes "no signal". Since news is present
on about 82 percent of stock-days this affects the level of the estimates
only mildly, but it biases them DOWNWARD (toward less persistence), so the
resulting half-life is a conservative input to an argument that wants
persistence to be small.

Outputs (results/persistence/): acf.csv, ar1.csv, persistence.png, and a
summary printed to stdout.

Run from the project root:
  python -m src.sentiment_pipeline.estimate_persistence
"""

import numpy as np
import pandas as pd

from src import config
from src.features.sentiment_grid import _aggregate_llm

OUT = config.RESULTS_DIR / "persistence"
MAX_LAG = 10


def autocorr(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of x at `lag` (zero-mean convention)."""
    if lag == 0:
        return 1.0
    a, b = x[:-lag], x[lag:]
    a, b = a - x.mean(), b - x.mean()
    denom = ((x - x.mean()) ** 2).sum()
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def half_life(phi: float) -> float:
    """Trading days until an AR(1) shock decays to half its initial size."""
    if not (0 < phi < 1):
        return np.nan  # no decay interpretation outside a stationary AR(1)
    return float(np.log(0.5) / np.log(phi))


def main() -> None:
    # A trading calendar is needed only to align headline dates; the price
    # tensor supplies exactly the calendar the agent trades on.
    from src.features.price_tensor import load_ohlc_tensor

    dates, _ = load_ohlc_tensor()
    scores = pd.read_csv(config.LLM_SCORES_FILE, parse_dates=["date"])

    # halflife=0 -> no exponential smoothing; the lag is applied later in the
    # pipeline and is deliberately not applied here (see the module docstring).
    grid = _aggregate_llm(scores, dates, halflife=0.0)
    direction = grid[:, :, config.LLM_CHANNELS.index("direction")]
    news_flag = grid[:, :, config.LLM_CHANNELS.index("news_flag")]

    print(f"aggregated direction grid: {direction.shape} "
          f"({dates[0].date()} -> {dates[-1].date()})")
    print(f"news coverage: {news_flag.mean():.1%} of stock-days\n")

    acf_rows, ar1_rows = [], []
    for i, ticker in enumerate(config.DOW30):
        series = direction[:, i].astype(float)
        acf = [autocorr(series, lag) for lag in range(1, MAX_LAG + 1)]
        acf_rows.append({"ticker": ticker,
                         **{f"lag{l}": v for l, v in enumerate(acf, start=1)}})
        # AR(1) by OLS through the origin on the demeaned series.
        x, y = series[:-1] - series.mean(), series[1:] - series.mean()
        phi = float((x @ y) / (x @ x)) if (x @ x) > 0 else np.nan
        ar1_rows.append({"ticker": ticker, "phi": phi,
                         "half_life_days": half_life(phi),
                         "coverage": float(news_flag[:, i].mean())})

    acf_df = pd.DataFrame(acf_rows)
    ar1_df = pd.DataFrame(ar1_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    acf_df.to_csv(OUT / "acf.csv", index=False)
    ar1_df.to_csv(OUT / "ar1.csv", index=False)

    mean_acf = acf_df[[f"lag{l}" for l in range(1, MAX_LAG + 1)]].mean()
    phi_med = ar1_df["phi"].median()
    hl = ar1_df["half_life_days"].dropna()

    print("Mean autocorrelation across the 30 tickers:")
    for lag, v in mean_acf.items():
        print(f"  {lag:>6}: {v:+.4f}")
    print(f"\nAR(1) coefficient phi: median {phi_med:.4f} "
          f"(range {ar1_df['phi'].min():.4f} to {ar1_df['phi'].max():.4f})")
    if len(hl):
        print(f"Implied half-life:     median {hl.median():.2f} trading days "
              f"(IQR {hl.quantile(.25):.2f}-{hl.quantile(.75):.2f}, "
              f"{len(hl)}/{len(ar1_df)} tickers with a stationary decay)")
    print(f"\nFor the chapter: sentiment innovations decay with a half-life of "
          f"about {hl.median():.1f} trading days." if len(hl) else "")

    _plot(mean_acf, hl)
    print(f"wrote acf.csv, ar1.csv, persistence.png -> {OUT}")


def _plot(mean_acf: pd.Series, hl: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    lags = range(1, len(mean_acf) + 1)
    ax1.bar(lags, mean_acf.values, color="tab:blue")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("lag (trading days)")
    ax1.set_ylabel("autocorrelation")
    ax1.set_title("Mean ACF of the daily direction signal")
    ax1.set_xticks(list(lags))

    if len(hl):
        ax2.hist(hl, bins=12, color="tab:blue")
        ax2.axvline(hl.median(), ls="--", color="grey",
                    label=f"median {hl.median():.1f} days")
        ax2.legend()
    ax2.set_xlabel("implied AR(1) half-life (trading days)")
    ax2.set_ylabel("tickers")
    ax2.set_title("Persistence across the Dow 30")

    fig.tight_layout()
    fig.savefig(OUT / "persistence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
