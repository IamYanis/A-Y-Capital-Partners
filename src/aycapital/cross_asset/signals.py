"""Causal, interpretable price signals and contemporaneous cross-sectional ranks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aycapital.cross_asset.config import METRICS
from aycapital.data import validate_prices


@dataclass(frozen=True)
class SignalSet:
    metrics: dict[str, pd.DataFrame]
    percentiles: dict[str, pd.DataFrame]
    score: pd.DataFrame


def rolling_max_drawdown(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Worst peak-to-trough drawdown wholly within each trailing price window."""
    return prices.rolling(window).apply(
        lambda values: np.min(values / np.maximum.accumulate(values) - 1), raw=True
    )


def calculate_signals(
    prices: pd.DataFrame, benchmark: str, weights: dict[str, float], periods_per_year: int = 252
) -> SignalSet:
    prices = validate_prices(prices)
    if benchmark not in prices:
        raise ValueError("Benchmark must be included in price panel")
    if not weights or not set(weights).issubset(METRICS):
        raise ValueError("Known ranking metrics required")
    if (
        not all(np.isfinite(v) for v in weights.values())
        or sum(abs(v) for v in weights.values()) == 0
    ):
        raise ValueError("Finite, nonzero ranking weights required")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year < 1
    ):
        raise ValueError("Annualization must be a positive integer")
    returns = prices.pct_change(fill_method=None)
    metrics = {f"momentum_{n}": prices / prices.shift(n) - 1 for n in (5, 20, 60, 120)}
    metrics.update({f"trend_{n}": prices / prices.rolling(n).mean() - 1 for n in (20, 50, 200)})
    metrics.update(
        {
            f"volatility_{n}": returns.rolling(n).std(ddof=1) * np.sqrt(periods_per_year)
            for n in (20, 60)
        }
    )
    metrics["volatility_adjusted_momentum"] = metrics["momentum_60"] / metrics[
        "volatility_60"
    ].replace(0, np.nan)
    # Relative growth ratio, rather than a forecast of relative performance.
    metrics["relative_strength"] = (1 + metrics["momentum_60"]).div(
        1 + metrics["momentum_60"][benchmark], axis=0
    ) - 1
    metrics["correlation_spy"] = returns.rolling(60).corr(returns[benchmark])
    metrics["rolling_max_drawdown"] = rolling_max_drawdown(prices, 60)
    metrics["downside_volatility"] = np.sqrt(
        returns.clip(upper=0).pow(2).rolling(60).mean()
    ) * np.sqrt(periods_per_year)
    percentiles = {}
    contributions = []
    complete = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    for name, weight in weights.items():
        if weight == 0:
            continue
        metric = metrics[name].replace([np.inf, -np.inf], np.nan)
        # Midrank scaled to [0,1]; ties share a score. No historical fitting.
        count = metric.notna().sum(axis=1)
        rank = (metric.rank(axis=1, method="average") - 1).div(
            (count - 1).replace(0, np.nan), axis=0
        )
        rank.loc[count == 1] = metric.loc[count == 1].notna().replace({True: 0.5, False: np.nan})
        percentiles[name] = rank
        # Negative weights invert preference: lower volatility is preferred.
        preferred = rank if weight > 0 else 1 - rank
        contributions.append(preferred * abs(weight))
        complete &= metric.notna()
    score = sum(contributions) / sum(abs(v) for v in weights.values())
    score = score.where(complete)
    return SignalSet(metrics, percentiles, score)


def latest_table(
    signals: SignalSet, asset_classes: dict[str, str], close_time: pd.Timestamp
) -> pd.DataFrame:
    date = signals.score.index[-1]
    table = pd.DataFrame({name: data.loc[date] for name, data in signals.metrics.items()})
    table["score"] = signals.score.loc[date]
    table["rank"] = table.score.rank(ascending=False, method="min")
    table["asset_class"] = pd.Series(asset_classes)
    table["signal_session"] = date.date().isoformat()
    table["signal_available_after_utc"] = close_time.isoformat()
    table["interpretation"] = "Historical evidence score; no return forecast"
    return table.sort_values(["score"], ascending=False, kind="stable")
