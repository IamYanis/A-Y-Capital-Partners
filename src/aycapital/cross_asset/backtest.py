"""Fixed-rule walk-forward simulation: post-close signal, next-session close fill."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from aycapital.cross_asset.config import positive_int
from aycapital.cross_asset.market import MarketPanel, validate_panel
from aycapital.cross_asset.signals import SignalSet
from aycapital.statistics import summary


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    turnover: pd.DataFrame
    statistics: pd.DataFrame
    trades: pd.DataFrame
    periods: pd.Series
    metadata: dict[str, Any]


def target_weights(scores: pd.Series, top_n: int) -> pd.Series:
    """Equal weights in top N complete scores; stable ticker-order tie breaking."""
    chosen = scores.dropna().sort_values(ascending=False, kind="stable").head(top_n)
    target = pd.Series(0.0, index=scores.index)
    if chosen.empty:
        raise ValueError("No complete signals available")
    target.loc[chosen.index] = 1 / len(chosen)
    return target


def rebalance_accounting(
    pre_trade_weights: pd.Series, target: pd.Series, cost_rate: float
) -> tuple[float, float]:
    """Solve post-cost NAV and actual traded notional, both per pre-trade NAV.

    u + c * sum(abs(u * target - old)) = 1; fractional holdings, symmetric fees.
    This includes initial cash purchases and fee-funded sales on retained assets.
    """
    if not 0 <= cost_rate < 1:
        raise ValueError("Cost rate must be in [0,1)")
    if not pre_trade_weights.index.equals(target.index):
        raise ValueError("Rebalance asset labels must match")
    old, desired = pre_trade_weights.to_numpy(), target.to_numpy()
    if (
        not np.isfinite(old).all()
        or not np.isfinite(desired).all()
        or (old < 0).any()
        or (desired < 0).any()
        or old.sum() > 1 + 1e-10
        or not np.isclose(desired.sum(), 1)
    ):
        raise ValueError("Invalid long-only rebalance weights")

    def residual(u: float) -> float:
        return float(u + cost_rate * np.abs(u * desired - old).sum() - 1)

    nav_fraction = brentq(residual, 0.0, 1.0, xtol=1e-14) if cost_rate else 1.0
    traded = float(np.abs(nav_fraction * desired - old).sum())
    if nav_fraction <= 0:
        raise ValueError("Transaction costs consume capital")
    return float(nav_fraction), traded


def _simulate(
    panel: MarketPanel,
    scores: pd.DataFrame,
    start: int,
    end: int,
    top_n: int,
    rebalance_sessions: int,
    cost_bps: float,
    benchmark: str,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Cash starts at one. Rebalance after applying the old holdings' session return.

    Turnover is buy+sell notional / pre-trade NAV (no division by two). Costs are
    deducted as a fraction of pre-trade NAV; new weights refer to post-cost NAV.
    """
    prices = panel.prices
    assets = prices.columns
    daily = prices.pct_change(fill_method=None)
    names = ["strategy", benchmark, "equal_weight"]
    holdings = {name: pd.Series(0.0, index=assets) for name in names}
    records, turnovers, trades = [], [], []
    cost_rate = cost_bps / 10000
    for position in range(start, end):
        date = prices.index[position]
        record, turnover_row = {}, {}
        for name in names:
            old = holdings[name]
            gross = float(old @ daily.iloc[position])
            drifted = old * (1 + daily.iloc[position]) / (1 + gross)
            traded = 0.0
            nav_fraction = 1.0
            rebalance = (position - start) % rebalance_sessions == 0
            if name == benchmark:
                rebalance = position == start  # Buy-and-hold benchmark, same entry cost.
            if rebalance:
                if name == "strategy":
                    target = target_weights(scores.iloc[position - 1], top_n)
                elif name == benchmark:
                    target = pd.Series(0.0, index=assets)
                    target[benchmark] = 1.0
                else:
                    target = pd.Series(1 / len(assets), index=assets)
                nav_fraction, traded = rebalance_accounting(drifted, target, cost_rate)
                signal_time = panel.closes_utc.iloc[position - 1]
                fill_time = panel.closes_utc.iloc[position]
                if not signal_time < fill_time:
                    raise ValueError("Signals must precede fills")
                trades.append(
                    {
                        "period": label,
                        "portfolio": name,
                        "signal_session": prices.index[position - 1],
                        "signal_available_after_utc": signal_time,
                        "execution_session": date,
                        "execution_close_utc": fill_time,
                        "traded_notional_fraction": traded,
                        "cost_fraction": traded * cost_rate,
                        **{f"weight_{a}": float(target[a]) for a in assets},
                    }
                )
                holdings[name] = target
            else:
                holdings[name] = drifted
            record[name] = (1 + gross) * nav_fraction - 1
            turnover_row[name] = traded
        records.append(record)
        turnovers.append(turnover_row)
    index = prices.index[start:end]
    return pd.DataFrame(records, index=index), pd.DataFrame(turnovers, index=index), trades


def walk_forward(
    panel: MarketPanel,
    signals: SignalSet,
    benchmark: str,
    *,
    top_n: int = 4,
    rebalance_sessions: int = 20,
    transaction_cost_bps: float = 10,
    train_fraction: float = 0.6,
    periods_per_year: int = 252,
) -> BacktestResult:
    panel = validate_panel(panel)
    prices = panel.prices
    for value, name in [
        (top_n, "top_n"),
        (rebalance_sessions, "rebalance_sessions"),
        (periods_per_year, "periods_per_year"),
    ]:
        positive_int(value, name)
    if top_n > len(prices.columns) or benchmark not in prices:
        raise ValueError("Invalid universe or benchmark")
    if benchmark in {"strategy", "equal_weight"}:
        raise ValueError("Benchmark name conflicts with portfolio names")
    if not signals.score.index.equals(prices.index) or not signals.score.columns.equals(
        prices.columns
    ):
        raise ValueError("Signal labels must exactly match market prices")
    if not np.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps < 10000:
        raise ValueError("Invalid transaction costs")
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("Invalid training fraction")
    if np.isinf(signals.score.to_numpy()).any():
        raise ValueError("Infinite signal scores are invalid")
    valid = signals.score.notna().all(axis=1)
    if not valid.any():
        raise ValueError("No complete cross-asset signal dates")
    first_signal = int(np.flatnonzero(valid.to_numpy())[0])
    start = max(first_signal + 1, 1)
    if not valid.iloc[first_signal:].all():
        raise ValueError("Signal gaps after warmup; cannot silently remove assets")
    split = start + int((len(prices) - start) * train_fraction)
    if split - start < 2 or len(prices) - split < 2:
        raise ValueError("Need at least two evaluation sessions per split")
    blocks, turnover_blocks, trade_records, labels = [], [], [], []
    stats = []
    for label, left, right in [("in_sample", start, split), ("out_of_sample", split, len(prices))]:
        daily, turnover, trades = _simulate(
            panel,
            signals.score,
            left,
            right,
            top_n,
            rebalance_sessions,
            transaction_cost_bps,
            benchmark,
            label,
        )
        blocks.append(daily)
        turnover_blocks.append(turnover)
        trade_records.extend(trades)
        labels.extend([label] * len(daily))
        performance = summary(daily, periods_per_year)
        elapsed_years = len(daily) / periods_per_year
        performance["cagr"] = (1 + daily).prod() ** (1 / elapsed_years) - 1
        performance["total_traded_notional"] = turnover.sum()
        performance["annualized_turnover"] = turnover.sum() / elapsed_years
        performance["period"] = label
        stats.append(performance.reset_index(names="portfolio").set_index(["period", "portfolio"]))
    daily = pd.concat(blocks)
    period_labels = pd.Series(labels, index=daily.index, name="period")
    # Each split is a fresh account; do not splice it into a misleading continuous NAV.
    equity = (1 + daily).groupby(period_labels, sort=False).cumprod()
    return BacktestResult(
        daily,
        equity,
        pd.concat(turnover_blocks),
        pd.concat(stats),
        pd.DataFrame(trade_records),
        period_labels,
        {
            "execution": "Signal after session t close; fill t+1 close; first earned return t+2",
            "split_session": str(prices.index[split].date()),
            "parameters": {
                "top_n": top_n,
                "rebalance_sessions": rebalance_sessions,
                "transaction_cost_bps": transaction_cost_bps,
                "train_fraction": train_fraction,
                "periods_per_year": periods_per_year,
            },
            "fitting": "Fixed ex-ante rules, no parameter search or full-sample fitting",
            "split_accounts": "Independent cash start; pre-split history used for OOS signals",
            "cost_model": "Self-financing proportional fees; excludes intraday slippage and impact",
            "currency": panel.metadata.get("currency"),
            "limitations": "Static universe; vendor adjustments; no predictive accuracy claim",
        },
    )
