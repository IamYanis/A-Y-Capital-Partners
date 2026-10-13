"""GBP benchmarks composed from existing signals, risk estimators and backtesting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aycapital.covariance import shrinkage_covariance
from aycapital.cross_asset.backtest import fixed_weight_history
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.market import MarketPanel, validate_panel
from aycapital.cross_asset.scenarios import allocation
from aycapital.cross_asset.signals import SignalSet, calculate_signals, latest_table
from aycapital.statistics import cumulative_returns, price_returns, rolling_returns, summary


@dataclass(frozen=True)
class BenchmarkReport:
    signals: SignalSet
    metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_risk: pd.DataFrame
    equity: pd.DataFrame
    correlation: pd.DataFrame
    covariance: pd.DataFrame
    turnover: pd.DataFrame
    trades: pd.DataFrame


def build_benchmarks(
    panel: MarketPanel, research: ResearchConfig, portfolios: dict[str, dict[str, float]]
) -> BenchmarkReport:
    panel = validate_panel(panel)
    if panel.metadata.get("currency") != "GBP":
        raise ValueError("Benchmark report requires explicitly converted GBP prices")
    annual = research.backtest["periods_per_year"]
    signals = calculate_signals(panel.prices, research.benchmark, research.ranking_weights, annual)
    classes = {ticker: data["asset_class"] for ticker, data in research.assets.items()}
    table = latest_table(signals, classes, panel.closes_utc.iloc[-1])
    assets = panel.prices.columns
    reserved = {research.benchmark, "Equal-weight benchmark"}
    if set(portfolios) & reserved:
        raise ValueError("Portfolio names conflict with benchmark names")
    allocations = {name: allocation(weights, assets) for name, weights in portfolios.items()}
    allocations[research.benchmark] = allocation({research.benchmark: 1.0}, assets)
    allocations["Equal-weight benchmark"] = pd.Series(1 / len(assets), index=assets)
    returns, turnover, trades = fixed_weight_history(
        panel,
        allocations,
        research.benchmark,
        research.backtest["rebalance_sessions"],
        research.backtest["transaction_cost_bps"],
    )
    daily = price_returns(panel.prices)
    asset_risk = summary(daily, annual)
    table = table.join(asset_risk[["max_drawdown", "annualized_volatility"]])
    for window in (5, 20, 40, 60):
        historical = rolling_returns(daily, window).iloc[-1]
        for name in (research.benchmark, "Equal-weight benchmark"):
            baseline = rolling_returns(returns[[name]], window).iloc[-1, 0]
            table[f"relative_{window}_vs_{name}"] = (1 + historical) / (1 + baseline) - 1
        table[f"return_{window}_gbp"] = historical
    table["last_observation_date"] = panel.prices.index[-1].date().isoformat()
    table["observations"] = len(panel.prices)
    table["currency"] = "GBP"
    table["instrument_role"] = "Research proxy; UMushroom eligibility unverified"
    correlation = daily.tail(60).corr()
    covariance = shrinkage_covariance(daily.tail(60), annual)
    portfolio_risk = summary(returns, annual)
    for name, weights in allocations.items():
        portfolio_risk.loc[name, "current_allocation_volatility_60"] = np.sqrt(
            weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()
        )
    equity = 1 + cumulative_returns(returns)
    return BenchmarkReport(
        signals, table, returns, portfolio_risk, equity, correlation, covariance, turnover, trades
    )
