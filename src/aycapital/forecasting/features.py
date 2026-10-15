"""Causal features using existing price signals, dated FX and vintage-aware macro."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aycapital.cross_asset.macro import MacroData, available_as_of
from aycapital.cross_asset.market import MarketPanel, validate_panel
from aycapital.cross_asset.signals import calculate_signals


@dataclass(frozen=True)
class FeatureSet:
    assets: dict[str, pd.DataFrame]
    regime_trend: pd.Series
    regime_volatility: pd.Series
    scope: str
    warnings: tuple[str, ...]


def build_features(
    panel: MarketPanel,
    fx_alignment: pd.DataFrame,
    benchmark: str,
    macro: dict[str, MacroData] | None = None,
    required_macro: tuple[str, ...] = (),
) -> FeatureSet:
    panel = validate_panel(panel)
    if panel.metadata.get("currency") != "GBP":
        raise ValueError("Forecasts require the existing explicitly converted GBP panel")
    if not fx_alignment.index.equals(panel.prices.index):
        raise ValueError("FX alignment must cover the exact asset sessions")
    fx = pd.to_numeric(fx_alignment.usd_per_gbp)
    available = pd.to_datetime(fx_alignment.fx_available_at_utc, utc=True)
    if (
        not np.isfinite(fx).all()
        or (fx <= 0).any()
        or available.isna().any()
        or (available > panel.closes_utc).any()
    ):
        raise ValueError("FX features contain invalid or unavailable quotes")
    signals = calculate_signals(panel.prices, benchmark, {"momentum_20": 1})
    common = pd.DataFrame(index=panel.prices.index)
    for window in (5, 20, 60):
        common[f"gbp_per_usd_change_{window}"] = (1 / fx).pct_change(window, fill_method=None)
    if "fx_age_calendar_days" in fx_alignment:
        common["fx_age_calendar_days"] = fx_alignment.fx_age_calendar_days
    common["benchmark_trend_200"] = signals.metrics["trend_200"][benchmark]
    common["benchmark_volatility_60"] = signals.metrics["volatility_60"][benchmark]
    warnings = []
    macro = macro or {}
    for series, data in macro.items():
        values = available_as_of(data, pd.DatetimeIndex(panel.closes_utc))
        values.index = common.index
        common[f"macro_{series}_known_level"] = values
        # Changes of already-known levels, not inflation YoY or an invented spread.
        common[f"macro_{series}_known_change_20"] = values.diff(20)
    if {"DGS2", "DGS10"}.issubset(macro):
        common["macro_yield_curve"] = common.macro_DGS10_known_level - common.macro_DGS2_known_level
    absent = sorted(set(required_macro) - set(macro))
    if absent:
        warnings.append("Unavailable publication-aware macro/credit series: " + ", ".join(absent))
    # A liquid-price credit proxy is useful but is never called an observed credit spread.
    if {"HYG", "LQD"}.issubset(panel.prices):
        common["credit_proxy_relative_momentum_20"] = (1 + signals.metrics["momentum_20"].HYG) / (
            1 + signals.metrics["momentum_20"].LQD
        ) - 1
    scope = (
        "price_fx_macro"
        if not absent and macro
        else "partial_macro_price_fx"
        if macro
        else "reduced_price_fx"
    )
    frames = {}
    for asset in panel.prices:
        frame = common.copy()
        for name in (
            "momentum_5",
            "momentum_20",
            "momentum_60",
            "momentum_120",
            "volatility_20",
            "volatility_60",
            "trend_50",
            "trend_200",
            "relative_strength",
            "downside_volatility",
            "rolling_max_drawdown",
        ):
            frame[name] = signals.metrics[name][asset]
        frames[asset] = frame.replace([np.inf, -np.inf], np.nan)
    return FeatureSet(
        frames, common.benchmark_trend_200, common.benchmark_volatility_60, scope, tuple(warnings)
    )


def forward_labels(prices: pd.DataFrame, benchmark: str, horizon: int) -> pd.DataFrame:
    """Signal t, next-close entry t+1, exit t+1+h. Never used as features."""
    if horizon < 1 or benchmark not in prices:
        raise ValueError("Positive horizon and included benchmark required")
    labels = prices.shift(-(horizon + 1)) / prices.shift(-1) - 1
    return labels
