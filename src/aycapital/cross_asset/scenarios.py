"""User-specified terminal shocks, not calibrated conditional returns or forecasts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def allocation(weights: dict[str, float], assets: pd.Index) -> pd.Series:
    if not weights or not set(weights).issubset(assets):
        raise ValueError("Portfolio contains unknown research assets or is empty")
    result = pd.Series(weights, dtype=float).reindex(assets, fill_value=0.0)
    if (
        not np.isfinite(result).all()
        or (result < 0).any()
        or not np.isclose(result.sum(), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("Long-only portfolio weights must be finite, nonnegative and sum to one")
    return result


@dataclass(frozen=True)
class ScenarioConfig:
    scenarios: dict[str, dict[str, Any]]
    portfolios: dict[str, dict[str, float]]
    horizon_sessions: int
    max_fx_age_days: int

    @classmethod
    def load(cls, path: str | Path, assets: pd.Index) -> "ScenarioConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if raw.get("kind") != "hypothetical":
            raise ValueError("Only explicitly hypothetical scenarios are supported")
        horizon = raw["horizon_sessions"]
        age = raw["max_fx_age_days"]
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 20 <= horizon <= 40
            or isinstance(age, bool)
            or not isinstance(age, int)
            or age < 1
        ):
            raise ValueError(
                "Scenario horizon must be 20–40 sessions; FX maximum age must be positive"
            )
        result = cls(raw["scenarios"], raw["portfolios"], horizon, age)
        scenario_returns(result, assets)
        return result


def scenario_returns(config: ScenarioConfig, assets: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Terminal GBP shocks = (1+native shock)*(1+GBP-per-USD shock)-1.

    Weighted terminal portfolio return assumes initial weights, no interim trading,
    no fees and unhedged USD assets. Shock units are fractions, not percentages.
    """
    if not config.scenarios or not config.portfolios:
        raise ValueError("Scenarios and comparison portfolios are required")
    rows = {}
    for name, scenario in config.scenarios.items():
        shocks = pd.Series(scenario["asset_returns_usd"], dtype=float)
        currency = scenario["gbp_per_usd_return"]
        if (
            set(shocks.index) != set(assets)
            or shocks.index.has_duplicates
            or not np.isfinite(shocks).all()
            or (shocks <= -1).any()
            or not np.isfinite(currency)
            or currency <= -1
        ):
            raise ValueError("Every asset needs a finite shock above -100%, with valid FX shock")
        rows[name] = (1 + shocks.reindex(assets)) * (1 + currency) - 1
    asset_returns = pd.DataFrame(rows).T
    portfolio_returns = pd.DataFrame(
        {
            name: asset_returns @ allocation(weights, assets)
            for name, weights in config.portfolios.items()
        }
    )
    return asset_returns, portfolio_returns
