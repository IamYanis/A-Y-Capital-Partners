"""Validated YAML research configuration. No implicit competition rules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

METRICS = {
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "trend_20",
    "trend_50",
    "trend_200",
    "volatility_adjusted_momentum",
    "relative_strength",
    "volatility_20",
    "volatility_60",
    "correlation_spy",
    "rolling_max_drawdown",
    "downside_volatility",
}


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ResearchConfig:
    assets: dict[str, dict[str, str]]
    benchmark: str
    history_years: int
    cache_dir: Path
    report_currency: str
    ranking_weights: dict[str, float]
    backtest: dict[str, Any]
    macro: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ResearchConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ResearchConfig":
        assets = raw["assets"]
        if not isinstance(assets, dict) or not assets or raw["benchmark"] not in assets:
            raise ValueError("A nonempty asset mapping must include the benchmark")
        for ticker, info in assets.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError("Tickers must be nonempty strings")
            if not all(
                isinstance(info.get(key), str) and info[key]
                for key in ("asset_class", "currency", "calendar")
            ):
                raise ValueError(f"Asset metadata incomplete: {ticker}")
        # This MVP deliberately limits alignment to a common exchange calendar.
        if len({a["calendar"] for a in assets.values()}) != 1:
            raise ValueError("Phase 1 requires one common listing calendar")
        currency = raw["report_currency"]
        if any(a["currency"] != currency for a in assets.values()):
            raise ValueError(
                "Mixed quote currencies require explicit FX conversion before research"
            )
        years = positive_int(raw["history_years"], "history_years")
        if years < 5:
            raise ValueError("Request at least five years of market history")
        weights = raw["ranking_weights"]
        if (
            not isinstance(weights, dict)
            or not weights
            or not set(weights).issubset(METRICS)
            or not all(np.isfinite(v) for v in weights.values())
            or sum(abs(v) for v in weights.values()) == 0
        ):
            raise ValueError("Ranking requires finite, nonzero weights for known metrics")
        settings = dict(raw["backtest"])
        n = positive_int(settings["top_n"], "top_n")
        if n > len(assets):
            raise ValueError("top_n exceeds universe size")
        positive_int(settings["rebalance_sessions"], "rebalance_sessions")
        positive_int(settings["periods_per_year"], "periods_per_year")
        if not 0 < settings["train_fraction"] < 1:
            raise ValueError("train_fraction must be between zero and one")
        cost = settings["transaction_cost_bps"]
        if not np.isfinite(cost) or not 0 <= cost < 10000:
            raise ValueError("transaction_cost_bps must be in [0, 10000)")
        macro = dict(raw.get("macro", {"series": {}, "mode": "latest"}))
        if macro.get("mode") not in {"latest", "vintages"}:
            raise ValueError("macro mode must be latest or vintages")
        return cls(
            assets,
            raw["benchmark"],
            years,
            Path(raw["cache_dir"]),
            currency,
            dict(weights),
            settings,
            macro,
        )

    @property
    def calendar(self) -> str:
        return next(iter(self.assets.values()))["calendar"]
