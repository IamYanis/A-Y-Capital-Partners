"""Fixed, validated forecast design. Hyperparameters are never selected on OOS data."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ForecastConfig:
    horizons: tuple[int, ...] = (5, 10, 20, 40)
    min_train_sessions: int = 252
    train_window_sessions: int | None = None
    min_regime_sessions: int = 100
    min_regime_blocks: int = 10
    min_oos_origins: int = 40
    min_calibration_origins: int = 20
    interval_coverage: float = 0.8
    ridge_alpha: float = 10.0
    logistic_c: float = 0.1
    bootstrap_draws: int = 4999
    bootstrap_block_origins: int = 5
    family_alpha: float = 0.05
    seed: int = 2026
    macro_policy: str = "optional"
    macro_series: tuple[str, ...] = ("DGS2", "DGS10", "BAMLH0A0HYM2", "CPIAUCSL", "DTWEXBGS")

    def __post_init__(self):
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Unique forecast horizons are required")
        for horizon in self.horizons:
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("Horizons must be positive session counts")
        for name in (
            "min_train_sessions",
            "min_regime_sessions",
            "min_regime_blocks",
            "min_oos_origins",
            "min_calibration_origins",
            "bootstrap_draws",
            "bootstrap_block_origins",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.train_window_sessions is not None:
            if (
                not isinstance(self.train_window_sessions, int)
                or self.train_window_sessions < self.min_train_sessions + max(self.horizons) + 2
            ):
                raise ValueError("Rolling window must cover training plus the horizon purge")
        if not 0 < self.interval_coverage < 1 or not 0 < self.family_alpha < 1:
            raise ValueError("Coverage and significance level must be in (0,1)")
        if (
            not np.isfinite(self.ridge_alpha)
            or self.ridge_alpha <= 0
            or not np.isfinite(self.logistic_c)
            or self.logistic_c <= 0
        ):
            raise ValueError("Positive finite regularization parameters required")
        if self.macro_policy not in {"optional", "require", "off"}:
            raise ValueError("macro_policy must be optional, require or off")

    @classmethod
    def load(cls, path: str | Path) -> "ForecastConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict) or not set(raw).issubset({f.name for f in fields(cls)}):
            raise ValueError("Unknown or malformed forecasting configuration")
        for name in ("horizons", "macro_series"):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)
