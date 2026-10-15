"""Purged expanding/rolling prequential forecasts with fixed simple models."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aycapital.forecasting.config import ForecastConfig
from aycapital.forecasting.features import FeatureSet, forward_labels
from aycapital.forecasting.uncertainty import empirical_distribution

MODELS = ("historical_average", "conditional_average", "ridge", "logistic")


@dataclass(frozen=True)
class ForecastRun:
    oos: pd.DataFrame
    latest: pd.DataFrame
    distributions: pd.DataFrame
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)


def purged_train_positions(
    origin: int, horizon: int, complete: np.ndarray, window: int | None = None
) -> np.ndarray:
    positions = np.arange(len(complete))
    # Strictly before decision close, including next-close entry offset.
    eligible = (positions + horizon + 1 < origin) & complete
    if window is not None:
        eligible &= positions >= origin - window
    return positions[eligible]


def nonoverlapping_positions(positions: np.ndarray, horizon: int) -> np.ndarray:
    selected, last = [], -horizon
    for pos in positions:
        if pos - last >= horizon:
            selected.append(pos)
            last = pos
    return np.asarray(selected, dtype=int)


def _frequency(values: np.ndarray) -> float:
    return float((np.count_nonzero(values > 0) + 1) / (len(values) + 2))


def _ridge(x: np.ndarray, y: np.ndarray, current: np.ndarray, alpha: float) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y)
    return float(model.predict(current)[0])


def _logistic(
    x: np.ndarray, y: np.ndarray, current: np.ndarray, regularization: float
) -> tuple[float, bool]:
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) < 2 or np.bincount(labels, minlength=2).min() < 5:
        return _frequency(y), False
    model = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=1000))
    model.fit(x, labels)
    return float(model.predict_proba(current)[0, 1]), True


def _regimes(features: FeatureSet, train: np.ndarray, origin: int) -> tuple[np.ndarray, str]:
    trend = features.regime_trend.to_numpy()
    volatility = features.regime_volatility.to_numpy()
    threshold = float(np.median(volatility[train]))
    categories = np.where(trend >= 0, "trend_up", "trend_down").astype(object)
    categories = categories + np.where(volatility > threshold, "/high_vol", "/low_vol")
    return categories, str(categories[origin])


def run_forecasts(
    prices: pd.DataFrame, features: FeatureSet, benchmark: str, config: ForecastConfig
) -> ForecastRun:
    if set(features.assets) != set(prices.columns):
        raise ValueError("Features must cover exactly the research assets")
    oos_rows, latest_rows, samples, regime_forecasts = [], [], [], []
    n = len(prices)
    for asset in prices:
        frame = features.assets[asset]
        if not frame.index.equals(prices.index):
            raise ValueError("Feature and price timestamps must match exactly")
        x = frame.to_numpy(dtype=float)
        x_complete = np.isfinite(x).all(axis=1)
        for horizon in config.horizons:
            labels = forward_labels(prices, benchmark, horizon)
            y = labels[asset].to_numpy()
            excess = y - labels[benchmark].to_numpy()
            complete = x_complete & np.isfinite(y) & np.isfinite(excess)
            first_valid = np.flatnonzero(x_complete)
            first = (
                (int(first_valid[0]) if len(first_valid) else n)
                + config.min_train_sessions
                + horizon
                + 1
            )
            origins = list(range(first, n - horizon - 1, horizon))
            history: dict[str, list[tuple]] = {model: [] for model in MODELS}
            for origin in origins + [n - 1]:
                latest = origin == n - 1
                train = purged_train_positions(
                    origin, horizon, complete, config.train_window_sessions
                )
                enough = len(train) >= config.min_train_sessions and x_complete[origin]
                rows = []
                if enough:
                    categories, regime = _regimes(features, train, origin)
                    conditional = train[categories[train] == regime]
                    condition_blocks = len(nonoverlapping_positions(conditional, horizon))
                    conditional_ok = (
                        len(conditional) >= config.min_regime_sessions
                        and condition_blocks >= config.min_regime_blocks
                    )
                    current = x[origin : origin + 1]
                    xx, yy, ee = x[train], y[train], excess[train]
                    expected = _ridge(xx, yy, current, config.ridge_alpha)
                    expected_excess = (
                        0.0 if asset == benchmark else _ridge(xx, ee, current, config.ridge_alpha)
                    )
                    pos_probability, pos_ok = _logistic(xx, yy, current, config.logistic_c)
                    out_probability, out_ok = (
                        (0.0, True)
                        if asset == benchmark
                        else _logistic(xx, ee, current, config.logistic_c)
                    )
                    positive_mean = float(yy[yy > 0].mean()) if (yy > 0).any() else 0.0
                    negative_mean = float(yy[yy <= 0].mean()) if (yy <= 0).any() else 0.0
                    excess_positive = float(ee[ee > 0].mean()) if (ee > 0).any() else 0.0
                    excess_negative = float(ee[ee <= 0].mean()) if (ee <= 0).any() else 0.0
                    estimates = {
                        "historical_average": (
                            float(yy.mean()),
                            float(ee.mean()),
                            _frequency(yy),
                            0.0 if asset == benchmark else _frequency(ee),
                            True,
                        ),
                        "conditional_average": (
                            float(y[conditional].mean()) if conditional_ok else float(yy.mean()),
                            float(excess[conditional].mean())
                            if conditional_ok
                            else float(ee.mean()),
                            _frequency(y[conditional]) if conditional_ok else _frequency(yy),
                            0.0
                            if asset == benchmark
                            else (
                                _frequency(excess[conditional])
                                if conditional_ok
                                else _frequency(ee)
                            ),
                            conditional_ok,
                        ),
                        "ridge": (expected, expected_excess, np.nan, np.nan, True),
                        "logistic": (
                            pos_probability * positive_mean + (1 - pos_probability) * negative_mean,
                            out_probability * excess_positive
                            + (1 - out_probability) * excess_negative,
                            pos_probability,
                            out_probability,
                            pos_ok and out_ok,
                        ),
                    }
                    for model, (
                        mean,
                        relative_mean,
                        p_positive,
                        p_outperform,
                        supported,
                    ) in estimates.items():
                        mean = max(-1.0, mean)
                        if asset == benchmark:
                            p_outperform = 0.0
                        matured = [
                            r
                            for r in history[model]
                            if r[0] < origin and (model != "conditional_average" or r[3] == regime)
                        ]
                        residuals = np.asarray([r[1] for r in matured], dtype=float)
                        relative_residuals = np.asarray([r[2] for r in matured], dtype=float)
                        low, high = np.nan, np.nan
                        if len(residuals) >= config.min_calibration_origins:
                            low, high, empirical_p = empirical_distribution(
                                mean, residuals, config.interval_coverage
                            )
                            if model == "ridge":
                                p_positive = empirical_p
                                p_outperform = (
                                    0.0
                                    if asset == benchmark
                                    else _frequency(relative_mean + relative_residuals)
                                )
                        row = {
                            "asset": asset,
                            "horizon": horizon,
                            "model": model,
                            "origin_position": origin,
                            "decision_session": prices.index[origin],
                            "label_end_position": origin + horizon + 1,
                            "expected_return_gbp": mean,
                            "expected_excess_return": relative_mean,
                            "probability_positive": p_positive,
                            "probability_outperform": p_outperform,
                            "interval_low": low,
                            "interval_high": high,
                            "interval_coverage_target": config.interval_coverage,
                            "train_sessions": len(train),
                            "train_nonoverlap_blocks": len(
                                nonoverlapping_positions(train, horizon)
                            ),
                            "train_last_label_end_position": int(train[-1] + horizon + 1),
                            "calibration_origins": len(residuals),
                            "regime": regime,
                            "regime_train_sessions": len(conditional),
                            "regime_train_blocks": condition_blocks,
                            "model_supported": bool(
                                supported
                                and len(nonoverlapping_positions(train, horizon))
                                >= config.min_regime_blocks
                            ),
                            "feature_scope": features.scope,
                            "actual_return": y[origin] if not latest else np.nan,
                            "actual_excess_return": excess[origin] if not latest else np.nan,
                        }
                        rows.append(row)
                        if latest and len(residuals) >= config.min_calibration_origins:
                            for value in np.maximum(-1.0, mean + residuals):
                                samples.append(
                                    {
                                        "asset": asset,
                                        "horizon": horizon,
                                        "model": model,
                                        "sample_return": float(value),
                                        "kind": "Approximate prequential residual distribution",
                                    }
                                )
                    if latest:
                        for state in (
                            "trend_up/high_vol",
                            "trend_up/low_vol",
                            "trend_down/high_vol",
                            "trend_down/low_vol",
                        ):
                            observations = train[categories[train] == state]
                            blocks = len(nonoverlapping_positions(observations, horizon))
                            supported = (
                                len(observations) >= config.min_regime_sessions
                                and blocks >= config.min_regime_blocks
                            )
                            regime_forecasts.append(
                                {
                                    "asset": asset,
                                    "horizon": horizon,
                                    "assumed_regime": state,
                                    "decision_session": prices.index[origin],
                                    "train_sessions": len(observations),
                                    "nonoverlap_blocks": blocks,
                                    "expected_return_gbp": float(y[observations].mean())
                                    if supported
                                    else np.nan,
                                    "probability_positive": _frequency(y[observations])
                                    if supported
                                    else np.nan,
                                    "probability_outperform": (
                                        0.0
                                        if asset == benchmark
                                        else _frequency(excess[observations])
                                    )
                                    if supported
                                    else np.nan,
                                    "status": "insufficient evidence: unvalidated regime",
                                    "sample_supported": supported,
                                }
                            )
                    if not latest:
                        for row in rows:
                            history[row["model"]].append(
                                (
                                    origin + horizon + 1,
                                    y[origin] - row["expected_return_gbp"],
                                    excess[origin] - row["expected_excess_return"],
                                    regime,
                                )
                            )
                elif latest:
                    for model in MODELS:
                        rows.append(
                            {
                                "asset": asset,
                                "horizon": horizon,
                                "model": model,
                                "origin_position": origin,
                                "decision_session": prices.index[origin],
                                "model_supported": False,
                                "feature_scope": features.scope,
                                "train_sessions": len(train),
                                "regime": "unavailable",
                                "calibration_origins": 0,
                            }
                        )
                (latest_rows if latest else oos_rows).extend(rows)
    return ForecastRun(
        pd.DataFrame(oos_rows),
        pd.DataFrame(latest_rows),
        pd.DataFrame(samples),
        pd.DataFrame(regime_forecasts),
    )
