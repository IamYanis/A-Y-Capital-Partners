"""Paired OOS skill evaluation; failed tests remain explicit insufficient evidence."""

import numpy as np
import pandas as pd

from aycapital.forecasting.config import ForecastConfig
from aycapital.forecasting.engine import ForecastRun
from aycapital.forecasting.uncertainty import block_inference, holm_adjust


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    bins = np.minimum((p * 5).astype(int), 4)
    rows, error = [], 0.0
    for group in range(5):
        mask = bins == group
        if mask.any():
            predicted, observed = float(p[mask].mean()), float(y[mask].mean())
            error += mask.mean() * abs(predicted - observed)
            rows.append(
                {
                    "bin": group,
                    "n": int(mask.sum()),
                    "mean_probability": predicted,
                    "observed_frequency": observed,
                }
            )
    return float(error), rows


def evaluate(
    run: ForecastRun, config: ForecastConfig, benchmark: str, source: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, calibration_rows, regime_rows = [], [], []
    keys = ["asset", "horizon", "model"]
    if run.oos.empty:
        grouped = []
    else:
        grouped = run.oos.groupby(keys, sort=False)
    for (asset, horizon, model), data in grouped:
        baseline = run.oos[
            (run.oos.asset == asset)
            & (run.oos.horizon == horizon)
            & (run.oos.model == "historical_average")
        ]
        paired = data.merge(
            baseline, on="origin_position", suffixes=("", "_base"), validate="one_to_one"
        )
        actual = paired.actual_return.to_numpy()
        predicted = paired.expected_return_gbp.to_numpy()
        errors = abs(actual - predicted)
        base_errors = abs(actual - paired.expected_return_gbp_base.to_numpy())
        row = {
            "asset": asset,
            "horizon": horizon,
            "model": model,
            "oos_origins": len(paired),
            "mae": float(errors.mean()),
            "baseline_mae": float(base_errors.mean()),
            "mae_improvement": float((base_errors - errors).mean()),
            "return_sign_accuracy": float(((predicted > 0) == (actual > 0)).mean()),
            "baseline_directional_accuracy": float(
                ((paired.probability_positive_base >= 0.5) == (actual > 0)).mean()
            ),
            "mean_forecast_excess_return": float(paired.expected_excess_return.mean()),
            "excess_return_mae": float(
                abs(paired.actual_excess_return - paired.expected_excess_return).mean()
            ),
            "baseline_excess_return_mae": float(
                abs(paired.actual_excess_return - paired.expected_excess_return_base).mean()
            ),
            "support_fraction": float(paired.model_supported.mean()),
        }
        gains = [base_errors - errors]
        probability_counts = []
        for endpoint, target, probability in [
            ("positive", (actual > 0).astype(float), "probability_positive"),
            (
                "outperform",
                (paired.actual_excess_return.to_numpy() > 0).astype(float),
                "probability_outperform",
            ),
        ]:
            valid = np.isfinite(paired[probability]) & np.isfinite(paired[probability + "_base"])
            yy = target[valid]
            pp = paired.loc[valid, probability].to_numpy()
            bb = paired.loc[valid, probability + "_base"].to_numpy()
            score = (pp - yy) ** 2
            base_score = (bb - yy) ** 2
            row[f"brier_{endpoint}"] = float(score.mean()) if len(pp) else np.nan
            row[f"baseline_brier_{endpoint}"] = float(base_score.mean()) if len(pp) else np.nan
            row[f"brier_{endpoint}_improvement"] = (
                float((base_score - score).mean()) if len(pp) else np.nan
            )
            row[f"probability_{endpoint}_origins"] = len(pp)
            if endpoint == "positive":
                row["directional_accuracy"] = (
                    float(((pp >= 0.5) == yy).mean()) if len(pp) else np.nan
                )
                row["baseline_directional_accuracy"] = (
                    float(((bb >= 0.5) == yy).mean()) if len(pp) else np.nan
                )
            else:
                row["outperformance_accuracy"] = (
                    float(((pp >= 0.5) == yy).mean()) if len(pp) else np.nan
                )
            if len(pp):
                error, calibration = _calibration(yy, pp)
                row[f"calibration_error_{endpoint}"] = error
                for bucket in calibration:
                    calibration_rows.append(
                        {
                            "asset": asset,
                            "horizon": horizon,
                            "model": model,
                            "endpoint": endpoint,
                            **bucket,
                        }
                    )
            if endpoint != "outperform" or asset != benchmark:
                gains.append(base_score - score)
                probability_counts.append(len(pp))
        # Self-outperformance is exactly impossible, not a classifier to validate.
        row["benchmark_self_comparison"] = asset == benchmark
        intervals = paired.interval_low.notna() & paired.interval_high.notna()
        covered = (paired.actual_return >= paired.interval_low) & (
            paired.actual_return <= paired.interval_high
        )
        row["interval_origins"] = int(intervals.sum())
        row["interval_empirical_coverage"] = (
            float(covered[intervals].mean()) if intervals.any() else np.nan
        )
        coverage_low, coverage_high, _ = block_inference(
            covered[intervals].to_numpy(dtype=float),
            config.bootstrap_draws,
            config.bootstrap_block_origins,
            config.seed + 10,
        )
        row["interval_coverage_ci_low"] = coverage_low
        row["interval_coverage_ci_high"] = coverage_high
        row["interval_mean_width"] = (
            float((paired.interval_high - paired.interval_low)[intervals].mean())
            if intervals.any()
            else np.nan
        )
        p_values = []
        for index, improvements in enumerate(gains):
            low, high, p = block_inference(
                improvements,
                config.bootstrap_draws,
                config.bootstrap_block_origins,
                config.seed + index,
            )
            row[f"gain_{index}_mean_ci_low"] = low
            row[f"gain_{index}_mean_ci_high"] = high
            p_values.append(p)
        row["joint_p_value"] = max(p_values)  # Intersection-union: all required endpoints improve.
        row["adequate_oos"] = bool(
            len(paired) >= config.min_oos_origins
            and min(probability_counts, default=0) >= config.min_oos_origins
            and row["interval_origins"] >= config.min_oos_origins
        )
        rows.append(row)
        for regime, subset in paired.groupby("regime"):
            regime_rows.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "model": model,
                    "regime": regime,
                    "oos_origins": len(subset),
                    "mae": float(abs(subset.actual_return - subset.expected_return_gbp).mean()),
                    "baseline_mae": float(
                        abs(subset.actual_return - subset.expected_return_gbp_base).mean()
                    ),
                    "status": "insufficient evidence: descriptive regime subgroup only",
                }
            )
    result = pd.DataFrame(rows)
    expected = run.latest[keys].drop_duplicates()
    if result.empty:
        result = expected.copy()
    else:
        result = expected.merge(result, how="left", on=keys, validate="one_to_one")
    for col, default in {
        "joint_p_value": 1.0,
        "adequate_oos": False,
        "interval_empirical_coverage": np.nan,
        "support_fraction": 0.0,
    }.items():
        if col not in result:
            result[col] = default
        result[col] = result[col].fillna(default) if not pd.isna(default) else result[col]
    candidates = result.model != "historical_average"
    result["holm_p_value"] = 1.0
    result.loc[candidates, "holm_p_value"] = holm_adjust(
        result.loc[candidates, "joint_p_value"].to_numpy()
    )
    statuses = []
    for row in result.itertuples():
        if row.model == "historical_average":
            status = "baseline reference; no skill claim"
        elif source != "yahoo":
            status = "insufficient evidence: synthetic/test data cannot establish market skill"
        elif not row.adequate_oos:
            status = "insufficient evidence: too few paired OOS or interval observations"
        elif row.support_fraction < 0.8:
            status = "insufficient evidence: conditional regime/class support inadequate"
        elif (
            not np.isfinite(row.interval_empirical_coverage)
            or row.interval_coverage_ci_low < config.interval_coverage - 0.1
        ):
            status = "insufficient evidence: interval coverage low or uncertain"
        elif row.holm_p_value > config.family_alpha:
            status = (
                "insufficient evidence: no adjusted improvement over return/probability baselines"
            )
        else:
            status = "retrospective OOS screen passed; prospective validation still required"
        statuses.append(status)
    result["validation_status"] = statuses
    return result, pd.DataFrame(calibration_rows), pd.DataFrame(regime_rows)
