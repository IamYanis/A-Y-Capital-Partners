"""Saved GBP snapshot -> forecasts, prequential validation and an honest skill report."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.macro import FredProvider, MacroData
from aycapital.cross_asset.market import MarketPanel, session_schedule, validate_panel
from aycapital.forecasting.config import ForecastConfig
from aycapital.forecasting.engine import run_forecasts
from aycapital.forecasting.features import build_features
from aycapital.forecasting.validation import evaluate


def _report(evaluation: pd.DataFrame, metadata: dict) -> str:
    passed = evaluation[evaluation.validation_status.str.startswith("retrospective OOS screen")]
    lines = [
        "# Version 0.4 forecast-validation report",
        "",
        f"Decision snapshot: **{metadata['decision_session']}**. Source: **{metadata['source']}**.",
        f"Feature scope: **{metadata['feature_scope']}**.",
        "",
        "## Credible skill assessment",
        "",
        "Retrospective research only; no prospective validation or investment recommendation.",
        "Design settings were fixed for this run, but the history has already "
        "been inspected in earlier project work. A fresh future holdout is still required.",
        "",
    ]
    if passed.empty:
        lines.append(
            "**No asset-class/horizon/model combination demonstrated credible skill under the "
            "configured joint, multiplicity-adjusted screen.**"
        )
    else:
        lines.append("The following combinations pass the retrospective statistical screen only:")
        lines.append("| Asset | Sessions | Model | Holm-adjusted p |")
        lines.append("|---|---:|---|---:|")
        for row in passed.itertuples():
            lines.append(f"| {row.asset} | {row.horizon} | {row.model} | {row.holm_p_value:.5f} |")
        lines.append(
            "These are candidates for fresh prospective testing, not validated trading forecasts."
        )
    lines += [
        "",
        "## Evidence by horizon and model",
        "",
        "| Horizon | Model | Asset combinations | Adequate OOS | Screen passed |",
        "|---:|---|---:|---:|---:|",
    ]
    for (horizon, model), group in evaluation.groupby(["horizon", "model"]):
        lines.append(
            f"| {horizon} | {model} | {len(group)} | {int(group.adequate_oos.sum())} | "
            f"{int(group.validation_status.str.startswith('retrospective OOS screen').sum())} |"
        )
    lines += [
        "",
        "## Design and safeguards",
        "",
        "- Targets: GBP simple returns from next-session-close entry to exit after h sessions.",
        "- Purged expanding/rolling fits: label endpoints precede the decision session.",
        "- Scalers, models and regime volatility thresholds fit on eligible training rows only.",
        "- OOS origins are spaced h sessions apart; label return intervals do not overlap.",
        "- Models: regime-conditional historical averages, fixed Ridge and logistic regressions.",
        "- Mean/frequency baselines use the same matured training set.",
        "- Sparse regime/class support triggers explicit fallback and insufficient evidence flags.",
        "- Intervals use previously matured, nonoverlapping OOS residuals only.",
        "- Approximate residual intervals assume stability; coverage is not guaranteed.",
        "- Paired moving-block bootstrap retains serial dependence across evaluation origins. "
        "MAE and both Brier losses must improve; self-outperformance is exactly zero.",
        "- Holm correction includes all configured asset/horizon/model candidates.",
        "- Regime subgroup metrics are descriptive, with no separate skill claim.",
        "- Calibration tables, MAE, directional accuracy, Brier losses, benchmark-excess MAE, "
        "interval coverage/width and paired confidence intervals are exported in CSV.",
        "- Forecast accuracy is not strategy P&L; a trading policy needs separate validation.",
        "",
        "## Limitations and unavailable inputs",
        "",
    ]
    lines += [f"- {warning}" for warning in metadata["warnings"]]
    lines += [
        "",
        "Full per-asset/horizon/model conclusions: validation.csv. "
        "Regime diagnostics: regime_validation.csv. Calibration: calibration.csv.",
        "Research estimates: latest_forecasts.csv. Predictions: oos_predictions.csv.",
        "Input hashes, source provenance, design configuration and feature names: metadata.json.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run --locked aycapital-forecast --input outputs/scenarios",
        "```",
        "",
        "A different snapshot or changed configuration is a different experiment. "
        "After tuning on these results, the same OOS outcomes are no longer untouched.",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline(
    input_dir: Path,
    output: Path,
    config: ForecastConfig,
    *,
    macro: dict[str, MacroData] | None = None,
) -> dict:
    metadata = json.loads((input_dir / "metadata.json").read_text())
    if metadata.get("currency") != "GBP":
        raise ValueError("Use an existing GBP benchmark/scenario snapshot")
    research = ResearchConfig.from_mapping(metadata["research_configuration"])
    prices = pd.read_csv(input_dir / "gbp_prices.csv", index_col=0, parse_dates=True)
    alignment = pd.read_csv(input_dir / "fx_alignment.csv", index_col=0, parse_dates=True)
    if list(prices.columns) != list(research.assets):
        raise ValueError("Saved prices and configured research universe differ")
    closes = session_schedule(
        research.calendar,
        str(prices.index[0].date()),
        str((prices.index[-1] + pd.Timedelta(days=1)).date()),
    )
    panel = validate_panel(
        MarketPanel(prices, closes, {"calendar": research.calendar, "currency": "GBP"})
    )
    warnings = list(metadata.get("warnings", []))
    macro = {} if config.macro_policy == "off" else dict(macro or {})
    if config.macro_policy != "off" and not macro and metadata["source"] == "yahoo":
        provider = FredProvider(research.cache_dir)
        for series in config.macro_series:
            try:
                macro[series] = provider.load(
                    series,
                    str(prices.index[0].date()),
                    str(prices.index[-1].date()),
                    str(prices.index[-1].date()),
                    mode="vintages",
                )
            except ValueError:
                warnings.append(
                    f"{series}: vintage retrieval unavailable; not replaced with revised history."
                )
    missing = set(config.macro_series) - set(macro)
    if config.macro_policy == "require" and missing:
        raise ValueError("Required vintage macro data unavailable: " + ", ".join(sorted(missing)))
    features = build_features(panel, alignment, research.benchmark, macro, config.macro_series)
    warnings += list(features.warnings)
    run = run_forecasts(prices, features, research.benchmark, config)
    validation, calibration, regimes = evaluate(run, config, research.benchmark, metadata["source"])
    validation["asset_class"] = validation.asset.map(
        {asset: info["asset_class"] for asset, info in research.assets.items()}
    )
    latest = run.latest.merge(
        validation[["asset", "horizon", "model", "validation_status"]],
        on=["asset", "horizon", "model"],
        how="left",
        validate="one_to_one",
    )
    for name in (
        "expected_return_gbp",
        "expected_excess_return",
        "probability_positive",
        "probability_outperform",
        "interval_low",
        "interval_high",
    ):
        if name not in latest:
            latest[name] = np.nan
    latest["current_evidence_status"] = latest.validation_status
    insufficient = (
        ~latest.model_supported.astype(bool)
        | (latest.calibration_origins < config.min_calibration_origins)
        | latest.expected_return_gbp.isna()
    )
    latest.loc[insufficient, "current_evidence_status"] = (
        "insufficient evidence: current training/regime/interval support"
    )
    latest["instrument_role"] = "Research proxy; UMushroom availability/eligibility unverified"
    latest["asset_class"] = latest.asset.map(
        {a: v["asset_class"] for a, v in research.assets.items()}
    )
    future = session_schedule(
        research.calendar,
        str(prices.index[-1].date()),
        str((prices.index[-1] + pd.Timedelta(days=180)).date()),
    )
    future = future[future.index > prices.index[-1]]
    latest["entry_session"] = str(future.index[0].date())
    latest["exit_session"] = latest.horizon.map(lambda h: str(future.index[h].date()))
    latest["available_after_utc"] = closes.iloc[-1].isoformat()
    warnings += [
        "Current vendor history is not a point-in-time corporate-action dataset.",
        "GBP targets inherit the disclosed asynchronous FX valuation convention.",
        "Prior audits inspected this history; a fresh prospective holdout is still needed.",
        "Bootstrap inference and residual intervals can fail under changing regimes.",
        "No trading, scenario probability or competition-eligibility claim is made.",
    ]
    if (
        config.bootstrap_draws + 1
        < 3 * len(prices.columns) * len(config.horizons) / config.family_alpha
    ):
        warnings.append(
            "Bootstrap resolution is too low for the most conservative Holm threshold; "
            "no strong significance claim is possible at that threshold."
        )
    result = {
        "version": "0.4.0",
        "source": metadata["source"],
        "decision_session": str(prices.index[-1].date()),
        "feature_scope": features.scope,
        "configuration": asdict(config),
        "feature_names": list(next(iter(features.assets.values())).columns),
        "input_provenance": metadata,
        "input_sha256": {
            name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
            for name in ("gbp_prices.csv", "fx_alignment.csv", "metadata.json")
        },
        "macro_provenance": {name: data.metadata for name, data in macro.items()},
        "warnings": warnings,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "latest_forecasts": latest,
        "oos_predictions": run.oos,
        "validation": validation,
        "calibration": calibration,
        "regime_validation": regimes,
        "predictive_samples": run.distributions,
        "conditional_regime_forecasts": run.regimes,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(result, indent=2, default=str))
    (output / "validation_report.md").write_text(_report(validation, result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/scenarios"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/forecasting.yaml"))
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.input, args.output or args.input / "forecasts", ForecastConfig.load(args.config)
        )
    except Exception as exc:
        parser.exit(1, f"Forecast research failed: {exc}\n")
    print(f"{result['source'].upper()} research forecasts as of {result['decision_session']}")
    print("Validation report:", (args.output or args.input / "forecasts") / "validation_report.md")
    print("Failed baselines or inadequate samples mean insufficient evidence.")


if __name__ == "__main__":
    main()
