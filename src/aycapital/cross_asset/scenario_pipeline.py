"""Build GBP evidence and hypothetical scenario artifacts for the dashboard."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from aycapital.cross_asset.benchmarks import build_benchmarks
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.fx import FxData, gbp_panel, load_fx
from aycapital.cross_asset.market import load_market, synthetic_panel
from aycapital.cross_asset.scenarios import ScenarioConfig, scenario_returns


def run_scenarios(
    research: ResearchConfig,
    scenarios: ScenarioConfig,
    start: str,
    end: str,
    output: Path,
    *,
    source: str = "yahoo",
    refresh: bool = False,
) -> dict:
    fx_start = (pd.Timestamp(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    if source == "yahoo":
        panel = load_market(research, start, end, refresh=refresh)
        fx = load_fx(fx_start, end, str(research.cache_dir / "fx"), refresh=refresh)
    elif source == "synthetic":
        panel = synthetic_panel(research, start, end)
        dates = pd.bdate_range(fx_start, pd.Timestamp(end) - pd.Timedelta(days=1))
        rates = pd.Series(1.25 * np.exp(0.00005 * np.arange(len(dates))), index=dates)
        times = pd.Series(dates.tz_localize("UTC") + pd.Timedelta(days=1), index=dates)
        fx = FxData(rates, times, {"source": "synthetic", "warning": "Synthetic FX illustration"})
    else:
        raise ValueError("Unsupported market source")
    converted, alignment = gbp_panel(panel, fx, scenarios.max_fx_age_days)
    report = build_benchmarks(converted, research, scenarios.portfolios)
    stress_assets, stress_portfolios = scenario_returns(scenarios, converted.prices.columns)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "gbp_prices": converted.prices,
        "fx_alignment": alignment,
        "historical_gbp_metrics": report.metrics,
        "historical_equity": report.equity,
        "historical_portfolio_returns": report.portfolio_returns,
        "historical_portfolio_risk": report.portfolio_risk,
        "historical_turnover": report.turnover,
        "correlation_60": report.correlation,
        "covariance_60": report.covariance,
        "historical_scores": report.signals.score,
        "hypothetical_asset_returns": stress_assets,
        "hypothetical_portfolio_returns": stress_portfolios,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv")
    report.trades.to_csv(output / "historical_paper_trades.csv", index=False)
    (output / "scenarios.yaml").write_text(
        yaml.safe_dump({"kind": "hypothetical", **asdict(scenarios)}, sort_keys=False)
    )
    warnings = list(converted.metadata["data_quality_warnings"]) + [
        "Vendor adjusted prices approximate total returns and can be revised.",
        "Static proxy universe; common history excludes unavailable leading observations.",
        "Historical portfolio comparisons are ex-post illustrations, not OOS validation.",
        "Scenario shocks are manually chosen assumptions, not forecasts or calibrated estimates.",
        "UMushroom instrument availability, eligibility and rules remain unverified.",
    ]
    excluded_fx = fx.metadata.get("excluded_out_of_range_dates", [])
    if excluded_fx:
        warnings.append(
            f"Provider returned {len(excluded_fx)} FX date(s) outside the requested "
            "interval; raw quotes retained and explicitly excluded from conversion."
        )
    missing_fx = fx.metadata.get("missing_observation_dates", [])
    if missing_fx:
        warnings.append(
            f"Provider FX contains {len(missing_fx)} missing quote dates; "
            "raw cache retains them, and matching uses only valid dated quotes."
        )
    if report.metrics.score.isna().any():
        warnings.append(
            "Some historical scores are undefined (warmup or zero volatility); "
            "they are left missing, not imputed."
        )
    latest = alignment.iloc[-1]
    metadata = {
        "source": source,
        "currency": "GBP",
        "market": converted.metadata,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "first_observation": str(converted.prices.index[0].date()),
        "last_observation": str(converted.prices.index[-1].date()),
        "asset_close_utc": latest.asset_close_utc.isoformat(),
        "fx_observation_date": str(latest.fx_observation_date.date()),
        "fx_available_at_utc": latest.fx_available_at_utc.isoformat(),
        "warnings": warnings,
        "research_configuration": asdict(research),
        "scenario_configuration": asdict(scenarios),
        "scenario_type": "Hypothetical terminal shock; no empirical calibration",
        "calibration_status": "Future enhancement requiring historical regimes and OOS validation",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/cross_asset.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/scenarios.yaml"))
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", type=Path, default=Path("outputs/scenarios"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        research = ResearchConfig.load(args.config)
        scenarios = ScenarioConfig.load(args.scenarios, pd.Index(research.assets))
        end = args.end or pd.Timestamp.now(tz=ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
        start = args.start or (
            pd.Timestamp(end) - pd.DateOffset(years=research.history_years)
        ).strftime("%Y-%m-%d")
        metadata = run_scenarios(
            research, scenarios, start, end, args.output, source=args.source, refresh=args.refresh
        )
    except Exception as exc:
        parser.exit(1, f"Scenario report failed: {exc}\n")
    print(f"{args.source.upper()} GBP evidence through {metadata['last_observation']}")
    print(f"Hypothetical scenarios and saved dashboard artifacts: {args.output}")


if __name__ == "__main__":
    main()
