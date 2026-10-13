"""CLI for reproducible cross-asset research. Live failures never trigger simulation."""

import argparse
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from aycapital.cross_asset.backtest import walk_forward
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.logging import event
from aycapital.cross_asset.macro import FredProvider, MacroData
from aycapital.cross_asset.market import load_market, synthetic_panel
from aycapital.cross_asset.report import export_report
from aycapital.cross_asset.signals import calculate_signals


def run_research(
    config: ResearchConfig,
    start: str,
    end: str,
    output: Path,
    *,
    source: str = "yahoo",
    include_macro: bool = False,
    refresh: bool = False,
) -> dict:
    if source == "synthetic":
        panel = synthetic_panel(config, start, end)
    elif source == "yahoo":
        panel = load_market(config, start, end, refresh=refresh)
    else:
        raise ValueError("Unknown market source")
    signals = calculate_signals(
        panel.prices, config.benchmark, config.ranking_weights, config.backtest["periods_per_year"]
    )
    backtest = walk_forward(panel, signals, config.benchmark, **config.backtest)
    macro: dict[str, MacroData] = {}
    if include_macro:
        if source == "synthetic":
            raise ValueError("Request real macro data separately from a synthetic market demo")
        fred = FredProvider(config.cache_dir)
        macro_end = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for series in config.macro["series"]:
            macro[series] = fred.load(
                series, start, macro_end, macro_end, mode=config.macro["mode"], refresh=refresh
            )
    metadata = export_report(config, panel, signals, backtest, output, macro)
    event(
        logging.INFO,
        "research_complete",
        output=str(output),
        source=source,
        latest_session=metadata["signal_session"],
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/cross_asset.yaml"))
    parser.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    parser.add_argument("--start", help="Defaults to history_years before end")
    parser.add_argument("--end", help="Exclusive; defaults to today's Europe/London date")
    parser.add_argument("--output", type=Path, default=Path("outputs/cross_asset"))
    parser.add_argument("--macro", action="store_true", help="Retrieve FRED charts; needs API key")
    parser.add_argument("--refresh", action="store_true", help="Download new immutable snapshots")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = ResearchConfig.load(args.config)
        end = args.end or pd.Timestamp.now(tz=ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
        start = args.start or (
            pd.Timestamp(end) - pd.DateOffset(years=config.history_years)
        ).strftime("%Y-%m-%d")
        if pd.Timestamp(start) > pd.Timestamp(end) - pd.DateOffset(years=5):
            raise ValueError("Research requests must cover at least five calendar years")
        result = run_research(
            config,
            start,
            end,
            args.output,
            source=args.source,
            include_macro=args.macro,
            refresh=args.refresh,
        )
    except Exception as exc:
        event(logging.ERROR, "research_failed", error_type=type(exc).__name__, message=str(exc))
        parser.exit(1, f"Research failed: {exc}\n")
    print(f"{args.source.upper()} research report: {args.output / 'report.html'}")
    print("Latest signal session:", result["signal_session"])
    print("Historical evidence only; no calibrated 4–8 week forecast or competition eligibility.")


if __name__ == "__main__":
    main()
