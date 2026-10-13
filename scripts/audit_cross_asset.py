"""Descriptive stability audit, never a parameter-fitting or forecast procedure."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from aycapital.cross_asset.backtest import walk_forward
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.market import load_market
from aycapital.cross_asset.signals import calculate_signals


def persistence(data: pd.DataFrame) -> pd.Series:
    ranks = data.rank(axis=1)
    return ranks.corrwith(ranks.shift(), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cross_asset.yaml", type=Path)
    parser.add_argument("--start", default="2020-10-13")
    parser.add_argument("--end", default="2026-10-13")
    parser.add_argument("--output", default="docs/audit/version_0_2", type=Path)
    args = parser.parse_args()
    config = ResearchConfig.load(args.config)
    panel = load_market(config, args.start, args.end)
    signals = calculate_signals(panel.prices, config.benchmark, config.ranking_weights)
    score = signals.score.dropna()
    momentum = signals.percentiles["momentum_60"].loc[score.index]
    risk = 1 - signals.percentiles["volatility_60"].loc[score.index]
    stats = pd.DataFrame(
        {
            "score_rank_persistence": persistence(score),
            "momentum_rank_persistence": persistence(momentum),
            "risk_rank_persistence": persistence(risk),
        }
    )
    selections = [
        set(row.sort_values(ascending=False, kind="stable").head(config.backtest["top_n"]).index)
        for _, row in score.iterrows()
    ]
    stats["top_n_retention"] = [np.nan] + [
        len(a & b) / config.backtest["top_n"] for a, b in zip(selections, selections[1:])
    ]
    for horizon in (20, 40):
        # Signal at t; hypothetical entry t+1 close; exit t+1+h close.
        forward = panel.prices.shift(-(horizon + 1)) / panel.prices.shift(-1) - 1
        forward_rank = forward.rank(axis=1).loc[score.index]
        ic = score.rank(axis=1).corrwith(forward_rank, axis=1)
        endpoints = pd.Series(panel.prices.index, index=panel.prices.index).shift(-(horizon + 1))
        within_year = endpoints.loc[score.index].dt.year.eq(score.index.year)
        stats[f"forward_{horizon}_session_rank_ic"] = ic.where(within_year)
    for change, description in [
        (0.8, "risk_weight_20pct_lower"),
        (1.2, "risk_weight_20pct_higher"),
    ]:
        weights = dict(config.ranking_weights)
        for metric in ["volatility_60", "rolling_max_drawdown"]:
            weights[metric] *= change
        alternative = calculate_signals(panel.prices, config.benchmark, weights).score.loc[
            score.index
        ]
        selected = [
            set(
                row.sort_values(ascending=False, kind="stable").head(config.backtest["top_n"]).index
            )
            for _, row in alternative.iterrows()
        ]
        stats[f"{description}_top_n_overlap"] = [
            len(a & b) / config.backtest["top_n"] for a, b in zip(selections, selected)
        ]
    yearly = []
    simple = panel.prices.pct_change(fill_method=None)
    for year, group in stats.groupby(stats.index.year):
        record = {
            "year": year,
            "first_signal": str(group.index[0].date()),
            "last_signal": str(group.index[-1].date()),
            "signal_sessions": len(group),
        }
        record.update(group.mean().to_dict())
        r = simple.loc[group.index, config.benchmark]
        record["benchmark_observed_return"] = float((1 + r).prod() - 1)
        record["benchmark_observed_volatility"] = float(r.std() * np.sqrt(252))
        for horizon in (20, 40):
            record[f"forward_{horizon}_session_ic_observations"] = int(
                group[f"forward_{horizon}_session_rank_ic"].count()
            )
        yearly.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    regime = pd.DataFrame(yearly).set_index("year")
    regime.to_csv(args.output / "regime_diagnostics.csv")
    metrics = pd.DataFrame(
        {name: data.loc[score.index].rank(axis=1).stack() for name, data in signals.metrics.items()}
    )
    correlation = metrics.corr()
    correlation.to_csv(args.output / "metric_rank_correlations.csv")
    costs = []
    for bps in (0, 10, 25, 50):
        settings = dict(config.backtest, transaction_cost_bps=bps)
        bt = walk_forward(panel, signals, config.benchmark, **settings)
        table = bt.statistics.reset_index()
        table["cost_bps"] = bps
        costs.append(table)
    pd.concat(costs).to_csv(args.output / "cost_sensitivity.csv", index=False)
    provenance = {
        "market": panel.metadata,
        "configuration": asdict(config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "classification": "Calendar-year diagnostic slices, not fitted macro regimes",
        "forward_ic": "Ex-post cross-sectional Spearman; overlapping 20/40-session "
        "windows, no significance or independence claim",
        "caveat": "Descriptive audit; post-hoc sensitivity is not OOS validation. "
        "Warmup excludes early sample. 2026 is incomplete.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(regime.round(4).to_string())
    print(
        "Momentum60/relative-strength rank correlation:",
        correlation.loc["momentum_60", "relative_strength"],
    )


if __name__ == "__main__":
    main()
