"""Run the frozen allocation study against an existing GBP snapshot, never a holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from aycapital.allocation_research.evaluation import paired_uncertainty, regime_flags, sharpe
from aycapital.allocation_research.macro import PublicVintageProvider, monthly_features
from aycapital.allocation_research.strategies import targets_for_horizon
from aycapital.cross_asset.backtest import allocation_history
from aycapital.cross_asset.market import MarketPanel, session_schedule, validate_panel
from aycapital.forecasting.uncertainty import holm_adjust
from aycapital.statistics import summary

REGISTERED_PLAN_SHA256 = "dbb12323befce5d59b91dbbd421f96130468c3736c284009e2760d897e597e70"
REGISTERED_HISTORICAL_END = pd.Timestamp("2026-10-14")

CANDIDATES = ["inverse_volatility", "risk_parity", "trend_following"]


def load_historical(directory: Path, plan: dict) -> tuple[MarketPanel, pd.DataFrame, dict]:
    """Reject holdout dates BEFORE loading any price or FX values."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("currency") != "GBP":
        raise ValueError("Need an existing GBP scenario snapshot")
    end = pd.Timestamp(plan["historical_end"])
    if end != REGISTERED_HISTORICAL_END or plan["prospective_start"] != "2026-10-15":
        raise ValueError("Registered holdout boundaries cannot be changed in a historical run")
    if end >= pd.Timestamp(plan["prospective_start"]):
        raise ValueError("Historical research and prospective holdout overlap")
    dates = pd.read_csv(directory / "gbp_prices.csv", usecols=[0]).iloc[:, 0]
    dates = pd.to_datetime(dates)
    if dates.max() > end or pd.Timestamp(metadata["last_observation"]) > end:
        raise ValueError("Sealed prospective period: refusing to load future price values")
    prices = pd.read_csv(directory / "gbp_prices.csv", index_col=0, parse_dates=True)
    fx_dates = pd.to_datetime(pd.read_csv(directory / "fx_alignment.csv", usecols=[0]).iloc[:, 0])
    if fx_dates.max() > end:
        raise ValueError("Sealed prospective period: refusing to load future FX values")
    alignment = pd.read_csv(directory / "fx_alignment.csv", index_col=0, parse_dates=True)
    if not alignment.index.equals(prices.index):
        raise ValueError("FX and asset dates must match exactly")
    closes = pd.to_datetime(alignment.asset_close_utc, utc=True)
    available = pd.to_datetime(alignment.fx_available_at_utc, utc=True)
    quote_dates = pd.to_datetime(alignment.fx_observation_date, utc=True)
    ages = (closes - quote_dates).dt.total_seconds() / 86400
    if (
        not np.isfinite(alignment.usd_per_gbp).all()
        or (alignment.usd_per_gbp <= 0).any()
        or (available > closes).any()
        or (quote_dates > available).any()
        or (ages > 6).any()
        or (ages < 0).any()
    ):
        raise ValueError("Invalid or unavailable GBP FX alignment")
    if not np.allclose(ages, alignment.fx_age_calendar_days):
        raise ValueError("Incorrect FX age metadata")
    panel = validate_panel(MarketPanel(prices, closes, metadata["market"]))
    # Reconstruct original USD adjusted closes using the SAME recorded conversion.
    native = prices.mul(alignment.usd_per_gbp, axis=0)
    return panel, native, metadata


def run_comparison(
    panel: MarketPanel, native: pd.DataFrame, macro: pd.DataFrame, plan: dict
) -> dict[str, pd.DataFrame]:
    """Sequential fixed-rule historical evidence; no model fitting or tuning."""
    first_execution = max(plan["estimation_sessions"], plan["trend_sessions"] - 1) + 1
    if not native.index.equals(panel.prices.index) or not native.columns.equals(
        panel.prices.columns
    ):
        raise ValueError("Native prices must match GBP observation labels")
    flags = regime_flags(panel.prices, macro, plan).shift(1)
    tables = {
        name: []
        for name in [
            "performance",
            "daily_returns",
            "turnover",
            "trades",
            "holding_returns",
            "rolling",
            "regimes",
            "uncertainty",
        ]
    }
    first_index = panel.prices.index[first_execution:]
    for horizon in plan["holding_sessions"]:
        targets = targets_for_horizon(panel.prices, plan, horizon)
        for cost in plan["cost_bps"]:
            daily, turnover, trades = allocation_history(
                panel, targets, plan["benchmark"], first_execution, horizon, cost
            )
            native_panel = MarketPanel(
                native, panel.closes_utc, {**panel.metadata, "currency": "USD"}
            )
            native_daily, _, _ = allocation_history(
                native_panel, targets, plan["benchmark"], first_execution, horizon, cost
            )
            performance = summary(daily)
            performance["zero_hurdle_sharpe"] = sharpe(daily.to_numpy())
            performance["cagr"] = (1 + daily).prod() ** (252 / len(daily)) - 1
            performance["annualized_turnover"] = turnover.sum() * 252 / len(daily)
            performance["native_usd_cumulative_return_same_targets"] = (1 + native_daily).prod() - 1
            performance["gbp_minus_native_total_return"] = (1 + daily).prod() - (
                1 + native_daily
            ).prod()
            performance["total_cost_fraction_sum"] = trades.groupby("portfolio").cost_fraction.sum()
            performance["observations"] = len(daily)
            # Each full holding block starts after a fill and ends at the next
            # fill, inclusive of that rebalancing fee. Entry fee is reported in
            # daily NAV but excluded from holding-block samples.
            complete = (len(daily) - 1) // horizon
            performance["complete_holding_blocks"] = complete
            for frame_name, frame in [
                ("performance", performance.reset_index(names="portfolio")),
                ("daily_returns", daily.reset_index(names="session")),
                ("turnover", turnover.reset_index(names="session")),
                ("trades", trades),
            ]:
                tables[frame_name].append(frame.assign(holding_sessions=horizon, cost_bps=cost))
            blocks = []
            for left in range(1, 1 + complete * horizon, horizon):
                block = (1 + daily.iloc[left : left + horizon]).prod() - 1
                blocks.append(
                    {
                        "start": daily.index[left],
                        "end": daily.index[left + horizon - 1],
                        **block.to_dict(),
                    }
                )
            tables["holding_returns"].append(
                pd.DataFrame(blocks).assign(holding_sessions=horizon, cost_bps=cost)
            )
            rolling_return = (1 + daily).rolling(252).apply(np.prod, raw=True) - 1
            rolling_sharpe = daily.rolling(252).mean() / daily.rolling(252).std() * np.sqrt(252)
            # Full 252-day window drawdown includes starting NAV of one.
            rolling_dd = daily.rolling(252).apply(
                lambda x: np.min(
                    np.r_[1.0, np.cumprod(1 + x)]
                    / np.maximum.accumulate(np.r_[1.0, np.cumprod(1 + x)])
                    - 1
                ),
                raw=True,
            )
            for name in daily:
                tables["rolling"].append(
                    pd.DataFrame(
                        {
                            "session": first_index,
                            "portfolio": name,
                            "return_252": rolling_return[name].to_numpy(),
                            "zero_hurdle_sharpe_252": rolling_sharpe[name].to_numpy(),
                            "max_drawdown_252": rolling_dd[name].to_numpy(),
                            "turnover_252": turnover[name].rolling(252).sum().to_numpy(),
                            "excess_return_252_equal_weight": (
                                rolling_return[name] - rolling_return.equal_weight
                            ).to_numpy(),
                            "excess_return_252_defensive": (
                                rolling_return[name] - rolling_return.defensive
                            ).to_numpy(),
                            "holding_sessions": horizon,
                            "cost_bps": cost,
                        }
                    )
                )
            for regime in flags:
                selected = flags[regime].reindex(daily.index).fillna(False).astype(bool)
                selected_returns = daily.loc[selected]
                # Scattered regime days are NOT a continuous tradable NAV.
                for name in daily:
                    values = selected_returns[name]
                    tables["regimes"].append(
                        pd.DataFrame(
                            [
                                {
                                    "holding_sessions": horizon,
                                    "cost_bps": cost,
                                    "regime": regime,
                                    "portfolio": name,
                                    "sessions": len(values),
                                    "episodes": int(
                                        (selected & ~selected.shift(1, fill_value=False)).sum()
                                    ),
                                    "annualized_arithmetic_return": values.mean() * 252,
                                    "annualized_volatility": values.std() * np.sqrt(252),
                                    "mean_daily_excess_equal_weight": (
                                        values - selected_returns.equal_weight
                                    ).mean(),
                                    "mean_daily_excess_defensive": (
                                        values - selected_returns.defensive
                                    ).mean(),
                                    "evidence": "descriptive_only"
                                    if len(values) >= plan["minimum_regime_sessions"]
                                    else "insufficient_evidence",
                                }
                            ]
                        )
                    )
            if cost in plan["cost_bps"][:2]:
                for candidate in CANDIDATES:
                    for reference in ["equal_weight", "defensive"]:
                        for block in plan["bootstrap_block_sessions"]:
                            stats = paired_uncertainty(
                                daily[candidate],
                                daily[reference],
                                plan["bootstrap_draws"],
                                block,
                                plan["bootstrap_seed"],
                            )
                            tables["uncertainty"].append(
                                pd.DataFrame(
                                    [
                                        {
                                            "portfolio": candidate,
                                            "reference": reference,
                                            "holding_sessions": horizon,
                                            "cost_bps": cost,
                                            "bootstrap_block_sessions": block,
                                            "complete_holding_blocks": complete,
                                            **stats,
                                        }
                                    ]
                                )
                            )
    result = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}
    gates = []
    for (horizon, candidate), group in result["uncertainty"].groupby(
        ["holding_sessions", "portfolio"]
    ):
        gates.append(
            {
                "holding_sessions": horizon,
                "portfolio": candidate,
                "worst_p_value": group.p_value.max(),
                "worst_sharpe_lower": group.sharpe_low.min(),
                "complete_holding_blocks": group.complete_holding_blocks.min(),
            }
        )
    gates = pd.DataFrame(gates)
    gates["holm_p_value"] = holm_adjust(gates.worst_p_value.to_numpy())
    gates["historical_gate_pass"] = (
        (gates.holm_p_value <= plan["alpha"])
        & (gates.worst_sharpe_lower > 0)
        & (gates.complete_holding_blocks >= plan["minimum_holding_blocks"])
    )
    gates["prospective_evidence"] = "insufficient_evidence_untouched_period_not_evaluated"
    result["evidence"] = gates
    result["macro_features"] = macro.reset_index(names="session")
    result["regime_flags"] = flags.reset_index(names="session")
    return result


def write_report(tables: dict, metadata: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    performance = tables["performance"]
    selected = performance.loc[performance.cost_bps.eq(10)]
    lines = [
        "# Fixed-rule allocation research",
        "",
        "**Retrospective historical evidence, not forecasts or prospective validation.**",
        "",
        f"Evaluation: {metadata['evaluation_start']} to {metadata['last_observation']}; "
        f"source: {metadata['source']}; valuation currency GBP.",
        "",
        "| Hold | Strategy | CAGR | Volatility | Max drawdown | Zero-hurdle Sharpe | "
        "Annual traded notional | GBP minus USD total return |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples():
        lines.append(
            f"| {row.holding_sessions} | {row.portfolio} | {row.cagr:.2%} | "
            f"{row.annualized_volatility:.2%} | {row.max_drawdown:.2%} | "
            f"{row.zero_hurdle_sharpe:.3f} | {row.annualized_turnover:.2f}× | "
            f"{row.gbp_minus_native_total_return:.2%} |"
        )
    lines += [
        "",
        "10 bps per traded side shown; 25/50 bps are in performance.csv. "
        "Turnover is buys plus sells, including initial cash entry. "
        "GBP minus USD is a total-return percentage-point difference with identical "
        "targets, not an additive FX P&L decomposition or hedged return.",
        "",
        "## Prespecified evidence gate",
        "",
        "| Hold | Candidate | Complete blocks | Worst Sharpe CI lower | Holm p | Historical gate |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in tables["evidence"].itertuples():
        lines.append(
            f"| {row.holding_sessions} | {row.portfolio} | "
            f"{row.complete_holding_blocks} | {row.worst_sharpe_lower:.3f} | "
            f"{row.holm_p_value:.3f} | {row.historical_gate_pass} |"
        )
    passes = int(tables["evidence"].historical_gate_pass.sum())
    lines += [
        "",
        f"{passes} of six candidate/horizon combinations pass the historical gate. "
        "Zero have prospective confirmation. The gate requires improvement over both "
        "equal-weight and defensive benchmarks at 10 AND 25 bps, under both 40- and "
        "80-session block bootstraps, positive lower 95% intervals, Holm family "
        "correction and at least 40 full holding blocks. See uncertainty.csv for "
        "paired return-difference intervals. These approximate tests assume local "
        "stationarity and cannot establish robustness to unseen structural breaks.",
        "",
        "## Decision",
        "",
    ]
    if not passes:
        lines += [
            "No tested dynamic allocation demonstrates a robust risk-adjusted advantage "
            "under the registered gate. Retain equal-weight with 40-session rebalancing "
            "as the simplest tested research default, chosen before results. Do not "
            "promote a point estimate or a favorable regime to predictive skill."
        ]
    else:
        lines += [
            "Historical differences justify further fixed-rule prospective research only. "
            "Retain the simple reference until the untouched evaluation is complete."
        ]
    lines += [
        "",
        "## Regime coverage and limitations",
        "",
        "Regime flags use previous-close information, including conservative macro "
        "availability. Buckets overlap; counts are sessions and episodes, not independent "
        "economic cycles. Scattered regime days are not compounded as a tradable NAV.",
        "",
        "| Regime | Sessions | Episodes | Evidence |",
        "|---|---:|---:|---|",
    ]
    regimes = tables["regimes"].query("holding_sessions == 20 and cost_bps == 10")
    for row in regimes.loc[regimes.portfolio.eq("equal_weight")].itertuples():
        lines.append(f"| {row.regime} | {row.sessions} | {row.episodes} | {row.evidence} |")
    lines += [
        "",
        "### Strategy behavior within regimes",
        "",
        "Mean daily excess returns below are basis points versus equal-weight, "
        "at 10 bps costs. They are descriptive associations, not forecasts or "
        "independent regime tests. Zero-session buckets cannot support conclusions.",
        "",
        "| Hold | Regime | Strategy | Excess bps/day | Daily-risk annualized |",
        "|---:|---|---|---:|---:|",
    ]
    regime_rows = tables["regimes"].loc[
        tables["regimes"].cost_bps.eq(10)
        & tables["regimes"].portfolio.isin(CANDIDATES + ["defensive"])
        & tables["regimes"].sessions.gt(0)
    ]
    for row in regime_rows.itertuples():
        lines.append(
            f"| {row.holding_sessions} | {row.regime} | {row.portfolio} | "
            f"{row.mean_daily_excess_equal_weight * 10000:.2f} | "
            f"{row.annualized_volatility:.2%} |"
        )
    lines += ["", "Macro warnings:"] + [f"- {warning}" for warning in metadata["macro_warnings"]]
    if not metadata["macro_warnings"]:
        lines.append(
            "- No vintage retrieval failures. See macro_features.csv for actual "
            "observation, knowledge and availability dates; unavailable states remain missing."
        )
    lines += ["", "Data/method limitations:"] + [f"- {warning}" for warning in metadata["warnings"]]
    lines += [
        "",
        "The prospective holdout starts 2026-10-15 and remains unscored. Its first "
        "252 sessions are sealed; even that provides too few independent 20/40-day "
        "blocks for strong confirmation. Do not tune on interim results. A future "
        "explicit reviewed change is required to unlock evaluation. See the frozen "
        "allocation_plan.md and configuration commit ef17a02.",
        "",
        "Macro source: [ALFRED historical-vintage help](https://alfred.stlouisfed.org/help). "
        "This study uses delayed month-end knowledge states, not observation-date alignment "
        "or guessed initial publication lags. Inflation and unemployment changes compare "
        "observations within the same vintage. NBER hindsight recession dates are not used.",
        "",
        "Competition eligibility is unverified. None of these proxy allocations is "
        "an executable UMushroom recommendation. SGOV retains USD/GBP exposure; "
        "equal ETF weights also overweight overlapping equity/credit exposures.",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/scenarios"))
    parser.add_argument("--plan", type=Path, default=Path("config/allocation_research.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/allocation_research"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Explicit incomplete-scope run; unsupported regimes remain unknown",
    )
    args = parser.parse_args()
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != REGISTERED_PLAN_SHA256:
        raise ValueError("Unregistered plan changes: register a separate study before analysis")
    plan = yaml.safe_load(args.plan.read_text())
    panel, native, source = load_historical(args.input, plan)
    frames, provenance, macro_warnings = [], [], []
    provider = PublicVintageProvider(args.cache)
    for series in plan["macro_series"]:
        if args.skip_macro:
            macro_warnings.append(
                f"{series}: intentionally not retrieved; regime evidence incomplete"
            )
            continue
        # Retrieval failures are explicit and never replaced with latest/revised data.
        try:
            frame, records = provider.load(
                series, str(panel.prices.index[0].date()), plan["historical_end"]
            )
            frames.append(frame)
            provenance.extend(records)
        except (ValueError, requests.RequestException) as exc:
            macro_warnings.append(f"{series}: vintage retrieval failed ({type(exc).__name__})")
    snapshots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    macro = (
        monthly_features(snapshots, panel.closes_utc)
        if frames
        else pd.DataFrame(index=panel.prices.index)
    )
    tables = run_comparison(panel, native, macro, plan)
    first = max(plan["estimation_sessions"], plan["trend_sessions"] - 1) + 1
    metadata = {
        "study": plan["study"],
        "source": source["source"],
        "currency": "GBP",
        "historical_evidence_only": True,
        "evaluation_start": str(panel.prices.index[first].date()),
        "first_observation": str(panel.prices.index[0].date()),
        "last_observation": str(panel.prices.index[-1].date()),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "prospective_start": plan["prospective_start"],
        "prospective_status": "sealed_unscored",
        "prospective_min_sessions": plan["prospective_min_sessions"],
        "prospective_last_session": str(
            session_schedule("XNYS", plan["prospective_start"], "2028-01-01")
            .index[plan["prospective_min_sessions"] - 1]
            .date()
        ),
        "plan": plan,
        "protocol_commit": "ef17a02",
        "input_sha256": {
            name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
            for name in ["gbp_prices.csv", "fx_alignment.csv", "metadata.json"]
        },
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "macro_provenance": provenance,
        "macro_warnings": macro_warnings,
        "warnings": [
            *[warning for warning in source["warnings"] if "Scenario shocks" not in warning],
            "No fresh holdout has been evaluated; all results are retrospective.",
            "Cost assumptions are unverified; no measured spread, impact or capacity.",
            "No taxes, securities lending, custody, FX entry/exit fees or terminal liquidation.",
            "Zero-hurdle Sharpe is not excess performance over investable GBP cash.",
            "Only one short common-history sample; regimes are overlapping diagnostics.",
            "Monthly vintage snapshots delay availability; not exact initial release times.",
            "Current-vendor adjusted histories are not point-in-time corporate-action data.",
        ],
    }
    if source["source"] == "synthetic":
        tables["evidence"]["historical_gate_pass"] = False
        metadata["warnings"].append("SYNTHETIC ILLUSTRATION: no empirical evidence.")
    write_report(tables, metadata, args.output)
    if frames:
        snapshots.to_csv(
            args.output / "macro_snapshots.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    print((args.output / "report.md").read_text())


if __name__ == "__main__":
    main()
