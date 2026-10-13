"""Saved GBP evidence and hypothetical scenario explorer; no trading or auto-downloads."""

import copy
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from aycapital.cross_asset.benchmarks import build_benchmarks
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.market import MarketPanel, session_schedule
from aycapital.cross_asset.scenarios import ScenarioConfig, allocation, scenario_returns

st.set_page_config(page_title="A-Y GBP Benchmarks & Scenarios", layout="wide")
st.title("Cross-Asset GBP Benchmarks & Scenarios")
st.caption("Historical evidence · Hypothetical stress tests · Research proxies")
directory = Path(st.sidebar.text_input("Saved report directory", "outputs/scenarios"))
if not (directory / "metadata.json").is_file():
    st.info("Generate data first: uv run --locked aycapital-scenarios")
    st.code(
        "uv run --locked aycapital-scenarios --source synthetic "
        "--start 2018-01-01 --end 2024-01-01 --output outputs/scenarios"
    )
    st.stop()

try:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("currency") != "GBP" or "scenario_configuration" not in metadata:
        raise ValueError("Select a GBP scenario report directory")
    research = ResearchConfig.from_mapping(metadata["research_configuration"])
    prices = pd.read_csv(directory / "gbp_prices.csv", index_col=0, parse_dates=True)
    classes = {ticker: asset["asset_class"] for ticker, asset in research.assets.items()}
    scenarios = ScenarioConfig.load(directory / "scenarios.yaml", prices.columns)
    alignment = pd.read_csv(directory / "fx_alignment.csv", index_col=0, parse_dates=True)
except (ValueError, KeyError, OSError) as exc:
    st.error(f"Report cannot be loaded: {exc}")
    st.stop()

if metadata["source"] == "synthetic":
    st.warning("SYNTHETIC market and FX illustration. These are not historical market results.")
st.info(
    f"GBP evidence: {metadata['first_observation']} to {metadata['last_observation']} · "
    f"Latest FX observation: {metadata['fx_observation_date']}"
)
st.caption(
    "This is a saved snapshot. Run aycapital-scenarios again (or --refresh) to update "
    "observations; the dashboard never silently substitutes or refreshes data."
)
with st.expander("Observation dates and data quality", expanded=False):
    st.write(f"Asset close: {metadata['asset_close_utc']}")
    st.write(f"FX available under research convention: {metadata['fx_available_at_utc']}")
    for warning in metadata["warnings"]:
        st.warning(warning)
    st.dataframe(alignment.tail(10))

historical, hypothetical, portfolios_tab, methods = st.tabs(
    ["Historical GBP evidence", "Hypothetical scenarios", "Portfolio comparisons", "Assumptions"]
)

with historical:
    st.subheader("Historical asset-class comparison")
    dates = list(prices.index[200:].strftime("%Y-%m-%d"))
    if not dates:
        st.error("Need at least 201 observations for the full signal set")
        st.stop()
    date = st.select_slider("Evidence as of session", options=dates, value=dates[-1])
    horizon = st.radio("Trailing evidence window (trading sessions)", [20, 40], horizontal=True)
    chosen_fx = alignment.loc[date]
    st.caption(
        f"Selected asset session: {date} · FX observation: "
        f"{chosen_fx.fx_observation_date} · FX available: {chosen_fx.fx_available_at_utc}"
    )
    cut = prices.loc[:date]
    closes = session_schedule(
        research.calendar,
        cut.index[0].strftime("%Y-%m-%d"),
        (cut.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    panel = MarketPanel(cut, closes, {"calendar": research.calendar, "currency": "GBP"})
    report = build_benchmarks(panel, research, scenarios.portfolios)
    st.caption(
        "Scores summarize trailing GBP momentum, trend and risk. They are not forecasts "
        "for the next 20–40 sessions. No calibrated attractiveness prediction is available."
    )
    view = report.metrics.copy()
    view["trailing_window_return_pct"] = view[f"return_{horizon}_gbp"] * 100
    for n in [5, 20, 60]:
        view[f"{n}_session_GBP_return_pct"] = view[f"return_{n}_gbp"] * 100
    columns = [
        "asset_class",
        "score",
        "rank",
        "trailing_window_return_pct",
        "5_session_GBP_return_pct",
        "20_session_GBP_return_pct",
        "60_session_GBP_return_pct",
        "volatility_20",
        "volatility_60",
        "max_drawdown",
        "correlation_spy",
        f"relative_{horizon}_vs_{research.benchmark}",
        f"relative_{horizon}_vs_Equal-weight benchmark",
        "last_observation_date",
    ]
    st.dataframe(view[columns], width="stretch")
    st.caption(
        "Returns marked pct are percentages; volatility, drawdown and relative metrics "
        "are fractions. Equal-weight benchmark rebalances on the configured schedule."
    )
    st.bar_chart(view.set_index("asset_class")["trailing_window_return_pct"])
    asset = st.selectbox(
        "Asset history", list(cut.columns), format_func=lambda a: f"{a} — {classes[a]}"
    )
    st.line_chart(report.signals.score[[asset]])
    st.caption(
        "Historical evidence score (dimensionless). Selected-date computations use only "
        "the prefix of saved observations, conditional on the vendor snapshot."
    )
    st.plotly_chart(
        px.imshow(
            report.correlation,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Trailing 60-session GBP return correlations",
        )
    )
    st.download_button(
        "Export selected-date historical metrics",
        view.to_csv(),
        file_name=f"historical_gbp_{date}.csv",
        mime="text/csv",
    )

with hypothetical:
    st.subheader("Hypothetical terminal scenario assumptions")
    st.warning(
        "Manual shocks are assumptions, not historical performance, probabilities, "
        "empirical estimates or validated predictions."
    )
    scenario_name = st.selectbox("Scenario", list(scenarios.scenarios))
    selected = scenarios.scenarios[scenario_name]
    st.write(selected["description"])
    st.caption(
        f"All shocks cover {scenarios.horizon_sessions} trading sessions. "
        "They are not rescaled when the historical evidence window changes."
    )
    fx_shock = st.number_input(
        "GBP-per-USD shock (%) — positive means USD strengthens versus GBP",
        min_value=-99.0,
        value=float(selected["gbp_per_usd_return"] * 100),
        step=0.5,
    )
    assumptions = pd.DataFrame(
        {
            "ticker": list(prices.columns),
            "asset_class": [classes[t] for t in prices.columns],
            "USD terminal return assumption (%)": [
                selected["asset_returns_usd"][t] * 100 for t in prices.columns
            ],
        }
    )
    edited = st.data_editor(
        assumptions,
        hide_index=True,
        disabled=["ticker", "asset_class"],
        column_config={
            "USD terminal return assumption (%)": st.column_config.NumberColumn(
                min_value=-99.0, required=True, step=0.5
            )
        },
        key="shock_editor",
    )
    amended = copy.deepcopy(scenarios.scenarios)
    amended[scenario_name]["gbp_per_usd_return"] = fx_shock / 100
    amended[scenario_name]["asset_returns_usd"] = dict(
        zip(edited.ticker, edited["USD terminal return assumption (%)"] / 100)
    )
    active = ScenarioConfig(
        amended, scenarios.portfolios, scenarios.horizon_sessions, scenarios.max_fx_age_days
    )
    try:
        asset_stress, portfolio_stress = scenario_returns(active, prices.columns)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.subheader("Hypothetical asset comparison (GBP terminal returns %)")
    comparison = asset_stress.T * 100
    comparison.index = [f"{t} — {classes[t]}" for t in comparison.index]
    st.dataframe(comparison)
    st.bar_chart(comparison[scenario_name].sort_values(ascending=False))
    st.caption(
        "Scenario ordering reflects the shocks entered above; it is not a predictive ranking."
    )

with portfolios_tab:
    st.subheader("Allocation comparison and stress tests")
    st.caption(
        "Configured allocations are illustrations. Edits affect historical comparisons "
        "and terminal stress calculations; neither validates a recommendation."
    )
    edit_portfolio = st.selectbox("Allocation to edit", list(active.portfolios))
    weight_rows = pd.DataFrame(
        {
            "ticker": list(prices.columns),
            "weight_pct": [
                active.portfolios[edit_portfolio].get(t, 0.0) * 100 for t in prices.columns
            ],
        }
    )
    edited_weights = st.data_editor(
        weight_rows,
        hide_index=True,
        disabled=["ticker"],
        column_config={
            "weight_pct": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, required=True, step=1.0
            )
        },
        key="allocation_editor",
    )
    allocations = copy.deepcopy(active.portfolios)
    allocations[edit_portfolio] = dict(zip(edited_weights.ticker, edited_weights.weight_pct / 100))
    st.caption(f"Edited allocation total: {edited_weights.weight_pct.sum():.2f}% (must equal 100%)")
    try:
        allocation(allocations[edit_portfolio], prices.columns)
        configured = ScenarioConfig(
            active.scenarios, allocations, active.horizon_sessions, active.max_fx_age_days
        )
        _, portfolio_stress = scenario_returns(configured, prices.columns)
        history = build_benchmarks(panel, research, allocations)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.dataframe(portfolio_stress * 100)
        st.bar_chart(portfolio_stress.loc[scenario_name] * 100)
        st.subheader(f"Historical GBP allocation paths through {date}")
        st.line_chart(history.equity)
        st.dataframe(history.portfolio_risk)
        stress_floor = portfolio_stress.min().rename(
            "Worst configured hypothetical terminal return"
        )
        st.dataframe(stress_floor)
        st.caption(
            "Worst configured scenario is a scenario-set minimum, not VaR, probability or "
            "path drawdown. Terminal shocks do not describe an interim price path."
        )
        st.download_button(
            "Export hypothetical portfolio comparison",
            portfolio_stress.to_csv(),
            "hypothetical_portfolios.csv",
            "text/csv",
        )

with methods:
    st.write(
        "Research ETFs remain separate from UMushroom-eligible instruments; platform "
        "availability and UK competition eligibility are unverified for all proxies."
    )
    st.write(
        "Empirical calibration is future work: predefine release-aware historical regimes, "
        "estimate asset/FX relationships on training data, and assess untouched OOS outcomes "
        "with uncertainty and multiple-testing controls. No such calibration is implemented."
    )
    st.write(
        "GBP prices use dated USD-per-GBP observations with an explicit availability convention. "
        "Bounded backward matching is disclosed; equity and FX closes are not synchronized."
    )
    st.write(
        "Historical allocation comparisons reuse the v0.2 ledger: next-session-close entry, "
        "weight drift, configured rebalancing and fees. Scenario shocks are terminal weighted "
        "returns with no interim rebalancing or fees. These are different calculations."
    )
    st.json(metadata["scenario_configuration"])
