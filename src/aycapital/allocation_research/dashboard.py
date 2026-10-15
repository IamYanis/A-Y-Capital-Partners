"""Saved fixed-rule allocation evidence in the existing dashboard."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _read(output: Path, name: str) -> pd.DataFrame:
    path = output / f"{name}.csv"
    return pd.read_csv(path if path.exists() else output / f"{name}.csv.gz")


def render_allocations(directory: Path) -> None:
    st.subheader("Fixed-rule allocation research")
    default = Path("outputs/allocation_research")
    if not (default / "report.md").exists():
        default = Path("docs/research/allocation_research_01")
    output = Path(st.text_input("Saved allocation research directory", str(default)))
    if not (output / "metadata.json").exists():
        st.info("Generate allocation evidence from an existing frozen GBP snapshot:")
        st.code(f"uv run --locked aycapital-allocations --input {directory}")
        return
    try:
        metadata = json.loads((output / "metadata.json").read_text())
        if any(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
            for name, digest in metadata["input_sha256"].items()
        ):
            st.warning(
                "Allocation results refer to a different saved input snapshot. "
                "Select their source snapshot to compare them here."
            )
            return
        performance = _read(output, "performance")
        evidence = _read(output, "evidence")
        rolling = _read(output, "rolling")
        regimes = _read(output, "regimes")
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Allocation report unavailable: {exc}")
        return
    st.warning(
        "Retrospective historical evidence. No prospective confirmation. "
        "No strategy ranking is a validated prediction or eligible competition allocation."
    )
    st.caption(
        f"{metadata['evaluation_start']} to {metadata['last_observation']} · GBP · "
        f"Source: {metadata['source']} · Holdout: {metadata['prospective_status']}"
    )
    horizon = st.selectbox("Allocation holding period", [20, 40])
    cost = st.selectbox("Per-side traded-notional cost (bps)", [10, 25, 50])
    selection = {"horizon": horizon, "cost": cost}
    selected = performance.query(
        "holding_sessions == @horizon and cost_bps == @cost", local_dict=selection
    )
    st.dataframe(
        selected[
            [
                "portfolio",
                "cagr",
                "annualized_volatility",
                "max_drawdown",
                "zero_hurdle_sharpe",
                "annualized_turnover",
                "gbp_minus_native_total_return",
            ]
        ],
        hide_index=True,
    )
    st.caption(
        "Sharpe uses a zero hurdle, not GBP cash. USD comparison uses identical GBP-selected "
        "targets; it is not a hedged strategy. Costs are assumptions, not measured spreads."
    )
    st.dataframe(
        evidence.query("holding_sessions == @horizon", local_dict=selection), hide_index=True
    )
    if not evidence.historical_gate_pass.any():
        st.info(
            "Insufficient evidence of robust advantage. Registered fallback: equal-weight "
            "with 40-session rebalancing, subject to universe and GBP risk review."
        )
    selected_rolling = rolling.query(
        "holding_sessions == @horizon and cost_bps == @cost", local_dict=selection
    )
    metric = st.selectbox(
        "Allocation rolling 252-session metric",
        [
            "return_252",
            "zero_hurdle_sharpe_252",
            "max_drawdown_252",
            "turnover_252",
            "excess_return_252_equal_weight",
            "excess_return_252_defensive",
        ],
    )
    st.plotly_chart(
        px.line(selected_rolling, x="session", y=metric, color="portfolio"),
        use_container_width=True,
    )
    st.caption("Overlapping rolling windows are descriptive, not independent test replications.")
    st.dataframe(
        regimes.query("holding_sessions == @horizon and cost_bps == @cost", local_dict=selection),
        hide_index=True,
    )
    st.caption(
        "Regimes overlap and may contain few economic episodes. Unknown macro states are "
        "not treated as recoveries. Sparse regimes provide insufficient evidence."
    )
    with st.expander("Macro availability, research report and safeguards"):
        st.markdown((output / "report.md").read_text())
        if any((output / f"macro_features{suffix}").exists() for suffix in [".csv", ".csv.gz"]):
            st.dataframe(_read(output, "macro_features").tail(10), hide_index=True)
        st.json(metadata)
