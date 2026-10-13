"""Read saved research outputs: uv run streamlit run scripts/cross_asset_dashboard.py."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="A-Y Cross-Asset Research", layout="wide")
st.title("A-Y Cross-Asset Research")
directory = Path(st.sidebar.text_input("Saved report directory", "outputs/cross_asset"))
if not (directory / "metadata.json").is_file():
    st.info("Generate a report with aycapital-research, then select its output directory.")
    st.stop()
metadata = json.loads((directory / "metadata.json").read_text())
st.caption(
    f"Source: {metadata['market']['source']} · "
    f"Currency: {metadata['market']['currency']} · "
    f"Signal session: {metadata['signal_session']}"
)
st.warning(metadata["disclaimer"])
st.subheader("Asset ranking and signal breakdown")
st.dataframe(pd.read_csv(directory / "latest_signals.csv"), hide_index=True)
st.subheader("Backtest statistics")
st.dataframe(pd.read_csv(directory / "performance.csv"), hide_index=True)
equity = pd.read_csv(directory / "equity.csv", index_col="session", parse_dates=True)
for label in ("in_sample", "out_of_sample"):
    st.subheader(f"{label}: independent account with transaction costs")
    st.line_chart(equity.loc[equity.period == label].drop(columns="period"))
st.subheader("Latest trailing return correlation")
st.dataframe(pd.read_csv(directory / "latest_correlation.csv", index_col="ticker"))
st.caption("Historical signal plots and macro charts are available in the standalone report.html.")
if (directory / "macro_display.csv").is_file():
    st.subheader("Descriptive macro snapshots")
    macro = pd.read_csv(
        directory / "macro_display.csv", index_col="observation_date", parse_dates=True
    )
    metric = st.selectbox("Macro series", list(macro.columns))
    st.line_chart(macro[[metric]])
