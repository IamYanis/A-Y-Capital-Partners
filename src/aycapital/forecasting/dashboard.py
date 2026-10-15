"""Forecast panel embedded in the existing dashboard; reads saved validation only."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def render_forecasts(directory: Path) -> None:
    st.subheader("Probabilistic GBP research forecasts — v0.4")
    output = directory / "forecasts"
    if not (output / "metadata.json").is_file():
        st.info("No forecasting validation is saved for this snapshot. Generate it first:")
        st.code(f"uv run --locked aycapital-forecast --input {directory}")
        return
    try:
        metadata = json.loads((output / "metadata.json").read_text())
        latest = _read(output / "latest_forecasts.csv")
        validation = _read(output / "validation.csv")
        if any(
            hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
            for name, digest in metadata["input_sha256"].items()
        ):
            st.error("Forecasts refer to a different price snapshot. Regenerate validation.")
            return
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Forecast outputs unavailable: {exc}")
        return
    st.caption(
        f"Forecast decision session: {metadata['decision_session']} · "
        f"Source: {metadata['source']} · Feature scope: {metadata['feature_scope']}"
    )
    st.caption(
        "Forecast dates are independent of the historical date slider. Entry is the "
        "next session close; h-session outcomes begin after entry."
    )
    st.warning(
        "Research estimates are not trading recommendations. Failed baseline comparisons, "
        "sparse samples or unsupported regimes mean insufficient evidence."
    )
    with st.expander("Forecast provenance and limitations"):
        for warning in metadata["warnings"]:
            st.warning(warning)
    horizon = st.selectbox("Forecast horizon (trading sessions)", sorted(latest.horizon.unique()))
    current = latest[latest.horizon == horizon]
    st.dataframe(
        current[
            [
                "asset",
                "asset_class",
                "model",
                "expected_return_gbp",
                "probability_positive",
                "probability_outperform",
                "interval_low",
                "interval_high",
                "entry_session",
                "exit_session",
                "regime",
                "calibration_origins",
                "current_evidence_status",
            ]
        ]
    )
    st.caption(
        "Returns and probabilities are fractions. Empty intervals/probabilities are "
        "unavailable estimates, never zeros. Intervals are approximate, not guaranteed."
    )
    asset = st.selectbox("Forecast asset", list(current.asset.unique()))
    model = st.selectbox("Forecast model", list(current.model.unique()))
    selected = current[(current.asset == asset) & (current.model == model)].iloc[0]
    st.warning(selected.current_evidence_status)
    st.subheader("OOS accuracy against matched historical baselines")
    metrics = validation[(validation.asset == asset) & (validation.horizon == horizon)]
    st.dataframe(metrics)
    st.caption(
        "Directional accuracy and Brier scores use realized OOS outcomes. Return/excess MAE "
        "are accuracy metrics, not net trading performance. Holm correction spans the full family."
    )
    calibration = _read(output / "calibration.csv")
    if not calibration.empty:
        view = calibration[
            (calibration.asset == asset)
            & (calibration.horizon == horizon)
            & (calibration.model == model)
        ]
        if not view.empty:
            st.plotly_chart(
                px.line(
                    view,
                    x="mean_probability",
                    y="observed_frequency",
                    color="endpoint",
                    markers=True,
                    title="OOS probability calibration (descriptive bins)",
                )
            )
            st.dataframe(view)
    samples = _read(output / "predictive_samples.csv")
    if not samples.empty:
        distribution = samples[
            (samples.asset == asset) & (samples.horizon == horizon) & (samples.model == model)
        ]
        if not distribution.empty:
            st.plotly_chart(
                px.histogram(
                    distribution,
                    x="sample_return",
                    nbins=25,
                    title="Approximate predictive distribution from matured OOS residuals",
                )
            )
        else:
            st.info("Insufficient matured OOS residuals for a predictive distribution.")
    oos = _read(output / "oos_predictions.csv")
    if not oos.empty:
        historical = oos[(oos.asset == asset) & (oos.horizon == horizon) & (oos.model == model)]
        st.plotly_chart(
            px.line(
                historical,
                x="decision_session",
                y=["expected_return_gbp", "actual_return"],
                title="Historical OOS predictions and outcomes",
            )
        )
    conditional = _read(output / "conditional_regime_forecasts.csv")
    if not conditional.empty:
        st.subheader("Alternative-regime conditional averages")
        st.dataframe(conditional[(conditional.asset == asset) & (conditional.horizon == horizon)])
        st.caption(
            "These use past outcomes for assumed trend/volatility regimes. Unsupported "
            "samples remain missing. Regime-specific predictive skill is not established."
        )
    st.download_button(
        "Download forecast-validation report",
        (output / "validation_report.md").read_text(),
        file_name="forecast_validation.md",
    )
