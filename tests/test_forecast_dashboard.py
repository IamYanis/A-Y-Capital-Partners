"""Optional forecast-tab integration using synthetic outputs, never market-skill fixtures."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.scenario_pipeline import run_scenarios
from aycapital.cross_asset.scenarios import ScenarioConfig
from aycapital.forecasting.config import ForecastConfig
from aycapital.forecasting.pipeline import run_pipeline


def test_forecast_tab_controls_and_stale_snapshot_detection(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    research = ResearchConfig.load("config/cross_asset.yaml")
    research = replace(
        research,
        assets={a: research.assets[a] for a in ["SPY", "GLD"]},
        backtest=dict(research.backtest, top_n=1),
    )
    scenario = ScenarioConfig(
        {"test": {"asset_returns_usd": {"SPY": 0.1, "GLD": 0.0}, "gbp_per_usd_return": 0.0}},
        {"example": {"SPY": 1.0}},
        40,
        6,
    )
    run_scenarios(research, scenario, "2020-01-01", "2022-01-01", tmp_path, source="synthetic")
    config = ForecastConfig(
        horizons=(5, 10),
        min_train_sessions=60,
        min_regime_sessions=10,
        min_regime_blocks=2,
        min_oos_origins=4,
        min_calibration_origins=3,
        bootstrap_draws=99,
    )
    run_pipeline(tmp_path, tmp_path / "forecasts", config)
    script = Path(__file__).resolve().parents[1] / "scripts/scenario_dashboard.py"
    app = AppTest.from_file(script, default_timeout=60).run()
    app.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not app.exception
    controls = {box.label: box for box in app.selectbox}
    controls["Forecast horizon (trading sessions)"].set_value(10).run()
    assert not app.exception
    controls = {box.label: box for box in app.selectbox}
    controls["Forecast model"].set_value("ridge").run()
    assert not app.exception
    assert any("insufficient evidence" in warning.value for warning in app.warning)
    fx = pd.read_csv(tmp_path / "fx_alignment.csv")
    fx.loc[0, "usd_per_gbp"] *= 1.01
    fx.to_csv(tmp_path / "fx_alignment.csv", index=False)
    app.run()
    assert any("different price snapshot" in error.value for error in app.error)
