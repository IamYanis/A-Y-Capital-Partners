"""Optional dashboard integration test; enabled with the dashboard extra."""

from pathlib import Path

import pandas as pd
import pytest

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.scenario_pipeline import run_scenarios
from aycapital.cross_asset.scenarios import ScenarioConfig


def test_dashboard_scenario_horizon_and_date_controls(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    research = ResearchConfig.load("config/cross_asset.yaml")
    config = ScenarioConfig.load("config/scenarios.yaml", pd.Index(research.assets))
    run_scenarios(research, config, "2021-01-01", "2024-01-01", tmp_path, source="synthetic")
    script = Path(__file__).resolve().parents[1] / "scripts/scenario_dashboard.py"
    app = AppTest.from_file(script, default_timeout=60).run()
    app.sidebar.text_input[0].set_value(str(tmp_path)).run()
    assert not app.exception
    app.selectbox[1].set_value("Growth shock / recession").run()
    assert not app.exception
    app.radio[0].set_value(40).run()
    assert not app.exception
    app.select_slider[0].set_value("2022-01-03").run()
    assert not app.exception
    assert any("SYNTHETIC" in warning.value for warning in app.warning)
    assert any("not historical performance" in warning.value for warning in app.warning)
