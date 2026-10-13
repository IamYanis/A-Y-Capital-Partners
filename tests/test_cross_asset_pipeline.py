import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.pipeline import run_research


def test_end_to_end_synthetic_report(tmp_path):
    config = ResearchConfig.load("config/cross_asset.yaml")
    metadata = run_research(config, "2018-01-01", "2024-01-01", tmp_path, source="synthetic")
    latest = pd.read_csv(tmp_path / "latest_signals.csv", index_col="ticker")
    assert len(latest) == 14 and latest.score.notna().all()
    assert latest.asset_class.notna().all()
    assert latest.source.eq("synthetic").all()
    json_rows = json.loads((tmp_path / "latest_signals.json").read_text())
    assert len(json_rows) == 14
    assert all("signal_available_after_utc" in row for row in json_rows)
    assert metadata["market"]["source"] == "synthetic"
    assert "no parameter search" in metadata["backtest"]["fitting"]
    performance = pd.read_csv(tmp_path / "performance.csv")
    assert set(performance.period) == {"in_sample", "out_of_sample"}
    assert len(performance) == 6
    for filename in [
        "report.html",
        "rolling_correlations.csv",
        "latest_correlation.csv",
        "equity.csv",
        "signal_history.csv",
        "metadata.json",
        "paper_trades.csv",
    ]:
        assert (tmp_path / filename).is_file()
    html = (tmp_path / "report.html").read_text()
    assert "SYNTHETIC" in html and "out_of_sample" in html
    assert "cdn.plot.ly" not in html.split("<script src=")[-1][:200]


def test_live_failure_never_falls_back(tmp_path, monkeypatch):
    config = ResearchConfig.load("config/cross_asset.yaml")

    def fail(*args, **kwargs):
        raise ValueError("Provider unavailable")

    monkeypatch.setattr("aycapital.cross_asset.pipeline.load_market", fail)
    with pytest.raises(ValueError, match="unavailable"):
        run_research(config, "2018-01-01", "2024-01-01", tmp_path, source="yahoo")
    assert not (tmp_path / "report.html").exists()


def test_configurable_weights_and_asset_class_metadata(tmp_path):
    raw = yaml.safe_load(Path("config/cross_asset.yaml").read_text())
    raw["ranking_weights"] = {"momentum_20": 1}
    raw["backtest"]["top_n"] = 2
    config = ResearchConfig.from_mapping(raw)
    assert config.ranking_weights == {"momentum_20": 1}
    assert config.assets["TLT"]["asset_class"] == "Long-term government bonds"
