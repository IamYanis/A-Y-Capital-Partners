import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.market import load_market, session_schedule, synthetic_panel


@pytest.fixture
def config(tmp_path):
    raw = yaml.safe_load(open("config/cross_asset.yaml"))
    raw["assets"] = {k: v for k, v in raw["assets"].items() if k in {"SPY", "QQQ"}}
    raw["backtest"]["top_n"] = 1
    raw["cache_dir"] = str(tmp_path)
    return ResearchConfig.from_mapping(raw)


def test_calendar_holidays_and_early_close():
    closes = session_schedule("XNYS", "2024-07-01", "2024-07-06")
    assert pd.Timestamp("2024-07-04") not in closes.index
    assert closes.loc["2024-07-03"] == pd.Timestamp("2024-07-03 17:00Z")
    assert closes.loc["2024-07-05"] == pd.Timestamp("2024-07-05 20:00Z")


def test_configuration_rejects_currency_and_calendar_mismatch():
    raw = yaml.safe_load(open("config/cross_asset.yaml"))
    for field, value in [("currency", "GBP"), ("calendar", "XLON")]:
        bad = copy.deepcopy(raw)
        bad["assets"]["QQQ"][field] = value
        with pytest.raises(ValueError):
            ResearchConfig.from_mapping(bad)
    raw["history_years"] = 4
    with pytest.raises(ValueError):
        ResearchConfig.from_mapping(raw)


class Provider:
    name = "test"

    def __init__(self, defect=None):
        self.calls = 0
        self.defect = defect

    def download(self, ticker, start, end):
        self.calls += 1
        dates = session_schedule("XNYS", start, end).index
        values = pd.Series(np.linspace(100, 110, len(dates)), index=dates, name=ticker)
        if ticker == "QQQ":
            if self.defect == "leading":
                return values.iloc[30:]
            if self.defect == "missing":
                values.iloc[100] = np.nan
            if self.defect == "negative":
                values.iloc[100] = -1
            if self.defect == "duplicate":
                return pd.concat([values.iloc[:1], values])
            if self.defect == "failed":
                raise RuntimeError("Unavailable")
        return values


def test_market_cache_and_leading_history(config, caplog):
    provider = Provider("leading")
    first = load_market(config, "2022-01-01", "2024-01-01", provider)
    second = load_market(config, "2022-01-01", "2024-01-01", provider)
    pd.testing.assert_frame_equal(first.prices, second.prices, check_freq=False)
    assert provider.calls == 2
    assert "short_history" in caplog.text
    assert len(first.prices) > 400
    assert first.metadata["assets"]["SPY"]["sha256"]
    snapshots = list((config.cache_dir / "market").glob("snapshot-*.csv"))
    load_market(config, "2022-01-01", "2024-01-01", provider, refresh=True)
    assert len(list((config.cache_dir / "market").glob("snapshot-*.csv"))) == len(snapshots) + 2
    pointer = next(
        p
        for p in (config.cache_dir / "market").glob("*.json")
        if not p.name.startswith("snapshot-")
    )
    metadata = json.loads(pointer.read_text())
    (pointer.parent / metadata["snapshot"]).write_text("corrupt")
    with pytest.raises(ValueError):
        load_market(config, "2022-01-01", "2024-01-01", provider)


@pytest.mark.parametrize("defect", ["missing", "negative", "duplicate", "failed"])
def test_market_failures_logged(config, defect, caplog):
    with pytest.raises(ValueError, match="QQQ"):
        load_market(config, "2022-01-01", "2024-01-01", Provider(defect))
    assert "market_request_failed" in caplog.text


def test_uncompleted_session_excluded(config):
    panel = load_market(
        config, "2022-01-01", "2024-01-05", Provider(), now=pd.Timestamp("2024-01-04 15:00Z")
    )
    assert panel.prices.index[-1] == pd.Timestamp("2024-01-03")


def test_synthetic_reproducible(config):
    first = synthetic_panel(config, "2022-01-01", "2024-01-01")
    second = synthetic_panel(config, "2022-01-01", "2024-01-01")
    pd.testing.assert_frame_equal(first.prices, second.prices)
    assert first.metadata["source"] == "synthetic"
