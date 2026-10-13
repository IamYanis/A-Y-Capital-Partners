from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aycapital.cross_asset.benchmarks import build_benchmarks
from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.fx import FxData, gbp_panel, load_fx
from aycapital.cross_asset.market import MarketPanel, session_schedule, synthetic_panel
from aycapital.cross_asset.scenario_pipeline import run_scenarios
from aycapital.cross_asset.scenarios import ScenarioConfig, allocation, scenario_returns


def fx_data(dates, rates):
    index = pd.DatetimeIndex(dates)
    return FxData(
        pd.Series(rates, index=index),
        pd.Series(index.tz_localize("UTC") + pd.Timedelta(days=1), index=index),
        {},
    )


def tiny_panel():
    closes = session_schedule("XNYS", "2024-01-02", "2024-01-04")
    return MarketPanel(
        pd.DataFrame({"SPY": [100.0, 110.0]}, index=closes.index),
        closes,
        {"calendar": "XNYS", "currency": "USD"},
    )


def test_prototype_fx_translation_with_explicit_availability():
    panel = tiny_panel()
    fx = fx_data(["2024-01-01", "2024-01-02"], [1.25, 1.375])
    converted, observations = gbp_panel(panel, fx)
    np.testing.assert_allclose(converted.prices.SPY, [80.0, 80.0])
    assert converted.metadata["currency"] == "GBP"
    assert observations.fx_observation_date.iloc[-1] == pd.Timestamp("2024-01-02")
    assert (observations.fx_available_at_utc <= observations.asset_close_utc).all()


def test_future_fx_cannot_affect_earlier_converted_prices():
    panel = tiny_panel()
    fx = fx_data(["2024-01-01", "2024-01-02", "2024-01-03"], [1.25, 1.375, 100.0])
    converted, _ = gbp_panel(panel, fx)
    np.testing.assert_allclose(converted.prices.SPY, [80.0, 80.0])


@pytest.mark.parametrize("rates", [[1.25, 0.0], [1.25, np.nan], [1.25, np.inf]])
def test_invalid_fx_rejected(rates):
    with pytest.raises(ValueError):
        gbp_panel(tiny_panel(), fx_data(["2024-01-01", "2024-01-02"], rates))


def test_missing_or_stale_fx_rejected():
    with pytest.raises(ValueError, match="already-available"):
        gbp_panel(tiny_panel(), fx_data(["2024-01-03", "2024-01-04"], [1.25, 1.25]))
    with pytest.raises(ValueError, match="maximum age"):
        gbp_panel(tiny_panel(), fx_data(["2023-12-01", "2023-12-02"], [1.25, 1.25]))
    with pytest.raises(ValueError):
        gbp_panel(tiny_panel(), fx_data(["2024-01-01", "2024-01-01"], [1.25, 1.25]))


def test_fx_cache_uses_existing_provider_interface(tmp_path):
    class Provider:
        name = "test"
        calls = 0

        def download(self, ticker, start, end):
            self.calls += 1
            assert ticker == "GBPUSD=X"
            return pd.Series([1.25, 1.3], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

    provider = Provider()
    first = load_fx("2024-01-01", "2024-01-03", str(tmp_path), provider)
    second = load_fx("2024-01-01", "2024-01-03", str(tmp_path), provider)
    np.testing.assert_allclose(first.usd_per_gbp, second.usd_per_gbp)
    assert provider.calls == 1


def small_scenarios():
    return ScenarioConfig(
        {"test": {"asset_returns_usd": {"SPY": 0.1, "GLD": -0.2}, "gbp_per_usd_return": 0.05}},
        {"mix": {"SPY": 0.6, "GLD": 0.4}},
        40,
        4,
    )


def test_scenario_compounding_and_portfolio_aggregation():
    assets = pd.Index(["SPY", "GLD"])
    config = small_scenarios()
    asset_returns, portfolios = scenario_returns(config, assets)
    assert asset_returns.loc["test", "SPY"] == pytest.approx(0.155)
    assert asset_returns.loc["test", "GLD"] == pytest.approx(-0.16)
    assert portfolios.loc["test", "mix"] == pytest.approx(0.6 * 0.155 + 0.4 * -0.16)
    assert asset_returns.loc["test", "SPY"] != pytest.approx(0.1 + 0.05)


@pytest.mark.parametrize(
    "weights",
    [{"SPY": -0.1, "GLD": 1.1}, {"SPY": 0.5}, {"SPY": np.nan, "GLD": 1.0}, {"UNKNOWN": 1.0}],
)
def test_invalid_allocations_fail_without_normalization(weights):
    with pytest.raises(ValueError):
        allocation(weights, pd.Index(["SPY", "GLD"]))


def test_incomplete_scenario_is_not_filled_with_zero():
    config = small_scenarios()
    with pytest.raises(ValueError):
        scenario_returns(config, pd.Index(["SPY", "GLD", "IEF"]))
    config.scenarios["test"]["asset_returns_usd"]["SPY"] = np.nan
    with pytest.raises(ValueError):
        scenario_returns(config, pd.Index(["SPY", "GLD"]))


def test_prototype_scenario_and_model_weights_preserved():
    assets = pd.Index(ResearchConfig.load("config/cross_asset.yaml").assets)
    config = ScenarioConfig.load("config/scenarios.yaml", assets)
    stress, portfolios = scenario_returns(config, assets)
    assert len(stress) == 4
    assert stress.loc["Disinflation / soft landing", "SPY"] == pytest.approx(1.08 * 0.97 - 1)
    weights = allocation(config.portfolios["Prototype allocation"], assets)
    np.testing.assert_allclose(portfolios["Prototype allocation"], stress @ weights)


def test_gbp_benchmark_metrics_and_no_silent_missing_prices():
    research = ResearchConfig.load("config/cross_asset.yaml")
    panel = synthetic_panel(research, "2021-01-01", "2024-01-01")
    dates = pd.bdate_range("2020-12-25", "2023-12-31")
    fx = fx_data(dates, np.full(len(dates), 1.25))
    converted, _ = gbp_panel(panel, fx)
    report = build_benchmarks(converted, research, {"example": {"SPY": 1.0}})
    for window in [5, 20, 60]:
        expected = converted.prices.SPY.iloc[-1] / converted.prices.SPY.iloc[-window - 1] - 1
        assert report.metrics.loc["SPY", f"return_{window}_gbp"] == pytest.approx(expected)
        assert report.metrics.loc["SPY", f"relative_{window}_vs_SPY"] == pytest.approx(
            0.0, abs=1e-12
        )
    np.testing.assert_allclose(report.portfolio_returns.example, report.portfolio_returns.SPY)
    assert report.metrics.currency.eq("GBP").all()
    bad = converted.prices.copy()
    bad.iloc[100, 0] = np.nan
    with pytest.raises(ValueError):
        build_benchmarks(
            MarketPanel(bad, converted.closes_utc, converted.metadata),
            research,
            {"example": {"SPY": 1.0}},
        )


def test_synthetic_pipeline_exports_dates_and_source_labels(tmp_path):
    research = ResearchConfig.load("config/cross_asset.yaml")
    config = ScenarioConfig.load("config/scenarios.yaml", pd.Index(research.assets))
    metadata = run_scenarios(
        research, config, "2021-01-01", "2024-01-01", tmp_path, source="synthetic"
    )
    assert metadata["source"] == "synthetic" and metadata["currency"] == "GBP"
    assert metadata["fx_observation_date"] < metadata["last_observation"]
    assert metadata["warnings"] and metadata["scenario_type"].startswith("Hypothetical")
    for name in [
        "historical_gbp_metrics",
        "fx_alignment",
        "historical_equity",
        "hypothetical_asset_returns",
        "hypothetical_portfolio_returns",
    ]:
        assert (tmp_path / f"{name}.csv").is_file()
    assert Path(tmp_path / "scenarios.yaml").is_file()


def test_missing_provider_fx_quotes_are_explicit_and_raw_cache_retained(tmp_path, caplog):
    class Provider:
        name = "test_missing"

        def download(self, *args):
            return pd.Series([1.25, np.nan, 1.3], index=pd.date_range("2024-01-01", periods=3))

    data = load_fx("2024-01-01", "2024-01-04", str(tmp_path), Provider())
    assert len(data.usd_per_gbp) == 2
    assert data.metadata["missing_observation_dates"] == ["2024-01-02"]
    assert "missing_fx_quotes" in caplog.text
    raw = pd.read_csv(tmp_path / data.metadata["snapshot"])
    assert raw.usd_per_gbp.isna().sum() == 1


def test_fx_exclusive_end_is_enforced_even_when_provider_includes_it(tmp_path, caplog):
    class Provider:
        name = "test_extra_date"

        def download(self, *args):
            return pd.Series([1.25, 1.3, 2.0], index=pd.date_range("2024-01-01", periods=3))

    data = load_fx("2024-01-01", "2024-01-03", str(tmp_path), Provider())
    assert data.usd_per_gbp.index[-1] == pd.Timestamp("2024-01-02")
    assert data.metadata["excluded_out_of_range_dates"] == ["2024-01-03"]
    assert "out_of_range_fx_quotes" in caplog.text


def test_long_weekend_quote_reuse_is_bounded_and_warned():
    closes = session_schedule("XNYS", "2025-04-22", "2025-04-24")
    panel = MarketPanel(
        pd.DataFrame({"SPY": [100.0, 110.0]}, index=closes.index),
        closes,
        {"calendar": "XNYS", "currency": "USD"},
    )
    fx = fx_data(["2025-04-17", "2025-04-22"], [1.2, 1.3])
    converted, dates = gbp_panel(panel, fx, max_age_days=6)
    assert dates.fx_observation_date.iloc[0] == pd.Timestamp("2025-04-17")
    assert any("four calendar days" in w for w in converted.metadata["data_quality_warnings"])
    with pytest.raises(ValueError, match="maximum age"):
        gbp_panel(panel, fx, max_age_days=5)
