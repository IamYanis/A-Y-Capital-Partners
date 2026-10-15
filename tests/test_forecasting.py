from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aycapital.cross_asset.config import ResearchConfig
from aycapital.cross_asset.macro import MacroData
from aycapital.cross_asset.market import MarketPanel, session_schedule
from aycapital.cross_asset.scenario_pipeline import run_scenarios
from aycapital.cross_asset.scenarios import ScenarioConfig
from aycapital.forecasting.config import ForecastConfig
from aycapital.forecasting.engine import purged_train_positions, run_forecasts
from aycapital.forecasting.features import FeatureSet, build_features, forward_labels
from aycapital.forecasting.pipeline import run_pipeline
from aycapital.forecasting.uncertainty import block_inference, empirical_distribution, holm_adjust
from aycapital.forecasting.validation import evaluate


@pytest.fixture
def model_case():
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2020-01-01", periods=230)
    innovations = rng.normal(0, 0.01, (230, 2))
    prices = pd.DataFrame(
        100 * np.exp(innovations.cumsum(axis=0)), index=dates, columns=["SPY", "GLD"]
    )
    x = pd.DataFrame(
        {"momentum": prices.SPY.pct_change(5).fillna(0), "noise": rng.normal(size=len(dates))},
        index=dates,
    )
    features = FeatureSet(
        {asset: x.copy() for asset in prices},
        prices.SPY.pct_change(10).fillna(0),
        pd.Series(0.15, index=dates),
        "test",
        (),
    )
    config = ForecastConfig(
        horizons=(5, 10),
        min_train_sessions=40,
        min_regime_sessions=10,
        min_regime_blocks=2,
        min_oos_origins=4,
        min_calibration_origins=3,
        bootstrap_draws=99,
    )
    return prices, features, config


def test_forward_label_entry_offset_and_purge():
    p = pd.DataFrame({"SPY": np.arange(1.0, 31.0)})
    target = forward_labels(p, "SPY", 5)
    assert target.SPY.iloc[0] == pytest.approx(7 / 2 - 1)
    assert target.SPY.iloc[-6:].isna().all()
    train = purged_train_positions(20, 5, np.ones(30, dtype=bool))
    assert train[-1] == 13
    assert ((train + 6) < 20).all()
    rolling = purged_train_positions(20, 5, np.ones(30, dtype=bool), 10)
    assert rolling[0] == 10


def test_future_prices_features_and_targets_cannot_change_prior_predictions(model_case):
    prices, features, config = model_case
    original = run_forecasts(prices, features, "SPY", config)
    changed = prices.copy()
    changed.iloc[150:, 1] *= 10
    changed_features = {a: frame.copy() for a, frame in features.assets.items()}
    for frame in changed_features.values():
        frame.iloc[150:] *= 1000
    future_vol = features.regime_volatility.copy()
    future_vol.iloc[150:] *= 1000
    future = run_forecasts(
        changed,
        replace(features, assets=changed_features, regime_volatility=future_vol),
        "SPY",
        config,
    )
    columns = [
        "asset",
        "horizon",
        "model",
        "origin_position",
        "expected_return_gbp",
        "probability_positive",
        "probability_outperform",
        "interval_low",
        "interval_high",
    ]
    pd.testing.assert_frame_equal(
        original.oos.loc[original.oos.origin_position < 150, columns],
        future.oos.loc[future.oos.origin_position < 150, columns],
    )
    assert (original.oos.train_last_label_end_position < original.oos.origin_position).all()
    for _, group in original.oos.groupby(["asset", "horizon", "model"]):
        assert group.origin_position.diff().dropna().ge(group.horizon.iloc[0]).all()


def test_calibration_only_uses_matured_nonoverlapping_oos_residuals(model_case):
    prices, features, config = model_case
    run = run_forecasts(prices, features, "SPY", config)
    rows = run.oos[(run.oos.asset == "GLD") & (run.oos.horizon == 5) & (run.oos.model == "ridge")]
    assert rows.iloc[0].calibration_origins == 0
    assert np.isnan(rows.iloc[0].interval_low)
    for row in rows.itertuples():
        known = rows[rows.label_end_position < row.origin_position]
        assert row.calibration_origins == len(known)
        if len(known) >= config.min_calibration_origins:
            residuals = (known.actual_return - known.expected_return_gbp).to_numpy()
            low, high, _ = empirical_distribution(
                row.expected_return_gbp, residuals, config.interval_coverage
            )
            assert row.interval_low == pytest.approx(low)
            assert row.interval_high == pytest.approx(high)
    benchmark = run.latest[run.latest.asset == "SPY"]
    assert benchmark.probability_outperform.eq(0).all()


def test_failed_baselines_and_synthetic_data_never_claim_market_skill(model_case):
    prices, features, config = model_case
    run = run_forecasts(prices, features, "SPY", config)
    validation, calibration, _ = evaluate(run, config, "SPY", "synthetic")
    assert (
        validation.loc[validation.model != "historical_average", "validation_status"]
        .str.startswith("insufficient evidence")
        .all()
    )
    assert calibration.n.gt(0).all()
    baseline = validation[validation.model == "historical_average"]
    np.testing.assert_allclose(baseline.mae_improvement, 0.0)
    assert validation.holm_p_value.between(0, 1).all()


def test_block_inference_and_multiplicity():
    low, high, p = block_inference(np.full(80, 0.1), 999, 5, 1)
    assert low == pytest.approx(0.1) and high == pytest.approx(0.1)
    assert p == pytest.approx(0.001)
    assert block_inference(np.full(80, -0.1), 999, 5, 1)[2] == 1
    np.testing.assert_allclose(holm_adjust(np.array([0.01, 0.03, 0.5])), [0.03, 0.06, 0.5])
    first = block_inference(np.sin(np.arange(100)), 99, 5, 2)
    assert first == block_inference(np.sin(np.arange(100)), 99, 5, 2)


def test_sparse_data_exports_insufficient_evidence(model_case):
    prices, features, config = model_case
    config = replace(config, min_train_sessions=1000)
    run = run_forecasts(prices, features, "SPY", config)
    assert run.oos.empty and not run.latest.model_supported.any()
    validation, _, _ = evaluate(run, config, "SPY", "yahoo")
    assert (
        validation.loc[validation.model != "historical_average", "validation_status"]
        .str.contains("insufficient evidence")
        .all()
    )


def test_vintage_macro_features_reject_revised_snapshots_and_future_revisions():
    closes = session_schedule("XNYS", "2023-01-01", "2024-05-01")
    prices = pd.DataFrame({"SPY": np.linspace(100, 140, len(closes))}, index=closes.index)
    panel = MarketPanel(prices, closes, {"calendar": "XNYS", "currency": "GBP"})
    fx = pd.DataFrame(
        {"usd_per_gbp": 1.25, "fx_available_at_utc": closes - pd.Timedelta(days=1)},
        index=closes.index,
    )
    observations = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-01-01"],
            "vintage_date": ["2024-02-01", "2024-03-01"],
            "realtime_end": ["2024-02-29", "9999-12-31"],
            "value": [100.0, 200.0],
        }
    )
    data = MacroData(observations, {"historical_use": True})
    old = build_features(panel, fx, "SPY", {"CPIAUCSL": data}, ("CPIAUCSL",))
    observations = observations.copy()
    observations.loc[1, "value"] = 999.0
    changed = build_features(
        panel,
        fx,
        "SPY",
        {"CPIAUCSL": MacroData(observations, {"historical_use": True})},
        ("CPIAUCSL",),
    )
    pd.testing.assert_frame_equal(
        old.assets["SPY"].loc[:"2024-02-29"], changed.assets["SPY"].loc[:"2024-02-29"]
    )
    assert old.assets["SPY"].loc["2024-02-15", "macro_CPIAUCSL_known_level"] == 100
    with pytest.raises(ValueError, match="snapshots"):
        build_features(panel, fx, "SPY", {"CPIAUCSL": MacroData(observations, {})})
    fx.fx_available_at_utc = closes + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="unavailable"):
        build_features(panel, fx, "SPY")


def test_pipeline_no_macro_substitution_and_real_export_contract(tmp_path):
    research = ResearchConfig.load("config/cross_asset.yaml")
    # Keep a genuine price benchmark and one asset, without fitting across assets.
    research = replace(
        research,
        assets={a: research.assets[a] for a in ["SPY", "GLD"]},
        backtest=dict(research.backtest, top_n=1),
    )
    scenario = ScenarioConfig(
        {"test": {"asset_returns_usd": {"SPY": 0.1, "GLD": 0.0}, "gbp_per_usd_return": 0.0}},
        {"sample": {"SPY": 1.0}},
        40,
        6,
    )
    source = tmp_path / "snapshot"
    run_scenarios(research, scenario, "2020-01-01", "2022-01-01", source, source="synthetic")
    config = ForecastConfig(
        horizons=(10,),
        min_train_sessions=60,
        min_regime_sessions=10,
        min_regime_blocks=2,
        min_oos_origins=4,
        min_calibration_origins=3,
        bootstrap_draws=99,
    )
    output = source / "forecasts"
    metadata = run_pipeline(source, output, config)
    assert metadata["feature_scope"] == "reduced_price_fx"
    assert metadata["input_sha256"]["gbp_prices.csv"]
    latest = pd.read_csv(output / "latest_forecasts.csv")
    assert set(latest.asset) == {"SPY", "GLD"} and len(latest) == 8
    assert (pd.to_datetime(latest.exit_session) > pd.to_datetime(latest.entry_session)).all()
    assert "No asset-class/horizon/model" in (output / "validation_report.md").read_text()
    with pytest.raises(ValueError, match="Required vintage"):
        run_pipeline(source, output, replace(config, macro_policy="require"))
    assert Path(output / "calibration.csv").is_file()


def test_skill_screen_has_power_but_rejects_a_tied_baseline():
    """Synthetic loss fixtures test the decision rule; they are not market forecasts."""
    from aycapital.forecasting.engine import ForecastRun

    rows = []
    for position in range(80):
        outcome = 0.1 if position % 2 else -0.1
        for model in ["historical_average", "ridge"]:
            good = model == "ridge"
            probability = (0.99 if outcome > 0 else 0.01) if good else 0.5
            rows.append(
                {
                    "asset": "GLD",
                    "horizon": 5,
                    "model": model,
                    "origin_position": position * 5,
                    "actual_return": outcome,
                    "expected_return_gbp": 0.99 * outcome if good else 0.0,
                    "actual_excess_return": outcome,
                    "expected_excess_return": 0.99 * outcome if good else 0.0,
                    "probability_positive": probability,
                    "probability_outperform": probability,
                    "interval_low": -0.2,
                    "interval_high": 0.2,
                    "model_supported": True,
                    "regime": "synthetic_test_fixture",
                }
            )
    oos = pd.DataFrame(rows)
    latest = oos[["asset", "horizon", "model"]].drop_duplicates()
    config = ForecastConfig(horizons=(5,), bootstrap_draws=999)
    data = ForecastRun(oos, latest, pd.DataFrame())
    result, _, _ = evaluate(data, config, "SPY", "yahoo")  # Mock the source gate only.
    assert (
        result.loc[result.model == "ridge", "validation_status"]
        .str.startswith("retrospective")
        .all()
    )
    for column in [
        "expected_return_gbp",
        "expected_excess_return",
        "probability_positive",
        "probability_outperform",
    ]:
        baseline = oos.loc[oos.model == "historical_average", column].to_numpy()
        oos.loc[oos.model == "ridge", column] = baseline
    tied, _, _ = evaluate(ForecastRun(oos, latest, pd.DataFrame()), config, "SPY", "yahoo")
    assert tied.loc[tied.model == "ridge", "validation_status"].str.startswith("insufficient").all()
