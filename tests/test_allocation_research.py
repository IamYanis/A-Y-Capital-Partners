"""Causal allocation research tests: deterministic fixtures, no network/skill claims."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aycapital.allocation_research.evaluation import paired_uncertainty, regime_flags
from aycapital.allocation_research.macro import PublicVintageProvider, monthly_features
from aycapital.allocation_research.pipeline import load_historical, run_comparison, write_report
from aycapital.allocation_research.strategies import risk_parity, targets_for_horizon
from aycapital.cross_asset.backtest import allocation_history, fixed_weight_history
from aycapital.cross_asset.market import MarketPanel, session_schedule


@pytest.fixture
def plan():
    config = yaml.safe_load(Path("config/allocation_research.yaml").read_text())
    return {
        **config,
        "defensive_weights": {"SGOV": 0.8, "SPY": 0.2},
        "holding_sessions": [20, 40],
        "bootstrap_draws": 49,
    }


@pytest.fixture
def panel():
    closes = session_schedule("XNYS", "2020-01-01", "2023-01-01")
    rng = np.random.default_rng(199)
    prices = pd.DataFrame(
        100
        * np.exp(
            rng.normal([0.0003, 0.0001, 0.0002], [0.012, 0.003, 0.008], (len(closes), 3))
        ).cumprod(axis=0),
        index=closes.index,
        columns=["SPY", "SGOV", "GLD"],
    )
    return MarketPanel(prices, closes, {"calendar": "XNYS", "currency": "GBP"})


def test_risk_parity_matches_inverse_vol_on_diagonal():
    covariance = pd.DataFrame(np.diag([0.04, 0.01, 0.09]), index=list("abc"), columns=list("abc"))
    weights = risk_parity(covariance)
    expected = np.array([1 / 0.2, 1 / 0.1, 1 / 0.3])
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=1e-5)
    contributions = weights * (covariance @ weights)
    np.testing.assert_allclose(contributions / contributions.sum(), 1 / 3, atol=1e-5)
    with pytest.raises(ValueError, match="positive-definite"):
        risk_parity(covariance * 0)


@pytest.mark.parametrize("horizon", [20, 40])
def test_future_perturbation_invariance_and_next_close(panel, plan, horizon):
    original = targets_for_horizon(panel.prices, plan, horizon)
    changed = panel.prices.copy()
    cutoff = 500
    changed.iloc[cutoff:, 0] *= 2
    perturbed = targets_for_horizon(changed, plan, horizon)
    for name in original:
        pd.testing.assert_frame_equal(original[name].iloc[:cutoff], perturbed[name].iloc[:cutoff])
    daily, turnover, trades = allocation_history(panel, original, "SPY", 200, horizon, 10)
    later = MarketPanel(changed, panel.closes_utc, panel.metadata)
    changed_daily, _, _ = allocation_history(later, perturbed, "SPY", 200, horizon, 10)
    pd.testing.assert_frame_equal(
        daily.loc[: panel.prices.index[cutoff - 1]],
        changed_daily.loc[: panel.prices.index[cutoff - 1]],
    )
    np.testing.assert_allclose(daily.iloc[0], 1 / 1.001 - 1)
    assert (trades.signal_available_after_utc < trades.execution_close_utc).all()
    assert trades.query("portfolio == 'SPY'").shape[0] == 1
    assert turnover.iloc[1].eq(0).all()


def test_dynamic_ledger_reproduces_fixed_allocation_ledger(panel):
    weights = pd.Series([0.5, 0.3, 0.2], index=panel.prices.columns)
    benchmark = pd.Series([1.0, 0.0, 0.0], index=weights.index)
    fixed = {"mix": weights, "SPY": benchmark}
    histories = {
        name: pd.DataFrame([w] * len(panel.prices), index=panel.prices.index)
        for name, w in fixed.items()
    }
    expected = fixed_weight_history(panel, fixed, "SPY", 20, 25)
    actual = allocation_history(panel, histories, "SPY", 1, 20, 25)
    pd.testing.assert_frame_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1])
    with pytest.raises(ValueError, match="costs"):
        allocation_history(panel, histories, "SPY", 1, 20, np.nan)
    with pytest.raises(ValueError, match="long-only"):
        allocation_history(panel, {**histories, "bad": histories["mix"] * 2}, "SPY", 1, 20, 0)


def test_macros_use_same_vintage_lags_and_next_day_availability():
    rows = []
    for vintage, recent in [("2021-06-30", 105), ("2021-07-15", 120)]:
        for date, value in [
            ("2020-05-01", 100),
            ("2020-02-01", 100),
            ("2021-02-01", 102),
            ("2021-05-01", recent),
        ]:
            # Historical denominator gets revised only in the later snapshot.
            if vintage == "2021-07-15" and date == "2020-05-01":
                value = 110
            rows.append(
                dict(series="CPIAUCSL", observation_date=date, known_by_date=vintage, value=value)
            )
    snapshots = pd.DataFrame(rows)
    dates = pd.to_datetime(["2021-06-30 20:00Z", "2021-07-01 20:00Z", "2021-07-16 20:00Z"])
    closes = pd.Series(dates, index=dates.tz_localize(None).normalize())
    features = monthly_features(snapshots, closes)
    assert np.isnan(features.CPIAUCSL_yoy.iloc[0])
    assert features.CPIAUCSL_yoy.iloc[1] == pytest.approx(5)
    assert features.CPIAUCSL_yoy.iloc[2] == pytest.approx((120 / 110 - 1) * 100)
    prior = monthly_features(snapshots.iloc[:4], closes)
    pd.testing.assert_frame_equal(features.iloc[:2], prior.iloc[:2])
    assert features.CPIAUCSL_observation_date.iloc[1] == pd.Timestamp("2021-05-01")
    later = pd.Series(pd.to_datetime(["2021-10-01 20:00Z"]))
    assert monthly_features(snapshots, later).CPIAUCSL_yoy.isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        monthly_features(pd.concat([snapshots, snapshots]), closes)


def test_missing_macros_do_not_imply_recovery(panel, plan):
    flags = regime_flags(panel.prices, pd.DataFrame(index=panel.prices.index), plan)
    assert flags.inflation_shock.isna().all()
    assert flags.economic_slowdown.isna().all()
    assert flags.recovery.isna().all()


def test_public_vintage_identity_cache_and_no_latest_fallback(tmp_path):
    class Response:
        text = "observation_date,CPIAUCSL_20210131\n2020-12-01,100\n"

        def raise_for_status(self):
            pass

    class Session:
        calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            assert "vintage_date" in params
            return Response()

    session = Session()
    provider = PublicVintageProvider(tmp_path, session)
    data, metadata = provider.load("CPIAUCSL", "2021-01-01", "2021-01-31")
    assert data.known_by_date.iloc[0] == pd.Timestamp("2021-01-31")
    assert metadata[0]["historical_use"]
    provider.load("CPIAUCSL", "2021-01-01", "2021-01-31")
    assert session.calls == 1
    Response.text = "observation_date,CPIAUCSL\n2020-12-01,999\n"
    with pytest.raises(ValueError, match="vintage identities"):
        provider.load("CPIAUCSL", "2021-02-01", "2021-02-28")


def save_input(directory, panel, usd_per_gbp=1.25):
    panel.prices.to_csv(directory / "gbp_prices.csv")
    metadata = {
        "currency": "GBP",
        "source": "synthetic",
        "market": panel.metadata,
        "last_observation": str(panel.prices.index[-1].date()),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    alignment = pd.DataFrame(index=panel.prices.index)
    alignment["asset_close_utc"] = panel.closes_utc
    alignment["usd_per_gbp"] = usd_per_gbp
    alignment["fx_observation_date"] = panel.prices.index - pd.Timedelta(days=1)
    alignment["fx_available_at_utc"] = panel.prices.index.tz_localize("UTC")
    alignment["fx_age_calendar_days"] = (
        panel.closes_utc
        - pd.Series(
            pd.to_datetime(alignment.fx_observation_date, utc=True), index=panel.prices.index
        )
    ).dt.total_seconds() / 86400
    alignment.to_csv(directory / "fx_alignment.csv")


def test_holdout_guard_precedes_price_value_parsing_and_fx_alignment(tmp_path, panel, plan):
    save_input(tmp_path, panel)
    loaded, native, _ = load_historical(tmp_path, plan)
    np.testing.assert_allclose(native / loaded.prices, 1.25)
    # Historical metadata lies: actual index still prevents loading holdout values.
    (tmp_path / "gbp_prices.csv").write_text("session,SPY\n2026-10-15,DO_NOT_PARSE\n")
    with pytest.raises(ValueError, match="Sealed prospective"):
        load_historical(tmp_path, plan)
    save_input(tmp_path, panel)
    fx = pd.read_csv(tmp_path / "fx_alignment.csv", index_col=0)
    fx.iloc[0, fx.columns.get_loc("fx_available_at_utc")] = "2026-10-15 00:00:00+00:00"
    fx.to_csv(tmp_path / "fx_alignment.csv")
    with pytest.raises(ValueError, match="unavailable"):
        load_historical(tmp_path, plan)


def test_paired_bootstrap_identical_and_reproducible():
    rng = np.random.default_rng(13)
    x = rng.normal(0.001, 0.01, 400)
    identical = paired_uncertainty(x, x, 99, 40, 9)
    assert identical["sharpe_low"] == 0
    assert identical["sharpe_high"] == 0
    assert identical["p_value"] == 1
    assert paired_uncertainty(x, x + 0.001, 99, 40, 9) == paired_uncertainty(
        x, x + 0.001, 99, 40, 9
    )
    with pytest.raises(ValueError, match="two full"):
        paired_uncertainty(x[:10], x[:10], 99, 40, 9)


def test_comparison_has_all_costs_no_false_small_sample_claims(panel, plan, tmp_path):
    tables = run_comparison(
        panel, panel.prices * 1.25, pd.DataFrame(index=panel.prices.index), plan
    )
    assert len(tables["performance"]) == 36
    assert len(tables["evidence"]) == 6
    assert not tables["evidence"].historical_gate_pass.any()
    assert tables["evidence"].prospective_evidence.str.contains("insufficient").all()
    performance = tables["performance"]
    np.testing.assert_allclose(performance.gbp_minus_native_total_return, 0, atol=1e-10)
    for _, group in performance.groupby(["holding_sessions", "portfolio"]):
        assert group.sort_values("cost_bps").cumulative_return.is_monotonic_decreasing
    assert not tables["regimes"].query("regime == 'economic_slowdown'").sessions.any()
    assert (tables["holding_returns"].end > tables["holding_returns"].start).all()
    metadata = dict(
        evaluation_start="2020-10-16",
        last_observation="2022-12-30",
        source="synthetic",
        macro_warnings=["missing"],
        warnings=["synthetic"],
    )
    write_report(tables, metadata, tmp_path)
    assert "No tested dynamic allocation" in (tmp_path / "report.md").read_text()


def test_dashboard_reads_saved_evidence_and_detects_changed_inputs(tmp_path):
    pytest.importorskip("streamlit")
    import hashlib

    from streamlit.testing.v1 import AppTest

    source = tmp_path / "source"
    source.mkdir()
    (source / "gbp_prices.csv").write_text("session,SPY\n2022-01-03,100\n")
    output = tmp_path / "allocations"
    output.mkdir()
    metadata = {
        "evaluation_start": "2021-07-30",
        "last_observation": "2022-12-30",
        "source": "synthetic",
        "prospective_status": "sealed_unscored",
        "input_sha256": {
            "gbp_prices.csv": hashlib.sha256((source / "gbp_prices.csv").read_bytes()).hexdigest()
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata))
    (output / "report.md").write_text("Synthetic fixture only")
    pd.DataFrame(
        [
            dict(
                portfolio="equal_weight",
                holding_sessions=h,
                cost_bps=c,
                cagr=0.01,
                annualized_volatility=0.1,
                max_drawdown=-0.1,
                zero_hurdle_sharpe=0.1,
                annualized_turnover=0.5,
                gbp_minus_native_total_return=0.01,
            )
            for h in [20, 40]
            for c in [10, 25, 50]
        ]
    ).to_csv(output / "performance.csv", index=False)
    pd.DataFrame([dict(holding_sessions=20, historical_gate_pass=False)]).to_csv(
        output / "evidence.csv", index=False
    )
    pd.DataFrame(
        [
            dict(
                holding_sessions=20,
                cost_bps=10,
                portfolio="equal_weight",
                session="2022-12-30",
                return_252=0.01,
            )
        ]
    ).to_csv(output / "rolling.csv", index=False)
    pd.DataFrame(
        [
            dict(
                holding_sessions=20,
                cost_bps=10,
                regime="slowdown",
                evidence="insufficient_evidence",
            )
        ]
    ).to_csv(output / "regimes.csv", index=False)
    script = (
        "from pathlib import Path\n"
        "from aycapital.allocation_research.dashboard import render_allocations\n"
        f"render_allocations(Path({str(source)!r}))\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()
    app.text_input[0].set_value(str(output)).run()
    assert not app.exception
    assert any("Insufficient evidence" in info.value for info in app.info)
    app.selectbox[0].set_value(40).run()
    assert not app.exception
    (source / "gbp_prices.csv").write_text("changed snapshot")
    app.run()
    assert not app.exception
    assert any("different saved input snapshot" in warning.value for warning in app.warning)


def test_registered_holdout_cannot_be_shifted_by_configuration(tmp_path, panel, plan):
    save_input(tmp_path, panel)
    with pytest.raises(ValueError, match="boundaries"):
        load_historical(
            tmp_path, {**plan, "historical_end": "2027-10-14", "prospective_start": "2027-10-15"}
        )


def test_currency_return_identity_with_identical_targets_and_constant_usd_assets():
    closes = session_schedule("XNYS", "2024-07-01", "2024-07-16")
    rate = pd.Series(np.linspace(1.2, 1.4, len(closes)), index=closes.index)
    usd = pd.DataFrame(100.0, index=closes.index, columns=["SPY", "GLD"])
    gbp = usd.div(rate, axis=0)
    histories = {
        "mix": pd.DataFrame({"SPY": 0.5, "GLD": 0.5}, index=closes.index),
        "SPY": pd.DataFrame({"SPY": 1.0, "GLD": 0.0}, index=closes.index),
    }
    gbp_result, _, _ = allocation_history(
        MarketPanel(gbp, closes, {"calendar": "XNYS", "currency": "GBP"}),
        histories,
        "SPY",
        1,
        20,
        0,
    )
    usd_result, _, _ = allocation_history(
        MarketPanel(usd, closes, {"calendar": "XNYS", "currency": "USD"}),
        histories,
        "SPY",
        1,
        20,
        0,
    )
    assert gbp_result.mix.iloc[0] == 0  # Fill only; no pre-entry FX return.
    np.testing.assert_allclose(
        gbp_result.mix.iloc[1:], (rate.shift(1) / rate - 1).iloc[2:], atol=1e-15
    )
    assert usd_result.eq(0).all().all()
    assert (1 + gbp_result.mix).prod() == pytest.approx(rate.iloc[1] / rate.iloc[-1])
