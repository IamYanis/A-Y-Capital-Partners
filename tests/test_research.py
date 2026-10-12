import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from aycapital.covariance import sample_covariance, shrinkage_covariance
from aycapital.data import load_yahoo, synthetic_prices, to_gbp, validate_prices
from aycapital.demo import main
from aycapital.portfolio import Constraints, optimize
from aycapital.statistics import (
    cumulative_returns,
    drawdowns,
    price_returns,
    rolling_returns,
    rolling_volatility,
    summary,
)


@pytest.fixture
def prices():
    return pd.DataFrame({"A": [100., 110., 99., 108.9], "B": [50., 55., 60.5, 54.45]},
                        index=pd.date_range("2024-01-01", periods=4))


def test_returns_and_compounding(prices):
    result = price_returns(prices)
    np.testing.assert_allclose(result.A, [0.1, -0.1, 0.1])
    np.testing.assert_allclose(price_returns(prices, log=True), np.log1p(result))
    assert cumulative_returns(result).A.iloc[-1] == pytest.approx(0.089)
    assert rolling_returns(result, 2).A.iloc[-1] == pytest.approx(-0.01)
    assert np.isnan(rolling_returns(result, 2).A.iloc[0])


def test_risk_statistics(prices):
    result = price_returns(prices)
    stats = summary(result, 252)
    assert stats.loc["A", "annualized_volatility"] == pytest.approx(
        np.std([.1, -.1, .1], ddof=1) * np.sqrt(252))
    assert stats.loc["A", "max_drawdown"] == pytest.approx(-.1)
    assert stats.loc["A", "sharpe"] == pytest.approx(
        result.A.mean() / result.A.std() * np.sqrt(252))
    assert stats.loc["A", "sortino"] == pytest.approx(
        result.A.mean() / np.sqrt(.01 / 3) * np.sqrt(252))
    np.testing.assert_allclose(rolling_volatility(result, 2).iloc[1:],
                               result.rolling(2).std().iloc[1:] * np.sqrt(252))
    losses = pd.DataFrame({"A": [-.2, .1, -.1]})
    assert drawdowns(losses).A.iloc[0] == pytest.approx(-.2)
    assert summary(pd.DataFrame({"A": [0., 0., 0.]})).loc["A", "sharpe"] is not None
    assert np.isnan(summary(pd.DataFrame({"A": [0., 0., 0.]})).loc["A", "sharpe"])


def test_risk_free_convention():
    data = pd.DataFrame({"A": [.02, -.01, .01, -.02, .03]})
    excess = data.A - (1.05 ** (1 / 12) - 1)
    stats = summary(data, 12, .05)
    assert stats.loc["A", "sharpe"] == pytest.approx(
        excess.mean() / data.A.std() * np.sqrt(12))
    assert stats.loc["A", "skewness"] == pytest.approx(data.A.skew())
    assert stats.loc["A", "excess_kurtosis"] == pytest.approx(data.A.kurt())


@pytest.mark.parametrize("defect", ["missing", "zero", "duplicate", "unsorted", "infinite"])
def test_invalid_prices(prices, defect):
    if defect == "missing":
        prices.iloc[1, 0] = np.nan
    elif defect == "zero":
        prices.iloc[1, 0] = 0
    elif defect == "duplicate":
        prices.index = [prices.index[0]] * len(prices)
    elif defect == "unsorted":
        prices = prices.iloc[::-1]
    else:
        prices.iloc[1, 0] = np.inf
    with pytest.raises(ValueError):
        price_returns(prices)


def test_covariance(prices):
    data = price_returns(prices)
    np.testing.assert_allclose(sample_covariance(data, 12), np.cov(data.T, ddof=1) * 12)
    shrink = shrinkage_covariance(data, 12)
    np.testing.assert_allclose(shrink, LedoitWolf().fit(data).covariance_ * 12)
    assert shrink.columns.equals(data.columns)
    assert np.linalg.eigvalsh(shrink).min() >= -1e-12
    with pytest.raises(ValueError):
        sample_covariance(data.assign(A=np.nan))


def test_minimum_variance_and_limits():
    cov = pd.DataFrame(np.diag([1., 4.]), index=["A", "B"], columns=["A", "B"])
    weights = optimize(cov, Constraints())
    np.testing.assert_allclose(weights, [.8, .2], atol=1e-5)
    limited = optimize(cov, Constraints(.6))
    np.testing.assert_allclose(limited, [.6, .4], atol=1e-5)
    grouped = optimize(cov, Constraints(exposure_caps={"group": (("A",), .3)}))
    np.testing.assert_allclose(grouped, [.3, .7], atol=1e-5)
    forecast = pd.Series([0., 5.], index=cov.index)
    assert optimize(cov, Constraints(), forecast).B > weights.B


@pytest.mark.parametrize("constraint", [
    Constraints(.4), Constraints(eligible_assets=("A",)),
    Constraints(exposure_caps={"all": (("A", "B"), .5)}),
    Constraints(exposure_caps={"unknown": (("C",), .5)}),
])
def test_reject_constraints(constraint):
    cov = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError):
        optimize(cov, constraint)


def test_invalid_covariance_and_forecasts():
    cov = pd.DataFrame([[1., 2.], [2., 1.]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError):
        optimize(cov, Constraints())
    cov = cov * 0 + np.eye(2)
    with pytest.raises(ValueError):
        optimize(cov, Constraints(), pd.Series([1., 2.], index=["B", "A"]))


def test_fx(prices):
    fx = prices * 0 + [0.01, 0.8]
    converted = to_gbp(prices, fx)
    assert converted.A.iloc[0] == 1
    assert converted.B.iloc[0] == 40
    with pytest.raises(ValueError):
        to_gbp(prices, fx.iloc[1:])
    # FX changes affect returns, not just the initial quote conversion.
    fx.loc[fx.index[1], "B"] = .9
    assert price_returns(to_gbp(prices, fx)).B.iloc[0] == pytest.approx(55 * .9 / 40 - 1)


def test_provider_and_cache(prices, tmp_path, monkeypatch):
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return pd.concat({"Close": prices}, axis=1)

    monkeypatch.setattr("aycapital.data.yf.download", download)
    first = load_yahoo(["A", "B"], "2024-01-01", "2024-01-05", tmp_path)
    second = load_yahoo(["A", "B"], "2024-01-01", "2024-01-05", tmp_path)
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert len(calls) == 1 and calls[0]["auto_adjust"] is True
    assert first.attrs["source"]["retrieved_at_utc"]
    next(tmp_path.glob("*.csv")).write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_yahoo(["A", "B"], "2024-01-01", "2024-01-05", tmp_path)


def test_provider_missing_and_single(prices, tmp_path, monkeypatch):
    monkeypatch.setattr("aycapital.data.yf.download", lambda *a, **k:
                        prices[["A"]].rename(columns={"A": "Close"}))
    assert load_yahoo(["A"], "2024-01-01", "2024-01-05", tmp_path).shape == (4, 1)
    monkeypatch.setattr("aycapital.data.yf.download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no data"):
        load_yahoo(["B"], "2024-01-01", "2024-01-05", tmp_path)
    with pytest.raises(ValueError):
        validate_prices(prices.assign(A=np.nan))


def test_synthetic_and_demo(tmp_path, monkeypatch):
    pd.testing.assert_frame_equal(synthetic_prices(["A", "B"]), synthetic_prices(["A", "B"]))
    monkeypatch.setattr(sys, "argv", ["aycapital-demo", "--output", str(tmp_path)])
    main()
    weights = pd.read_csv(tmp_path / "weights.csv", index_col=0).weight
    assert weights.sum() == pytest.approx(1)
    assert weights.max() <= .6000001
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["provenance"]["source"] == "synthetic"
    assert metadata["configuration"]["competition"]["rules_configured"] is False


@pytest.mark.parametrize("annual", [0, -1, 2.5, True])
def test_invalid_annualization(annual):
    with pytest.raises(ValueError):
        summary(pd.DataFrame({"A": [.01, -.02]}), annual)


def test_loss_and_missing_return_rejected():
    with pytest.raises(ValueError):
        cumulative_returns(pd.DataFrame({"A": [-1., .1]}))
    with pytest.raises(ValueError):
        cumulative_returns(pd.DataFrame({"A": [np.nan, .1]}))


def test_duplicate_assets_and_invalid_dates(prices):
    prices.columns = ["A", "A"]
    with pytest.raises(ValueError):
        validate_prices(prices)
    prices.columns = ["A", "B"]
    prices.index = pd.DatetimeIndex([pd.NaT, *prices.index[1:]])
    with pytest.raises(ValueError):
        validate_prices(prices)


def test_provider_rejects_missing_assets_and_gaps(prices, tmp_path, monkeypatch):
    monkeypatch.setattr("aycapital.data.yf.download", lambda *a, **k:
                        pd.concat({"Close": prices[["A"]]}, axis=1))
    with pytest.raises(ValueError, match="omitted"):
        load_yahoo(["A", "B"], "2024-01-01", "2024-01-05", tmp_path)
    prices.iloc[1, 0] = np.nan
    monkeypatch.setattr("aycapital.data.yf.download", lambda *a, **k:
                        pd.concat({"Close": prices}, axis=1))
    with pytest.raises(ValueError, match="Missing"):
        load_yahoo(["A", "B"], "2024-01-01", "2024-01-05", tmp_path)
