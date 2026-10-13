import numpy as np
import pandas as pd
import pytest

from aycapital.cross_asset.market import MarketPanel, session_schedule, validate_panel
from aycapital.cross_asset.signals import calculate_signals, rolling_max_drawdown


@pytest.fixture
def panel():
    closes = session_schedule("XNYS", "2021-01-01", "2024-01-01")
    rng = np.random.default_rng(17)
    increments = rng.normal(0.0004, 0.012, (len(closes), 3))
    prices = pd.DataFrame(
        100 * np.exp(increments.cumsum(axis=0)), index=closes.index, columns=["SPY", "QQQ", "GLD"]
    )
    return MarketPanel(prices, closes, {"calendar": "XNYS", "currency": "USD"})


def test_signal_definitions(panel):
    prices = panel.prices
    signals = calculate_signals(prices, "SPY", {"momentum_60": 1, "volatility_60": -1})
    m = signals.metrics
    assert m["momentum_20"].SPY.iloc[-1] == pytest.approx(
        prices.SPY.iloc[-1] / prices.SPY.iloc[-21] - 1
    )
    assert m["trend_200"].QQQ.iloc[-1] == pytest.approx(
        prices.QQQ.iloc[-1] / prices.QQQ.iloc[-200:].mean() - 1
    )
    assert m["relative_strength"].SPY.dropna().abs().max() == 0
    assert m["relative_strength"].QQQ.iloc[-1] == pytest.approx(
        (prices.QQQ.iloc[-1] / prices.QQQ.iloc[-61]) / (prices.SPY.iloc[-1] / prices.SPY.iloc[-61])
        - 1
    )
    r = prices.pct_change(fill_method=None).iloc[-60:]
    assert m["volatility_60"].GLD.iloc[-1] == pytest.approx(r.GLD.std() * np.sqrt(252))
    assert m["downside_volatility"].GLD.iloc[-1] == pytest.approx(
        np.sqrt(np.mean(np.minimum(r.GLD, 0) ** 2)) * np.sqrt(252)
    )
    assert m["correlation_spy"].QQQ.iloc[-1] == pytest.approx(r.QQQ.corr(r.SPY))
    assert signals.score.max().max() <= 1 and signals.score.min().min() >= 0
    expected = (signals.percentiles["momentum_60"] + 1 - signals.percentiles["volatility_60"]) / 2
    pd.testing.assert_frame_equal(signals.score, expected)


def test_prefix_invariance_and_future_perturbation(panel):
    weights = {"momentum_120": 1, "trend_200": 1, "volatility_60": -1}
    whole = calculate_signals(panel.prices, "SPY", weights)
    cut = 350
    prefix = calculate_signals(panel.prices.iloc[:cut], "SPY", weights)
    changed = panel.prices.copy()
    changed.iloc[cut:] *= np.linspace(1, 100, len(changed) - cut)[:, None]
    future = calculate_signals(changed, "SPY", weights)
    for name in whole.metrics:
        pd.testing.assert_frame_equal(prefix.metrics[name], whole.metrics[name].iloc[:cut])
        pd.testing.assert_frame_equal(prefix.metrics[name], future.metrics[name].iloc[:cut])
    pd.testing.assert_frame_equal(prefix.score, whole.score.iloc[:cut])
    pd.testing.assert_frame_equal(prefix.score, future.score.iloc[:cut])


def test_trailing_drawdown_definition():
    prices = pd.DataFrame({"A": [100.0, 80.0, 90.0, 110.0, 99.0, 120.0]})
    dd = rolling_max_drawdown(prices, 3)
    assert dd.A.iloc[2] == pytest.approx(-0.2)
    assert dd.A.iloc[3] == 0  # Earlier peak has left the window.
    assert dd.A.iloc[-1] == pytest.approx(-0.1)


def test_rank_ties_and_warmup():
    prices = pd.DataFrame(
        {"SPY": np.arange(1.0, 251.0), "QQQ": np.arange(1.0, 251.0)},
        index=pd.bdate_range("2020-01-01", periods=250),
    )
    scores = calculate_signals(prices, "SPY", {"trend_200": 1}).score
    assert scores.iloc[:199].isna().all().all()
    np.testing.assert_allclose(scores.iloc[199:], 0.5)


def test_panel_rejects_calendar_gaps(panel):
    broken = MarketPanel(
        panel.prices.drop(panel.prices.index[100]),
        panel.closes_utc.drop(panel.prices.index[100]),
        panel.metadata,
    )
    with pytest.raises(ValueError, match="sessions"):
        validate_panel(broken)
