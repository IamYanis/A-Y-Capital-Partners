import numpy as np
import pandas as pd
import pytest

from aycapital.cross_asset.backtest import walk_forward
from aycapital.cross_asset.market import MarketPanel, session_schedule
from aycapital.cross_asset.signals import SignalSet, calculate_signals


def tiny_case():
    closes = session_schedule("XNYS", "2024-07-01", "2024-07-16")
    prices = pd.DataFrame(
        {
            "SPY": [100.0] * len(closes),
            "QQQ": [100.0, 200.0, 400.0, *([800.0] * (len(closes) - 3))],
        },
        index=closes.index,
    )
    # Constant ex-ante preference, deliberately no price-dependent fitting.
    scores = pd.DataFrame(
        {"SPY": [0.0] * len(closes), "QQQ": [1.0] * len(closes)}, index=closes.index
    )
    return MarketPanel(prices, closes, {"calendar": "XNYS", "currency": "USD"}), scores


def test_next_session_close_no_early_earnings_and_costs():
    panel, scores = tiny_case()
    result = walk_forward(
        panel,
        SignalSet({}, {}, scores),
        "SPY",
        top_n=1,
        rebalance_sessions=2,
        transaction_cost_bps=100,
        train_fraction=0.6,
    )
    first = result.returns.iloc[0]
    assert first.strategy == pytest.approx(1 / 1.01 - 1)  # t+1 fill only, misses t+1 doubling.
    assert result.returns.strategy.iloc[1] == pytest.approx(1.0)  # t+2 first earned return.
    assert result.turnover.strategy.iloc[0] == pytest.approx(1 / 1.01)
    trades = result.trades
    assert (trades.signal_available_after_utc < trades.execution_close_utc).all()
    assert trades.iloc[0].execution_session == pd.Timestamp("2024-07-02")
    # July 4 holiday is not a rebalance or return session.
    assert pd.Timestamp("2024-07-04") not in result.returns.index
    assert len(result.statistics) == 6
    out = result.returns.loc[result.periods == "out_of_sample"]
    assert out.strategy.iloc[0] == pytest.approx(1 / 1.01 - 1)  # independent fresh cash account


def test_drift_turnover_not_just_target_changes():
    panel, scores = tiny_case()
    result = walk_forward(
        panel,
        SignalSet({}, {}, scores),
        "SPY",
        top_n=2,
        rebalance_sessions=2,
        transaction_cost_bps=0,
        train_fraction=0.6,
    )
    # Equal weights bought at July 2 close; subsequent QQQ growth drifts holdings.
    assert result.turnover.equal_weight.iloc[2] > 0
    assert result.turnover.SPY.iloc[1:].sum() == 1  # only OOS entry adds a trade
    assert (result.returns >= -1).all().all()


def test_future_prices_cannot_change_prior_trades_or_returns():
    closes = session_schedule("XNYS", "2020-01-01", "2024-01-01")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        100 * np.exp(rng.normal(0, 0.01, (len(closes), 3)).cumsum(axis=0)),
        index=closes.index,
        columns=["SPY", "QQQ", "GLD"],
    )
    weights = {"momentum_60": 1, "trend_200": 1}
    original_panel = MarketPanel(prices, closes, {"calendar": "XNYS"})
    original = walk_forward(
        original_panel, calculate_signals(prices, "SPY", weights), "SPY", top_n=1
    )
    cutoff = 800
    future_prices = prices.copy()
    future_prices.iloc[cutoff:, 1] *= 50
    future_panel = MarketPanel(future_prices, closes, original_panel.metadata)
    changed = walk_forward(
        future_panel, calculate_signals(future_prices, "SPY", weights), "SPY", top_n=1
    )
    date = prices.index[cutoff]
    pd.testing.assert_frame_equal(
        original.returns.loc[: prices.index[cutoff - 1]],
        changed.returns.loc[: prices.index[cutoff - 1]],
    )
    pd.testing.assert_frame_equal(
        original.trades.loc[original.trades.execution_session < date],
        changed.trades.loc[changed.trades.execution_session < date],
    )
    stats = original.statistics.loc[("out_of_sample", "strategy")]
    daily = original.returns.strategy.loc[original.periods == "out_of_sample"]
    assert stats.cagr == pytest.approx((1 + daily).prod() ** (252 / len(daily)) - 1)
    assert stats.annualized_turnover >= 0


def test_invalid_backtest_parameters():
    panel, scores = tiny_case()
    for kwargs in (
        {"transaction_cost_bps": -1},
        {"train_fraction": 1},
        {"top_n": 10},
        {"rebalance_sessions": 0},
    ):
        with pytest.raises(ValueError):
            walk_forward(panel, SignalSet({}, {}, scores), "SPY", **kwargs)


def test_exact_fee_cash_purchase_and_asset_rotation():
    from aycapital.cross_asset.backtest import rebalance_accounting

    labels = ["SPY", "QQQ"]
    target = pd.Series([0.0, 1.0], index=labels)
    u, turnover = rebalance_accounting(pd.Series([0.0, 0.0], index=labels), target, 0.1)
    assert u == pytest.approx(1 / 1.1)
    assert turnover == pytest.approx(u)
    u, turnover = rebalance_accounting(pd.Series([1.0, 0.0], index=labels), target, 0.1)
    assert u == pytest.approx(0.9 / 1.1)
    assert turnover == pytest.approx(1 + u)
    assert u + 0.1 * turnover == pytest.approx(1)


def test_fee_accounting_against_independent_share_ledger():
    panel, scores = tiny_case()
    result = walk_forward(
        panel,
        SignalSet({}, {}, scores),
        "SPY",
        top_n=2,
        rebalance_sessions=2,
        transaction_cost_bps=75,
        train_fraction=0.6,
    )
    for label in ["in_sample", "out_of_sample"]:
        dates = result.returns.index[result.periods == label]
        cash, shares, nav_before = 1.0, np.zeros(2), 1.0
        for i, date in enumerate(dates):
            prices = panel.prices.loc[date].to_numpy()
            value = cash + shares @ prices
            if i % 2 == 0:
                # Independent bisection in currency amounts, not portfolio weights.
                low, high = 0.0, value
                for _ in range(60):
                    budget = (low + high) / 2
                    fee = 0.0075 * np.abs(budget / 2 - shares * prices).sum()
                    if budget + fee > value:
                        high = budget
                    else:
                        low = budget
                cash = 0.0
                shares = ((low + high) / 2) / 2 / prices
                value = shares @ prices
            assert result.returns.loc[date, "equal_weight"] == pytest.approx(value / nav_before - 1)
            nav_before = value


def test_infinite_signal_scores_rejected():
    panel, scores = tiny_case()
    scores.iloc[2, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        walk_forward(panel, SignalSet({}, {}, scores), "SPY", top_n=1)
