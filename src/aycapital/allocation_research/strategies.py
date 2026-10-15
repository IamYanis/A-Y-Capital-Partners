"""Prespecified, trailing-only GBP allocations. No return prediction or tuning."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from aycapital.covariance import shrinkage_covariance
from aycapital.cross_asset.scenarios import allocation
from aycapital.data import validate_prices


def risk_parity(covariance: pd.DataFrame) -> pd.Series:
    """Equal positive risk budgets via convex log-barrier risk budgeting.

    Solve in volatility-scaled coordinates, then normalize to a fully invested
    long-only portfolio. Require positive-definite covariance and verify budgets.
    """
    if (
        not covariance.index.equals(covariance.columns)
        or not np.isfinite(covariance.to_numpy()).all()
    ):
        raise ValueError("Invalid covariance labels or values")
    cov = covariance.to_numpy()
    if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Risk parity needs positive-definite covariance")
    vol = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(vol, vol)
    n = len(vol)
    result = minimize(
        lambda x: 0.5 * x @ correlation @ x - np.log(x).sum() / n,
        np.ones(n) / np.sqrt(n),
        jac=lambda x: correlation @ x - 1 / (n * x),
        bounds=[(1e-10, None)] * n,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000},
    )
    weights = result.x / vol
    weights /= weights.sum()
    contributions = weights * (cov @ weights)
    if not result.success or not np.allclose(contributions / contributions.sum(), 1 / n, atol=1e-5):
        raise ValueError("Risk parity failed risk-budget verification")
    return pd.Series(weights, index=covariance.index)


def targets_for_horizon(prices: pd.DataFrame, plan: dict, horizon: int) -> dict[str, pd.DataFrame]:
    """Compute only scheduled decision rows; carry targets between decisions.

    The ledger reads the PREVIOUS row. Risk estimates include the decision close,
    never the execution close. Leading warmup rows stay missing.
    """
    prices = validate_prices(prices)
    lookback, trend_window = plan["estimation_sessions"], plan["trend_sessions"]
    first = max(lookback, trend_window - 1)
    if len(prices) <= first + 1 or horizon not in plan["holding_sessions"]:
        raise ValueError("Insufficient history or unregistered holding period")
    returns = prices.pct_change(fill_method=None)
    trend = prices / prices.rolling(trend_window).mean() - 1
    names = [
        "equal_weight",
        "inverse_volatility",
        "risk_parity",
        "trend_following",
        "defensive",
        plan["benchmark"],
    ]
    if plan["benchmark"] not in prices or plan["defensive_asset"] not in prices:
        raise ValueError("Missing benchmark or defensive asset")
    histories = {
        name: pd.DataFrame(np.nan, index=prices.index, columns=prices.columns) for name in names
    }
    equal = pd.Series(1 / len(prices.columns), index=prices.columns)
    defensive = allocation(plan["defensive_weights"], prices.columns)
    benchmark = allocation({plan["benchmark"]: 1.0}, prices.columns)
    for position in range(first, len(prices), horizon):
        trailing = returns.iloc[position - lookback + 1 : position + 1]
        vol = trailing.std(ddof=1)
        if vol.isna().any() or (vol <= 0).any():
            raise ValueError("Missing or zero risk estimate; cannot silently reallocate")
        inverse = (1 / vol) / (1 / vol).sum()
        parity = risk_parity(shrinkage_covariance(trailing))
        active = trend.iloc[position] > 0
        trend_weights = equal * active
        trend_weights[plan["defensive_asset"]] += 1 - trend_weights.sum()
        targets = [equal, inverse, parity, trend_weights, defensive, benchmark]
        for name, weights in zip(names, targets, strict=True):
            histories[name].iloc[position] = weights
    return {name: history.ffill() for name, history in histories.items()}
