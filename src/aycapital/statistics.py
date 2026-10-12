"""Historical statistics, using simple returns and explicit annualization."""

import numpy as np
import pandas as pd

from aycapital.validation import frame, periods, returns


def price_returns(prices: pd.DataFrame, *, log: bool = False) -> pd.DataFrame:
    prices = frame(prices)
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    result = prices.pct_change(fill_method=None).iloc[1:]
    return np.log1p(result) if log else result


def cumulative_returns(data: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns(data)).cumprod() - 1


def drawdowns(data: pd.DataFrame) -> pd.DataFrame:
    wealth = 1 + cumulative_returns(data)
    # Include starting wealth of one, including a loss on the first observation.
    peak = wealth.cummax().clip(lower=1)
    return wealth / peak - 1


def rolling_returns(data: pd.DataFrame, window: int) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be positive")
    return (1 + returns(data)).rolling(window).apply(np.prod, raw=True) - 1


def rolling_volatility(data: pd.DataFrame, window: int, periods_per_year: int = 252):
    if window < 2:
        raise ValueError("Volatility window must be at least two")
    return returns(data).rolling(window).std(ddof=1) * np.sqrt(periods(periods_per_year))


def summary(data: pd.DataFrame, periods_per_year: int = 252,
            risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Risk-free rate is annual effective; ratios use daily excess simple returns.

    Sortino downside deviation includes all observations, with positive excess set
    to zero. Undefined ratios (zero risk) are NaN. Kurtosis is excess kurtosis.
    """
    data = returns(data)
    annual = periods(periods_per_year)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("Invalid annual risk-free rate")
    excess = data - np.expm1(np.log1p(risk_free_rate) / annual)
    std = data.std(ddof=1)
    downside = np.sqrt(excess.clip(upper=0).pow(2).mean())
    return pd.DataFrame({
        "cumulative_return": cumulative_returns(data).iloc[-1],
        "annualized_volatility": std * np.sqrt(annual),
        "max_drawdown": drawdowns(data).min(),
        "skewness": data.skew(),
        "excess_kurtosis": data.kurt(),
        "sharpe": excess.mean() / std.replace(0, np.nan) * np.sqrt(annual),
        "sortino": excess.mean() / downside.replace(0, np.nan) * np.sqrt(annual),
    })
