"""Paired dependence-aware uncertainty and descriptive regime diagnostics."""

import numpy as np
import pandas as pd

from aycapital.forecasting.uncertainty import block_inference


def sharpe(values: np.ndarray, axis=0):
    """Annualized return/volatility with zero hurdle; NOT GBP cash excess Sharpe."""
    vol = np.std(values, axis=axis, ddof=1)
    return np.divide(
        np.mean(values, axis=axis) * np.sqrt(252),
        vol,
        out=np.full_like(vol, np.nan, dtype=float),
        where=vol > 0,
    )


def paired_uncertainty(candidate, reference, draws, block, seed) -> dict:
    """Circular paired block bootstrap of recomputed Sharpe differences.

    Center the bootstrap statistic around the observed estimate to approximate
    the null; serial dependence within blocks is retained. Local stationarity is
    an assumption, not a guarantee. CI and p are approximate, not exact tests.
    """
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if len(candidate) != len(reference) or len(candidate) < 2 * block:
        raise ValueError("Need paired observations and at least two full bootstrap blocks")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite paired observations")
    rng = np.random.default_rng(seed)
    n = len(candidate)
    starts = rng.integers(n, size=(draws, int(np.ceil(n / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n] % n
    estimate = float(sharpe(candidate) - sharpe(reference))
    sampled = sharpe(candidate[indices], axis=1) - sharpe(reference[indices], axis=1)
    if not np.isfinite(sampled).all() or not np.isfinite(estimate):
        return {
            "sharpe_difference": np.nan,
            "sharpe_low": np.nan,
            "sharpe_high": np.nan,
            "p_value": 1.0,
            "annual_return_difference_low": np.nan,
            "annual_return_difference_high": np.nan,
        }
    low, high = np.quantile(sampled, [0.025, 0.975])
    p = (1 + np.count_nonzero(sampled - estimate >= estimate)) / (draws + 1)
    mean_low, mean_high, _ = block_inference(candidate - reference, draws, block, seed)
    return {
        "sharpe_difference": estimate,
        "sharpe_low": low,
        "sharpe_high": high,
        "p_value": p,
        "annual_return_difference_low": mean_low * 252,
        "annual_return_difference_high": mean_high * 252,
    }


def regime_flags(prices: pd.DataFrame, macro: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Decision-known overlapping diagnostic buckets, not allocation switches.

    Missing macro means unknown, never a benign macro state. The recovery flag
    requires both macro flags known and false. Regime return attribution uses
    the PREVIOUS session's flags; same-close information never labels old returns.
    """
    spy = prices[plan["benchmark"]]
    flags = pd.DataFrame(index=prices.index)
    flags["equity_correction"] = (
        (spy / spy.rolling(252).max() - 1)
        .le(plan["correction_drawdown"])
        .astype("boolean")
        .where(spy.rolling(252).count().eq(252))
    )
    flags["inflation_shock"] = pd.Series(pd.NA, index=prices.index, dtype="boolean")
    flags["economic_slowdown"] = pd.Series(pd.NA, index=prices.index, dtype="boolean")
    if {"CPIAUCSL_yoy", "CPIAUCSL_yoy_change_3m"}.issubset(macro):
        known = macro.CPIAUCSL_yoy.notna() & macro.CPIAUCSL_yoy_change_3m.notna()
        flags["inflation_shock"] = (
            (
                (macro.CPIAUCSL_yoy >= plan["inflation_yoy_threshold"])
                & (macro.CPIAUCSL_yoy_change_3m >= plan["inflation_acceleration_pp"])
            )
            .astype("boolean")
            .where(known)
        )
    if "UNRATE_change_3m" in macro:
        flags["economic_slowdown"] = (
            (macro.UNRATE_change_3m >= plan["unemployment_3m_increase_pp"])
            .astype("boolean")
            .where(macro.UNRATE_change_3m.notna())
        )
    positive_trend = (spy.pct_change(60, fill_method=None) > 0) & (spy > spy.rolling(200).mean())
    flags["recovery"] = positive_trend & ~flags.any(axis=1)
    flags["recovery"] = flags.recovery.astype("boolean").where(flags.notna().all(axis=1))
    return flags
