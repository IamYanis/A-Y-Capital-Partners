"""Paired moving-block uncertainty and multiplicity control for OOS losses."""

import numpy as np


def block_inference(
    improvements: np.ndarray, draws: int, block_length: int, seed: int
) -> tuple[float, float, float]:
    """One-sided centered-null block-bootstrap p and percentile mean CI.

    Chronological, paired loss improvements. Evaluation origins already have
    nonoverlapping labels; contiguous blocks retain residual serial dependence.
    Approximate inference assumes local stationarity, not exact finite-sample validity.
    """
    values = np.asarray(improvements, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan, np.nan, 1.0
    block = min(block_length, max(1, len(values) // 2))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(draws, int(np.ceil(len(values) / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, : len(values)] % len(
        values
    )
    means = values[indices].mean(axis=1)
    centered_means = (values - values.mean())[indices].mean(axis=1)
    p_value = (1 + np.count_nonzero(centered_means >= values.mean())) / (draws + 1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high), float(p_value)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def empirical_distribution(
    expected: float, residuals: np.ndarray, coverage: float
) -> tuple[float, float, float]:
    """Approximate empirical predictive quantiles and positive-return probability."""
    samples = np.maximum(-1.0, expected + np.asarray(residuals))
    alpha = (1 - coverage) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    probability = (1 + np.count_nonzero(samples > 0)) / (len(samples) + 2)
    return float(low), float(high), float(probability)
