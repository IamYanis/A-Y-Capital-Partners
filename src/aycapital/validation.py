"""Shared strict validation. Missing observations are never silently filled."""

import numpy as np
import pandas as pd


def frame(data: pd.DataFrame, *, minimum: int = 2) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.shape[1] == 0 or len(data) < minimum:
        raise ValueError(f"Expected a DataFrame with assets and at least {minimum} rows")
    if data.columns.has_duplicates or data.index.has_duplicates:
        raise ValueError("Duplicate assets or observations")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Observations must be sorted")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite observations; align data explicitly")
    return data.astype(float)


def returns(data: pd.DataFrame) -> pd.DataFrame:
    data = frame(data)
    if (data <= -1).any().any():
        raise ValueError("Simple returns must exceed -100%")
    return data


def periods(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    return value
