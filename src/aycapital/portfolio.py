"""Long-only research allocations with explicit, validated constraints."""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Constraints:
    """Exposure groups map group names to assets and maximum total weights."""

    max_weight: float = 1.0
    eligible_assets: tuple[str, ...] | None = None
    exposure_caps: dict[str, tuple[tuple[str, ...], float]] = field(default_factory=dict)


def optimize(covariance: pd.DataFrame, constraints: Constraints,
             expected_returns: pd.Series | None = None,
             risk_aversion: float = 1.0) -> pd.Series:
    """Minimum variance, or mean-variance if explicit annual forecasts are supplied.

    Covariance and expected returns must share an annualization convention.
    Historical sample means are never automatically treated as forecasts.
    """
    labels = covariance.columns
    if (len(labels) == 0 or labels.has_duplicates
            or not covariance.index.equals(labels)):
        raise ValueError("Covariance must be square with matching unique labels")
    matrix = covariance.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Covariance must be finite and symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite")
    cap = constraints.max_weight
    if not np.isfinite(cap) or not 0 < cap <= 1 or len(labels) * cap < 1 - 1e-12:
        raise ValueError("Invalid or infeasible maximum position weight")
    if constraints.eligible_assets is not None:
        if not set(labels).issubset(constraints.eligible_assets):
            raise ValueError("Universe contains ineligible assets")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    weights = cp.Variable(len(labels))
    restrictions = [weights >= 0, weights <= cap, cp.sum(weights) == 1]
    for assets, limit in constraints.exposure_caps.values():
        if len(set(assets)) != len(assets) or not set(assets).issubset(labels):
            raise ValueError("Exposure groups contain duplicate or unknown assets")
        if not np.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("Invalid exposure cap")
        restrictions.append(cp.sum(weights[[labels.get_loc(a) for a in assets]]) <= limit)
    objective = risk_aversion * cp.quad_form(weights, cp.psd_wrap(matrix))
    if expected_returns is not None:
        if not expected_returns.index.equals(labels):
            raise ValueError("Expected returns must match covariance asset order")
        if not np.isfinite(expected_returns.to_numpy()).all():
            raise ValueError("Expected returns must be finite")
        objective -= expected_returns.to_numpy() @ weights
    problem = cp.Problem(cp.Minimize(objective), restrictions)
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError(f"Optimization failed: {problem.status}")
    result = pd.Series(weights.value, index=labels, name="weight")
    tolerance = 1e-7
    if (result.min() < -tolerance or result.max() > cap + tolerance
            or abs(result.sum() - 1) > tolerance):
        raise ValueError("Solver output violates allocation constraints")
    for assets, limit in constraints.exposure_caps.values():
        if result.loc[list(assets)].sum() > limit + tolerance:
            raise ValueError("Solver output violates exposure cap")
    return result
