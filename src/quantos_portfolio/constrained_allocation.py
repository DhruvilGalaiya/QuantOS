from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.quantos_portfolio.multi_market_risk import (
    build_annualized_covariance_matrix,
)


def constrained_minimum_variance(
    covariance: pd.DataFrame,
    max_weight: float = 0.25,
) -> pd.Series:
    """
    Long-only minimum-variance portfolio with
    an explicit maximum position constraint.

    Constraints:
        weights >= 0
        weights <= max_weight
        sum(weights) = 1
    """

    if max_weight <= 0 or max_weight > 1:
        raise ValueError(
            "max_weight must be between 0 and 1."
        )

    assets = covariance.index
    n = len(assets)

    if max_weight * n < 1.0:
        raise ValueError(
            "Maximum weight is too restrictive "
            "for the number of assets."
        )

    cov = covariance.to_numpy(
        dtype=float
    )

    initial = np.ones(n) / n

    def objective(weights):
        return float(
            weights @ cov @ weights
        )

    constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1.0,
    }

    bounds = [
        (0.0, max_weight)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 3000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Constrained minimum-variance "
            f"optimization failed: {result.message}"
        )

    weights = pd.Series(
        result.x,
        index=assets,
        name="CONSTRAINED_MIN_VARIANCE",
    )

    return weights / weights.sum()


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("QUANTOS CONSTRAINED MINIMUM VARIANCE")
    print("=" * 80)

    covariance = (
        build_annualized_covariance_matrix()
    )

    weights = constrained_minimum_variance(
        covariance,
        max_weight=0.25,
    )

    portfolio_variance = (
        weights.to_numpy()
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    print()
    print("Maximum individual weight: 25%")

    print()
    print("Weights:")

    print(
        weights
        .sort_values(ascending=False)
        .round(4)
    )

    print()
    print(
        f"Weight total: "
        f"{weights.sum():.6f}"
    )

    print(
        f"Portfolio volatility: "
        f"{portfolio_volatility:.4%}"
    )

    print()
    print(
        f"Maximum actual weight: "
        f"{weights.max():.4%}"
    )