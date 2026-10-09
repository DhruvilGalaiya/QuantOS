from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.quantos_portfolio.multi_market_risk import (
    build_annualized_covariance_matrix,
)


def normalize_weights(weights: pd.Series) -> pd.Series:
    """
    Normalize weights so they sum to 1.
    """

    weights = weights.astype(float)

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    return weights / total


def equal_weight(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Equal-weight allocation.
    """

    assets = covariance.index

    weights = pd.Series(
        1.0 / len(assets),
        index=assets,
        name="equal_weight",
    )

    return weights


def portfolio_variance(
    weights: np.ndarray,
    covariance: pd.DataFrame,
) -> float:
    """
    Portfolio variance.
    """

    cov = covariance.to_numpy(
        dtype=float
    )

    return float(
        weights @ cov @ weights
    )


def portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Annualized portfolio volatility.
    """

    weights = weights.reindex(
        covariance.index
    )

    variance = portfolio_variance(
        weights.to_numpy(dtype=float),
        covariance,
    )

    return float(
        np.sqrt(max(variance, 0.0))
    )


def minimum_variance(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Long-only minimum-variance portfolio.

    Constraints:
        weights >= 0
        sum(weights) = 1
    """

    assets = covariance.index
    n = len(assets)

    initial = np.ones(n) / n

    bounds = [
        (0.0, 1.0)
        for _ in range(n)
    ]

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    result = minimize(
        portfolio_variance,
        initial,
        args=(covariance,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum variance optimization failed: "
            + result.message
        )

    weights = pd.Series(
        result.x,
        index=assets,
        name="minimum_variance",
    )

    return normalize_weights(weights)


def risk_parity(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Long-only equal-risk-contribution portfolio.

    Each asset is targeted to contribute approximately
    equally to total portfolio volatility.
    """

    assets = covariance.index
    n = len(assets)

    initial = np.ones(n) / n

    bounds = [
        (1e-8, 1.0)
        for _ in range(n)
    ]

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    def objective(weights):

        cov = covariance.to_numpy(
            dtype=float
        )

        portfolio_var = (
            weights @ cov @ weights
        )

        if portfolio_var <= 0:
            return 1e10

        marginal = cov @ weights

        contributions = (
            weights * marginal
        )

        target = (
            portfolio_var / n
        )

        return float(
            np.sum(
                (contributions - target) ** 2
            )
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 3000,
            "ftol": 1e-14,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Risk parity optimization failed: "
            + result.message
        )

    weights = pd.Series(
        result.x,
        index=assets,
        name="risk_parity",
    )

    return normalize_weights(weights)


def allocation_table(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate all baseline allocations.
    """

    ew = equal_weight(covariance)

    mv = minimum_variance(covariance)

    rp = risk_parity(covariance)

    table = pd.DataFrame({
        "EQUAL_WEIGHT": ew,
        "MIN_VARIANCE": mv,
        "RISK_PARITY": rp,
    })

    return table


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("QUANTOS 14-ASSET ALLOCATION ENGINE")
    print("=" * 80)

    covariance = (
        build_annualized_covariance_matrix()
    )

    allocations = allocation_table(
        covariance
    )

    print()
    print("Allocation weights:")
    print(
        allocations.round(4)
    )

    print()
    print("Weight totals:")

    print(
        allocations.sum()
    )

    print()
    print("Portfolio volatility:")

    for strategy in allocations.columns:

        volatility = portfolio_volatility(
            allocations[strategy],
            covariance,
        )

        print(
            f"{strategy:<20} "
            f"{volatility:.4%}"
        )

    print()
    print("Largest allocations:")

    for strategy in allocations.columns:

        print()
        print(strategy)

        print(
            allocations[strategy]
            .sort_values(
                ascending=False
            )
            .head(5)
            .round(4)
        )