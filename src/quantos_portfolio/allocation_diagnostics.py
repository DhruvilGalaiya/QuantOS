from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantos_portfolio.allocation_engine import (
    allocation_table,
)
from src.quantos_portfolio.multi_market_risk import (
    build_annualized_covariance_matrix,
)


def calculate_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:

    weights = weights.reindex(
        covariance.index
    )

    w = weights.to_numpy(
        dtype=float
    )

    cov = covariance.to_numpy(
        dtype=float
    )

    portfolio_variance = (
        w @ cov @ w
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    marginal = (
        cov @ w
    ) / portfolio_volatility

    contribution = (
        w * marginal
    ) / portfolio_volatility

    return pd.Series(
        contribution,
        index=covariance.index,
        name="risk_contribution",
    )


def calculate_concentration(
    weights: pd.Series,
) -> float:
    """
    Herfindahl-Hirschman-style concentration measure.

    1/N = perfectly equal allocation.
    Higher values = greater concentration.
    """

    return float(
        np.sum(
            weights.to_numpy() ** 2
        )
    )


def effective_number_of_assets(
    weights: pd.Series,
) -> float:
    """
    Effective number of assets implied by weights.
    """

    concentration = calculate_concentration(
        weights
    )

    return float(
        1.0 / concentration
    )


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("QUANTOS ALLOCATION DIAGNOSTICS")
    print("=" * 80)

    covariance = (
        build_annualized_covariance_matrix()
    )

    allocations = allocation_table(
        covariance
    )

    for strategy in allocations.columns:

        weights = allocations[strategy]

        risk = calculate_risk_contribution(
            weights,
            covariance,
        )

        concentration = (
            calculate_concentration(
                weights
            )
        )

        effective_assets = (
            effective_number_of_assets(
                weights
            )
        )

        print()
        print("=" * 60)
        print(strategy)
        print("=" * 60)

        print()
        print(
            f"Concentration: "
            f"{concentration:.4f}"
        )

        print(
            f"Effective assets: "
            f"{effective_assets:.2f}"
        )

        print()
        print("Weights:")

        print(
            weights
            .sort_values(
                ascending=False
            )
            .round(4)
        )

        print()
        print("Risk contribution:")

        print(
            risk
            .sort_values(
                ascending=False
            )
            .round(4)
        )

        print()
        print(
            f"Risk contribution sum: "
            f"{risk.sum():.6f}"
        )