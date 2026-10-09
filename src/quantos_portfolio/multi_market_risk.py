from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantos_portfolio.multi_market_returns import (
    build_raw_return_matrix,
)


TRADING_DAYS = 252


def build_correlation_matrix() -> pd.DataFrame:
    """
    Cross-asset correlation matrix.

    Uses pairwise available observations rather than
    forcing all assets onto the same trading dates.
    """

    returns = build_raw_return_matrix()

    return returns.corr()


def build_annualized_covariance_matrix() -> pd.DataFrame:
    """
    Annualized covariance matrix.

    Pairwise available observations are used for each
    asset pair.
    """

    returns = build_raw_return_matrix()

    covariance = returns.cov()

    return covariance * TRADING_DAYS


def portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Annualized portfolio volatility.
    """

    weights = weights.reindex(covariance.index)

    w = weights.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)

    variance = w @ cov @ w

    return float(np.sqrt(max(variance, 0.0)))


def risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Marginal and total contribution of each asset
    to portfolio volatility.
    """

    weights = weights.reindex(covariance.index)

    w = weights.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)

    portfolio_variance = w @ cov @ w

    if portfolio_variance <= 0:
        raise ValueError(
            "Portfolio variance must be positive."
        )

    portfolio_vol = np.sqrt(portfolio_variance)

    marginal_contribution = (
        cov @ w
    ) / portfolio_vol

    contribution = (
        w * marginal_contribution
    )

    contribution_pct = (
        contribution / portfolio_vol
    )

    return pd.Series(
        contribution_pct,
        index=covariance.index,
        name="risk_contribution",
    )


if __name__ == "__main__":

    print()
    print("=" * 80)
    print("QUANTOS MULTI-MARKET RISK ENGINE")
    print("=" * 80)

    correlation = build_correlation_matrix()

    covariance = (
        build_annualized_covariance_matrix()
    )

    print()
    print("Correlation matrix:")
    print(correlation.round(3))

    print()
    print("Annualized covariance matrix:")
    print(covariance.round(6))

    print()
    print("Correlation range:")

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1,
        ).astype(bool)
    )

    print(
        f"Minimum: {upper.min().min():.4f}"
    )

    print(
        f"Maximum: {upper.max().max():.4f}"
    )

    print()
    print("Covariance matrix shape:")
    print(covariance.shape)