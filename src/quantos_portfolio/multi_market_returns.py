from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantos_portfolio.multi_market_data import (
    build_inr_price_matrix,
)


def build_raw_return_matrix() -> pd.DataFrame:
    """
    Build raw daily returns.

    Missing values are preserved because different markets
    have different trading calendars.
    """

    prices = build_inr_price_matrix()

    returns = prices.pct_change(
        fill_method=None
    )

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return returns


def build_portfolio_return_matrix() -> pd.DataFrame:
    """
    Build a portfolio-ready daily return matrix.

    Once an asset has started trading, a missing return on a
    day when its exchange is closed is treated as 0%.

    The first valid observation for each asset remains NaN.
    """

    returns = build_raw_return_matrix()

    portfolio_returns = returns.copy()

    for column in portfolio_returns.columns:

        first_valid = portfolio_returns[column].first_valid_index()

        if first_valid is None:
            continue

        after_start = (
            portfolio_returns.index >= first_valid
        )

        portfolio_returns.loc[
            after_start,
            column,
        ] = portfolio_returns.loc[
            after_start,
            column,
        ].fillna(0.0)

    return portfolio_returns


if __name__ == "__main__":

    raw = build_raw_return_matrix()

    portfolio = build_portfolio_return_matrix()

    print()
    print("=" * 80)
    print("QUANTOS PORTFOLIO RETURN MATRIX")
    print("=" * 80)

    print()
    print("Shape:")
    print(portfolio.shape)

    print()
    print("Assets:")
    print(list(portfolio.columns))

    print()
    print("Date range:")
    print(
        portfolio.index.min().date(),
        "→",
        portfolio.index.max().date(),
    )

    print()
    print("Raw missing returns:")
    print(raw.isna().sum())

    print()
    print("Portfolio missing returns:")
    print(portfolio.isna().sum())

    print()
    print("First valid return date:")
    print(portfolio.apply(
        lambda x: x.first_valid_index()
    ))

    print()
    print("Return statistics:")

    print(
        portfolio.describe().T[
            ["mean", "std", "min", "max"]
        ]
    )