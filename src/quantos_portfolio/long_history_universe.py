from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantos_portfolio.multi_market_returns import (
    build_raw_return_matrix,
)


LONG_HISTORY_ASSETS = [
    "NIFTY_50",
    "NIFTY_BANK",
    "NIFTY_MIDCAP_100",
    "NIFTY_NEXT_50",
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "NFLX",
]


def build_long_history_returns() -> pd.DataFrame:

    returns = build_raw_return_matrix()

    missing_assets = [
        asset
        for asset in LONG_HISTORY_ASSETS
        if asset not in returns.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Missing assets: {missing_assets}"
        )

    result = returns[
        LONG_HISTORY_ASSETS
    ].copy()

    return result


def validate_long_history_returns(
    returns: pd.DataFrame,
):

    print("=" * 80)
    print("QUANTOS LONG-HISTORY 13-ASSET RETURN MATRIX")
    print("=" * 80)

    print("\nShape:")
    print(returns.shape)

    print("\nAssets:")
    for i, asset in enumerate(
        returns.columns,
        start=1,
    ):
        print(f"{i:2d}. {asset}")

    print("\nDate range:")
    print(
        f"{returns.index.min().date()} "
        f"-> "
        f"{returns.index.max().date()}"
    )

    print("\nMissing returns:")
    print(returns.isna().sum())

    print("\nInfinite returns:")

    infinite_counts = pd.Series(
        np.isinf(
            returns.to_numpy()
        ).sum(axis=0),
        index=returns.columns,
    )

    print(infinite_counts)

    print("\nFirst valid return date:")

    first_valid = returns.apply(
        lambda series: series.first_valid_index()
    )

    print(first_valid)

    print("\nLast valid return date:")

    last_valid = returns.apply(
        lambda series: series.last_valid_index()
    )

    print(last_valid)

    print("\nReturn statistics:")

    print(
        returns.describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )

    print("\nValidation:")

    observed_values = returns.to_numpy()

    finite_mask = np.isfinite(observed_values)
    missing_mask = np.isnan(observed_values)

    valid_observed_mask = (
    finite_mask | missing_mask
)

    print(
      "All non-missing returns finite:",
      finite_mask[~missing_mask].all(),
    )

    print(
        "All assets present:",
        list(returns.columns)
        == LONG_HISTORY_ASSETS,
    )

    print(
        "PLTR excluded:",
        "PLTR" not in returns.columns,
    )

    print("=" * 80)


if __name__ == "__main__":

    returns = build_long_history_returns()

    validate_long_history_returns(
        returns
    )