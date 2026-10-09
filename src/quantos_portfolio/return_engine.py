from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .market_data import (
    load_native_price_matrix,
    load_mixed_price_matrix,
    validate_price_matrix,
)


ROOT = Path(__file__).resolve().parents[2]


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Convert a price matrix into a return matrix.

    method:
        "log"    -> logarithmic returns
        "simple" -> simple percentage returns
    """
    prices = validate_price_matrix(prices)

    if method == "log":
        returns = np.log(
            prices / prices.shift(1)
        )

    elif method == "simple":
        returns = prices.pct_change()

    else:
        raise ValueError(
            "method must be 'log' or 'simple'."
        )

    return returns


def build_price_matrix(
    symbols: Iterable[str],
) -> pd.DataFrame:
    """
    Build a price matrix from native QuantOS assets.
    """
    symbols = [
        symbol.upper().strip()
        for symbol in symbols
    ]

    if not symbols:
        raise ValueError(
            "No assets supplied."
        )

    return load_native_price_matrix(
        symbols
    )


def build_return_matrix(
    symbols: Iterable[str],
    method: str = "log",
    min_assets_per_day: int = 1,
) -> pd.DataFrame:
    """
    Build a return matrix from native QuantOS assets.
    """
    prices = build_price_matrix(symbols)

    returns = calculate_returns(
        prices,
        method=method,
    )

    returns = returns.dropna(
        how="all"
    )

    valid_counts = returns.notna().sum(
        axis=1
    )

    returns = returns.loc[
        valid_counts >= min_assets_per_day
    ]

    return returns


def build_mixed_price_matrix(
    native_symbols: list[str] | None = None,
    csv_assets: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    """
    Build a price matrix containing both
    native QuantOS assets and user-provided CSV assets.
    """
    return load_mixed_price_matrix(
        native_symbols=native_symbols,
        csv_assets=csv_assets,
    )


def build_mixed_return_matrix(
    native_symbols: list[str] | None = None,
    csv_assets: dict[str, str | Path] | None = None,
    method: str = "log",
    min_assets_per_day: int = 1,
) -> pd.DataFrame:
    """
    Build a return matrix containing both
    native and user-provided assets.
    """
    prices = build_mixed_price_matrix(
        native_symbols=native_symbols,
        csv_assets=csv_assets,
    )

    returns = calculate_returns(
        prices,
        method=method,
    )

    returns = returns.dropna(
        how="all"
    )

    valid_counts = returns.notna().sum(
        axis=1
    )

    returns = returns.loc[
        valid_counts >= min_assets_per_day
    ]

    return returns


def load_user_price_csv(
    path: str | Path,
    date_column: str = "Date",
    price_column: str = "Close",
    symbol: str | None = None,
) -> pd.Series:
    """
    Compatibility wrapper for loading a user CSV.
    """
    from .market_data import load_csv_prices

    return load_csv_prices(
        path=path,
        symbol=symbol,
        date_column=date_column,
        price_column=price_column,
    )


def build_returns_from_prices(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Build returns directly from an existing
    price matrix.
    """
    return calculate_returns(
        prices,
        method=method,
    )