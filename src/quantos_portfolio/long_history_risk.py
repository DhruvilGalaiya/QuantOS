from __future__ import annotations

import numpy as np
import pandas as pd

from .long_history_universe import build_long_history_returns


ROLLING_WINDOW = 504


def build_long_history_covariance(
    returns: pd.DataFrame | None = None,
    window: int = ROLLING_WINDOW,
) -> dict[pd.Timestamp, pd.DataFrame]:

    if returns is None:
        returns = build_long_history_returns()

    returns = returns.sort_index()

    covariance_matrices = {}

    for i in range(window, len(returns)):
        date = returns.index[i]

        window_returns = returns.iloc[i - window:i]

        # Keep only assets with sufficient observations.
        valid_assets = [
            col
            for col in window_returns.columns
            if window_returns[col].notna().sum() >= int(window * 0.90)
        ]

        if len(valid_assets) < 2:
            continue

        covariance = window_returns[valid_assets].cov()

        covariance_matrices[date] = covariance

    return covariance_matrices


def build_long_history_volatility(
    returns: pd.DataFrame | None = None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:

    if returns is None:
        returns = build_long_history_returns()

    returns = returns.sort_index()

    # Require a minimum number of valid observations per asset.
    min_periods = int(window * 0.90)

    rolling_vol = (
        returns
        .rolling(window=window, min_periods=min_periods)
        .std()
        * np.sqrt(252)
    )

    return rolling_vol

def build_long_history_correlation(
    returns: pd.DataFrame | None = None,
    window: int = ROLLING_WINDOW,
) -> dict[pd.Timestamp, pd.DataFrame]:

    if returns is None:
        returns = build_long_history_returns()

    returns = returns.sort_index()

    correlation_matrices = {}

    for i in range(window, len(returns)):
        date = returns.index[i]

        window_returns = returns.iloc[i - window:i]

        valid_assets = [
            col
            for col in window_returns.columns
            if window_returns[col].notna().sum() >= int(window * 0.90)
        ]

        if len(valid_assets) < 2:
            continue

        correlation = window_returns[valid_assets].corr()

        correlation_matrices[date] = correlation

    return correlation_matrices


def validate_long_history_risk():

    returns = build_long_history_returns()

    covariance = build_long_history_covariance(returns)
    volatility = build_long_history_volatility(returns)
    correlation = build_long_history_correlation(returns)

    print("=" * 80)
    print("QUANTOS LONG-HISTORY RISK ENGINE")
    print("=" * 80)

    print("\nReturn matrix:")
    print("Shape:", returns.shape)
    print(
        "Date range:",
        returns.index.min().date(),
        "->",
        returns.index.max().date(),
    )

    print("\nRolling window:")
    print(f"{ROLLING_WINDOW} observations")

    # ---------------------------------------------------------
    # COVARIANCE
    # ---------------------------------------------------------

    print("\nCovariance matrices:")
    print("Count:", len(covariance))

    if covariance:

        first_date = min(covariance)
        last_date = max(covariance)

        print(
            "First covariance date:",
            first_date.date(),
        )

        print(
            "Last covariance date:",
            last_date.date(),
        )

        print("\nLatest covariance matrix:")
        print(
            covariance[last_date].round(6)
        )

    # ---------------------------------------------------------
    # VOLATILITY
    # ---------------------------------------------------------

    print("\nLatest annualized volatility:")

    valid_volatility = volatility.dropna(how="all")

    if valid_volatility.empty:

        print(
            "No valid rolling volatility estimates available."
        )

    else:

        latest_vol = (
            valid_volatility
            .iloc[-1]
            .dropna()
        )

        if latest_vol.empty:

            print(
                "No assets have a valid latest volatility estimate."
            )

        else:

            print(
                latest_vol
                .sort_values(ascending=False)
            )

    # ---------------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------------

    print("\nCorrelation matrices:")
    print("Count:", len(correlation))

    if correlation:

        latest_corr_date = max(correlation)

        print("\nLatest correlation matrix:")
        print(
            correlation[latest_corr_date].round(3)
        )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\nValidation:")

    if covariance:

        all_covariance_finite = all(
            np.isfinite(matrix.to_numpy()).all()
            for matrix in covariance.values()
        )

    else:

        all_covariance_finite = False

    print(
        "All covariance values finite:",
        all_covariance_finite,
    )

    print(
        "Covariance matrices available:",
        len(covariance) > 0,
    )

    print(
        "Valid volatility observations:",
        len(valid_volatility),
    )

    if not valid_volatility.empty:

        latest_vol_count = int(
            valid_volatility.iloc[-1].notna().sum()
        )

    else:

        latest_vol_count = 0

    print(
        "Latest volatility estimates:",
        latest_vol_count,
    )

    print(
        "Correlation matrices available:",
        len(correlation) > 0,
    )

    print("=" * 80)

if __name__ == "__main__":
    validate_long_history_risk()