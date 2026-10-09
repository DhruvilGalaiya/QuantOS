"""
QuantOS - Update Volatility Data to Latest Market Date

Purpose
-------
Rebuild the NIFTY-50 realized-volatility dataset using the current
daily market data, so the volatility artifact is synchronized with
the latest available market close.

This script intentionally does NOT modify the existing production
volatility model implementation.

It:
    1. Reads the authoritative NIFTY daily price data.
    2. Recalculates realized volatility features through the latest date.
    3. Preserves the same volatility definitions used by QuantOS.
    4. Writes the updated volatility parquet.

GARCH forecasting remains handled separately by the existing
QuantOS volatility engine.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PRICE_PATH = Path(
    "data/regime/daily/nifty_50.parquet"
)

OUTPUT_PATH = Path(
    "data/regime/volatility/nifty_50.parquet"
)


# ============================================================
# CONFIG
# ============================================================

ANNUALIZATION = 252


# ============================================================
# REALIZED VOLATILITY
# ============================================================

def realized_variance(
    returns,
    window,
):
    """
    Rolling realized variance from daily log returns.
    """

    return (
        returns
        .pow(2)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )


def build_volatility_dataset():

    if not PRICE_PATH.exists():

        raise FileNotFoundError(
            f"Missing NIFTY daily data:\n"
            f"{PRICE_PATH}"
        )

    df = pd.read_parquet(
        PRICE_PATH
    ).copy()

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Normalize timestamps.
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Convert timezone-aware timestamps to naive timestamps.
    if getattr(
        df["timestamp"].dt,
        "tz",
        None,
    ) is not None:

        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize(None)
        )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Returns.
    # --------------------------------------------------------

    df["log_return_1d"] = (
        np.log(
            df["close"]
            / df["close"].shift(1)
        )
    )

    # --------------------------------------------------------
    # Realized variance.
    # --------------------------------------------------------

    df["rv_1d"] = (
        df["log_return_1d"]
        .pow(2)
    )

    df["rv_5d"] = (
        realized_variance(
            df["log_return_1d"],
            5,
        )
    )

    df["rv_21d"] = (
        realized_variance(
            df["log_return_1d"],
            21,
        )
    )

    # Daily realized variance.
    df["rv_daily"] = (
        df["rv_1d"]
    )

    # Weekly realized variance.
    df["rv_weekly"] = (
        df["rv_5d"]
    )

    # Monthly realized variance.
    df["rv_monthly"] = (
        df["rv_21d"]
    )

    # --------------------------------------------------------
    # Annualized realized volatility.
    # --------------------------------------------------------

    df["realized_vol_1d"] = (
        np.sqrt(
            df["rv_1d"]
            * ANNUALIZATION
        )
    )

    df["realized_vol_5d"] = (
        np.sqrt(
            df["rv_5d"]
            * ANNUALIZATION
        )
    )

    df["realized_vol_21d"] = (
        np.sqrt(
            df["rv_21d"]
            * ANNUALIZATION
        )
    )

    # --------------------------------------------------------
    # Forward realized variance.
    #
    # These are useful for historical model evaluation.
    # They are deliberately shifted into the future and therefore
    # must never be used as today's input.
    # --------------------------------------------------------

    df["forward_rv_1d"] = (
        df["rv_1d"]
        .shift(-1)
    )

    df["forward_rv_5d"] = (
        df["rv_1d"]
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
        .shift(-4)
    )

    df["forward_rv_21d"] = (
        df["rv_1d"]
        .rolling(
            21,
            min_periods=21,
        )
        .mean()
        .shift(-20)
    )

    df["forward_vol_1d"] = (
        np.sqrt(
            df["forward_rv_1d"]
            * ANNUALIZATION
        )
    )

    df["forward_vol_5d"] = (
        np.sqrt(
            df["forward_rv_5d"]
            * ANNUALIZATION
        )
    )

    df["forward_vol_21d"] = (
        np.sqrt(
            df["forward_rv_21d"]
            * ANNUALIZATION
        )
    )

    # --------------------------------------------------------
    # Preserve original market columns and required order.
    # --------------------------------------------------------

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "log_return_1d",
        "rv_1d",
        "rv_5d",
        "rv_21d",
        "rv_daily",
        "rv_weekly",
        "rv_monthly",
        "forward_rv_1d",
        "forward_rv_5d",
        "forward_rv_21d",
        "realized_vol_1d",
        "realized_vol_5d",
        "realized_vol_21d",
        "forward_vol_1d",
        "forward_vol_5d",
        "forward_vol_21d",
    ]

    # Add missing optional columns safely.
    for column in columns:

        if column not in df.columns:

            df[column] = np.nan

    df = df[columns]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "QUANTOS VOLATILITY DATA UPDATE"
    )
    print("=" * 78)

    print()

    print(
        f"Source : {PRICE_PATH}"
    )

    print(
        f"Output : {OUTPUT_PATH}"
    )

    print()

    df = build_volatility_dataset()

    latest = df.iloc[-1]

    print(
        "UPDATED VOLATILITY DATA"
    )

    print("-" * 78)

    print(
        f"Rows        : {len(df):,}"
    )

    print(
        f"Period      : "
        f"{df['timestamp'].iloc[0].date()} "
        f"→ "
        f"{df['timestamp'].iloc[-1].date()}"
    )

    print(
        f"Latest close: "
        f"₹{latest['close']:,.2f}"
    )

    print(
        f"1D RV       : "
        f"{latest['realized_vol_1d'] * 100:.2f}%"
    )

    print(
        f"5D RV       : "
        f"{latest['realized_vol_5d'] * 100:.2f}%"
    )

    print(
        f"21D RV      : "
        f"{latest['realized_vol_21d'] * 100:.2f}%"
    )

    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "FILE SAVED"
    )

    print("-" * 78)

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "VOLATILITY DATA UPDATE COMPLETE."
    )


if __name__ == "__main__":

    main()