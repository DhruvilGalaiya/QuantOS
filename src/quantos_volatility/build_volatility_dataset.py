"""
QuantOS | Volatility Dataset Builder
-------------------------------------

Builds leakage-safe realized-variance features and forward targets
from the refreshed Upstox daily volatility dataset.

INPUT:
    data/regime/volatility_daily/*.parquet

OUTPUT:
    data/regime/volatility/*.parquet

Current volatility universe:
    NIFTY 50
    NIFTY BANK
    SENSEX

Definitions
-----------
r_t       = log(C_t / C_{t-1})

RV_1,t    = r_t^2

RV_5,t    = sum of the previous 5 squared returns

RV_21,t   = sum of the previous 21 squared returns

Forward targets:
    fRV_h,t = sum(r_{t+1}^2 ... r_{t+h}^2)

The model features use only information available at time t.
Future observations are used only for forward targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# File is:
# QuantOS/src/quantos_volatility/build_volatility_dataset.py
#
# parents[0] = quantos_volatility
# parents[1] = src
# parents[2] = QuantOS
ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# The refreshed Upstox volatility data is stored here.
INPUT_DIR = ROOT / "data" / "regime" / "volatility_daily"

OUTPUT_DIR = ROOT / "data" / "regime" / "volatility"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VOLATILITY UNIVERSE
# ============================================================

FILES = {
    "NIFTY_50": "nifty_50.parquet",
    "NIFTY_BANK": "nifty_bank.parquet",
    "SENSEX": "sensex.parquet",
}


# ============================================================
# MODEL SETTINGS
# ============================================================

HORIZONS = (1, 5, 21)

TRADING_DAYS = 252


# ============================================================
# FORWARD REALIZED VARIANCE
# ============================================================

def forward_sum(
    series: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    Calculate the NEXT h squared returns.

    At time t:

        forward_rv_1d
            = r_(t+1)^2

        forward_rv_5d
            = r_(t+1)^2 + ... + r_(t+5)^2

        forward_rv_21d
            = r_(t+1)^2 + ... + r_(t+21)^2

    Future observations therefore appear only in targets,
    never in model features.
    """

    return (
        series
        .shift(-1)
        .rolling(
            horizon,
            min_periods=horizon,
        )
        .sum()
        .shift(-(horizon - 1))
    )


# ============================================================
# BUILD ONE SYMBOL
# ============================================================

def build_symbol(
    symbol: str,
    filename: str,
) -> pd.DataFrame:

    path = INPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{symbol}: input file not found: {path}"
        )

    df = pd.read_parquet(path).copy()

    # --------------------------------------------------------
    # Validate input columns
    # --------------------------------------------------------

    required = [
        "timestamp",
        "close",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean types
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Sort and remove duplicates
    # --------------------------------------------------------

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "close",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"{symbol}: input dataset is empty."
        )

    # --------------------------------------------------------
    # Close-to-close log return
    # --------------------------------------------------------

    df["log_return_1d"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    squared_return = (
        df["log_return_1d"].pow(2)
    )

    # ========================================================
    # BACKWARD REALIZED VARIANCE FEATURES
    # ========================================================

    # 1-day realized variance
    df["rv_1d"] = squared_return

    # 5-day realized variance
    df["rv_5d"] = (
        squared_return
        .rolling(
            5,
            min_periods=5,
        )
        .sum()
    )

    # 21-day realized variance
    df["rv_21d"] = (
        squared_return
        .rolling(
            21,
            min_periods=21,
        )
        .sum()
    )

    # ========================================================
    # HAR-RV FEATURES
    # ========================================================

    # Daily component
    df["rv_daily"] = df["rv_1d"]

    # Weekly component
    df["rv_weekly"] = (
        squared_return
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
    )

    # Monthly component
    df["rv_monthly"] = (
        squared_return
        .rolling(
            21,
            min_periods=21,
        )
        .mean()
    )

    # ========================================================
    # FORWARD TARGETS
    # ========================================================

    for horizon in HORIZONS:

        df[f"forward_rv_{horizon}d"] = (
            forward_sum(
                squared_return,
                horizon,
            )
        )

    # ========================================================
    # ANNUALISED REALIZED VOLATILITY
    # ========================================================

    for horizon in HORIZONS:

        df[f"realized_vol_{horizon}d"] = np.sqrt(
            df[f"rv_{horizon}d"]
            * TRADING_DAYS
            / horizon
        )

    # ========================================================
    # FORWARD ANNUALISED VOLATILITY
    # ========================================================

    for horizon in HORIZONS:

        df[f"forward_vol_{horizon}d"] = np.sqrt(
            df[f"forward_rv_{horizon}d"]
            * TRADING_DAYS
            / horizon
        )

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    feature_cols = [
        "log_return_1d",
        "rv_1d",
        "rv_5d",
        "rv_21d",
        "rv_daily",
        "rv_weekly",
        "rv_monthly",
    ]

    # ========================================================
    # MODEL TARGETS
    # ========================================================

    target_cols = [
        f"forward_rv_{horizon}d"
        for horizon in HORIZONS
    ]

    # ========================================================
    # REMOVE INVALID MODEL ROWS
    # ========================================================

    out = (
        df
        .dropna(
            subset=feature_cols
        )
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(
            f"{symbol}: no usable volatility rows."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if out["timestamp"].duplicated().any():
        raise AssertionError(
            f"{symbol}: duplicate timestamps remain."
        )

    if (
        out[feature_cols]
        .isna()
        .any()
        .any()
    ):
        raise AssertionError(
            f"{symbol}: NaNs remain in model features."
        )

    # Make absolutely sure the dataset is sorted.
    out = (
        out
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "QuantOS VOLATILITY DATASET BUILDER | "
        "3 UNDERLYINGS"
    )
    print("=" * 70)

    print(
        f"\nInput directory:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print(
        "\nUniverse: "
        "NIFTY 50 | NIFTY BANK | SENSEX"
    )

    success = 0

    # --------------------------------------------------------
    # Process each underlying
    # --------------------------------------------------------

    for symbol, filename in FILES.items():

        print("\n" + "-" * 70)
        print(symbol)
        print("-" * 70)

        try:

            out = build_symbol(
                symbol,
                filename,
            )

            output_path = (
                OUTPUT_DIR / filename
            )

            out.to_parquet(
                output_path,
                index=False,
            )

            success += 1

            print(
                f"Rows:         {len(out):,}"
            )

            print(
                f"Period:       "
                f"{out['timestamp'].min()} -> "
                f"{out['timestamp'].max()}"
            )

            print(
                f"Latest close: "
                f"{out['close'].iloc[-1]:,.2f}"
            )

            print(
                f"Latest 1D RV: "
                f"{out['rv_1d'].iloc[-1]:.8f}"
            )

            print(
                f"Latest 21D RV: "
                f"{out['rv_21d'].iloc[-1]:.8f}"
            )

            print(
                f"Saved:        "
                f"{output_path}"
            )

        except Exception as exc:

            print(
                f"ERROR: {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VOLATILITY DATASET BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Successful symbols: "
        f"{success}/{len(FILES)}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()