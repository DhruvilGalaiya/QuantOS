import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Make backend/ available for imports
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BACKEND_DIR)
)


from app.database import engine


def run_feature_quality_checks():

    print("=" * 70)
    print("FEATURE QUALITY & LEAKAGE CHECKS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load feature data
    # ---------------------------------------------------------

    query = """
        SELECT *
        FROM market_features
        ORDER BY symbol, timestamp;
    """

    data = pd.read_sql(
        query,
        engine
    )

    print(f"\nRows loaded: {len(data)}")
    print(f"Symbols: {data['symbol'].nunique()}")

    # ---------------------------------------------------------
    # 1. Duplicate check
    # ---------------------------------------------------------

    duplicates = data.duplicated(
        subset=["symbol", "timestamp"]
    ).sum()

    print("\n1. Duplicate observations")
    print(f"Duplicates: {duplicates}")

    # ---------------------------------------------------------
    # 2. Timestamp ordering
    # ---------------------------------------------------------

    ordered = True

    for symbol, group in data.groupby("symbol"):

        group = group.sort_values("timestamp")

        if not group["timestamp"].is_monotonic_increasing:
            ordered = False

            print(
                f"Timestamp ordering problem: {symbol}"
            )

    print("\n2. Timestamp ordering")
    print(f"Ordered correctly: {ordered}")

    # ---------------------------------------------------------
    # 3. Required feature columns
    # ---------------------------------------------------------

    required_features = [
        "return_1d",
        "log_return_1d",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "ema_50",
        "volatility_20",
        "rsi_14",
        "volume_sma_20",
        "volume_ratio",
        "high_low_range",
        "high_low_range_pct",
        "atr_14",
        "close_vs_sma20",
        "close_vs_sma50",
    ]

    missing_features = [
        column
        for column in required_features
        if column not in data.columns
    ]

    print("\n3. Required feature columns")

    if missing_features:

        print(
            f"Missing features: {missing_features}"
        )

    else:

        print(
            "All required features present."
        )

    # ---------------------------------------------------------
    # 4. Null values
    # ---------------------------------------------------------

    null_counts = data[
        required_features
    ].isnull().sum()

    print("\n4. Feature null counts")

    nonzero_nulls = null_counts[
        null_counts > 0
    ]

    if len(nonzero_nulls) == 0:

        print("No feature nulls found.")

    else:

        print(nonzero_nulls)

    # ---------------------------------------------------------
    # 5. RSI range
    # ---------------------------------------------------------

    invalid_rsi = data[
        data["rsi_14"].notnull()
        &
        (
            (data["rsi_14"] < 0)
            |
            (data["rsi_14"] > 100)
        )
    ]

    print("\n5. RSI range check")

    print(
        f"Invalid RSI rows: "
        f"{len(invalid_rsi)}"
    )

    # ---------------------------------------------------------
    # 6. Volatility range
    # ---------------------------------------------------------

    invalid_volatility = data[
        data["volatility_20"].notnull()
        &
        (data["volatility_20"] < 0)
    ]

    print("\n6. Volatility check")

    print(
        f"Negative volatility rows: "
        f"{len(invalid_volatility)}"
    )

    # ---------------------------------------------------------
    # 7. SMA relationship checks
    # ---------------------------------------------------------

    invalid_sma20 = data[
    data["close_vs_sma20"].notnull()
    &
    data["sma_20"].notnull()
    &
    (
        (
            data["close_vs_sma20"]
            -
            (
                data["close"]
                / data["sma_20"]
                - 1
            )
        ).abs() > 1e-6
      )
    ]

    invalid_sma50 = data[
    data["close_vs_sma50"].notnull()
    &
    data["sma_50"].notnull()
    &
    (
        (
            data["close_vs_sma50"]
            -
            (
                data["close"]
                / data["sma_50"]
                - 1
            )
        ).abs() > 1e-6
      )
    ]
    print("\n7. Price / SMA consistency")

    print(
        f"Invalid close_vs_sma20: "
        f"{len(invalid_sma20)}"
    )

    print(
        f"Invalid close_vs_sma50: "
        f"{len(invalid_sma50)}"
    )

    # ---------------------------------------------------------
    # 8. Return consistency
    # ---------------------------------------------------------

    data["calculated_return"] = (
        data
        .groupby("symbol")["close"]
        .pct_change()
    )

    return_difference = (
        data["return_1d"]
        -
        data["calculated_return"]
    ).abs()

    invalid_returns = data[
        data["return_1d"].notnull()
        &
        data["calculated_return"].notnull()
        &
        (return_difference > 1e-6)
    ]

    print("\n8. Return consistency")

    print(
        f"Incorrect return rows: "
        f"{len(invalid_returns)}"
    )

    # ---------------------------------------------------------
    # 9. Future leakage inspection
    # ---------------------------------------------------------

    leakage_features = [
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "ema_50",
        "volatility_20",
        "volume_sma_20",
    ]

    print("\n9. Future leakage inspection")

    leakage_found = False

    for symbol, group in data.groupby("symbol"):

        group = group.sort_values("timestamp").copy()

        for feature in leakage_features:

            if feature == "sma_10":

                expected = (
                    group["close"]
                    .rolling(10)
                    .mean()
                )

            elif feature == "sma_20":

                expected = (
                    group["close"]
                    .rolling(20)
                    .mean()
                )

            elif feature == "sma_50":

                expected = (
                    group["close"]
                    .rolling(50)
                    .mean()
                )

            elif feature == "ema_20":

                expected = (
                    group["close"]
                    .ewm(
                        span=20,
                        adjust=False
                    )
                    .mean()
                )

            elif feature == "ema_50":

                expected = (
                    group["close"]
                    .ewm(
                        span=50,
                        adjust=False
                    )
                    .mean()
                )

            elif feature == "volatility_20":

                expected = (
                    group["log_return_1d"]
                    .rolling(20)
                    .std()
                )

            elif feature == "volume_sma_20":

                expected = (
                    group["volume"]
                    .rolling(20)
                    .mean()
                )

            else:

                continue

            actual = group[feature]

            valid = (
                actual.notnull()
                &
                expected.notnull()
            )

            if valid.any():

                difference = (
                    actual[valid]
                    -
                    expected[valid]
                ).abs()

                if (difference > 1e-6).any():

                    leakage_found = True

                    print(
                        f"Potential calculation issue: "
                        f"{symbol} -> {feature}"
                    )

    if not leakage_found:

        print(
            "No leakage detected in "
            "rolling feature calculations."
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    run_feature_quality_checks()