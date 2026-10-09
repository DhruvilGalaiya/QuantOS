import os
import pandas as pd

from src.config import (
    REGIME_INSTRUMENTS,
    VALIDATION_INSTRUMENTS,
)


ALL_INSTRUMENTS = {
    **REGIME_INSTRUMENTS,
    **VALIDATION_INSTRUMENTS,
}


DATA_DIR = "data/regime/daily"


print("\nQuantOS Regime Dataset Validation")
print("=" * 70)


for name in ALL_INSTRUMENTS:

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    path = os.path.join(
        DATA_DIR,
        f"{filename}.parquet"
    )

    print(f"\n{name}")
    print("-" * 50)

    if not os.path.exists(path):
        print("ERROR: File not found")
        continue

    df = pd.read_parquet(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print(
        "Date range:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max()
    )

    print(
        "Duplicate timestamps:",
        df["timestamp"].duplicated().sum()
    )

    print(
        "Missing OHLC values:",
        df[["open", "high", "low", "close"]]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Invalid OHLC:",
        (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()
    )

    print(
        "Close min/max:",
        df["close"].min(),
        "/",
        df["close"].max()
    )