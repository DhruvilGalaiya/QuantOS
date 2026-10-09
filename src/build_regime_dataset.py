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
OUTPUT_DIR = "data/regime"

os.makedirs(OUTPUT_DIR, exist_ok=True)


series = []


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

    df = pd.read_parquet(path)

    df["date"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = df[
        ["date", "close"]
    ].copy()

    df = df.rename(
        columns={
            "close": name
        }
    )

    series.append(df)


# Combine all instruments on date
panel = series[0]

for df in series[1:]:
    panel = panel.merge(
        df,
        on="date",
        how="inner"
    )


panel = (
    panel
    .sort_values("date")
    .reset_index(drop=True)
)


print("\nQuantOS Regime Price Panel")
print("=" * 70)

print("Shape:", panel.shape)

print(
    "Date range:",
    panel["date"].min().date(),
    "→",
    panel["date"].max().date()
)

print("\nMissing values:")
print(panel.isna().sum())

print("\nFirst 5 rows:")
print(panel.head())

print("\nLast 5 rows:")
print(panel.tail())


output_path = (
    "data/regime/"
    "regime_price_panel.parquet"
)

panel.to_parquet(
    output_path,
    index=False
)

print(
    f"\nSaved successfully: {output_path}"
)