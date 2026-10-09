import os
import numpy as np
import pandas as pd


INPUT_PATH = "data/regime/regime_price_panel.parquet"
OUTPUT_PATH = "data/regime/regime_features.parquet"


# ---------------------------------------------------------
# 1. Load price panel
# ---------------------------------------------------------

df = pd.read_parquet(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. Define groups
# ---------------------------------------------------------

market = "NIFTY_50"

sectors = [
    "NIFTY_BANK",
    "NIFTY_IT",
    "NIFTY_PHARMA",
    "NIFTY_AUTO",
    "NIFTY_FIN_SERVICE",
]

vix = "INDIA_VIX"


# ---------------------------------------------------------
# 3. Daily log returns
# ---------------------------------------------------------

for column in [
    market,
    *sectors,
    vix,
]:

    df[f"{column}_RET_1D"] = np.log(
        df[column] / df[column].shift(1)
    )


# ---------------------------------------------------------
# 4. NIFTY return / momentum
# ---------------------------------------------------------

df["NIFTY_RET_5D"] = np.log(
    df[market] / df[market].shift(5)
)

df["NIFTY_RET_20D"] = np.log(
    df[market] / df[market].shift(20)
)


# ---------------------------------------------------------
# 5. Realized volatility
# ---------------------------------------------------------

df["NIFTY_VOL_20D"] = (
    df["NIFTY_50_RET_1D"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

df["NIFTY_VOL_60D"] = (
    df["NIFTY_50_RET_1D"]
    .rolling(60)
    .std()
    * np.sqrt(252)
)


# ---------------------------------------------------------
# 6. Drawdown
# ---------------------------------------------------------

rolling_peak = (
    df[market]
    .rolling(60)
    .max()
)

df["NIFTY_DRAWDOWN_60D"] = (
    df[market] / rolling_peak - 1
)


# ---------------------------------------------------------
# 7. Sector relative performance
# ---------------------------------------------------------

for sector in sectors:

    df[f"{sector}_REL_20D"] = (
        np.log(
            df[sector] / df[sector].shift(20)
        )
        -
        df["NIFTY_RET_20D"]
    )


# ---------------------------------------------------------
# 8. Cross-sectional sector dispersion
# ---------------------------------------------------------

sector_returns = df[
    [f"{sector}_RET_1D" for sector in sectors]
]

df["SECTOR_DISPERSION_1D"] = (
    sector_returns.std(axis=1)
)


# ---------------------------------------------------------
# 9. India VIX features
# ---------------------------------------------------------

df["VIX_LEVEL"] = np.log(
    df[vix]
)

df["VIX_RET_5D"] = np.log(
    df[vix] / df[vix].shift(5)
)

df["VIX_MA_20D"] = (
    df[vix]
    .rolling(20)
    .mean()
)

df["VIX_REL_20D"] = (
    df[vix] / df["VIX_MA_20D"] - 1
)


# ---------------------------------------------------------
# 10. Select HMM features
# ---------------------------------------------------------

FEATURE_COLUMNS = [

    # Market direction
    "NIFTY_50_RET_1D",
    "NIFTY_RET_5D",
    "NIFTY_RET_20D",

    # Market volatility
    "NIFTY_VOL_20D",
    "NIFTY_VOL_60D",

    # Market stress
    "NIFTY_DRAWDOWN_60D",

    # Sector rotation
    "NIFTY_BANK_REL_20D",
    "NIFTY_IT_REL_20D",
    "NIFTY_PHARMA_REL_20D",
    "NIFTY_AUTO_REL_20D",
    "NIFTY_FIN_SERVICE_REL_20D",

    # Cross-sectional behavior
    "SECTOR_DISPERSION_1D",

    # Implied volatility / fear
    "VIX_LEVEL",
    "VIX_RET_5D",
    "VIX_REL_20D",
]


features = df[
    ["date"] + FEATURE_COLUMNS
].copy()


# ---------------------------------------------------------
# 11. Remove warm-up period
# ---------------------------------------------------------

features = features.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# 12. Save
# ---------------------------------------------------------

os.makedirs(
    "data/regime",
    exist_ok=True
)

features.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# 13. Diagnostics
# ---------------------------------------------------------

print("\nQuantOS HMM Feature Dataset")
print("=" * 70)

print("Shape:", features.shape)

print(
    "Date range:",
    features["date"].min().date(),
    "→",
    features["date"].max().date()
)

print("\nFeatures:")
for column in FEATURE_COLUMNS:
    print(f"  {column}")

print("\nMissing values:")
print(features.isna().sum())

print("\nFeature summary:")
print(
    features[FEATURE_COLUMNS]
    .describe()
    .T[
        ["mean", "std", "min", "max"]
    ]
)

print(
    f"\nSaved successfully: {OUTPUT_PATH}"
)