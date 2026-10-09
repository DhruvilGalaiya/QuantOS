import os

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/regime/regime_features.parquet"
OUTPUT_DIR = "data/regime/hmm"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. Load feature dataset
# =========================================================

df = pd.read_parquet(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# =========================================================
# 2. Final HMM feature set
# =========================================================

FEATURE_COLUMNS = [
    "NIFTY_50_RET_1D",
    "NIFTY_RET_20D",
    "NIFTY_VOL_20D",
    "NIFTY_DRAWDOWN_60D",
    "NIFTY_BANK_REL_20D",
    "NIFTY_IT_REL_20D",
    "SECTOR_DISPERSION_1D",
    "VIX_REL_20D",
]


X = df[FEATURE_COLUMNS].copy()

dates = df["date"].copy()


# =========================================================
# 3. Chronological split
# =========================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


X_train = X.iloc[:train_end].copy()
X_validation = X.iloc[train_end:validation_end].copy()
X_test = X.iloc[validation_end:].copy()

dates_train = dates.iloc[:train_end].copy()
dates_validation = dates.iloc[train_end:validation_end].copy()
dates_test = dates.iloc[validation_end:].copy()


# =========================================================
# 4. Standardize using TRAINING data only
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 5. Save datasets
# =========================================================

train_df = pd.DataFrame(
    X_train_scaled,
    columns=FEATURE_COLUMNS,
)

train_df.insert(
    0,
    "date",
    dates_train.values,
)


validation_df = pd.DataFrame(
    X_validation_scaled,
    columns=FEATURE_COLUMNS,
)

validation_df.insert(
    0,
    "date",
    dates_validation.values,
)


test_df = pd.DataFrame(
    X_test_scaled,
    columns=FEATURE_COLUMNS,
)

test_df.insert(
    0,
    "date",
    dates_test.values,
)


train_df.to_parquet(
    f"{OUTPUT_DIR}/train.parquet",
    index=False,
)

validation_df.to_parquet(
    f"{OUTPUT_DIR}/validation.parquet",
    index=False,
)

test_df.to_parquet(
    f"{OUTPUT_DIR}/test.parquet",
    index=False,
)


# Save scaler for later live inference

joblib.dump(
    scaler,
    f"{OUTPUT_DIR}/scaler.joblib",
)


# Save feature list

with open(
    f"{OUTPUT_DIR}/feature_columns.txt",
    "w",
) as f:

    for feature in FEATURE_COLUMNS:
        f.write(feature + "\n")


# =========================================================
# 6. Diagnostics
# =========================================================

print("\nQuantOS HMM Dataset Preparation")
print("=" * 70)

print("\nFeatures:", len(FEATURE_COLUMNS))

for feature in FEATURE_COLUMNS:
    print(f"  {feature}")


print("\nDataset sizes")
print("-" * 70)

print(
    f"TRAIN      : {len(train_df):4d} rows | "
    f"{dates_train.min().date()} → "
    f"{dates_train.max().date()}"
)

print(
    f"VALIDATION : {len(validation_df):4d} rows | "
    f"{dates_validation.min().date()} → "
    f"{dates_validation.max().date()}"
)

print(
    f"TEST       : {len(test_df):4d} rows | "
    f"{dates_test.min().date()} → "
    f"{dates_test.max().date()}"
)


print("\nScaled training feature means:")
print(
    train_df[FEATURE_COLUMNS]
    .mean()
    .round(4)
)


print("\nScaled training feature std:")
print(
    train_df[FEATURE_COLUMNS]
    .std()
    .round(4)
)


print(
    f"\nSaved HMM datasets to: {OUTPUT_DIR}"
)