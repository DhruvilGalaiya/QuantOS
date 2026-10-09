"""
QuantOS ML dataset builder.

Joins engineered market features with the leakage-aware
train / validation / test target splits.

IMPORTANT:
- Features come from market_features.
- Targets and dataset membership come from dataset_train,
  dataset_validation and dataset_test.
- The final 2026 test observations are never used for fitting.
"""

from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text


# ============================================================
# PATH SETUP
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.database import engine


# ============================================================
# FEATURE / TARGET DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "return_1d",
    "log_return_1d",
    "sma_10",
    "sma_20",
    "sma_50",
    "volatility_20",
    "rsi_14",
    "volume_sma_20",
    "volume_ratio",
    "atr_14",
    "close_vs_sma20",
    "close_vs_sma50",
]


TARGET_COLUMNS = [
    "target_return_1d",
    "target_return_5d",
    "target_direction_1d",
    "target_direction_5d",
]


IDENTIFIER_COLUMNS = [
    "timestamp",
    "symbol",
    "dataset_split",
]


# ============================================================
# SQL
# ============================================================

BASE_QUERY = """
SELECT
    f.timestamp,
    f.symbol,

    f.return_1d,
    f.log_return_1d,
    f.sma_10,
    f.sma_20,
    f.sma_50,
    f.volatility_20,
    f.rsi_14,
    f.volume_sma_20,
    f.volume_ratio,
    f.atr_14,
    f.close_vs_sma20,
    f.close_vs_sma50,

    t.target_return_1d,
    t.target_return_5d,
    t.target_direction_1d,
    t.target_direction_5d,

    t.dataset_split

FROM market_features f

INNER JOIN {target_table} t
    ON f.timestamp = t.timestamp
    AND f.symbol = t.symbol

ORDER BY
    f.symbol,
    f.timestamp
"""


# ============================================================
# LOAD ONE DATASET
# ============================================================

def load_dataset(target_table: str) -> pd.DataFrame:
    """
    Load one leakage-aware split and join it with market features.
    """

    query = text(
        BASE_QUERY.format(target_table=target_table)
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise RuntimeError(
            f"{target_table} produced zero rows after feature join."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame, name: str):
    """
    Validate the joined ML dataset.
    """

    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET VALIDATION")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Symbols: {df['symbol'].nunique()}")

    print(
        f"Date range: "
        f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = (
        IDENTIFIER_COLUMNS
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"{name}: missing columns: {missing_columns}"
        )

    print("Required columns: PASS")

    # --------------------------------------------------------
    # Duplicate observations
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=["timestamp", "symbol"]
    ).sum()

    print(f"Duplicate timestamp/symbol rows: {duplicates}")

    if duplicates != 0:
        raise RuntimeError(
            f"{name}: duplicate timestamp/symbol observations found."
        )

    # --------------------------------------------------------
    # Missing features
    # --------------------------------------------------------

    feature_nulls = df[FEATURE_COLUMNS].isnull().sum()

    print("\nFeature null counts:")

    nonzero_feature_nulls = feature_nulls[
        feature_nulls > 0
    ]

    if len(nonzero_feature_nulls) == 0:
        print("  None")
    else:
        print(nonzero_feature_nulls.to_string())

    # --------------------------------------------------------
    # Missing targets
    # --------------------------------------------------------

    target_nulls = df[TARGET_COLUMNS].isnull().sum()

    print("\nTarget null counts:")

    nonzero_target_nulls = target_nulls[
        target_nulls > 0
    ]

    if len(nonzero_target_nulls) == 0:
        print("  None")
    else:
        print(nonzero_target_nulls.to_string())

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    print("\nDataset split distribution:")

    print(
        df["dataset_split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Direction targets
    # --------------------------------------------------------

    print("\nDirection 1D distribution:")

    print(
        df["target_direction_1d"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nDirection 5D distribution:")

    print(
        df["target_direction_5d"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print(f"\n{name.upper()} VALIDATION COMPLETE")


# ============================================================
# LOAD ALL DATASETS
# ============================================================

def load_all_datasets():

    print("\n" + "=" * 70)
    print("BUILDING ML DATASETS")
    print("=" * 70)

    print("\nLoading training dataset...")
    train = load_dataset("dataset_train")

    print("Loading validation dataset...")
    validation = load_dataset("dataset_validation")

    print("Loading test dataset...")
    test = load_dataset("dataset_test")

    validate_dataset(train, "train")
    validate_dataset(validation, "validation")
    validate_dataset(test, "test")

    return train, validation, test


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train, validation, test = load_all_datasets()

    print("\n" + "=" * 70)
    print("ML DATASET SUMMARY")
    print("=" * 70)

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")

    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(f"  - {column}")

    print("\nTarget columns:")
    for column in TARGET_COLUMNS:
        print(f"  - {column}")

    print("\nML DATASET BUILD COMPLETE")
    print("=" * 70)