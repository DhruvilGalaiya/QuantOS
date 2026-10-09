"""
QuantOS - Histogram Gradient Boosting Baseline

Purpose:
    Train HistGradientBoostingClassifier models for:
        1. 1D direction prediction
        2. 5D direction prediction

Evaluation:
    - Training set
    - Validation set
    - Final out-of-sample test set

Important:
    Final test results are diagnostic only.
    Do not tune hyperparameters using final test performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PATH SETUP
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.database import engine


# ============================================================
# FEATURES
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


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "target_direction_1d": "1D_DIRECTION",
    "target_direction_5d": "5D_DIRECTION",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load dataset from PostgreSQL and join market_features.
    """

    print(f"Loading {dataset_name} from PostgreSQL...")

    query = f"""
        SELECT
            d.timestamp,
            d.symbol,

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

            d.target_direction_1d,
            d.target_direction_5d,

            d.dataset_split

        FROM {dataset_name} d

        JOIN market_features f
            ON d.timestamp = f.timestamp
            AND d.symbol = f.symbol

        ORDER BY d.timestamp, d.symbol
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df):,} rows from {dataset_name}.")

    return df


def load_all_datasets() -> pd.DataFrame:

    print("=" * 70)
    print("QUANTOS HISTOGRAM GRADIENT BOOSTING MODEL")
    print("=" * 70)

    print("Loading ML datasets from PostgreSQL...")

    train = load_dataset("dataset_train")
    validation = load_dataset("dataset_validation")
    test = load_dataset("dataset_test")

    df = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    print()
    print("=" * 70)
    print("COMBINED ML DATASET")
    print("-" * 70)

    print(f"Total rows:  {len(df):,}")
    print(f"Columns:     {len(df.columns)}")
    print(f"Symbols:     {df['symbol'].nunique()}")

    print()
    print("Dataset splits:")
    print(df["dataset_split"].value_counts().sort_index())

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    target_column: str,
):

    required_columns = FEATURE_COLUMNS + [
        target_column,
        "dataset_split",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # Remove rows where target itself is unavailable.
    df = df.dropna(
        subset=[target_column]
    ).copy()

    # --------------------------------------------------------
    # TEMPORAL SPLITS
    # --------------------------------------------------------

    train = df[
        df["dataset_split"].isin(
            [
                "historical_train",
                "recent_train",
            ]
        )
    ].copy()

    validation = df[
        df["dataset_split"].isin(
            [
                "historical_validation",
                "recent_validation",
            ]
        )
    ].copy()

    test = df[
        df["dataset_split"].isin(
            [
                "historical_test",
                "final_test",
            ]
        )
    ].copy()

    if len(train) == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(validation) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if len(test) == 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    # --------------------------------------------------------
    # FEATURES / TARGETS
    # --------------------------------------------------------

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train[target_column].astype(int)

    X_val = validation[FEATURE_COLUMNS].copy()
    y_val = validation[target_column].astype(int)

    X_test = test[FEATURE_COLUMNS].copy()
    y_test = test[target_column].astype(int)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train,
        validation,
        test,
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


# ============================================================
# TRAIN TARGET
# ============================================================

def train_target(
    df: pd.DataFrame,
    target_column: str,
    target_name: str,
):

    print()
    print("#" * 70)
    print(
        f"TRAINING HISTOGRAM GRADIENT BOOSTING: "
        f"{target_name}"
    )
    print("#" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_df,
        val_df,
        test_df,
    ) = prepare_data(
        df,
        target_column,
    )

    print()
    print("DATASET SIZES")
    print("-" * 70)

    print(
        f"Training rows:       {len(X_train):,}"
    )

    print(
        f"Validation rows:     {len(X_val):,}"
    )

    print(
        f"Test rows:           {len(X_test):,}"
    )

    print()
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print("TRAIN:")
    print(
        y_train.value_counts()
        .sort_index()
    )

    print()
    print("VALIDATION:")
    print(
        y_val.value_counts()
        .sort_index()
    )

    print()
    print("TEST:")
    print(
        y_test.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    print()
    print("HistGradientBoosting configuration:")
    print("max_iter:           300")
    print("learning_rate:      0.05")
    print("max_leaf_nodes:     15")
    print("min_samples_leaf:   30")
    print("l2_regularization:  1.0")

    model = build_model()

    print()
    print(
        "Fitting HistGradientBoosting "
        "on TRAINING data only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "HistGradientBoosting fitting complete."
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Rows:               {len(X_val):,}"
    )

    print(
        f"Accuracy:           "
        f"{accuracy_score(y_val, val_pred):.4f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{balanced_accuracy_score(y_val, val_pred):.4f}"
    )

    print(
        f"Precision:           "
        f"{precision_score(y_val, val_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:              "
        f"{recall_score(y_val, val_pred, zero_division=0):.4f}"
    )

    print(
        f"F1 score:            "
        f"{f1_score(y_val, val_pred, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC:             "
        f"{roc_auc_score(y_val, val_prob):.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_val,
            val_pred,
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    print()
    print("=" * 70)
    print("FINAL TEST - OUT OF SAMPLE RESULTS")
    print("=" * 70)

    print(
        f"Rows:               {len(X_test):,}"
    )

    print(
        f"Accuracy:           "
        f"{accuracy_score(y_test, test_pred):.4f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{balanced_accuracy_score(y_test, test_pred):.4f}"
    )

    print(
        f"Precision:           "
        f"{precision_score(y_test, test_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:              "
        f"{recall_score(y_test, test_pred, zero_division=0):.4f}"
    )

    print(
        f"F1 score:            "
        f"{f1_score(y_test, test_pred, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC:             "
        f"{roc_auc_score(y_test, test_prob):.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            test_pred,
        )
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print()
    print("FEATURE IMPORTANCE")
    print("-" * 70)

    # HistGradientBoosting does not expose the same
    # feature_importances_ attribute as RandomForest.
    #
    # We therefore use permutation importance on the
    # VALIDATION set.

    from sklearn.inspection import permutation_importance

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importance_series = pd.Series(
        importance.importances_mean,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    for feature, value in importance_series.items():

        print(
            f"{feature:<20} {value:.6f}"
        )

    return {
        "model": model,
        "validation_auc": roc_auc_score(
            y_val,
            val_prob,
        ),
        "validation_f1": f1_score(
            y_val,
            val_pred,
            zero_division=0,
        ),
        "test_auc": roc_auc_score(
            y_test,
            test_prob,
        ),
        "test_f1": f1_score(
            y_test,
            test_pred,
            zero_division=0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_all_datasets()

    results = {}

    # --------------------------------------------------------
    # TRAIN BOTH TARGETS
    # --------------------------------------------------------

    for target_column, target_name in TARGETS.items():

        results[target_column] = train_target(
            df=df,
            target_column=target_column,
            target_name=target_name,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HISTOGRAM GRADIENT BOOSTING TRAINING COMPLETE")
    print("=" * 70)

    for target_column, result in results.items():

        print()
        print(target_column)

        print(
            f"Validation ROC-AUC: "
            f"{result['validation_auc']:.4f}"
        )

        print(
            f"Validation F1:      "
            f"{result['validation_f1']:.4f}"
        )

        print(
            f"Test ROC-AUC:       "
            f"{result['test_auc']:.4f}"
        )

        print(
            f"Test F1:            "
            f"{result['test_f1']:.4f}"
        )

    print()
    print("IMPORTANT:")
    print(
        "Final test metrics are diagnostic only."
    )
    print(
        "Do not tune the model using final test results."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()