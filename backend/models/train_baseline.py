"""
QuantOS baseline ML training pipeline.

Purpose:
    Train leakage-safe baseline classification models for:
        - 1-day market direction
        - 5-day market direction

Important:
    - Train data is used for fitting.
    - Validation data is used for model selection/evaluation.
    - Test data is NEVER used for fitting or model selection.
    - Missing feature values are imputed using TRAINING data only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


# ============================================================
# DATABASE
# ============================================================

from app.database import engine


# ============================================================
# CONFIGURATION
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

TARGETS = {
    "target_direction_1d": "1D_DIRECTION",
    "target_direction_5d": "5D_DIRECTION",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(dataset_name):
    """
    Load an ML dataset from PostgreSQL.

    dataset_train / dataset_validation / dataset_test contain
    targets and dataset split information.

    market_features contains the engineered feature columns.

    The two tables are joined on timestamp + symbol.
    """

    print(f"Loading {dataset_name} from PostgreSQL...")

    query = text(f"""
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

            d.target_return_1d,
            d.target_return_5d,

            d.target_end_timestamp_1d,
            d.target_end_timestamp_5d,

            d.dataset_split

        FROM {dataset_name} d

        INNER JOIN market_features f
            ON d.timestamp = f.timestamp
           AND d.symbol = f.symbol

        ORDER BY d.timestamp, d.symbol
    """)

    df = pd.read_sql(query, engine)

    if df.empty:
        raise RuntimeError(
            f"No rows returned when loading {dataset_name}."
        )

    print(
        f"Loaded {len(df):,} rows from {dataset_name} "
        f"after joining market_features."
    )

    return df


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df: pd.DataFrame, target_column: str):
    """
    Prepare X/y while preserving the temporal split.

    Missing values are handled later through a pipeline whose
    imputer is fitted ONLY on training data.
    """

    required_columns = FEATURE_COLUMNS + [
        target_column,
        "dataset_split",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=[target_column]
    ).copy()

    train = df[
        df["dataset_split"].isin(
            ["historical_train", "recent_train"]
        )
    ].copy()

    validation = df[
        df["dataset_split"].isin(
            ["historical_validation", "recent_validation"]
        )
    ].copy()

    test = df[
        df["dataset_split"].isin(
            ["historical_test", "final_test"]
        )
    ].copy()

    if len(train) == 0:
        raise RuntimeError("Training dataset is empty.")

    if len(validation) == 0:
        raise RuntimeError("Validation dataset is empty.")

    if len(test) == 0:
        raise RuntimeError("Test dataset is empty.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_column].astype(int)

    X_val = validation[FEATURE_COLUMNS]
    y_val = validation[target_column].astype(int)

    X_test = test[FEATURE_COLUMNS]
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
    """
    Leakage-safe preprocessing + logistic regression.

    Median imputation is fitted only on the training set because
    the entire preprocessing stack is inside the Pipeline.
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name: str,
):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            y,
            probabilities,
        )
    except ValueError:
        auc = float("nan")

    matrix = confusion_matrix(
        y,
        predictions,
    )

    print()
    print("=" * 70)
    print(f"{dataset_name} RESULTS")
    print("=" * 70)

    print(f"Rows:               {len(y):,}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Balanced accuracy:   {balanced_accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 score:            {f1:.4f}")
    print(f"ROC-AUC:             {auc:.4f}")

    print()
    print("Confusion matrix:")
    print(matrix)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


# ============================================================
# TRAIN ONE TARGET
# ============================================================

def train_target(
    df: pd.DataFrame,
    target_column: str,
    target_name: str,
):
    print()
    print("#" * 70)
    print(f"TRAINING BASELINE MODEL: {target_name}")
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
    print(f"Training rows:       {len(X_train):,}")
    print(f"Validation rows:     {len(X_val):,}")
    print(f"Test rows:           {len(X_test):,}")

    print()
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print("TRAIN:")
    print(y_train.value_counts().sort_index())

    print()
    print("VALIDATION:")
    print(y_val.value_counts().sort_index())

    print()
    print("TEST:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()

    print()
    print("Fitting model on TRAINING data only...")

    model.fit(
        X_train,
        y_train,
    )

    print("Model fitting complete.")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # TEST
    #
    # We print test results for diagnostic purposes only.
    # They must NOT be used for model selection.
    # --------------------------------------------------------

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "FINAL TEST - OUT OF SAMPLE",
    )

    # --------------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------------

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = pd.Series(
        classifier.coef_[0],
        index=FEATURE_COLUMNS,
    ).sort_values(
        key=np.abs,
        ascending=False,
    )

    print()
    print("FEATURE COEFFICIENTS")
    print("-" * 70)

    print(coefficients)

    return {
        "model": model,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("QUANTOS BASELINE ML TRAINING")
    print("=" * 70)

    print("Loading ML datasets from PostgreSQL...")

    print("Loading dataset_train...")
    train_df = load_dataset("dataset_train")

    print("Loading dataset_validation...")
    validation_df = load_dataset("dataset_validation")

    print("Loading dataset_test...")
    test_df = load_dataset("dataset_test")

    # Combine the three temporal datasets.
    # dataset_split keeps the original train/validation/test boundaries.
    df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    # Always sort chronologically before any downstream processing.
    df = df.sort_values(
        ["timestamp", "symbol"]
    ).reset_index(drop=True)

    print()
    print("COMBINED ML DATASET")
    print("-" * 70)
    print(f"Total rows:  {len(df):,}")
    print(f"Columns:     {len(df.columns)}")
    print(f"Symbols:     {df['symbol'].nunique()}")

    print()
    print("Dataset splits:")
    print(
        df["dataset_split"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # TRAIN BOTH TARGETS
    # --------------------------------------------------------

    results = {}

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
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)

    for target_column, result in results.items():

        print()
        print(
            target_column
        )

        val = result[
            "validation_metrics"
        ]

        test = result[
            "test_metrics"
        ]

        print(
            f"Validation ROC-AUC: "
            f"{val['roc_auc']:.4f}"
        )

        print(
            f"Validation F1:      "
            f"{val['f1']:.4f}"
        )

        print(
            f"Test ROC-AUC:       "
            f"{test['roc_auc']:.4f}"
        )

        print(
            f"Test F1:            "
            f"{test['f1']:.4f}"
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Final test metrics are diagnostic only."
    )
    print(
        "Do not tune the model using the final test results."
    )


if __name__ == "__main__":
    main()