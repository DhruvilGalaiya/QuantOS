"""
QuantOS Random Forest baseline.

Purpose:
    Train a nonlinear tree-based classifier using the exact same
    temporal train / validation / test datasets used by the
    logistic regression baseline.

Important:
    - No test-set tuning.
    - Imputer is fitted on training data only.
    - Validation is used for model assessment.
    - Final test remains out-of-sample.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
# IMPORT EXISTING QUANTOS DATASET DEFINITIONS
# ============================================================

from data.ml_dataset import (
    load_dataset,
    FEATURE_COLUMNS,
)


# ============================================================
# TARGET DEFINITIONS
# ============================================================

TARGETS = {
    "target_direction_1d": "1D_DIRECTION",
    "target_direction_5d": "5D_DIRECTION",
}


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500

MAX_DEPTH = 10

MIN_SAMPLES_LEAF = 10


# ============================================================
# COMBINE DATASETS
# ============================================================

def load_all_datasets():

    print("Loading ML datasets from PostgreSQL...")

    print("Loading dataset_train...")
    train_df = load_dataset("dataset_train")

    print("Loading dataset_validation...")
    validation_df = load_dataset("dataset_validation")

    print("Loading dataset_test...")
    test_df = load_dataset("dataset_test")

    df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    df = (
        df
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )

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

    return df


# ============================================================
# PREPARE TEMPORAL DATA
# ============================================================

def prepare_data(df, target_column):

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

    df = df.dropna(
        subset=[target_column]
    ).copy()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train = df[
        df["dataset_split"].isin(
            [
                "historical_train",
                "recent_train",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation = df[
        df["dataset_split"].isin(
            [
                "historical_validation",
                "recent_validation",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

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
# BUILD RANDOM FOREST
# ============================================================

def build_random_forest():

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
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
    dataset_name,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

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

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    cm = confusion_matrix(
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
    print(f"ROC-AUC:             {roc_auc:.4f}")

    print()
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(model):

    rf = model.named_steps["model"]

    importance = pd.Series(
        rf.feature_importances_,
        index=FEATURE_COLUMNS,
    )

    importance = (
        importance
        .sort_values(ascending=False)
    )

    print()
    print("=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    for feature, value in importance.items():

        print(
            f"{feature:<20} {value:.6f}"
        )


# ============================================================
# TRAIN ONE TARGET
# ============================================================

def train_target(
    df,
    target_column,
    target_name,
):

    print()
    print("#" * 70)
    print(
        f"TRAINING RANDOM FOREST: {target_name}"
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
        validation_df,
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
        y_train
        .value_counts()
        .sort_index()
    )

    print()
    print("VALIDATION:")
    print(
        y_val
        .value_counts()
        .sort_index()
    )

    print()
    print("TEST:")
    print(
        y_test
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_random_forest()

    print()
    print("Random Forest configuration:")
    print(
        f"n_estimators:       {N_ESTIMATORS}"
    )
    print(
        f"max_depth:          {MAX_DEPTH}"
    )
    print(
        f"min_samples_leaf:   {MIN_SAMPLES_LEAF}"
    )
    print(
        f"max_features:       sqrt"
    )

    print()
    print(
        "Fitting Random Forest on TRAINING data only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Random Forest fitting complete."
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "FINAL TEST - OUT OF SAMPLE",
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print_feature_importance(
        model
    )

    return {
        "validation": validation_results,
        "test": test_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QUANTOS RANDOM FOREST MODEL")
    print("=" * 70)

    df = load_all_datasets()

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
    print("RANDOM FOREST TRAINING COMPLETE")
    print("=" * 70)

    for target_column, result in results.items():

        print()
        print(target_column)

        print(
            f"Validation ROC-AUC: "
            f"{result['validation']['roc_auc']:.4f}"
        )

        print(
            f"Validation F1:      "
            f"{result['validation']['f1']:.4f}"
        )

        print(
            f"Test ROC-AUC:       "
            f"{result['test']['roc_auc']:.4f}"
        )

        print(
            f"Test F1:            "
            f"{result['test']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()