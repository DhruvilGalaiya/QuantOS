"""
QuantOS - HMM V3.1
Proper Walk-Forward Independent Regime Detection

V3.1 fixes the V3 validation design by ensuring:

2019-2020 -> validation
2021-2022 -> validation
2023-2024 -> validation
2025-2026 -> FINAL TEST ONLY

The final test is never used as a validation period.

Each index receives its own independently trained HMM.

Important:
- Future returns are NEVER used to train the HMM.
- State labels are determined using training data only.
- Validation data is genuinely unseen.
- Final test data is completely untouched until the end.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sqlalchemy import text
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# ============================================================
# PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database import engine


# ============================================================
# CONFIGURATION
# ============================================================

N_STATES = 3

N_ITER = 300

RANDOM_STATE = 42

MIN_TRAIN_ROWS = 750


# ------------------------------------------------------------
# Validation periods
# ------------------------------------------------------------

VALIDATION_PERIODS = [
    (2019, 2020),
    (2021, 2022),
    (2023, 2024),
]


# ------------------------------------------------------------
# FINAL TEST
# ------------------------------------------------------------

FINAL_TEST_START = 2025

FINAL_TEST_END = 2026


# ------------------------------------------------------------
# Indices
# ------------------------------------------------------------

EQUITY_SYMBOLS = [
    "NIFTY50",
    "NIFTYBANK",
    "NIFTYIT",
    "NIFTYAUTO",
    "NIFTYPHARMA",
    "NIFTYNEXT50",
    "NASDAQ100",
    "SP500",
]


# ============================================================
# HMM FEATURES
# ============================================================

HMM_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING EQUITY INDEX DATA")
    print("=" * 70)

    placeholders = ", ".join(
        f":symbol_{i}"
        for i in range(len(EQUITY_SYMBOLS))
    )

    query = text(
        f"""
        SELECT
            timestamp,
            symbol,
            close,
            return_1d,
            volatility_20,
            rsi_14,
            volume_ratio,
            close_vs_sma20,
            close_vs_sma50

        FROM market_features

        WHERE symbol IN ({placeholders})

        ORDER BY
            symbol,
            timestamp
        """
    )

    params = {
        f"symbol_{i}": symbol
        for i, symbol in enumerate(EQUITY_SYMBOLS)
    }

    with engine.connect() as connection:

        df = pd.read_sql(
            query,
            connection,
            params=params,
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df
        .sort_values(
            ["symbol", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df):,} rows."
    )

    print(
        f"Symbols: {df['symbol'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['timestamp'].min().date()} "
        f"-> "
        f"{df['timestamp'].max().date()}"
    )

    return df


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):

    print()
    print("=" * 70)
    print("BUILDING HISTORICAL REGIME FEATURES")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Historical returns
    # --------------------------------------------------------

    df["return_5d"] = (
        df.groupby("symbol")["close"]
        .pct_change(5)
    )

    df["return_20d"] = (
        df.groupby("symbol")["close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # FUTURE RETURNS
    #
    # These are evaluation-only.
    # They NEVER enter HMM training.
    # --------------------------------------------------------

    df["forward_1d"] = (
        df.groupby("symbol")["close"]
        .shift(-1)
        / df["close"]
        - 1
    )

    df["forward_5d"] = (
        df.groupby("symbol")["close"]
        .shift(-5)
        / df["close"]
        - 1
    )

    df["forward_20d"] = (
        df.groupby("symbol")["close"]
        .shift(-20)
        / df["close"]
        - 1
    )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def fit_preprocessor(train_df):

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    X_train = train_df[
        HMM_FEATURES
    ]

    X_train = imputer.fit_transform(
        X_train
    )

    X_train = scaler.fit_transform(
        X_train
    )

    return (
        imputer,
        scaler,
        X_train,
    )


# ============================================================
# TRANSFORM FEATURES
# ============================================================

def transform_features(
    df,
    imputer,
    scaler,
):

    X = df[
        HMM_FEATURES
    ]

    X = imputer.transform(
        X
    )

    X = scaler.transform(
        X
    )

    return X


# ============================================================
# TRAIN HMM
# ============================================================

def train_hmm(X):

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    model.fit(X)

    return model


# ============================================================
# STATE INTERPRETATION
# ============================================================

def determine_state_labels(
    train_df,
    states,
):

    temp = train_df.copy()

    temp["hmm_state"] = states

    rows = []

    for state in sorted(
        temp["hmm_state"].unique()
    ):

        state_df = temp[
            temp["hmm_state"] == state
        ]

        rows.append(
            {
                "state": state,

                "observations":
                    len(state_df),

                "return_20d":
                    state_df[
                        "return_20d"
                    ].mean(),

                "close_vs_sma50":
                    state_df[
                        "close_vs_sma50"
                    ].mean(),

                "rsi":
                    state_df[
                        "rsi_14"
                    ].mean(),

                "volatility":
                    state_df[
                        "volatility_20"
                    ].mean(),
            }
        )

    stats = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Rank states using TRAINING data only
    # --------------------------------------------------------

    return_rank = (
        stats["return_20d"]
        .rank(
            method="average"
        )
    )

    trend_rank = (
        stats["close_vs_sma50"]
        .rank(
            method="average"
        )
    )

    stats["directional_score"] = (
        0.60 * return_rank
        +
        0.40 * trend_rank
    )

    stats = (
        stats
        .sort_values(
            "directional_score"
        )
        .reset_index(drop=True)
    )

    bear_state = int(
        stats.iloc[0]["state"]
    )

    bull_state = int(
        stats.iloc[-1]["state"]
    )

    middle_states = [
        int(x)
        for x in stats["state"]
        if int(x)
        not in [
            bear_state,
            bull_state,
        ]
    ]

    side_state = middle_states[0]

    labels = {
        bear_state: "BEAR",
        side_state: "SIDE",
        bull_state: "BULL",
    }

    return (
        labels,
        stats,
    )


# ============================================================
# STATE DURATIONS
# ============================================================

def calculate_durations(
    states
):

    if len(states) == 0:

        return {
            "mean_duration": 0.0,
            "median_duration": 0.0,
            "transitions": 0,
            "one_day_states_pct": 0.0,
        }

    durations = []

    current_state = states[0]

    duration = 1

    transitions = 0

    for state in states[1:]:

        if state == current_state:

            duration += 1

        else:

            durations.append(
                duration
            )

            transitions += 1

            current_state = state

            duration = 1

    durations.append(
        duration
    )

    durations = np.array(
        durations
    )

    return {
        "mean_duration":
            float(
                durations.mean()
            ),

        "median_duration":
            float(
                np.median(durations)
            ),

        "transitions":
            transitions,

        "one_day_states_pct":
            float(
                (durations == 1).mean()
                * 100
            ),
    }


# ============================================================
# REGIME PERFORMANCE
# ============================================================

def evaluate_regimes(
    df,
    states,
    labels,
):

    result = df.copy()

    result["hmm_state"] = states

    result["regime"] = (
        result["hmm_state"]
        .map(labels)
    )

    rows = []

    for regime in [
        "BULL",
        "SIDE",
        "BEAR",
    ]:

        regime_df = result[
            result["regime"] == regime
        ]

        if len(regime_df) == 0:

            rows.append(
                {
                    "regime": regime,
                    "observations": 0,
                    "forward_1d": np.nan,
                    "forward_5d": np.nan,
                    "forward_20d": np.nan,
                    "win_rate_5d": np.nan,
                    "volatility_20":
                        np.nan,
                }
            )

            continue

        rows.append(
            {
                "regime": regime,

                "observations":
                    len(regime_df),

                "forward_1d":
                    regime_df[
                        "forward_1d"
                    ].mean(),

                "forward_5d":
                    regime_df[
                        "forward_5d"
                    ].mean(),

                "forward_20d":
                    regime_df[
                        "forward_20d"
                    ].mean(),

                "win_rate_5d":
                    (
                        regime_df[
                            "forward_5d"
                        ] > 0
                    ).mean(),

                "volatility_20":
                    regime_df[
                        "volatility_20"
                    ].mean(),
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    durations = calculate_durations(
        states
    )

    return (
        result,
        metrics,
        durations,
    )


# ============================================================
# RUN VALIDATION PERIOD
# ============================================================

def run_validation_period(
    symbol,
    symbol_df,
    validation_start,
    validation_end,
):

    print()
    print("=" * 70)

    print(
        f"{symbol} | "
        f"VALIDATION "
        f"{validation_start}-"
        f"{validation_end}"
    )

    print("=" * 70)

    validation_start_date = pd.Timestamp(
        f"{validation_start}-01-01"
    )

    validation_end_date = pd.Timestamp(
        f"{validation_end}-12-31"
    )

    # --------------------------------------------------------
    # TRAIN
    #
    # Everything before validation period.
    # --------------------------------------------------------

    train_df = symbol_df[
        symbol_df["timestamp"]
        < validation_start_date
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_df = symbol_df[
        (
            symbol_df["timestamp"]
            >= validation_start_date
        )
        &
        (
            symbol_df["timestamp"]
            <= validation_end_date
        )
    ].copy()

    if len(train_df) < MIN_TRAIN_ROWS:

        print(
            f"Skipping. "
            f"Training rows: "
            f"{len(train_df)}"
        )

        return None

    if len(validation_df) == 0:

        print(
            "Skipping. "
            "Validation dataset empty."
        )

        return None

    print(
        f"Training rows:   "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # FIT PREPROCESSOR ON TRAINING ONLY
    # --------------------------------------------------------

    (
        imputer,
        scaler,
        X_train,
    ) = fit_preprocessor(
        train_df
    )

    X_validation = transform_features(
        validation_df,
        imputer,
        scaler,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_hmm(
        X_train
    )

    # --------------------------------------------------------
    # DECODE TRAINING STATES
    # --------------------------------------------------------

    train_states = model.predict(
        X_train
    )

    (
        labels,
        state_stats,
    ) = determine_state_labels(
        train_df,
        train_states,
    )

    print()
    print(
        "TRAINING STATE MAPPING"
    )

    for state in sorted(labels):

        print(
            f"State {state} -> "
            f"{labels[state]}"
        )

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    validation_states = model.predict(
        X_validation
    )

    (
        evaluated_df,
        metrics,
        durations,
    ) = evaluate_regimes(
        validation_df,
        validation_states,
        labels,
    )

    print()
    print(
        "OUT-OF-SAMPLE REGIME PERFORMANCE"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "STATE STABILITY"
    )

    print(
        f"Mean duration: "
        f"{durations['mean_duration']:.2f}"
    )

    print(
        f"Median duration: "
        f"{durations['median_duration']:.2f}"
    )

    print(
        f"Transitions: "
        f"{durations['transitions']}"
    )

    print(
        f"1-day states: "
        f"{durations['one_day_states_pct']:.2f}%"
    )

    return {
        "symbol": symbol,
        "period": (
            validation_start,
            validation_end,
        ),
        "metrics": metrics,
        "durations": durations,
        "labels": labels,
        "state_stats": state_stats,
        "data": evaluated_df,
    }


# ============================================================
# FINAL TEST
# ============================================================

def run_final_test(
    symbol,
    symbol_df,
):

    print()
    print("=" * 70)

    print(
        f"{symbol} | "
        f"FINAL TEST "
        f"{FINAL_TEST_START}-"
        f"{FINAL_TEST_END}"
    )

    print("=" * 70)

    final_test_start = pd.Timestamp(
        f"{FINAL_TEST_START}-01-01"
    )

    final_test_end = pd.Timestamp(
        f"{FINAL_TEST_END}-12-31"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_df = symbol_df[
        (
            symbol_df["timestamp"]
            >= final_test_start
        )
        &
        (
            symbol_df["timestamp"]
            <= final_test_end
        )
    ].copy()

    # --------------------------------------------------------
    # TRAIN USING EVERYTHING BEFORE FINAL TEST
    #
    # This includes:
    # 2000-2024
    #
    # But NEVER 2025-2026.
    # --------------------------------------------------------

    train_df = symbol_df[
        symbol_df["timestamp"]
        < final_test_start
    ].copy()

    if len(train_df) < MIN_TRAIN_ROWS:

        print(
            "Insufficient final-test "
            "training data."
        )

        return None

    if len(test_df) == 0:

        print(
            "Final test dataset is empty."
        )

        return None

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Final test rows: "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # FIT PREPROCESSOR
    # --------------------------------------------------------

    (
        imputer,
        scaler,
        X_train,
    ) = fit_preprocessor(
        train_df
    )

    X_test = transform_features(
        test_df,
        imputer,
        scaler,
    )

    # --------------------------------------------------------
    # TRAIN HMM
    # --------------------------------------------------------

    model = train_hmm(
        X_train
    )

    # --------------------------------------------------------
    # DETERMINE LABELS FROM TRAINING ONLY
    # --------------------------------------------------------

    train_states = model.predict(
        X_train
    )

    (
        labels,
        state_stats,
    ) = determine_state_labels(
        train_df,
        train_states,
    )

    print()
    print(
        "FINAL TRAINING STATE MAPPING"
    )

    for state in sorted(labels):

        print(
            f"State {state} -> "
            f"{labels[state]}"
        )

    # --------------------------------------------------------
    # FINAL TEST PREDICTION
    # --------------------------------------------------------

    test_states = model.predict(
        X_test
    )

    (
        evaluated_df,
        metrics,
        durations,
    ) = evaluate_regimes(
        test_df,
        test_states,
        labels,
    )

    print()
    print(
        "FINAL OUT-OF-SAMPLE TEST"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "FINAL TEST STABILITY"
    )

    print(
        f"Mean duration: "
        f"{durations['mean_duration']:.2f}"
    )

    print(
        f"Median duration: "
        f"{durations['median_duration']:.2f}"
    )

    print(
        f"Transitions: "
        f"{durations['transitions']}"
    )

    print(
        f"1-day states: "
        f"{durations['one_day_states_pct']:.2f}%"
    )

    return {
        "metrics": metrics,
        "durations": durations,
        "labels": labels,
        "state_stats": state_stats,
        "data": evaluated_df,
    }


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    symbol_df,
):

    print()
    print()
    print("#" * 70)
    print(
        f"QUANTOS HMM V3.1: {symbol}"
    )
    print("#" * 70)

    validation_results = []

    # --------------------------------------------------------
    # WALK-FORWARD VALIDATION
    # --------------------------------------------------------

    for (
        validation_start,
        validation_end,
    ) in VALIDATION_PERIODS:

        result = run_validation_period(
            symbol,
            symbol_df,
            validation_start,
            validation_end,
        )

        if result is not None:

            validation_results.append(
                result
            )

    # --------------------------------------------------------
    # FINAL TEST
    #
    # This happens ONLY after all validation
    # periods have been evaluated.
    # --------------------------------------------------------

    final_test = run_final_test(
        symbol,
        symbol_df,
    )

    return {
        "validation": validation_results,
        "final_test": final_test,
    }


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    all_results
):

    print()
    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V3.1 SUMMARY"
    )
    print("=" * 70)

    rows = []

    for symbol, result in (
        all_results.items()
    ):

        for period_result in (
            result["validation"]
        ):

            metrics = (
                period_result[
                    "metrics"
                ]
            )

            for _, row in (
                metrics.iterrows()
            ):

                rows.append(
                    {
                        "symbol":
                            symbol,

                        "period":
                            (
                                f"{period_result['period'][0]}"
                                "-"
                                f"{period_result['period'][1]}"
                            ),

                        "regime":
                            row["regime"],

                        "observations":
                            row["observations"],

                        "forward_5d":
                            row["forward_5d"],

                        "forward_20d":
                            row["forward_20d"],

                        "win_rate_5d":
                            row["win_rate_5d"],

                        "volatility_20":
                            row["volatility_20"],
                    }
                )

    if rows:

        summary = pd.DataFrame(
            rows
        )

        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print(
        "V3.1 VALIDATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "VALIDATION TIMELINE"
    )

    print(
        "2019-2020 -> validation"
    )

    print(
        "2021-2022 -> validation"
    )

    print(
        "2023-2024 -> validation"
    )

    print(
        "2025-2026 -> FINAL TEST ONLY"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "1. HMM states are trained independently "
        "for each index."
    )

    print(
        "2. State labels use training data only."
    )

    print(
        "3. Future returns are evaluation-only."
    )

    print(
        "4. Validation periods are unseen during "
        "their corresponding training."
    )

    print(
        "5. 2025-2026 is NOT part of validation."
    )

    print(
        "6. Final test results must not be used "
        "to tune the model."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V3.1"
    )
    print(
        "PROPER WALK-FORWARD REGIME ENGINE"
    )
    print("=" * 70)

    df = load_data()

    df = build_features(
        df
    )

    all_results = {}

    for symbol in EQUITY_SYMBOLS:

        symbol_df = df[
            df["symbol"] == symbol
        ].copy()

        if len(symbol_df) < MIN_TRAIN_ROWS:

            print(
                f"Skipping {symbol}: "
                f"insufficient data."
            )

            continue

        all_results[symbol] = (
            process_symbol(
                symbol,
                symbol_df,
            )
        )

    print_summary(
        all_results
    )

    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V3.1 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()