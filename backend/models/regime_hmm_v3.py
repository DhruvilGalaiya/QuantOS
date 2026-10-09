"""
QuantOS - HMM V3
Walk-Forward Independent Equity Regime Detection

Purpose
-------
Research-grade regime detection using an independent
3-state Gaussian HMM for each equity index.

Design
------
1. Build historical-only regime features.
2. Train HMM using an expanding historical window.
3. Interpret states using TRAINING data only.
4. Freeze the state -> BULL/SIDE/BEAR mapping.
5. Apply the frozen model to unseen validation data.
6. Evaluate future returns only after prediction.
7. Measure regime stability and duration.
8. Keep the final test period untouched.

No future-return target is used during HMM fitting.
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

VALIDATION_YEARS = [
    (2019, 2020),
    (2021, 2022),
    (2023, 2024),
    (2025, 2026),
]

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

    print("Loading equity index data from PostgreSQL...")

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

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} rows."
    )

    return df


# ============================================================
# BUILD HISTORICAL FEATURES
# ============================================================

def build_features(df):

    print()
    print("=" * 70)
    print("BUILDING HISTORICAL REGIME FEATURES")
    print("=" * 70)

    df = df.copy()

    df["return_5d"] = (
        df.groupby("symbol")["close"]
        .pct_change(5)
    )

    df["return_20d"] = (
        df.groupby("symbol")["close"]
        .pct_change(20)
    )

    # Future returns are evaluation-only.
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
# PREPARE TRAINING FEATURES
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
# TRANSFORM NEW DATA
# ============================================================

def transform_features(
    df,
    imputer,
    scaler,
):

    X = df[
        HMM_FEATURES
    ]

    X = imputer.transform(X)

    X = scaler.transform(X)

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

                "observations":
                    len(state_df),
            }
        )

    stats = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Directional score
    #
    # ONLY training information is used here.
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

    stats = stats.sort_values(
        "directional_score"
    ).reset_index(drop=True)

    bear_state = int(
        stats.iloc[0]["state"]
    )

    bull_state = int(
        stats.iloc[-1]["state"]
    )

    middle = [
        int(x)
        for x in stats["state"]
        if int(x)
        not in [
            bear_state,
            bull_state,
        ]
    ]

    side_state = middle[0]

    labels = {
        bull_state: "BULL",
        side_state: "SIDE",
        bear_state: "BEAR",
    }

    return (
        labels,
        stats,
    )


# ============================================================
# STATE DURATIONS
# ============================================================

def calculate_durations(states):

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
            durations.mean(),

        "median_duration":
            np.median(durations),

        "transitions":
            transitions,

        "one_day_states_pct":
            (
                (durations == 1).mean()
                * 100
            ),
    }


# ============================================================
# VALIDATION METRICS
# ============================================================

def evaluate_validation(
    validation_df,
    states,
    labels,
):

    result = validation_df.copy()

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
            }
        )

    metrics = pd.DataFrame(
        rows
    )

    durations = (
        calculate_durations(
            states
        )
    )

    return (
        result,
        metrics,
        durations,
    )


# ============================================================
# SINGLE WALK-FORWARD PERIOD
# ============================================================

def run_period(
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
        f"{validation_start}-{validation_end}"
    )
    print("=" * 70)

    validation_start_date = pd.Timestamp(
        f"{validation_start}-01-01"
    )

    validation_end_date = pd.Timestamp(
        f"{validation_end}-12-31"
    )

    # --------------------------------------------------------
    # Expanding training window
    # --------------------------------------------------------

    train_df = symbol_df[
        symbol_df["timestamp"]
        < validation_start_date
    ].copy()

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
            f"Skipping period. "
            f"Only {len(train_df)} training rows."
        )

        return None

    if len(validation_df) == 0:

        print(
            "Skipping period. "
            "No validation observations."
        )

        return None

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
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
    # TRAIN HMM
    # --------------------------------------------------------

    model = train_hmm(
        X_train
    )

    # --------------------------------------------------------
    # TRAINING STATE DECODING
    # --------------------------------------------------------

    train_states = model.predict(
        X_train
    )

    (
        labels,
        train_stats,
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

    validation_states = (
        model.predict(
            X_validation
        )
    )

    (
        evaluated_df,
        metrics,
        durations,
    ) = evaluate_validation(
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
        f"{durations['mean_duration']:.2f} days"
    )

    print(
        f"Median duration: "
        f"{durations['median_duration']:.2f} days"
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
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "metrics": metrics,
        "durations": durations,
        "labels": labels,
        "train_stats": train_stats,
        "model": model,
        "validation_states": validation_states,
        "validation_df": evaluated_df,
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
        f"{symbol} | FINAL TEST 2025-2026"
    )
    print("=" * 70)

    train_end = pd.Timestamp(
        "2025-01-01"
    )

    test_df = symbol_df[
        symbol_df["timestamp"]
        >= train_end
    ].copy()

    train_df = symbol_df[
        symbol_df["timestamp"]
        < train_end
    ].copy()

    if len(test_df) == 0:

        print(
            "No final test data."
        )

        return None

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

    model = train_hmm(
        X_train
    )

    train_states = model.predict(
        X_train
    )

    (
        labels,
        train_stats,
    ) = determine_state_labels(
        train_df,
        train_states,
    )

    test_states = model.predict(
        X_test
    )

    (
        evaluated_df,
        metrics,
        durations,
    ) = evaluate_validation(
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
        f"1-day states: "
        f"{durations['one_day_states_pct']:.2f}%"
    )

    return {
        "metrics": metrics,
        "durations": durations,
        "labels": labels,
        "model": model,
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
        f"QUANTOS HMM V3: {symbol}"
    )
    print("#" * 70)

    results = []

    # --------------------------------------------------------
    # Walk-forward validation periods
    # --------------------------------------------------------

    for (
        validation_start,
        validation_end,
    ) in VALIDATION_YEARS:

        result = run_period(
            symbol,
            symbol_df,
            validation_start,
            validation_end,
        )

        if result is not None:

            results.append(
                result
            )

    # --------------------------------------------------------
    # Final untouched test
    # --------------------------------------------------------

    final_test = run_final_test(
        symbol,
        symbol_df,
    )

    return {
        "walk_forward": results,
        "final_test": final_test,
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def print_summary(
    all_results
):

    print()
    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V3 SUMMARY"
    )
    print("=" * 70)

    summary_rows = []

    for symbol, result in (
        all_results.items()
    ):

        for period_result in (
            result["walk_forward"]
        ):

            metrics = (
                period_result[
                    "metrics"
                ]
            )

            durations = (
                period_result[
                    "durations"
                ]
            )

            for _, row in (
                metrics.iterrows()
            ):

                summary_rows.append(
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

                        "mean_duration":
                            durations[
                                "mean_duration"
                            ],
                    }
                )

    if summary_rows:

        summary = pd.DataFrame(
            summary_rows
        )

        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print(
        "HMM V3 WALK-FORWARD VALIDATION COMPLETE"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "State labels were determined using "
        "training data only."
    )

    print(
        "Validation periods were never used "
        "to fit their corresponding HMM."
    )

    print(
        "Future returns are evaluation-only."
    )

    print(
        "The final 2025-2026 period is kept "
        "as the final diagnostic test."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V3"
    )
    print(
        "WALK-FORWARD INDEPENDENT REGIME ENGINE"
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
        "QUANTOS HMM V3 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()