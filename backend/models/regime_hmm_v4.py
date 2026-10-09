"""
QuantOS - HMM V4
Probabilistic Market Regime Engine

V4 objectives:

1. Independent HMM for each index.
2. Four latent market regimes.
3. No hard-coded BULL/SIDE/BEAR interpretation.
4. Regimes characterized using training data only.
5. Walk-forward validation:
       2019-2020
       2021-2022
       2023-2024
6. Completely untouched final test:
       2025-2026
7. Regime persistence analysis.
8. Transition probability analysis.
9. Trend / volatility / momentum characterization.
10. Future-return analysis ONLY for evaluation.
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

N_STATES = 4

N_ITER = 500

RANDOM_STATE = 42

MIN_TRAIN_ROWS = 750


VALIDATION_PERIODS = [
    (2019, 2020),
    (2021, 2022),
    (2023, 2024),
]


FINAL_TEST_START = 2025
FINAL_TEST_END = 2026


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
# FEATURES
# ============================================================

HMM_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "atr_14",
    "close_vs_sma20",
    "close_vs_sma50",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING INDEX DATA")
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
            atr_14,
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
        f"Loaded rows: {len(df):,}"
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
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    print()
    print("=" * 70)
    print("BUILDING HMM FEATURES")
    print("=" * 70)

    df = df.copy()

    grouped = df.groupby("symbol")["close"]

    df["return_5d"] = grouped.pct_change(5)

    df["return_20d"] = grouped.pct_change(20)

    # --------------------------------------------------------
    # Evaluation-only future returns
    # --------------------------------------------------------

    df["forward_1d"] = (
        grouped.shift(-1)
        / df["close"]
        - 1
    )

    df["forward_5d"] = (
        grouped.shift(-5)
        / df["close"]
        - 1
    )

    df["forward_20d"] = (
        grouped.shift(-20)
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

    X = train_df[
        HMM_FEATURES
    ]

    X = imputer.fit_transform(X)

    X = scaler.fit_transform(X)

    return (
        imputer,
        scaler,
        X,
    )


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
# HMM TRAINING
# ============================================================

def train_hmm(
    X,
):

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
# STATE CHARACTERIZATION
# ============================================================

def characterize_states(
    train_df,
    states,
):

    temp = train_df.copy()

    temp["hmm_state"] = states

    rows = []

    for state in range(N_STATES):

        state_df = temp[
            temp["hmm_state"] == state
        ]

        if len(state_df) == 0:
            continue

        rows.append(
            {
                "state": state,

                "observations":
                    len(state_df),

                "return_1d":
                    state_df[
                        "return_1d"
                    ].mean(),

                "return_5d":
                    state_df[
                        "return_5d"
                    ].mean(),

                "return_20d":
                    state_df[
                        "return_20d"
                    ].mean(),

                "volatility":
                    state_df[
                        "volatility_20"
                    ].mean(),

                "rsi":
                    state_df[
                        "rsi_14"
                    ].mean(),

                "volume_ratio":
                    state_df[
                        "volume_ratio"
                    ].mean(),

                "atr":
                    state_df[
                        "atr_14"
                    ].mean(),

                "close_vs_sma20":
                    state_df[
                        "close_vs_sma20"
                    ].mean(),

                "close_vs_sma50":
                    state_df[
                        "close_vs_sma50"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DESCRIPTIVE REGIME CLASSIFICATION
# ============================================================

def classify_regime(
    row,
    all_states,
):

    """
    Descriptive classification only.

    These labels are NOT directional predictions.

    The classification is based entirely on relative
    characteristics of the training states.
    """

    trend_score = (
        0.50 * row["return_20d"]
        +
        0.30 * row["close_vs_sma50"]
        +
        0.20 * row["close_vs_sma20"]
    )

    momentum_score = (
        0.60 * row["return_5d"]
        +
        0.40 * (
            row["rsi"] - 50
        )
    )

    volatility_score = (
        0.60 * row["volatility"]
        +
        0.40 * row["atr"]
    )

    # --------------------------------------------------------
    # Relative ranks
    # --------------------------------------------------------

    trend_rank = (
        all_states["trend_score"]
        .rank(
            pct=True
        )
    )

    volatility_rank = (
        all_states["volatility_score"]
        .rank(
            pct=True
        )
    )

    momentum_rank = (
        all_states["momentum_score"]
        .rank(
            pct=True
        )
    )

    state = row["state"]

    t = trend_rank.loc[
        all_states["state"] == state
    ].iloc[0]

    v = volatility_rank.loc[
        all_states["state"] == state
    ].iloc[0]

    m = momentum_rank.loc[
        all_states["state"] == state
    ].iloc[0]

    # --------------------------------------------------------
    # Descriptive regime
    # --------------------------------------------------------

    if v >= 0.75 and t < 0.50:

        label = "HIGH_VOLATILITY"

    elif v >= 0.75 and t >= 0.50:

        label = "HIGH_VOL_TREND"

    elif t >= 0.75 and m >= 0.50:

        label = "TRENDING_BULLISH"

    elif t <= 0.25 and m <= 0.50:

        label = "TRENDING_BEARISH"

    elif v <= 0.35:

        label = "LOW_VOLATILITY"

    else:

        label = "TRANSITION"

    return label


# ============================================================
# ENRICH STATE CHARACTERISTICS
# ============================================================

def build_regime_profiles(
    state_stats,
):

    stats = state_stats.copy()

    stats["trend_score"] = (
        0.50 * stats["return_20d"]
        +
        0.30 * stats["close_vs_sma50"]
        +
        0.20 * stats["close_vs_sma20"]
    )

    stats["momentum_score"] = (
        0.60 * stats["return_5d"]
        +
        0.40 * (
            stats["rsi"] - 50
        )
    )

    stats["volatility_score"] = (
        0.60 * stats["volatility"]
        +
        0.40 * stats["atr"]
    )

    labels = []

    for _, row in stats.iterrows():

        labels.append(
            classify_regime(
                row,
                stats,
            )
        )

    stats["regime"] = labels

    return stats


# ============================================================
# TRANSITION MATRIX
# ============================================================

def empirical_transition_matrix(
    states,
):

    matrix = np.zeros(
        (
            N_STATES,
            N_STATES,
        ),
        dtype=float,
    )

    for previous, current in zip(
        states[:-1],
        states[1:],
    ):

        matrix[
            int(previous),
            int(current)
        ] += 1

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    return matrix


# ============================================================
# PERSISTENCE
# ============================================================

def calculate_persistence(
    states,
):

    if len(states) == 0:

        return pd.DataFrame()

    durations = []

    current = states[0]

    duration = 1

    for state in states[1:]:

        if state == current:

            duration += 1

        else:

            durations.append(
                (
                    current,
                    duration,
                )
            )

            current = state

            duration = 1

    durations.append(
        (
            current,
            duration,
        )
    )

    rows = []

    for state in range(N_STATES):

        state_durations = [
            d
            for s, d in durations
            if s == state
        ]

        if not state_durations:

            continue

        rows.append(
            {
                "state": state,

                "episodes":
                    len(state_durations),

                "mean_duration":
                    np.mean(
                        state_durations
                    ),

                "median_duration":
                    np.median(
                        state_durations
                    ),

                "max_duration":
                    np.max(
                        state_durations
                    ),

                "one_day_pct":
                    np.mean(
                        np.array(
                            state_durations
                        ) == 1
                    ) * 100,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REGIME PERFORMANCE
# ============================================================

def evaluate_regimes(
    df,
    states,
    regime_map,
):

    result = df.copy()

    result["hmm_state"] = states

    result["regime"] = (
        result["hmm_state"]
        .map(regime_map)
    )

    rows = []

    for state in range(N_STATES):

        state_df = result[
            result["hmm_state"] == state
        ]

        if len(state_df) == 0:

            continue

        rows.append(
            {
                "state": state,

                "regime":
                    regime_map[state],

                "observations":
                    len(state_df),

                "forward_1d":
                    state_df[
                        "forward_1d"
                    ].mean(),

                "forward_5d":
                    state_df[
                        "forward_5d"
                    ].mean(),

                "forward_20d":
                    state_df[
                        "forward_20d"
                    ].mean(),

                "win_rate_5d":
                    (
                        state_df[
                            "forward_5d"
                        ] > 0
                    ).mean(),

                "volatility":
                    state_df[
                        "volatility_20"
                    ].mean(),
            }
        )

    return (
        result,
        pd.DataFrame(rows),
    )


# ============================================================
# SINGLE PERIOD
# ============================================================

def run_period(
    symbol,
    symbol_df,
    train_end,
    evaluation_start,
    evaluation_end,
    title,
):

    print()
    print("=" * 70)

    print(
        f"{symbol} | {title}"
    )

    print("=" * 70)

    train_end = pd.Timestamp(
        f"{train_end}-12-31"
    )

    evaluation_start = pd.Timestamp(
        f"{evaluation_start}-01-01"
    )

    evaluation_end = pd.Timestamp(
        f"{evaluation_end}-12-31"
    )

    train_df = symbol_df[
        symbol_df["timestamp"]
        <= train_end
    ].copy()

    evaluation_df = symbol_df[
        (
            symbol_df["timestamp"]
            >= evaluation_start
        )
        &
        (
            symbol_df["timestamp"]
            <= evaluation_end
        )
    ].copy()

    if len(train_df) < MIN_TRAIN_ROWS:

        print(
            f"SKIPPED: only "
            f"{len(train_df)} training rows."
        )

        return None

    if len(evaluation_df) == 0:

        print(
            "SKIPPED: no evaluation rows."
        )

        return None

    print(
        f"Training rows:   "
        f"{len(train_df):,}"
    )

    print(
        f"Evaluation rows: "
        f"{len(evaluation_df):,}"
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    (
        imputer,
        scaler,
        X_train,
    ) = fit_preprocessor(
        train_df
    )

    X_eval = transform_features(
        evaluation_df,
        imputer,
        scaler,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_hmm(
        X_train
    )

    print()
    print(
        "HMM CONVERGENCE"
    )

    print(
        f"Converged: "
        f"{model.monitor_.converged}"
    )

    print(
        f"Iterations: "
        f"{model.monitor_.iter}"
    )

    # --------------------------------------------------------
    # TRAIN STATE DECODING
    # --------------------------------------------------------

    train_states = model.predict(
        X_train
    )

    state_stats = characterize_states(
        train_df,
        train_states,
    )

    profiles = build_regime_profiles(
        state_stats
    )

    regime_map = {
        int(row["state"]):
        row["regime"]
        for _, row in profiles.iterrows()
    }

    # --------------------------------------------------------
    # DISPLAY TRAINING REGIMES
    # --------------------------------------------------------

    print()
    print(
        "LEARNED REGIME PROFILES"
    )

    display_columns = [
        "state",
        "regime",
        "observations",
        "return_5d",
        "return_20d",
        "volatility",
        "rsi",
        "volume_ratio",
        "trend_score",
        "momentum_score",
    ]

    print(
        profiles[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    transition = (
        empirical_transition_matrix(
            train_states
        )
    )

    transition_df = pd.DataFrame(
        transition,
        index=[
            f"STATE_{i}"
            for i in range(N_STATES)
        ],
        columns=[
            f"STATE_{i}"
            for i in range(N_STATES)
        ],
    )

    print()
    print(
        "EMPIRICAL TRAINING TRANSITION MATRIX"
    )

    print(
        transition_df.to_string(
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    persistence = calculate_persistence(
        train_states
    )

    print()
    print(
        "REGIME PERSISTENCE"
    )

    print(
        persistence.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    eval_states = model.predict(
        X_eval
    )

    (
        evaluated_df,
        performance,
    ) = evaluate_regimes(
        evaluation_df,
        eval_states,
        regime_map,
    )

    print()
    print(
        "OUT-OF-SAMPLE REGIME PERFORMANCE"
    )

    print(
        performance.to_string(
            index=False
        )
    )

    return {
        "performance": performance,
        "profiles": profiles,
        "transition": transition_df,
        "persistence": persistence,
        "data": evaluated_df,
        "regime_map": regime_map,
    }


# ============================================================
# FINAL TEST
# ============================================================

def run_final_test(
    symbol,
    symbol_df,
):

    return run_period(
        symbol=symbol,
        symbol_df=symbol_df,
        train_end=2024,
        evaluation_start=2025,
        evaluation_end=2026,
        title="FINAL TEST 2025-2026",
    )


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
        f"QUANTOS HMM V4: {symbol}"
    )

    print("#" * 70)

    results = {}

    # --------------------------------------------------------
    # WALK-FORWARD VALIDATION
    # --------------------------------------------------------

    validation_schedule = [
        (
            2018,
            2019,
            2020,
        ),
        (
            2020,
            2021,
            2022,
        ),
        (
            2022,
            2023,
            2024,
        ),
    ]

    for (
        train_end,
        evaluation_start,
        evaluation_end,
    ) in validation_schedule:

        key = (
            f"{evaluation_start}-"
            f"{evaluation_end}"
        )

        results[key] = run_period(
            symbol=symbol,
            symbol_df=symbol_df,
            train_end=train_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            title=(
                f"VALIDATION "
                f"{evaluation_start}-"
                f"{evaluation_end}"
            ),
        )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    results["FINAL_TEST"] = (
        run_final_test(
            symbol,
            symbol_df,
        )
    )

    return results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    all_results,
):

    print()
    print()
    print("=" * 70)

    print(
        "QUANTOS HMM V4 SUMMARY"
    )

    print("=" * 70)

    rows = []

    for symbol, symbol_results in (
        all_results.items()
    ):

        for period, result in (
            symbol_results.items()
        ):

            if result is None:
                continue

            performance = result[
                "performance"
            ]

            for _, row in (
                performance.iterrows()
            ):

                rows.append(
                    {
                        "symbol":
                            symbol,

                        "period":
                            period,

                        "state":
                            row["state"],

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
        "V4 VALIDATION DESIGN"
    )

    print("=" * 70)

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
        "HMM states are latent regimes."
    )

    print(
        "Regime labels are descriptive,"
        " not direct price predictions."
    )

    print(
        "State characterization uses"
        " training data only."
    )

    print(
        "Future returns are evaluation-only."
    )

    print(
        "Final test results must not be"
        " used for model tuning."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V4"
    )
    print(
        "PROBABILISTIC MARKET REGIME ENGINE"
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

    print_final_summary(
        all_results
    )

    print()
    print("=" * 70)

    print(
        "QUANTOS HMM V4 COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()