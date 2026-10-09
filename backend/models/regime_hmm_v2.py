"""
QuantOS - HMM V2: Independent Equity Index Regime Detection

Purpose:
    Discover Bull / Side / Bear-like market regimes independently
    for each equity index.

Design:
    - One HMM per symbol
    - Three hidden states per symbol
    - Historical features only
    - No future information used during HMM fitting
    - Future returns are used ONLY after fitting to validate
      and interpret the discovered states

Equity indices:
    NIFTY50
    NIFTYBANK
    NIFTYIT
    NIFTYAUTO
    NIFTYPHARMA
    NIFTYNEXT50
    NASDAQ100
    SP500
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
# PATH SETUP
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.database import engine


# ============================================================
# CONFIGURATION
# ============================================================

N_STATES = 3

RANDOM_STATE = 42

N_ITER = 300

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
# HMM OBSERVATION FEATURES
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
# LOAD DATA
# ============================================================

def load_equity_data():

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

    print(
        f"Loaded {len(df):,} rows."
    )

    return df


# ============================================================
# BUILD ADDITIONAL HISTORICAL FEATURES
# ============================================================

def build_regime_features(df):

    print()
    print("=" * 70)
    print("BUILDING HMM REGIME FEATURES")
    print("=" * 70)

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Calculate multi-horizon historical returns.
    #
    # These use only prices up to the current observation.
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
    # Basic validation
    # --------------------------------------------------------

    print()
    print("Features used by HMM:")

    for feature in HMM_FEATURES:
        print(f"  - {feature}")

    print()
    print("Feature null counts:")

    print(
        df[HMM_FEATURES]
        .isnull()
        .sum()
    )

    return df


# ============================================================
# PREPARE ONE SYMBOL
# ============================================================

def prepare_symbol_data(symbol_df):

    symbol_df = (
        symbol_df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # We require a target-independent observation matrix.
    #
    # Rows with missing observations are handled using median
    # imputation fitted only on this symbol's data.

    X_raw = symbol_df[HMM_FEATURES]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(
        X_raw
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_imputed
    )

    return (
        symbol_df,
        X_scaled,
        imputer,
        scaler,
    )


# ============================================================
# TRAIN ONE HMM
# ============================================================

def train_symbol_hmm(
    symbol,
    X,
):

    print()
    print("-" * 70)
    print(
        f"TRAINING HMM: {symbol}"
    )
    print("-" * 70)

    print(
        f"Observations: {len(X):,}"
    )

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    model.fit(X)

    print(
        f"Converged: {model.monitor_.converged}"
    )

    print(
        f"Iterations: {model.monitor_.iter}"
    )

    return model


# ============================================================
# DECODE STATES
# ============================================================

def decode_states(
    model,
    X,
):

    states = model.predict(X)

    probabilities = model.predict_proba(X)

    return (
        states,
        probabilities,
    )


# ============================================================
# STATE CHARACTERISTICS
# ============================================================

def calculate_state_statistics(
    symbol_df,
):

    rows = []

    for state in sorted(
        symbol_df["hmm_state"].unique()
    ):

        state_df = symbol_df[
            symbol_df["hmm_state"] == state
        ]

        rows.append(
            {
                "state": state,

                "observations":
                    len(state_df),

                "mean_return_1d":
                    state_df[
                        "return_1d"
                    ].mean(),

                "mean_return_5d":
                    state_df[
                        "return_5d"
                    ].mean(),

                "mean_return_20d":
                    state_df[
                        "return_20d"
                    ].mean(),

                "mean_volatility":
                    state_df[
                        "volatility_20"
                    ].mean(),

                "mean_rsi":
                    state_df[
                        "rsi_14"
                    ].mean(),

                "mean_volume_ratio":
                    state_df[
                        "volume_ratio"
                    ].mean(),

                "mean_close_vs_sma20":
                    state_df[
                        "close_vs_sma20"
                    ].mean(),

                "mean_close_vs_sma50":
                    state_df[
                        "close_vs_sma50"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STATE INTERPRETATION
# ============================================================

def interpret_states(stats):

    """
    Interpret discovered states using their characteristics.

    IMPORTANT:
        This does not use future returns.

    We primarily use the combination of:
        - 20D return
        - trend relative to SMA50
        - volatility

    The interpretation is diagnostic and will later be
    validated against forward returns.
    """

    stats = stats.copy()

    # --------------------------------------------------------
    # Create a directional score.
    #
    # Positive:
    #   positive long-term return
    #   price above SMA50
    #
    # Negative:
    #   negative long-term return
    #   price below SMA50
    #
    # We standardize within the three states so that the
    # relative ranking matters rather than raw units.
    # --------------------------------------------------------

    def rank_score(series):

        return (
            series.rank(
                method="average"
            )
            - 1
        ) / max(
            len(series) - 1,
            1,
        )

    return_score = rank_score(
        stats["mean_return_20d"]
    )

    trend_score = rank_score(
        stats["mean_close_vs_sma50"]
    )

    directional_score = (
        0.60 * return_score
        +
        0.40 * trend_score
    )

    stats["directional_score"] = (
        directional_score
    )

    stats = stats.sort_values(
        "directional_score"
    ).reset_index(drop=True)

    # Lowest directional state = bearish-like
    bear_state = int(
        stats.iloc[0]["state"]
    )

    # Highest directional state = bullish-like
    bull_state = int(
        stats.iloc[-1]["state"]
    )

    remaining_states = [
        int(state)
        for state in stats["state"]
        if int(state)
        not in [
            bear_state,
            bull_state,
        ]
    ]

    side_state = remaining_states[0]

    labels = {
        bear_state: "BEAR",
        side_state: "SIDE",
        bull_state: "BULL",
    }

    return (
        stats,
        labels,
    )


# ============================================================
# FORWARD VALIDATION
# ============================================================

def calculate_forward_validation(
    symbol_df,
    labels,
):

    print()
    print(
        "FORWARD PERFORMANCE VALIDATION"
    )

    validation_rows = []

    for state in sorted(
        symbol_df["hmm_state"].unique()
    ):

        state_df = symbol_df[
            symbol_df["hmm_state"] == state
        ]

        # ----------------------------------------------------
        # Forward returns are calculated here ONLY for
        # evaluating the discovered state.
        # They were NOT used as HMM observations.
        # ----------------------------------------------------

        forward_1d = (
            state_df["close"]
            .shift(-1)
            /
            state_df["close"]
            - 1
        )

        forward_5d = (
            state_df["close"]
            .shift(-5)
            /
            state_df["close"]
            - 1
        )

        forward_20d = (
            state_df["close"]
            .shift(-20)
            /
            state_df["close"]
            - 1
        )

        validation_rows.append(
            {
                "state": state,

                "regime":
                    labels[state],

                "observations":
                    len(state_df),

                "forward_1d":
                    forward_1d.mean(),

                "forward_5d":
                    forward_5d.mean(),

                "forward_20d":
                    forward_20d.mean(),

                "forward_1d_win_rate":
                    (
                        forward_1d > 0
                    ).mean(),

                "forward_5d_win_rate":
                    (
                        forward_5d > 0
                    ).mean(),
            }
        )

    return pd.DataFrame(
        validation_rows
    )


# ============================================================
# PRINT TRANSITION MATRIX
# ============================================================

def print_hmm_transition_matrix(
    model,
    labels,
):

    print()
    print(
        "HMM TRANSITION MATRIX"
    )
    print("-" * 70)

    ordered_states = [
        state
        for state in [
            next(
                s for s, label
                in labels.items()
                if label == "BULL"
            ),
            next(
                s for s, label
                in labels.items()
                if label == "SIDE"
            ),
            next(
                s for s, label
                in labels.items()
                if label == "BEAR"
            ),
        ]
    ]

    matrix = model.transmat_

    matrix_df = pd.DataFrame(
        matrix[
            np.ix_(
                ordered_states,
                ordered_states,
            )
        ],
        index=[
            labels[state]
            for state in ordered_states
        ],
        columns=[
            labels[state]
            for state in ordered_states
        ],
    )

    print(
        matrix_df
        .round(4)
    )

    return matrix_df


# ============================================================
# PROCESS ONE SYMBOL
# ============================================================

def process_symbol(symbol_df):

    symbol = (
        symbol_df["symbol"]
        .iloc[0]
    )

    print()
    print()
    print("=" * 70)
    print(
        f"QUANTOS HMM V2: {symbol}"
    )
    print("=" * 70)

    print(
        f"Rows: {len(symbol_df):,}"
    )

    print(
        f"Date range: "
        f"{symbol_df['timestamp'].min()} "
        f"-> "
        f"{symbol_df['timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        symbol_df,
        X,
        imputer,
        scaler,
    ) = prepare_symbol_data(
        symbol_df
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_symbol_hmm(
        symbol,
        X,
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    states, probabilities = (
        decode_states(
            model,
            X,
        )
    )

    symbol_df["hmm_state"] = states

    # --------------------------------------------------------
    # State probabilities
    # --------------------------------------------------------

    for state in range(N_STATES):

        symbol_df[
            f"hmm_prob_state_{state}"
        ] = probabilities[:, state]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_state_statistics(
        symbol_df
    )

    print()
    print(
        "DISCOVERED STATE CHARACTERISTICS"
    )
    print("-" * 70)

    print(
        stats.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Interpret
    # --------------------------------------------------------

    stats, labels = interpret_states(
        stats
    )

    print()
    print(
        "REGIME INTERPRETATION"
    )
    print("-" * 70)

    for state in sorted(labels):

        print(
            f"State {state} -> "
            f"{labels[state]}"
        )

    symbol_df["regime"] = (
        symbol_df["hmm_state"]
        .map(labels)
    )

    # --------------------------------------------------------
    # Forward validation
    # --------------------------------------------------------

    validation = (
        calculate_forward_validation(
            symbol_df,
            labels,
        )
    )

    print()
    print(
        validation.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------

    transition_matrix = (
        print_hmm_transition_matrix(
            model,
            labels,
        )
    )

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    latest = (
        symbol_df
        .sort_values("timestamp")
        .iloc[-1]
    )

    print()
    print(
        "LATEST REGIME"
    )
    print("-" * 70)

    print(
        f"Date:       {latest['timestamp']}"
    )

    print(
        f"State:      "
        f"{int(latest['hmm_state'])}"
    )

    print(
        f"Regime:     "
        f"{latest['regime']}"
    )

    print()
    print(
        "Current state probabilities:"
    )

    for state in range(N_STATES):

        print(
            f"State {state}: "
            f"{latest[f'hmm_prob_state_{state}']:.4f}"
        )

    return {
        "symbol": symbol,
        "data": symbol_df,
        "stats": stats,
        "validation": validation,
        "labels": labels,
        "transition_matrix": transition_matrix,
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V2 - "
        "INDEPENDENT EQUITY INDEX REGIMES"
    )
    print("=" * 70)

    print()
    print(
        "Indices:"
    )

    for symbol in EQUITY_SYMBOLS:
        print(
            f"  - {symbol}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_equity_data()

    if len(df) == 0:
        raise RuntimeError(
            "No equity index data found."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = build_regime_features(
        df
    )

    # --------------------------------------------------------
    # Process independently
    # --------------------------------------------------------

    results = {}

    for symbol in EQUITY_SYMBOLS:

        symbol_df = df[
            df["symbol"] == symbol
        ].copy()

        if len(symbol_df) < 500:

            print()
            print(
                f"Skipping {symbol}: "
                f"only {len(symbol_df)} rows."
            )

            continue

        results[symbol] = (
            process_symbol(
                symbol_df
            )
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print(
        "QUANTOS HMM V2 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "LATEST REGIME SUMMARY"
    )

    print("-" * 70)

    for symbol, result in results.items():

        latest = (
            result["data"]
            .sort_values("timestamp")
            .iloc[-1]
        )

        print(
            f"{symbol:12s} "
            f"{latest['regime']:5s} "
            f"State={int(latest['hmm_state'])}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "HMM states were discovered independently "
        "for each index."
    )

    print(
        "Forward returns were used only to validate "
        "the discovered states."
    )

    print(
        "The transition matrices will be used for "
        "the next regime-probability layer."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()