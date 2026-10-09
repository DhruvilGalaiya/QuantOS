"""
QuantOS - Hidden Markov Model Regime Detection

Purpose:
    Discover latent market regimes from market features.

The HMM discovers 3 statistical states:
    State 0
    State 1
    State 2

We then analyze each state's:
    - return
    - volatility
    - RSI
    - trend characteristics
    - forward returns
    - average duration

and map the statistical states to:
    BULL / SIDE / BEAR

IMPORTANT:
    The HMM is trained only on dataset_train.
    Validation/test data are NOT used to fit the model.
"""

from pathlib import Path
import sys

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

FEATURE_COLUMNS = [
    "return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "atr_14",
    "close_vs_sma20",
    "close_vs_sma50",
]

TARGET_COLUMNS = [
    "target_return_1d",
    "target_return_5d",
]

REGIME_NAMES = {
    0: "STATE_0",
    1: "STATE_1",
    2: "STATE_2",
}


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load historical training observations from PostgreSQL.

    We intentionally use dataset_train only for HMM fitting.
    """

    print("Loading training data from PostgreSQL...")

    query = text(
        """
        SELECT
            d.timestamp,
            d.symbol,

            f.return_1d,
            f.volatility_20,
            f.rsi_14,
            f.volume_ratio,
            f.atr_14,
            f.close_vs_sma20,
            f.close_vs_sma50,

            d.target_return_1d,
            d.target_return_5d,

            d.dataset_split

        FROM dataset_train d

        INNER JOIN market_features f
            ON d.timestamp = f.timestamp
            AND d.symbol = f.symbol

        ORDER BY
            d.symbol,
            d.timestamp
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    print(f"Loaded {len(df):,} training rows.")

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df):
    print()
    print("=" * 70)
    print("HMM DATA VALIDATION")
    print("=" * 70)

    print(f"Rows:       {len(df):,}")
    print(f"Symbols:    {df['symbol'].nunique()}")
    print(
        f"Date range: {df['timestamp'].min()} "
        f"-> {df['timestamp'].max()}"
    )

    print()
    print("Feature null counts:")
    print(df[FEATURE_COLUMNS].isnull().sum())

    print()

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing HMM features: {missing_columns}"
        )

    if len(df) == 0:
        raise RuntimeError("HMM dataset is empty.")

    print("Validation complete.")


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Impute and standardize the HMM observation features.

    The imputer and scaler are fitted ONLY on training data.
    """

    X = df[FEATURE_COLUMNS].copy()

    print()
    print("=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    print("Features:")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X)

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imputed)

    print()
    print("Imputation: median")
    print("Scaling:    StandardScaler")
    print("Feature preparation complete.")

    return X_scaled, imputer, scaler


# ============================================================
# TRAIN HMM
# ============================================================

def train_hmm(X, lengths):
    """
    Train a 3-state Gaussian HMM.

    lengths tells the HMM where each symbol's sequence ends.
    This prevents the model from treating the last observation
    of one asset and the first observation of another asset
    as a real market transition.
    """

    print()
    print("=" * 70)
    print("TRAINING HIDDEN MARKOV MODEL")
    print("=" * 70)

    print(f"Number of states: {N_STATES}")
    print("Model:            GaussianHMM")
    print("Covariance:       diagonal")
    print("Iterations:       300")
    print(f"Random state:     {RANDOM_STATE}")

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=300,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    model.fit(
        X,
        lengths=lengths,
    )

    print()
    print("HMM training complete.")
    print(f"Converged: {model.monitor_.converged}")
    print(f"Iterations: {model.monitor_.iter}")

    return model


# ============================================================
# DECODE STATES
# ============================================================

def decode_states(model, X, lengths):
    """
    Determine the most likely hidden state for every observation.
    """

    print()
    print("Decoding hidden states...")

    states = model.predict(
        X,
        lengths=lengths,
    )

    print("State decoding complete.")

    return states


# ============================================================
# STATE STATISTICS
# ============================================================

def calculate_state_statistics(df):
    """
    Calculate descriptive statistics for each discovered state.
    """

    print()
    print("=" * 70)
    print("HMM STATE CHARACTERISTICS")
    print("=" * 70)

    results = []

    for state in sorted(df["hmm_state"].unique()):

        state_df = df[
            df["hmm_state"] == state
        ]

        mean_return = state_df["return_1d"].mean()

        mean_volatility = state_df["volatility_20"].mean()

        mean_rsi = state_df["rsi_14"].mean()

        mean_volume_ratio = state_df["volume_ratio"].mean()

        mean_close_vs_sma20 = (
            state_df["close_vs_sma20"].mean()
        )

        mean_close_vs_sma50 = (
            state_df["close_vs_sma50"].mean()
        )

        mean_forward_1d = (
            state_df["target_return_1d"].mean()
        )

        mean_forward_5d = (
            state_df["target_return_5d"].mean()
        )

        results.append(
            {
                "state": state,
                "observations": len(state_df),
                "mean_return_1d": mean_return,
                "mean_volatility_20": mean_volatility,
                "mean_rsi_14": mean_rsi,
                "mean_volume_ratio": mean_volume_ratio,
                "mean_close_vs_sma20": mean_close_vs_sma20,
                "mean_close_vs_sma50": mean_close_vs_sma50,
                "mean_forward_return_1d": mean_forward_1d,
                "mean_forward_return_5d": mean_forward_5d,
            }
        )

    stats = pd.DataFrame(results)

    print()

    for _, row in stats.iterrows():

        print("-" * 70)

        print(
            f"STATE {int(row['state'])}"
        )

        print(
            f"Observations:          "
            f"{int(row['observations']):,}"
        )

        print(
            f"Mean 1D return:        "
            f"{row['mean_return_1d']:.6f}"
        )

        print(
            f"Mean volatility:        "
            f"{row['mean_volatility_20']:.6f}"
        )

        print(
            f"Mean RSI:               "
            f"{row['mean_rsi_14']:.2f}"
        )

        print(
            f"Mean volume ratio:      "
            f"{row['mean_volume_ratio']:.4f}"
        )

        print(
            f"Close vs SMA20:         "
            f"{row['mean_close_vs_sma20']:.6f}"
        )

        print(
            f"Close vs SMA50:         "
            f"{row['mean_close_vs_sma50']:.6f}"
        )

        print(
            f"Forward 1D return:      "
            f"{row['mean_forward_return_1d']:.6f}"
        )

        print(
            f"Forward 5D return:      "
            f"{row['mean_forward_return_5d']:.6f}"
        )

    return stats


# ============================================================
# LABEL STATES
# ============================================================

def label_states(stats):
    """
    Map HMM states to BULL / SIDE / BEAR.

    We primarily rank states using their forward 5D return.
    The state with the highest forward return becomes BULL.
    The lowest becomes BEAR.
    The remaining state becomes SIDE.

    NOTE:
        These forward returns are used ONLY for interpreting
        the discovered states after HMM fitting. They are not
        included in the HMM input features.
    """

    sorted_stats = stats.sort_values(
        "mean_forward_return_5d"
    )

    bear_state = int(
        sorted_stats.iloc[0]["state"]
    )

    bull_state = int(
        sorted_stats.iloc[-1]["state"]
    )

    side_state = int(
        sorted_stats.iloc[1]["state"]
    )

    labels = {
        bull_state: "BULL",
        side_state: "SIDE",
        bear_state: "BEAR",
    }

    print()
    print("=" * 70)
    print("REGIME INTERPRETATION")
    print("=" * 70)

    for state in sorted(labels):
        print(
            f"State {state} -> {labels[state]}"
        )

    return labels


# ============================================================
# STATE DISTRIBUTION
# ============================================================

def print_state_distribution(df, labels):
    print()
    print("=" * 70)
    print("REGIME DISTRIBUTION")
    print("=" * 70)

    counts = (
        df["hmm_state"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    for state, count in counts.items():

        percentage = (
            count / total * 100
        )

        print(
            f"{labels[state]:5s} "
            f"{count:8,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QUANTOS HIDDEN MARKOV MODEL REGIME DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_data(df)

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, imputer, scaler = prepare_features(df)

    # --------------------------------------------------------
    # BUILD SEQUENCE LENGTHS
    # --------------------------------------------------------

    lengths = (
        df.groupby("symbol")
        .size()
        .tolist()
    )

    print()
    print("HMM sequences:")
    print(f"Number of symbol sequences: {len(lengths)}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_hmm(
        X,
        lengths,
    )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    states = decode_states(
        model,
        X,
        lengths,
    )

    df["hmm_state"] = states

    # --------------------------------------------------------
    # STATE STATISTICS
    # --------------------------------------------------------

    stats = calculate_state_statistics(
        df
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    labels = label_states(
        stats
    )

    df["regime"] = df["hmm_state"].map(
        labels
    )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    print_state_distribution(
        df,
        labels,
    )

    # --------------------------------------------------------
    # MODEL TRANSITION MATRIX
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HMM LEARNED TRANSITION MATRIX")
    print("=" * 70)

    transition_matrix = model.transmat_

    transition_df = pd.DataFrame(
        transition_matrix,
        index=[
            REGIME_NAMES[i]
            for i in range(N_STATES)
        ],
        columns=[
            REGIME_NAMES[i]
            for i in range(N_STATES)
        ],
    )

    print(
        transition_df.round(4)
    )

    # --------------------------------------------------------
    # CURRENT STATE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LATEST DISCOVERED REGIME")
    print("=" * 70)

    latest_rows = (
        df.sort_values(
            ["symbol", "timestamp"]
        )
        .groupby("symbol")
        .tail(1)
    )

    for _, row in latest_rows.iterrows():

        print(
            f"{row['symbol']:10s} "
            f"{row['timestamp']}  "
            f"State={int(row['hmm_state'])}  "
            f"Regime={row['regime']}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HMM REGIME DISCOVERY COMPLETE")
    print("=" * 70)

    print()
    print("Discovered states:")
    print(
        stats[
            [
                "state",
                "observations",
                "mean_return_1d",
                "mean_forward_return_5d",
                "mean_volatility_20",
            ]
        ].to_string(index=False)
    )

    print()
    print("IMPORTANT:")
    print(
        "The transition matrix printed above is the HMM's "
        "learned state transition matrix."
    )

    print(
        "We will build the empirical regime transition "
        "matrix separately after validating the regimes."
    )


if __name__ == "__main__":
    main()