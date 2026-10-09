from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "regime"
    / "daily"
    / "nifty_50.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "regime"
    / "portfolio_hmm_signals.parquet"
)


STATE_NAMES = [
    "BULL",
    "SIDE",
    "BEAR",
]

N_STATES = 3

N_RESTARTS = 12

RANDOM_SEED = 42

# Match the portfolio walk-forward training horizon.
TRAIN_WINDOW = 504

# Match portfolio rebalance frequency.
REBALANCE_FREQUENCY = 21

# HMM fitting parameters.
MAX_ITER = 300

TOLERANCE = 1e-4


# ================================================================
# DATA LOADING
# ================================================================

def load_nifty_data() -> pd.DataFrame:

    df = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "timestamp",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

    rs = (
        average_gain
        /
        average_loss.replace(
            0,
            np.nan,
        )
    )

    return (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    close = df["close"].astype(float)

    daily_return = (
        close.pct_change()
    )

    df["return_1d"] = daily_return

    df["return_5d"] = (
        close.pct_change(5)
    )

    df["return_20d"] = (
        close.pct_change(20)
    )

    df["volatility"] = (
        daily_return
        .rolling(20)
        .std()
    )

    df["rsi"] = calculate_rsi(
        close
    )

    sma20 = (
        close
        .rolling(20)
        .mean()
    )

    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    df["close_vs_sma20"] = (
        close / sma20 - 1
    )

    df["close_vs_sma50"] = (
        close / sma50 - 1
    )

    df["trend_score"] = (
        0.5
        *
        df["close_vs_sma20"]
        +
        0.5
        *
        df["close_vs_sma50"]
    )

    df["momentum_score"] = (
        100
        *
        df["return_20d"]
    )

    feature_columns = [
        "return_5d",
        "return_20d",
        "volatility",
        "rsi",
        "trend_score",
        "momentum_score",
    ]

    df = df.dropna(
        subset=feature_columns
    ).reset_index(
        drop=True
    )

    return df


FEATURE_COLUMNS = [
    "return_5d",
    "return_20d",
    "volatility",
    "rsi",
    "trend_score",
    "momentum_score",
]


# ================================================================
# HMM FITTING
# ================================================================

def fit_best_hmm(
    X: np.ndarray,
):
    """
    Fit multiple HMM initializations and keep the best
    converged model.

    Diagonal covariance is deliberately used here because
    this historical allocator is repeatedly fitting models
    on rolling 504-observation windows.
    """

    best_model = None

    best_score = -np.inf

    best_seed = None

    converged_models = 0

    for restart in range(
        N_RESTARTS
    ):

        seed = (
            RANDOM_SEED
            +
            restart
        )

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=MAX_ITER,
            tol=TOLERANCE,
            random_state=seed,
            verbose=False,
        )

        try:

            model.fit(X)

            if not model.monitor_.converged:
                continue

            score = model.score(X)

            converged_models += 1

            if score > best_score:

                best_score = score

                best_model = model

                best_seed = seed

        except Exception:
            continue

    if best_model is None:

        raise RuntimeError(
            "No HMM restart converged."
        )

    return (
        best_model,
        best_score,
        best_seed,
        converged_models,
    )


# ================================================================
# CAUSAL FILTERED PROBABILITY
# ================================================================

def calculate_filtered_probability(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Forward-only HMM filtering.

    The probability at time t uses observations
    available through t only.

    No future observations are used.
    """

    log_transition = np.log(
        model.transmat_
        + 1e-300
    )

    log_start = np.log(
        model.startprob_
        + 1e-300
    )

    log_emission = (
        model._compute_log_likelihood(X)
    )

    n_observations = len(X)

    probabilities = np.zeros(
        (
            n_observations,
            N_STATES,
        )
    )

    alpha = (
        log_start
        +
        log_emission[0]
    )

    max_value = np.max(alpha)

    normalized = np.exp(
        alpha - max_value
    )

    probabilities[0] = (
        normalized
        /
        normalized.sum()
    )

    for t in range(
        1,
        n_observations,
    ):

        previous = (
            alpha[:, None]
            +
            log_transition
        )

        alpha = (
            log_emission[t]
            +
            np.logaddexp.reduce(
                previous,
                axis=0,
            )
        )

        max_value = np.max(alpha)

        normalized = np.exp(
            alpha - max_value
        )

        probabilities[t] = (
            normalized
            /
            normalized.sum()
        )

    return probabilities


# ================================================================
# ECONOMIC STATE MAPPING
# ================================================================

def map_states_to_regimes(
    training_df: pd.DataFrame,
    hidden_states: np.ndarray,
) -> dict:

    work = training_df.copy()

    work["state"] = hidden_states

    profile = (
        work
        .groupby("state")
        .agg(
            mean_return_20d=(
                "return_20d",
                "mean",
            ),
            mean_volatility=(
                "volatility",
                "mean",
            ),
            mean_rsi=(
                "rsi",
                "mean",
            ),
            mean_trend=(
                "trend_score",
                "mean",
            ),
        )
    )

    if len(profile) != N_STATES:

        raise RuntimeError(
            "HMM did not populate all three states."
        )

    # Highest medium-term return = BULL
    bull_state = int(
        profile[
            "mean_return_20d"
        ].idxmax()
    )

    # Lowest medium-term return = BEAR
    bear_state = int(
        profile[
            "mean_return_20d"
        ].idxmin()
    )

    remaining = [
        state
        for state in profile.index
        if state not in [
            bull_state,
            bear_state,
        ]
    ]

    if len(remaining) != 1:

        raise RuntimeError(
            "Unable to uniquely map SIDE state."
        )

    side_state = int(
        remaining[0]
    )

    return {
        bull_state: "BULL",
        side_state: "SIDE",
        bear_state: "BEAR",
    }


# ================================================================
# REORDER STATE PROBABILITIES
# ================================================================

def reorder_probabilities(
    probabilities: np.ndarray,
    mapping: dict,
) -> np.ndarray:

    state_for_regime = {
        regime: state
        for state, regime
        in mapping.items()
    }

    order = [
        state_for_regime[
            "BULL"
        ],
        state_for_regime[
            "SIDE"
        ],
        state_for_regime[
            "BEAR"
        ],
    ]

    return probabilities[
        order
    ]


# ================================================================
# SINGLE HISTORICAL SIGNAL
# ================================================================

def generate_signal(
    feature_df: pd.DataFrame,
    end_index: int,
):
    """
    Fit an HMM using ONLY observations ending at end_index.

    The latest observation in the training window is therefore
    the information set available at the rebalance date.
    """

    start_index = (
        end_index
        -
        TRAIN_WINDOW
        +
        1
    )

    if start_index < 0:

        return None

    training = (
        feature_df
        .iloc[
            start_index:
            end_index + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(training) < TRAIN_WINDOW:

        return None

    # ------------------------------------------------------------
    # Fit scaler ONLY on historical training window
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        training[
            FEATURE_COLUMNS
        ]
    )

    # ------------------------------------------------------------
    # Fit HMM ONLY on historical training window
    # ------------------------------------------------------------

    (
        model,
        train_score,
        best_seed,
        converged_models,
    ) = fit_best_hmm(
        X_train
    )

    # ------------------------------------------------------------
    # State mapping
    # ------------------------------------------------------------

    hidden_states = (
        model.predict(
            X_train
        )
    )

    mapping = (
        map_states_to_regimes(
            training,
            hidden_states,
        )
    )

    # ------------------------------------------------------------
    # Forward-only filtering
    # ------------------------------------------------------------

    filtered = (
        calculate_filtered_probability(
            model,
            X_train,
        )
    )

    current_raw = (
        filtered[-1]
    )

    current_ordered = (
        reorder_probabilities(
            current_raw,
            mapping,
        )
    )

    # ------------------------------------------------------------
    # One-step transition forecast
    # ------------------------------------------------------------

      

    state_for_regime = {
        regime: state
        for state, regime
        in mapping.items()
    }

    ordered_states = [
        state_for_regime[
            "BULL"
        ],
        state_for_regime[
            "SIDE"
        ],
        state_for_regime[
            "BEAR"
        ],
    ]

    transition_ordered = (
        model.transmat_[
            ordered_states
        ][:, ordered_states]
    )

    next_ordered = (
        current_ordered
        @ transition_ordered
    )

    current_regime = (
        STATE_NAMES[
            int(
                np.argmax(
                    current_ordered
                )
            )
        ]
    )

    next_regime = (
        STATE_NAMES[
            int(
                np.argmax(
                    next_ordered
                )
            )
        ]
    )

    date = (
        training[
            "timestamp"
        ].iloc[-1]
    )

    return {
        "timestamp": date,

        "current_regime":
            current_regime,

        "bull_probability":
            float(
                current_ordered[0]
            ),

        "side_probability":
            float(
                current_ordered[1]
            ),

        "bear_probability":
            float(
                current_ordered[2]
            ),

        "next_regime":
            next_regime,

        "next_bull_probability":
            float(
                next_ordered[0]
            ),

        "next_side_probability":
            float(
                next_ordered[1]
            ),

        "next_bear_probability":
            float(
                next_ordered[2]
            ),

        "train_start":
            training[
                "timestamp"
            ].iloc[0],

        "train_end":
            training[
                "timestamp"
            ].iloc[-1],

        "train_observations":
            len(training),

        "hmm_log_likelihood":
            float(train_score),

        "best_seed":
            int(best_seed),

        "converged_restarts":
            int(converged_models),
    }


# ================================================================
# REBALANCE DATE GENERATION
# ================================================================

def get_rebalance_indices(
    feature_df: pd.DataFrame,
) -> list[int]:

    first_index = (
        TRAIN_WINDOW - 1
    )

    if first_index >= len(
        feature_df
    ):

        return []

    indices = list(
        range(
            first_index,
            len(feature_df),
            REBALANCE_FREQUENCY,
        )
    )

    # Always include the final available observation.
    if (
        indices
        and
        indices[-1]
        != len(feature_df) - 1
    ):

        indices.append(
            len(feature_df) - 1
        )

    return indices


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("QUANTOS CAUSAL HISTORICAL HMM SIGNAL ENGINE")
    print("=" * 80)

    print()
    print(
        "Model: 3-state Gaussian HMM"
    )

    print(
        "States: BULL / SIDE / BEAR"
    )

    print(
        f"Training window: "
        f"{TRAIN_WINDOW} observations"
    )

    print(
        f"Rebalance frequency: "
        f"{REBALANCE_FREQUENCY} observations"
    )

    print()
    print(
        f"Input:  {INPUT_PATH}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    raw = load_nifty_data()

    print()
    print(
        f"Raw observations: "
        f"{len(raw)}"
    )

    feature_df = build_features(
        raw
    )

    print(
        f"Feature observations: "
        f"{len(feature_df)}"
    )

    print(
        f"Feature period: "
        f"{feature_df['timestamp'].min().date()} "
        f"-> "
        f"{feature_df['timestamp'].max().date()}"
    )

    # ------------------------------------------------------------
    # Generate causal signals
    # ------------------------------------------------------------

    rebalance_indices = (
        get_rebalance_indices(
            feature_df
        )
    )

    print()
    print(
        f"Historical signal dates: "
        f"{len(rebalance_indices)}"
    )

    results = []

    for counter, index in enumerate(
        rebalance_indices,
        start=1,
    ):

        date = (
            feature_df[
                "timestamp"
            ].iloc[index]
        )

        print(
            f"\rGenerating signal "
            f"{counter:>3}/"
            f"{len(rebalance_indices)}"
            f" | {date.date()}",
            end="",
            flush=True,
        )

        try:

            signal = generate_signal(
                feature_df,
                index,
            )

            if signal is not None:
                results.append(
                    signal
                )

        except Exception as exc:

            print()
            print(
                f"WARNING: "
                f"{date.date()} failed: "
                f"{exc}"
            )

    print()
    print()

    if not results:

        raise RuntimeError(
            "No historical HMM signals generated."
        )

    signals = pd.DataFrame(
        results
    )

    signals = (
        signals
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Probability validation
    # ------------------------------------------------------------

    current_probability_sum = (
        signals[
            [
                "bull_probability",
                "side_probability",
                "bear_probability",
            ]
        ]
        .sum(axis=1)
    )

    next_probability_sum = (
        signals[
            [
                "next_bull_probability",
                "next_side_probability",
                "next_bear_probability",
            ]
        ]
        .sum(axis=1)
    )

    current_error = (
        np.abs(
            current_probability_sum
            - 1.0
        ).max()
    )

    next_error = (
        np.abs(
            next_probability_sum
            - 1.0
        ).max()
    )

    if current_error > 1e-8:

        raise RuntimeError(
            "Current HMM probabilities "
            "do not sum to 1."
        )

    if next_error > 1e-8:

        raise RuntimeError(
            "Next-state HMM probabilities "
            "do not sum to 1."
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    signals.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print("=" * 80)
    print("CAUSAL HMM SIGNAL RESULTS")
    print("=" * 80)

    print()
    print(
        f"Signal rows: "
        f"{len(signals)}"
    )

    print(
        f"Signal period: "
        f"{signals['timestamp'].min().date()} "
        f"-> "
        f"{signals['timestamp'].max().date()}"
    )

    print()
    print("CURRENT REGIME COUNTS")

    print(
        signals[
            "current_regime"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("NEXT REGIME COUNTS")

    print(
        signals[
            "next_regime"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "MAX CURRENT PROBABILITY "
        f"SUM ERROR: {current_error:.2e}"
    )

    print(
        "MAX NEXT PROBABILITY "
        f"SUM ERROR: {next_error:.2e}"
    )

    print()
    print("LATEST CAUSAL SIGNAL")

    print(
        signals.tail(1)
        .T
        .to_string()
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("=" * 80)
    print("CAUSAL HMM SIGNAL ENGINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()