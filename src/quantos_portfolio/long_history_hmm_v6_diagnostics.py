"""
QuantOS HMM V6
Rigorous Causal HMM Regime Diagnostics

Purpose
-------
This version does NOT attempt to improve the HMM's score artificially.

It answers five questions:

1. Are the latent states statistically distinct?
2. Is the filtered state posterior excessively confident?
3. Are transition probabilities excessively sticky?
4. Does the regime contain forward-return / volatility information?
5. Are predicted probabilities calibrated out-of-sample?

Important
---------
This is a diagnostic experiment.

The HMM is fitted only on information available at each prediction date.
No future observations are used for model fitting.

The model is NOT optimized for directional accuracy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

N_STATES = 3
HMM_WINDOW = 504

SEEDS = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]

MIN_FORWARD_DAYS = 1
FORWARD_HORIZONS = [1, 5, 10, 21]

FEATURES = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "return_zscore",
    "vol_ratio",
]

STATE_NAMES = ["BULL", "SIDE", "BEAR"]

DATA_ROOT = Path("data")
OUTPUT_DIR = DATA_ROOT / "regime" / "hmm_v6"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA DISCOVERY
# ============================================================

def find_nifty50_file() -> Path:
    """
    Find the actual NIFTY 50 daily dataset.

    We explicitly avoid fuzzy matching such as selecting
    nifty_next_50.parquet merely because the filename contains
    'nifty'.
    """

    candidates = [
        DATA_ROOT / "regime" / "daily" / "nifty_50.parquet",
        DATA_ROOT / "regime" / "daily" / "nifty50.parquet",
        DATA_ROOT / "regime" / "daily" / "NIFTY_50.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = []

    for path in DATA_ROOT.rglob("*.parquet"):
        name = path.name.lower()

        if (
            "nifty_50" in name
            or "nifty50" in name
            or "nifty-50" in name
        ):
            if "next" not in name and "midcap" not in name:
                matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "Could not find the NIFTY 50 parquet file."
        )

    if len(matches) > 1:
        print("\nCandidate NIFTY 50 files:")
        for path in matches:
            print("  ", path)

        print(
            "\nUsing the first exact NIFTY 50 candidate:"
        )

    return matches[0]


# ============================================================
# DATA LOADING
# ============================================================

def load_nifty50() -> pd.DataFrame:

    path = find_nifty50_file()

    print("=" * 70)
    print("NIFTY 50 DATA")
    print("=" * 70)
    print("File:", path)

    df = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
    else:

        date_candidates = [
            "date",
            "Date",
            "datetime",
            "Datetime",
            "timestamp",
            "Timestamp",
        ]

        date_col = None

        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError(
                "Could not identify date column."
            )

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Remove timezone if present
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    # Find close column
    close_candidates = [
        "close",
        "Close",
        "closing_price",
        "Close Price",
    ]

    close_col = None

    for col in close_candidates:
        if col in df.columns:
            close_col = col
            break

    if close_col is None:
        raise ValueError(
            f"Could not find close column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[close_col]].rename(
        columns={close_col: "close"}
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df = df.dropna()

    df = df[df["close"] > 0]

    print("Rows:", len(df))
    print("Period:", df.index.min(), "->", df.index.max())

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    out = pd.DataFrame(index=df.index)

    close = df["close"]

    log_close = np.log(close)
    log_return = log_close.diff()

    out["return_1d"] = log_return

    out["momentum_5d"] = (
        log_close - log_close.shift(5)
    )

    out["momentum_20d"] = (
        log_close - log_close.shift(20)
    )

    out["vol_5d"] = (
        log_return.rolling(5).std()
    )

    out["vol_20d"] = (
        log_return.rolling(20).std()
    )

    out["vol_60d"] = (
        log_return.rolling(60).std()
    )

    rolling_mean = (
        log_return
        .rolling(60)
        .mean()
    )

    rolling_std = (
        log_return
        .rolling(60)
        .std()
    )

    out["return_zscore"] = (
        log_return - rolling_mean
    ) / rolling_std.replace(0, np.nan)

    out["vol_ratio"] = (
        out["vol_20d"]
        / out["vol_60d"].replace(0, np.nan)
    )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan
    )

    out = out.dropna()

    return out


# ============================================================
# STATE MAPPING
# ============================================================

def map_states(
    model: GaussianHMM,
    scaler: RobustScaler,
) -> dict:
    """
    Map anonymous HMM state IDs to economic regimes.

    Mapping is based on the scaled model means for the
    20-day momentum feature.

    Highest momentum -> BULL
    Middle -> SIDE
    Lowest -> BEAR
    """

    feature_idx = FEATURES.index("momentum_20d")

    means_scaled = model.means_[:, feature_idx]

    order = np.argsort(means_scaled)

    mapping = {
        int(order[2]): "BULL",
        int(order[1]): "SIDE",
        int(order[0]): "BEAR",
    }

    return mapping


# ============================================================
# CAUSAL FILTER
# ============================================================

# ============================================================
# CAUSAL FILTER
# ============================================================

# ============================================================
# CAUSAL FILTER
# ============================================================

def causal_filter(
    model: GaussianHMM,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Causal forward filter for GaussianHMM.

    Computes:

        P(S_t | X_1, ..., X_t)

    Supports hmmlearn covariance representations for:

        covariance_type = "diag"
            covars shape may be:
                (n_states, n_features)
            OR
                (n_states, n_features, n_features)

        covariance_type = "full"
            covars shape:
                (n_states, n_features, n_features)

    Future observations are never used.
    """

    n_obs = X_scaled.shape[0]
    n_states = model.n_components
    n_features = X_scaled.shape[1]

    alpha = np.zeros(
        (n_obs, n_states),
        dtype=float,
    )

    # --------------------------------------------------------
    # Start probabilities
    # --------------------------------------------------------

    log_start = np.log(
        np.clip(
            np.asarray(
                model.startprob_,
                dtype=float,
            ),
            1e-300,
            None,
        )
    )

    # --------------------------------------------------------
    # Transition probabilities
    # --------------------------------------------------------

    transition_matrix = np.asarray(
        model.transmat_,
        dtype=float,
    )

    log_trans = np.log(
        np.clip(
            transition_matrix,
            1e-300,
            None,
        )
    )

    log_2pi = np.log(
        2.0 * np.pi
    )

    # --------------------------------------------------------
    # Normalize covariance representation
    # --------------------------------------------------------

    covars = np.asarray(
        model.covars_,
        dtype=float,
    )

    covariance_type = (
        model.covariance_type
    )

    if covariance_type == "diag":

        # ----------------------------------------------------
        # Case 1:
        # (states, features)
        # ----------------------------------------------------

        if covars.ndim == 2:

            if covars.shape != (
                n_states,
                n_features,
            ):
                raise ValueError(
                    "Unexpected diagonal covariance "
                    f"shape: {covars.shape}. "
                    f"Expected "
                    f"{(n_states, n_features)}."
                )

            diagonal_variances = (
                covars.copy()
            )

        # ----------------------------------------------------
        # Case 2:
        # hmmlearn stores diagonal covariance as
        # (states, features, features)
        #
        # Extract ONLY the diagonal.
        # ----------------------------------------------------

        elif covars.ndim == 3:

            if covars.shape != (
                n_states,
                n_features,
                n_features,
            ):
                raise ValueError(
                    "Unexpected diagonal covariance "
                    f"shape: {covars.shape}. "
                    f"Expected "
                    f"{(n_states, n_features, n_features)}."
                )

            diagonal_variances = (
                np.diagonal(
                    covars,
                    axis1=1,
                    axis2=2,
                )
            )

        else:

            raise ValueError(
                "Unsupported diagonal covariance "
                f"dimensions: {covars.ndim}"
            )

        diagonal_variances = np.maximum(
            diagonal_variances,
            1e-8,
        )

        log_det = np.sum(
            np.log(
                diagonal_variances
            ),
            axis=1,
        )

        inverse_covariance = None

    # --------------------------------------------------------
    # FULL COVARIANCE
    # --------------------------------------------------------

    elif covariance_type == "full":

        if covars.shape != (
            n_states,
            n_features,
            n_features,
        ):
            raise ValueError(
                "Unexpected full covariance "
                f"shape: {covars.shape}. "
                f"Expected "
                f"{(n_states, n_features, n_features)}."
            )

        inverse_covariance = np.zeros_like(
            covars
        )

        log_det = np.zeros(
            n_states,
            dtype=float,
        )

        for state in range(
            n_states
        ):

            covariance = (
                covars[state].copy()
            )

            # Numerical regularisation
            covariance += (
                np.eye(n_features)
                * 1e-8
            )

            sign, determinant = (
                np.linalg.slogdet(
                    covariance
                )
            )

            if sign <= 0:
                raise ValueError(
                    "Covariance matrix for "
                    f"state {state} is not "
                    "positive definite."
                )

            log_det[state] = determinant

            inverse_covariance[state] = (
                np.linalg.inv(
                    covariance
                )
            )

    else:

        raise ValueError(
            "Unsupported covariance_type: "
            f"{covariance_type}"
        )

    # --------------------------------------------------------
    # Gaussian emission probability
    # --------------------------------------------------------

    def emission_log_prob(
        x: np.ndarray,
    ) -> np.ndarray:

        log_probability = np.empty(
            n_states,
            dtype=float,
        )

        for state in range(
            n_states
        ):

            diff = (
                x
                - model.means_[state]
            )

            # -----------------------------------------------
            # Diagonal covariance
            # -----------------------------------------------

            if covariance_type == "diag":

                mahalanobis = np.sum(
                    (
                        diff ** 2
                    )
                    / diagonal_variances[
                        state
                    ]
                )

            # -----------------------------------------------
            # Full covariance
            # -----------------------------------------------

            else:

                mahalanobis = (
                    diff
                    @ inverse_covariance[state]
                    @ diff
                )

            log_probability[state] = (
                -0.5
                * (
                    n_features
                    * log_2pi
                    + log_det[state]
                    + mahalanobis
                )
            )

        return log_probability

    # ========================================================
    # FORWARD ALGORITHM
    # ========================================================

    # First observation
    log_alpha = (
        log_start
        + emission_log_prob(
            X_scaled[0]
        )
    )

    log_alpha -= (
        np.logaddexp.reduce(
            log_alpha
        )
    )

    alpha[0] = np.exp(
        log_alpha
    )

    # --------------------------------------------------------
    # Remaining observations
    # --------------------------------------------------------

    for t in range(
        1,
        n_obs,
    ):

        emission = (
            emission_log_prob(
                X_scaled[t]
            )
        )

        next_log_alpha = np.empty(
            n_states,
            dtype=float,
        )

        for state in range(
            n_states
        ):

            next_log_alpha[state] = (
                np.logaddexp.reduce(
                    log_alpha
                    + log_trans[:, state]
                )
                + emission[state]
            )

        # Normalize
        next_log_alpha -= (
            np.logaddexp.reduce(
                next_log_alpha
            )
        )

        log_alpha = (
            next_log_alpha
        )

        alpha[t] = np.exp(
            log_alpha
        )

    return alpha

# ============================================================
# ENTROPY
# ============================================================

def entropy(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    return float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )


# ============================================================
# MODEL FIT
# ============================================================

def fit_best_hmm(
    X: pd.DataFrame,
):
    """
    Fit multiple random seeds.

    The best converged model by log likelihood is retained.

    No future observations are involved because X contains
    only the historical training window.
    """

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(
        X[FEATURES]
    )

    best_model = None
    best_score = -np.inf
    best_seed = None

    for seed in SEEDS:

        try:

            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=500,
                tol=1e-4,
                random_state=seed,
                init_params="stmc",
                verbose=False,
            )

            model.fit(X_scaled)

            print(
    f"Seed {seed}: "
    f"covariance_type={model.covariance_type}, "
    f"covars_shape={model.covars_.shape}"
)

            score = model.score(
    X_scaled
)

            if (
                np.isfinite(score)
                and score > best_score
            ):

                best_score = score
                best_model = model
                best_seed = seed

        except Exception as exc:

            print(
                f"Seed {seed} failed: {exc}"
            )

    if best_model is None:
        raise RuntimeError(
            "All HMM seeds failed."
        )

    return (
        best_model,
        scaler,
        best_seed,
        best_score,
    )


# ============================================================
# FORWARD REGIME STATISTICS
# ============================================================

def calculate_forward_statistics(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    current_state: str,
):
    """
    Calculate genuinely out-of-sample forward returns
    starting from the current OOS date.

    The HMM is never given these observations during fitting.
    They are used only for evaluation.
    """

    close = df["close"]

    current_position = df.index.get_loc(
        current_date
    )

    result = {
        "date": current_date,
        "current_state": current_state,
    }

    for horizon in FORWARD_HORIZONS:

        future_position = (
            current_position + horizon
        )

        if future_position >= len(df):
            continue

        future_date = df.index[
            future_position
        ]

        current_price = close.loc[
            current_date
        ]

        future_price = close.loc[
            future_date
        ]

        future_log_return = np.log(
            future_price / current_price
        )

        future_simple_return = (
            future_price / current_price
            - 1.0
        )

        result[
            f"forward_{horizon}d_log_return"
        ] = future_log_return

        result[
            f"forward_{horizon}d_return"
        ] = future_simple_return

    return result

# ============================================================
# MAIN OOS LOOP
# ============================================================

def run_oos():

    df = load_nifty50()

    features = build_features(df)

    print("\n" + "=" * 70)
    print("FEATURE DATA")
    print("=" * 70)

    print("Rows:", len(features))
    print(
        "Period:",
        features.index.min(),
        "->",
        features.index.max(),
    )

    results = []

    print("\n" + "=" * 70)
    print("CAUSAL WALK-FORWARD HMM")
    print("=" * 70)

    first_position = HMM_WINDOW

    for pos in range(
        first_position,
        len(features) - 1,
    ):

        train_start = (
            pos - HMM_WINDOW
        )

        train_end = pos

        train = features.iloc[
            train_start:train_end
        ]

        current_date = features.index[pos]

        # ----------------------------------------------------
        # Fit only on historical data
        # ----------------------------------------------------

        try:

            (
                model,
                scaler,
                seed,
                train_score,
            ) = fit_best_hmm(train)

        except Exception as exc:

            print(
                f"Skipping {current_date}: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Filter entire training window causally
        # ----------------------------------------------------

        X_scaled = scaler.transform(
            train[FEATURES]
        )

        filtered = causal_filter(
            model,
            X_scaled,
        )

        current_prob_raw = filtered[-1]

        # ----------------------------------------------------
        # State mapping
        # ----------------------------------------------------

        mapping = map_states(
            model,
            scaler,
        )

        inverse_mapping = {
            name: state_id
            for state_id, name
            in mapping.items()
        }

        current_prob = np.zeros(
            N_STATES
        )

        for state_id, state_name in mapping.items():
            current_prob[
                STATE_NAMES.index(state_name)
            ] = current_prob_raw[state_id]

        current_state_idx = int(
            np.argmax(current_prob)
        )

        current_state = STATE_NAMES[
            current_state_idx
        ]

        # ----------------------------------------------------
        # Transition matrix reordered
        # ----------------------------------------------------

        transition = np.zeros(
            (N_STATES, N_STATES)
        )

        for i, state_i in enumerate(
            STATE_NAMES
        ):

            original_i = inverse_mapping[
                state_i
            ]

            for j, state_j in enumerate(
                STATE_NAMES
            ):

                original_j = inverse_mapping[
                    state_j
                ]

                transition[i, j] = (
                    model.transmat_[
                        original_i,
                        original_j,
                    ]
                )

        next_prob = (
            current_prob
            @ transition
        )

        # ----------------------------------------------------
        # Next-day binary direction
        # ----------------------------------------------------

        actual_next_return = (
            np.log(
                df["close"].iloc[
                    df.index.get_loc(
                        current_date
                    ) + 1
                ]
                / df["close"].loc[
                    current_date
                ]
            )
        )

        actual_direction = int(
            actual_next_return > 0
        )

        bull_prob = current_prob[
            STATE_NAMES.index("BULL")
        ]

        bear_prob = current_prob[
            STATE_NAMES.index("BEAR")
        ]

        side_prob = current_prob[
            STATE_NAMES.index("SIDE")
        ]

        # Binary directional probability.
        #
        # IMPORTANT:
        # SIDE is explicitly retained as uncertainty.
        # We do NOT pretend that SIDE means DOWN or UP.
        directional_mass = (
            bull_prob + bear_prob
        )

        if directional_mass > 1e-12:

            prob_up = (
                bull_prob
                / directional_mass
            )

        else:

            prob_up = 0.5

        predicted_direction = int(
            prob_up >= 0.5
        )

        # ----------------------------------------------------
        # State persistence
        # ----------------------------------------------------

        state_id_original = inverse_mapping[
            current_state
        ]

        persistence = transition[
            current_state_idx,
            current_state_idx
        ]

        # ----------------------------------------------------
        # Forward returns
        # ----------------------------------------------------

        forward_stats = (
            calculate_forward_statistics(
                df,
                current_date,
                current_state,
            )
        )

        row = {
            "date": current_date,
            "seed": seed,
            "train_loglik": train_score,

            "current_state": current_state,

            "p_bull": bull_prob,
            "p_side": side_prob,
            "p_bear": bear_prob,

            "current_entropy": entropy(
                current_prob
            ),

            "next_p_bull": next_prob[0],
            "next_p_side": next_prob[1],
            "next_p_bear": next_prob[2],

            "next_entropy": entropy(
                next_prob
            ),

            "prob_up": prob_up,

            "actual_next_return":
                actual_next_return,

            "actual_direction":
                actual_direction,

            "predicted_direction":
                predicted_direction,

            "state_persistence":
                persistence,

            "raw_transition_max":
                transition.max(),

        }

        row.update(
            {
                key: value
                for key, value
                in forward_stats.items()
                if key != "date"
            }
        )

        results.append(row)

        if (
            len(results) % 100
            == 0
        ):

            print(
                f"Processed {len(results)} "
                f"OOS observations | "
                f"{current_date.date()}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df, features, results_df


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(
    df: pd.DataFrame,
    results: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("QUANTOS HMM V6 DIAGNOSTICS")
    print("=" * 70)

    print(
        "\nOOS observations:",
        len(results),
    )

    print(
        "Period:",
        results["date"].min(),
        "->",
        results["date"].max(),
    )

    # --------------------------------------------------------
    # State distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STATE DISTRIBUTION")
    print("-" * 70)

    counts = (
        results["current_state"]
        .value_counts()
        .reindex(
            STATE_NAMES,
            fill_value=0,
        )
    )

    percentages = (
        counts / counts.sum()
    )

    for state in STATE_NAMES:

        print(
            f"{state:>5}: "
            f"{counts[state]:4d} "
            f"({percentages[state]:.2%})"
        )

    # --------------------------------------------------------
    # Posterior confidence
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("POSTERIOR CONFIDENCE")
    print("-" * 70)

    print(
        "Mean current entropy:",
        f"{results['current_entropy'].mean():.4f}",
    )

    print(
        "Median current entropy:",
        f"{results['current_entropy'].median():.4f}",
    )

    print(
        "Maximum possible entropy:",
        f"{np.log(N_STATES):.4f}",
    )

    print(
        "Mean next entropy:",
        f"{results['next_entropy'].mean():.4f}",
    )

    # --------------------------------------------------------
    # Transition persistence
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STATE PERSISTENCE")
    print("-" * 70)

    print(
        results
        .groupby("current_state")
        ["state_persistence"]
        .mean()
        .reindex(STATE_NAMES)
    )

    # --------------------------------------------------------
    # Directional performance
    # --------------------------------------------------------

    y_true = results[
        "actual_direction"
    ].astype(int)

    y_pred = results[
        "predicted_direction"
    ].astype(int)

    prob_up = results[
        "prob_up"
    ].clip(
        1e-6,
        1 - 1e-6,
    )

    print("\n" + "-" * 70)
    print("ONE-DAY DIRECTIONAL TEST")
    print("-" * 70)

    print(
        "Accuracy:",
        f"{accuracy_score(y_true, y_pred):.4f}",
    )

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy_score(y_true, y_pred):.4f}",
    )

    print(
        "Brier Score:",
        f"{brier_score_loss(y_true, prob_up):.4f}",
    )

    print(
        "Log Loss:",
        f"{log_loss(y_true, prob_up):.4f}",
    )

    actual_up_frequency = y_true.mean()

    always_up_accuracy = (
        actual_up_frequency
    )

    always_down_accuracy = (
        1.0 - actual_up_frequency
    )

    print(
        "\nAlways UP Accuracy:",
        f"{always_up_accuracy:.4f}",
    )

    print(
        "Always DOWN Accuracy:",
        f"{always_down_accuracy:.4f}",
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PROBABILITY CALIBRATION")
    print("-" * 70)

    bins = [
        0.0,
        0.4,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
        1.01,
    ]

    calibration_rows = []

    for low, high in zip(
        bins[:-1],
        bins[1:],
    ):

        mask = (
            (prob_up > low)
            & (prob_up <= high)
        )

        if mask.sum() == 0:
            continue

        observed = (
            y_true[mask].mean()
        )

        predicted = (
            prob_up[mask].mean()
        )

        calibration_rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "observations": int(
                    mask.sum()
                ),
                "mean_predicted":
                    predicted,
                "observed_frequency":
                    observed,
                "calibration_gap":
                    observed - predicted,
            }
        )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    print(
        calibration_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # State-conditioned forward returns
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FORWARD RETURN BY CURRENT STATE")
    print("-" * 70)

    for horizon in FORWARD_HORIZONS:

        column = (
            f"forward_{horizon}d_return"
        )

        if column not in results.columns:
            continue

        summary = (
            results
            .groupby("current_state")
            [column]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                ]
            )
            .reindex(STATE_NAMES)
        )

        print(
            f"\n{horizon}-day forward return:"
        )

        print(
            summary.to_string()
        )

    # --------------------------------------------------------
    # State-conditioned volatility
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FORWARD RETURN DISPERSION BY STATE")
    print("-" * 70)

    for horizon in [5, 21]:

        column = (
            f"forward_{horizon}d_log_return"
        )

        if column not in results.columns:
            continue

        summary = (
            results
            .groupby("current_state")
            [column]
            .std()
            .reindex(STATE_NAMES)
        )

        print(
            f"{horizon}-day return std:"
        )

        print(summary)

    # --------------------------------------------------------
    # Probability state relationship
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CURRENT STATE PROBABILITIES")
    print("-" * 70)

    print(
        results[
            [
                "p_bull",
                "p_side",
                "p_bear",
            ]
        ].describe()
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "hmm_v6_oos_results.csv"
    )

    calibration_path = (
        OUTPUT_DIR
        / "hmm_v6_calibration.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        results_path
    )

    print(
        calibration_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df, features, results = (
        run_oos()
    )

    print_diagnostics(
        df,
        results,
    )


if __name__ == "__main__":
    main()