"""
QuantOS HMM V5 - One-Day Causal Out-of-Sample Evaluation
==========================================================

Purpose
-------
Evaluate whether the causal HMM provides useful short-horizon
information for the NEXT trading day.

This is the primary predictive HMM experiment for QuantOS.

The HMM itself remains a 3-state model:

    BULL
    SIDE
    BEAR

The evaluation target is intentionally binary because a single
trading day does not support the same +/-5% regime definition
used for a 21-day cumulative return.

Target:

    next-day log return > 0  -> UP
    next-day log return < 0  -> DOWN

The HMM probabilities are retained in their original three-state
form and aggregated as:

    P(UP)   = P(BULL)
    P(DOWN) = P(BEAR)

The SIDE probability is retained as an uncertainty/context signal.

Important
---------
Every forecast is strictly causal.

At date t:

    training data = previous HMM_WINDOW observations
    filtering     = observations available through t
    prediction    = transition to t+1
    target        = return at t+1

No future observations are used during model fitting or filtering.

The output is intended to become an input to the QuantOS
allocation engine rather than a standalone trading recommendation.

Author: QuantOS
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PORTFOLIO_DIR = DATA_DIR / "portfolio"

OUTPUT_FILE = (
    PORTFOLIO_DIR /
    "nifty50_hmm_v5_one_day_oos_results.csv"
)

# ------------------------------------------------------------
# HMM
# ------------------------------------------------------------

HMM_WINDOW = 504

N_STATES = 3

SEEDS = [
    7,
    17,
    27,
    37,
    47,
]

N_ITER = 300

TOL = 1e-3

# ------------------------------------------------------------
# Signal frequency
#
# Use every trading day because this is a one-day prediction
# experiment.
# ------------------------------------------------------------

SIGNAL_EVERY = 1

# ------------------------------------------------------------
# Transition shrinkage
# ------------------------------------------------------------

TRANSITION_SHRINKAGE = 0.35

# ------------------------------------------------------------
# Probability temperature
#
# Retained from V3/V4 for comparability.
#
# IMPORTANT:
# This is NOT tuned on the OOS test.
# ------------------------------------------------------------

PROBABILITY_TEMPERATURE = 2.5

# ------------------------------------------------------------
# State names
# ------------------------------------------------------------

STATE_NAMES = [
    "BULL",
    "SIDE",
    "BEAR",
]

# ------------------------------------------------------------
# Numerical convergence diagnostics
# ------------------------------------------------------------

CONVERGENCE_RELATIVE_NOISE = 1e-8

CONVERGENCE_ABSOLUTE_NOISE = 1e-5


# ============================================================
# DATA DISCOVERY
# ============================================================

def find_input_file() -> Path:
    """
    Locate the actual long-history NIFTY 50 dataset.

    Prefer the exact NIFTY 50 file over NIFTY Next 50.
    """

    search_roots = [
        DATA_DIR,
        PROJECT_ROOT / "src",
    ]

    candidates = []

    for root in search_roots:

        if not root.exists():
            continue

        for pattern in [
            "*nifty*50*.csv",
            "*NIFTY*50*.csv",
            "*nifty*50*.parquet",
            "*NIFTY*50*.parquet",
        ]:

            candidates.extend(
                root.rglob(pattern)
            )

    candidates = list(
        dict.fromkeys(candidates)
    )

    excluded_names = {
        "nifty50_hmm_oos_results.csv",
        "nifty50_hmm_v4_oos_results.csv",
        "nifty50_hmm_v5_one_day_oos_results.csv",
    }

    candidates = [
        path
        for path in candidates
        if path.name not in excluded_names
    ]

    if not candidates:

        raise FileNotFoundError(
            "\nNo NIFTY 50 dataset found.\n"
        )

    inspected = []

    for path in candidates:

        try:

            if path.suffix.lower() == ".parquet":

                df = pd.read_parquet(
                    path
                )

            elif path.suffix.lower() == ".csv":

                df = pd.read_csv(
                    path
                )

            else:

                continue

            if df.empty:
                continue

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
                continue

            dates = pd.to_datetime(
                df[date_col],
                errors="coerce"
            ).dropna()

            if dates.empty:
                continue

            inspected.append(
                (
                    len(dates),
                    dates.min(),
                    dates.max(),
                    path,
                )
            )

        except Exception:

            continue

    if not inspected:

        raise FileNotFoundError(
            "\nNIFTY files were found but none could "
            "be read successfully.\n"
        )

    # --------------------------------------------------------
    # Prefer exact NIFTY 50 names
    # --------------------------------------------------------

    preferred_names = {
        "nifty_50.parquet",
        "nifty50.parquet",
        "nifty_50.csv",
        "nifty50.csv",
    }

    preferred = [
        item
        for item in inspected
        if item[3].name.lower()
        in preferred_names
    ]

    if preferred:

        preferred.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = preferred[0]

    else:

        inspected.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = inspected[0]

    (
        count,
        start_date,
        end_date,
        path,
    ) = selected

    minimum_required = (
        HMM_WINDOW + 100
    )

    if count < minimum_required:

        raise ValueError(
            f"\nSelected dataset has only {count} observations.\n"
            f"At least {minimum_required} are required.\n"
        )

    print()
    print("=" * 70)
    print("NIFTY 50 DATASET")
    print("=" * 70)

    print(
        f"File         : {path}"
    )

    print(
        f"Observations : {count}"
    )

    print(
        f"Period       : "
        f"{start_date.date()} -> "
        f"{end_date.date()}"
    )

    print("=" * 70)

    return path


# ============================================================
# LOAD NIFTY 50 RETURNS
# ============================================================

def load_nifty50_returns() -> pd.Series:

    input_file = find_input_file()

    print()
    print(
        f"Loading data: {input_file}"
    )

    if input_file.suffix.lower() == ".parquet":

        df = pd.read_parquet(
            input_file
        )

    elif input_file.suffix.lower() == ".csv":

        df = pd.read_csv(
            input_file
        )

    else:

        raise ValueError(
            f"Unsupported file format: "
            f"{input_file.suffix}"
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

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

        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[date_col]
    )

    df = df.sort_values(
        date_col
    )

    df = df.drop_duplicates(
        subset=[date_col]
    )

    df = df.set_index(
        date_col
    )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    price_candidates = [
        "close",
        "Close",
        "adj_close",
        "Adj Close",
        "adjusted_close",
        "Adjusted Close",
    ]

    price_col = None

    for col in price_candidates:

        if col in df.columns:

            price_col = col
            break

    if price_col is None:

        numeric_cols = (
            df.select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        excluded = {
            "volume",
            "Volume",
            "open",
            "Open",
            "high",
            "High",
            "low",
            "Low",
        }

        numeric_cols = [
            col
            for col in numeric_cols
            if col not in excluded
        ]

        if not numeric_cols:

            raise ValueError(
                "Could not identify NIFTY 50 price column."
            )

        price_col = numeric_cols[0]

    print(
        f"Using price column: {price_col}"
    )

    prices = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )

    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan
    )

    prices = prices.dropna()

    if (prices <= 0).any():

        raise ValueError(
            "NIFTY 50 prices contain non-positive values."
        )

    # --------------------------------------------------------
    # Log returns
    # --------------------------------------------------------

    log_returns = np.log(
        prices
    ).diff()

    log_returns = log_returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    log_returns = log_returns.dropna()

    log_returns.index = pd.to_datetime(
        log_returns.index
    )

    if getattr(
        log_returns.index,
        "tz",
        None
    ) is not None:

        log_returns.index = (
            log_returns.index
            .tz_localize(None)
        )

    log_returns = (
        log_returns
        .sort_index()
    )

    log_returns = (
        log_returns[
            ~log_returns.index.duplicated(
                keep="last"
            )
        ]
    )

    log_returns.name = "log_return"

    print()
    print(
        f"Observations loaded: "
        f"{len(log_returns)}"
    )

    print(
        f"Period: "
        f"{log_returns.index.min().date()} "
        f"-> "
        f"{log_returns.index.max().date()}"
    )

    print(
        f"Mean daily log return: "
        f"{log_returns.mean():.8f}"
    )

    print(
        f"Daily volatility: "
        f"{log_returns.std():.8f}"
    )

    return log_returns


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def rolling_zscore(
    series: pd.Series,
    window: int
) -> pd.Series:

    mean = series.rolling(
        window
    ).mean()

    std = series.rolling(
        window
    ).std()

    return (
        (series - mean)
        /
        std.replace(
            0,
            np.nan
        )
    )


def build_hmm_features(
    returns: pd.Series
) -> pd.DataFrame:

    r = returns.copy()

    features = pd.DataFrame(
        index=r.index
    )

    features["return_1d"] = r

    features["momentum_5d"] = (
        r.rolling(5).sum()
    )

    features["momentum_20d"] = (
        r.rolling(20).sum()
    )

    features["vol_5d"] = (
        r.rolling(5).std()
    )

    features["vol_20d"] = (
        r.rolling(20).std()
    )

    features["vol_60d"] = (
        r.rolling(60).std()
    )

    features["return_zscore"] = (
        rolling_zscore(
            r,
            60
        )
    )

    short_vol = (
        r.rolling(20).std()
    )

    long_vol = (
        r.rolling(60).std()
    )

    features["vol_ratio"] = (
        short_vol
        /
        long_vol.replace(
            0,
            np.nan
        )
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features.dropna()


# ============================================================
# STATE MAPPING
# ============================================================

def map_states_to_regimes(
    model: GaussianHMM,
    scaler: RobustScaler,
) -> dict:

    means_raw = scaler.inverse_transform(
        model.means_
    )

    momentum_20 = means_raw[:, 2]

    order = np.argsort(
        momentum_20
    )

    mapping = {
        int(order[0]): "BEAR",
        int(order[1]): "SIDE",
        int(order[2]): "BULL",
    }

    return mapping


def reorder_probabilities(
    probabilities: np.ndarray,
    mapping: dict,
) -> np.ndarray:

    output = np.zeros(
        N_STATES
    )

    for state_id, regime in mapping.items():

        index = STATE_NAMES.index(
            regime
        )

        output[index] = (
            probabilities[state_id]
        )

    return output


def reorder_transition(
    transition: np.ndarray,
    mapping: dict,
) -> np.ndarray:

    output = np.zeros(
        (
            N_STATES,
            N_STATES
        )
    )

    for old_i, regime_i in mapping.items():

        new_i = STATE_NAMES.index(
            regime_i
        )

        for old_j, regime_j in mapping.items():

            new_j = STATE_NAMES.index(
                regime_j
            )

            output[
                new_i,
                new_j
            ] = transition[
                old_i,
                old_j
            ]

    output /= (
        output.sum(
            axis=1,
            keepdims=True
        )
    )

    return output


# ============================================================
# TRANSITION SHRINKAGE
# ============================================================

def shrink_transition_matrix(
    transition: np.ndarray,
    shrinkage: float
) -> np.ndarray:

    n = transition.shape[0]

    uniform = (
        np.ones(
            (
                n,
                n
            )
        )
        /
        n
    )

    shrunk = (
        (1.0 - shrinkage)
        *
        transition
        +
        shrinkage
        *
        uniform
    )

    shrunk = np.clip(
        shrunk,
        1e-12,
        None
    )

    shrunk /= (
        shrunk.sum(
            axis=1,
            keepdims=True
        )
    )

    return shrunk


# ============================================================
# TEMPERATURE
# ============================================================

def temperature_scale(
    probabilities: np.ndarray,
    temperature: float
) -> np.ndarray:

    p = np.asarray(
        probabilities,
        dtype=float
    )

    p = np.clip(
        p,
        1e-12,
        1.0
    )

    log_p = np.log(
        p
    )

    scaled = (
        log_p
        /
        temperature
    )

    scaled -= np.max(
        scaled
    )

    scaled = np.exp(
        scaled
    )

    scaled /= scaled.sum()

    return scaled


# ============================================================
# CAUSAL FORWARD FILTER
# ============================================================

def causal_filter(
    model: GaussianHMM,
    X_scaled: np.ndarray
) -> np.ndarray:

    means = model.means_

    covars = model.covars_

    if model.covariance_type == "diag":

        if covars.ndim == 3:

            variances = np.diagonal(
                covars,
                axis1=1,
                axis2=2
            )

        else:

            variances = covars

    else:

        raise ValueError(
            "Only diag covariance is supported."
        )

    variances = np.clip(
        variances,
        1e-8,
        None
    )

    alpha = np.asarray(
        model.startprob_,
        dtype=float
    )

    alpha = np.clip(
        alpha,
        1e-300,
        None
    )

    alpha /= alpha.sum()

    transition = model.transmat_

    def log_emission(x):

        diff = (
            x[None, :]
            -
            means
        )

        log_det = np.sum(
            np.log(
                variances
            ),
            axis=1
        )

        mahalanobis = np.sum(
            diff ** 2
            /
            variances,
            axis=1
        )

        dimension = x.shape[0]

        return (
            -0.5
            *
            (
                dimension
                *
                np.log(
                    2.0 * np.pi
                )
                +
                log_det
                +
                mahalanobis
            )
        )

    log_alpha = (
        np.log(
            np.clip(
                alpha,
                1e-300,
                None
            )
        )
        +
        log_emission(
            X_scaled[0]
        )
    )

    log_alpha -= np.max(
        log_alpha
    )

    alpha = np.exp(
        log_alpha
    )

    alpha /= alpha.sum()

    for t in range(
        1,
        len(X_scaled)
    ):

        predicted = (
            alpha
            @
            transition
        )

        predicted = np.clip(
            predicted,
            1e-300,
            None
        )

        predicted /= predicted.sum()

        log_alpha = (
            np.log(
                predicted
            )
            +
            log_emission(
                X_scaled[t]
            )
        )

        log_alpha -= np.max(
            log_alpha
        )

        alpha = np.exp(
            log_alpha
        )

        alpha /= alpha.sum()

    return alpha


# ============================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================

def convergence_diagnostics(
    model: GaussianHMM
) -> dict:

    history = list(
        getattr(
            model.monitor_,
            "history",
            []
        )
    )

    iterations = len(
        history
    )

    converged = bool(
        getattr(
            model.monitor_,
            "converged",
            False
        )
    )

    if iterations == 0:

        return {
            "iterations": 0,
            "final_delta": np.nan,
            "relative_delta": np.nan,
            "status": "UNKNOWN",
        }

    if iterations >= 2:

        delta = (
            history[-1]
            -
            history[-2]
        )

        scale = max(
            abs(
                history[-2]
            ),
            1.0
        )

        relative_delta = (
            abs(delta)
            /
            scale
        )

        if (
            delta < 0
            and
            abs(delta)
            <= CONVERGENCE_ABSOLUTE_NOISE
            and
            relative_delta
            <= CONVERGENCE_RELATIVE_NOISE
        ):

            status = (
                "NUMERICAL_NOISE"
            )

        elif delta >= 0:

            status = (
                "TRUE_CONVERGENCE"
                if converged
                else "LIKELIHOOD_STABLE"
            )

        else:

            status = (
                "NOT_CONVERGED"
            )

    else:

        delta = np.nan
        relative_delta = np.nan

        status = (
            "TRUE_CONVERGENCE"
            if converged
            else "UNKNOWN"
        )

    return {
        "iterations": iterations,
        "final_delta": float(delta),
        "relative_delta": float(
            relative_delta
        )
        if np.isfinite(
            relative_delta
        )
        else np.nan,
        "status": status,
    }


# ============================================================
# FIT ONE MODEL
# ============================================================

def fit_single_model(
    X_train: np.ndarray,
    seed: int
):

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=seed,
        init_params="stmc",
        params="stmc",
        verbose=False,
    )

    sink_out = io.StringIO()
    sink_err = io.StringIO()

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        with redirect_stdout(
            sink_out
        ), redirect_stderr(
            sink_err
        ):

            model.fit(
                X_scaled
            )

    diagnostics = (
        convergence_diagnostics(
            model
        )
    )

    diagnostics[
        "seed"
    ] = seed

    diagnostics[
        "train_score"
    ] = float(
        model.score(
            X_scaled
        )
    )

    return (
        model,
        scaler,
        diagnostics,
        X_scaled
    )


# ============================================================
# FIT BEST MODEL
# ============================================================

def fit_best_model(
    X_train: np.ndarray
):

    candidates = []

    for seed in SEEDS:

        try:

            (
                model,
                scaler,
                diagnostics,
                X_scaled,
            ) = fit_single_model(
                X_train,
                seed
            )

            if not np.isfinite(
                diagnostics[
                    "train_score"
                ]
            ):
                continue

            status = diagnostics[
                "status"
            ]

            if status in [
                "TRUE_CONVERGENCE",
                "LIKELIHOOD_STABLE",
            ]:

                quality = 2

            elif status == "NUMERICAL_NOISE":

                quality = 1

            else:

                quality = 0

            candidates.append(
                (
                    quality,
                    diagnostics[
                        "train_score"
                    ],
                    model,
                    scaler,
                    diagnostics,
                    X_scaled,
                )
            )

        except Exception:

            continue

    if not candidates:

        raise RuntimeError(
            "All HMM seeds failed."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True
    )

    (
        quality,
        score,
        model,
        scaler,
        diagnostics,
        X_scaled,
    ) = candidates[0]

    diagnostics[
        "candidate_count"
    ] = len(
        candidates
    )

    return (
        model,
        scaler,
        diagnostics,
        X_scaled
    )


# ============================================================
# ENTROPY
# ============================================================

def entropy(
    probabilities: np.ndarray
) -> float:

    p = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    return float(
        -np.sum(
            p * np.log(
                p
            )
        )
    )


# ============================================================
# MAIN OOS ENGINE
# ============================================================

def run_oos():

    print()
    print("=" * 70)
    print("QuantOS HMM V5 - ONE-DAY CAUSAL OOS TEST")
    print("=" * 70)

    print()
    print(
        "Forecast target: NEXT trading day"
    )

    print(
        f"Training window: {HMM_WINDOW} observations"
    )

    print(
        "Signal frequency: every trading day"
    )

    print(
        "No future observations are used."
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    returns = load_nifty50_returns()

    features = build_hmm_features(
        returns
    )

    aligned_returns = returns.reindex(
        features.index
    )

    valid = aligned_returns.notna()

    features = features.loc[
        valid
    ]

    aligned_returns = (
        aligned_returns.loc[
            valid
        ]
    )

    print()
    print(
        f"HMM feature observations: "
        f"{len(features)}"
    )

    print(
        f"HMM features: "
        f"{features.shape[1]}"
    )

    print(
        f"Feature period: "
        f"{features.index.min().date()} "
        f"-> "
        f"{features.index.max().date()}"
    )

    # --------------------------------------------------------
    # OOS
    # --------------------------------------------------------

    results = []

    start_position = (
        HMM_WINDOW
    )

    final_position = (
        len(features) - 1
    )

    for counter, position in enumerate(
        range(
            start_position,
            final_position,
            SIGNAL_EVERY
        ),
        start=1
    ):

        signal_date = (
            features.index[
                position
            ]
        )

        train_start = (
            position
            -
            HMM_WINDOW
        )

        train_end = position

        X_train = (
            features.iloc[
                train_start:train_end
            ]
            .values
        )

        try:

            (
                model,
                scaler,
                diagnostics,
                X_scaled,
            ) = fit_best_model(
                X_train
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # Current filtered state
        # ----------------------------------------------------

        current_raw = causal_filter(
            model,
            X_scaled
        )

        mapping = (
            map_states_to_regimes(
                model,
                scaler
            )
        )

        current_prob = (
            reorder_probabilities(
                current_raw,
                mapping
            )
        )

        # ----------------------------------------------------
        # Transition matrix
        # ----------------------------------------------------

        shrunk = (
            shrink_transition_matrix(
                model.transmat_,
                TRANSITION_SHRINKAGE
            )
        )

        transition = (
            reorder_transition(
                shrunk,
                mapping
            )
        )

        # ----------------------------------------------------
        # NEXT-DAY probability
        # ----------------------------------------------------

        next_prob = (
            current_prob
            @
            transition
        )

        next_prob = (
            temperature_scale(
                next_prob,
                PROBABILITY_TEMPERATURE
            )
        )

        # ----------------------------------------------------
        # Next-day target
        # ----------------------------------------------------

        next_position = (
            position + 1
        )

        if (
            next_position
            >= len(
                aligned_returns
            )
        ):
            break

        next_return = float(
            aligned_returns.iloc[
                next_position
            ]
        )

        if next_return > 0:

            actual_direction = "UP"

        elif next_return < 0:

            actual_direction = "DOWN"

        else:

            actual_direction = "FLAT"

        # ----------------------------------------------------
        # Binary probability
        #
        # Ignore SIDE as a directional class.
        #
        # Normalize BULL and BEAR so:
        #
        # P(UP) + P(DOWN) = 1
        #
        # SIDE remains an uncertainty measure.
        # ----------------------------------------------------

        bull_prob = (
            next_prob[0]
        )

        side_prob = (
            next_prob[1]
        )

        bear_prob = (
            next_prob[2]
        )

        directional_mass = (
            bull_prob
            +
            bear_prob
        )

        if directional_mass > 0:

            up_probability = (
                bull_prob
                /
                directional_mass
            )

            down_probability = (
                bear_prob
                /
                directional_mass
            )

        else:

            up_probability = 0.5
            down_probability = 0.5

        predicted_direction = (
            "UP"
            if up_probability >= 0.5
            else "DOWN"
        )

        current_regime = (
            STATE_NAMES[
                int(
                    np.argmax(
                        current_prob
                    )
                )
            ]
        )

        next_regime = (
            STATE_NAMES[
                int(
                    np.argmax(
                        next_prob
                    )
                )
            ]
        )

        results.append(
            {
                "signal_date": signal_date,

                "next_date": (
                    aligned_returns.index[
                        next_position
                    ]
                ),

                "actual_direction":
                    actual_direction,

                "next_day_log_return":
                    next_return,

                "current_regime":
                    current_regime,

                "next_regime":
                    next_regime,

                "current_prob_bull":
                    current_prob[0],

                "current_prob_side":
                    current_prob[1],

                "current_prob_bear":
                    current_prob[2],

                "next_prob_bull":
                    bull_prob,

                "next_prob_side":
                    side_prob,

                "next_prob_bear":
                    bear_prob,

                "directional_up_probability":
                    up_probability,

                "directional_down_probability":
                    down_probability,

                "predicted_direction":
                    predicted_direction,

                "current_entropy":
                    entropy(
                        current_prob
                    ),

                "next_entropy":
                    entropy(
                        next_prob
                    ),

                "raw_max_transition":
                    float(
                        np.max(
                            model.transmat_
                        )
                    ),

                "shrunk_max_transition":
                    float(
                        np.max(
                            transition
                        )
                    ),

                "selected_seed":
                    diagnostics[
                        "seed"
                    ],

                "candidate_count":
                    diagnostics[
                        "candidate_count"
                    ],

                "iterations":
                    diagnostics[
                        "iterations"
                    ],

                "train_log_likelihood":
                    diagnostics[
                        "train_score"
                    ],

                "final_delta":
                    diagnostics[
                        "final_delta"
                    ],

                "relative_delta":
                    diagnostics[
                        "relative_delta"
                    ],

                "convergence_status":
                    diagnostics[
                        "status"
                    ],
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            counter <= 5
            or counter % 250 == 0
        ):

            print(
                f"[{counter:04d}] "
                f"{signal_date.date()} | "
                f"Current={current_regime:<4} | "
                f"Next={next_regime:<4} | "
                f"Actual={actual_direction:<5} | "
                f"P(UP)={up_probability:.3f} | "
                f"P(SIDE)={side_prob:.3f}"
            )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:

        raise RuntimeError(
            "No OOS observations generated."
        )

    result_df[
        "signal_date"
    ] = pd.to_datetime(
        result_df[
            "signal_date"
        ]
    )

    result_df = result_df.sort_values(
        "signal_date"
    )

    PORTFOLIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    return result_df


# ============================================================
# REPORT
# ============================================================

def print_report(
    df: pd.DataFrame
):

    print()
    print()
    print("=" * 70)
    print("HMM V5 OOS RESULTS")
    print("=" * 70)

    print()
    print(
        f"Observations: {len(df)}"
    )

    print(
        f"Period: "
        f"{df['signal_date'].min().date()} "
        f"-> "
        f"{df['signal_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Directional evaluation
    # --------------------------------------------------------

    direction_df = df[
        df[
            "actual_direction"
        ].isin(
            [
                "UP",
                "DOWN",
            ]
        )
    ].copy()

    y_true = (
        direction_df[
            "actual_direction"
        ]
        .map(
            {
                "DOWN": 0,
                "UP": 1,
            }
        )
        .values
    )

    y_pred = (
        direction_df[
            "predicted_direction"
        ]
        .map(
            {
                "DOWN": 0,
                "UP": 1,
            }
        )
        .values
    )

    p_up = (
        direction_df[
            "directional_up_probability"
        ]
        .values
    )

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    brier = (
        brier_score_loss(
            y_true,
            p_up
        )
    )

    logloss = (
        log_loss(
            y_true,
            np.column_stack(
                [
                    1.0 - p_up,
                    p_up,
                ]
            ),
            labels=[
                0,
                1,
            ]
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
            zero_division=0
        )
    )

    print()
    print("=" * 70)
    print("ONE-DAY DIRECTIONAL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy            : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Brier Score         : "
        f"{brier:.4f}"
    )

    print(
        f"Log Loss            : "
        f"{logloss:.4f}"
    )

    print()
    print(
        "DOWN:"
    )

    print(
        f"  Precision: {precision[0]:.4f}"
    )

    print(
        f"  Recall   : {recall[0]:.4f}"
    )

    print(
        f"  F1       : {f1[0]:.4f}"
    )

    print(
        f"  Support  : {support[0]}"
    )

    print()
    print(
        "UP:"
    )

    print(
        f"  Precision: {precision[1]:.4f}"
    )

    print(
        f"  Recall   : {recall[1]:.4f}"
    )

    print(
        f"  F1       : {f1[1]:.4f}"
    )

    print(
        f"  Support  : {support[1]}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ]
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                "DOWN",
                "UP",
            ],
            columns=[
                "DOWN",
                "UP",
            ]
        )
    )

    # --------------------------------------------------------
    # Simple baselines
    # --------------------------------------------------------

    up_fraction = (
        y_true.mean()
    )

    always_up = np.ones_like(
        y_true
    )

    always_down = np.zeros_like(
        y_true
    )

    always_up_accuracy = (
        accuracy_score(
            y_true,
            always_up
        )
    )

    always_down_accuracy = (
        accuracy_score(
            y_true,
            always_down
        )
    )

    print()
    print("=" * 70)
    print("BASELINES")
    print("=" * 70)

    print(
        f"Actual UP frequency: "
        f"{up_fraction:.4f}"
    )

    print(
        f"Always UP accuracy : "
        f"{always_up_accuracy:.4f}"
    )

    print(
        f"Always DOWN accuracy: "
        f"{always_down_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # HMM uncertainty
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HMM SIGNAL CHARACTERISTICS")
    print("=" * 70)

    print(
        "Mean current entropy: "
        f"{df['current_entropy'].mean():.4f}"
    )

    print(
        "Mean next entropy: "
        f"{df['next_entropy'].mean():.4f}"
    )

    print(
        "Mean P(SIDE): "
        f"{df['next_prob_side'].mean():.4f}"
    )

    print(
        "Mean P(BULL): "
        f"{df['next_prob_bull'].mean():.4f}"
    )

    print(
        "Mean P(BEAR): "
        f"{df['next_prob_bear'].mean():.4f}"
    )

    print(
        "Mean raw max transition: "
        f"{df['raw_max_transition'].mean():.4f}"
    )

    print(
        "Mean shrunk max transition: "
        f"{df['shrunk_max_transition'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Probability buckets
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIRECTIONAL PROBABILITY BUCKETS")
    print("=" * 70)

    buckets = pd.cut(
        df[
            "directional_up_probability"
        ],
        bins=[
            0.0,
            0.45,
            0.50,
            0.55,
            0.60,
            0.70,
            1.01,
        ],
        include_lowest=True
    )

    bucket_table = (
        df.assign(
            probability_bucket=buckets
        )
        .groupby(
            "probability_bucket",
            observed=False
        )
        .agg(
            observations=(
                "actual_direction",
                "size"
            ),
            actual_up_rate=(
                "actual_direction",
                lambda x:
                    np.mean(
                        x == "UP"
                    )
            ),
            mean_predicted_up=(
                "directional_up_probability",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        bucket_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Convergence
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERGENCE")
    print("=" * 70)

    print(
        df[
            "convergence_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        f"\nMean iterations: "
        f"{df['iterations'].mean():.2f}"
    )

    # --------------------------------------------------------
    # Latest observations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LAST 10 SIGNALS")
    print("=" * 70)

    columns = [
        "signal_date",
        "current_regime",
        "next_regime",
        "actual_direction",
        "directional_up_probability",
        "next_prob_bull",
        "next_prob_side",
        "next_prob_bear",
        "current_entropy",
    ]

    print(
        df.tail(10)[
            columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("END HMM V5")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    results = run_oos()

    print_report(
        results
    )