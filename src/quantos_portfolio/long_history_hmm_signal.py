from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler


# ============================================================
# QUANTOS CAUSAL HMM V3
# ============================================================
#
# Purpose:
#   Long-history causal market-regime inference for NIFTY 50.
#
# Important design principles:
#
#   1. No future observations are used.
#   2. No artificial max-probability clipping.
#   3. No raw sticky transition matrix is trusted directly.
#   4. Transition probabilities are shrunk toward uniformity.
#   5. HMM probabilities are temperature-calibrated.
#   6. Multiple restarts are used.
#   7. Convergence is explicitly checked.
#   8. Calibration is evaluated out-of-sample.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

HMM_WINDOW = 504

N_STATES = 3

REBALANCE_EVERY = 21

RANDOM_SEEDS = [
    7,
    17,
    27,
    37,
    47,
    57,
    67,
    77,
]

MIN_COVAR = 1e-4

# ------------------------------------------------------------
# Transition shrinkage
# ------------------------------------------------------------
#
# 0.0 = trust HMM transition matrix completely
# 1.0 = completely uniform transitions
#
# We use moderate shrinkage rather than forcing a hard cap.
#
TRANSITION_SHRINKAGE = 0.35


# ------------------------------------------------------------
# Probability temperature
# ------------------------------------------------------------
#
# Temperature > 1 makes probabilities less confident.
#
# We do NOT use an arbitrary cap such as 0.94.
#
PROBABILITY_TEMPERATURE = 2.50


# ------------------------------------------------------------
# Convergence
# ------------------------------------------------------------

HMM_MAX_ITER = 1000

HMM_TOL = 1e-3


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_hmm_features(
    returns: pd.Series,
) -> pd.DataFrame:

    returns = (
        returns
        .astype(float)
        .sort_index()
    )

    features = pd.DataFrame(
        index=returns.index
    )

    # --------------------------------------------------------
    # Return / momentum features
    # --------------------------------------------------------

    features["return_1d"] = returns

    features["momentum_5d"] = (
        returns
        .rolling(5)
        .sum()
    )

    features["momentum_20d"] = (
        returns
        .rolling(20)
        .sum()
    )

    # --------------------------------------------------------
    # Volatility features
    # --------------------------------------------------------

    features["vol_5d"] = (
        returns
        .rolling(5)
        .std()
    )

    features["vol_20d"] = (
        returns
        .rolling(20)
        .std()
    )

    features["vol_60d"] = (
        returns
        .rolling(60)
        .std()
    )

    # --------------------------------------------------------
    # Return z-score
    # --------------------------------------------------------

    rolling_mean = (
        returns
        .rolling(20)
        .mean()
    )

    rolling_std = (
        returns
        .rolling(20)
        .std()
    )

    features["return_zscore"] = (
        (
            returns
            - rolling_mean
        )
        /
        rolling_std.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Volatility ratio
    # --------------------------------------------------------

    features["vol_ratio"] = (
        features["vol_5d"]
        /
        features["vol_20d"].replace(
            0,
            np.nan,
        )
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


# ============================================================
# HMM FITTING
# ============================================================

def fit_best_hmm(
    X: np.ndarray,
    seeds: list[int] | None = None,
) -> tuple[GaussianHMM, RobustScaler, float]:

    if seeds is None:
        seeds = RANDOM_SEEDS

    # --------------------------------------------------------
    # RobustScaler is less sensitive to extreme market days
    # than ordinary StandardScaler.
    # --------------------------------------------------------

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    if not np.isfinite(
        X_scaled
    ).all():

        raise ValueError(
            "Non-finite values found "
            "after feature scaling."
        )

    best_model = None
    best_score = -np.inf

    successful_models = 0

    for seed in seeds:

        # ----------------------------------------------------
        # Deterministic initialization
        #
        # This avoids some of the unstable random parameter
        # initialization that can produce poor EM trajectories.
        # ----------------------------------------------------

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=HMM_MAX_ITER,
            tol=HMM_TOL,
            random_state=seed,
            min_covar=MIN_COVAR,
            init_params="stmc",
            params="stmc",
            verbose=False,
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            try:

                model.fit(
                    X_scaled
                )

                score = float(
                    model.score(
                        X_scaled
                    )
                )

                if not np.isfinite(
                    score
                ):
                    continue

                # ------------------------------------------------
                # Check actual convergence flag.
                #
                # A model that technically fits but does not
                # converge should not automatically become the
                # production candidate.
                # ------------------------------------------------

                converged = bool(
                    getattr(
                        model.monitor_,
                        "converged",
                        False,
                    )
                )

                if not converged:
                    continue

                successful_models += 1

                if score > best_score:

                    best_score = score
                    best_model = model

            except Exception:

                continue

    # --------------------------------------------------------
    # Fallback:
    #
    # If none of the models reports convergence, fit one final
    # conservative model with a larger tolerance.
    #
    # This is preferable to silently accepting arbitrary EM
    # output.
    # --------------------------------------------------------

    if best_model is None:

        fallback = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=1500,
            tol=5e-3,
            random_state=7,
            min_covar=MIN_COVAR,
            init_params="stmc",
            params="stmc",
            verbose=False,
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            fallback.fit(
                X_scaled
            )

        fallback_score = float(
            fallback.score(
                X_scaled
            )
        )

        if not np.isfinite(
            fallback_score
        ):

            raise RuntimeError(
                "HMM fitting failed for "
                "all random seeds and "
                "fallback configuration."
            )

        best_model = fallback
        best_score = fallback_score

    return (
        best_model,
        scaler,
        best_score,
    )


# ============================================================
# HMM GAUSSIAN EMISSION PROBABILITIES
# ============================================================

def gaussian_log_emission_probabilities(
    X: np.ndarray,
    model: GaussianHMM,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    means = np.asarray(
        model.means_,
        dtype=np.float64,
    )

    covars = np.asarray(
        model.covars_,
        dtype=np.float64,
    )

    n_observations = X.shape[0]
    n_features = X.shape[1]
    n_states = means.shape[0]

    log_probs = np.zeros(
        (
            n_observations,
            n_states,
        ),
        dtype=np.float64,
    )

    # ========================================================
    # DIAGONAL COVARIANCE
    # ========================================================

    if model.covariance_type == "diag":

        if covars.ndim == 3:

            variances = np.diagonal(
                covars,
                axis1=1,
                axis2=2,
            )

        elif covars.ndim == 2:

            variances = covars

        else:

            raise ValueError(
                "Unexpected covariance shape: "
                f"{covars.shape}"
            )

        variances = np.maximum(
            variances,
            MIN_COVAR,
        )

        constant = (
            n_features
            * np.log(
                2.0 * np.pi
            )
        )

        for state in range(
            n_states
        ):

            diff = (
                X
                - means[state]
            )

            log_det = np.sum(
                np.log(
                    variances[state]
                )
            )

            quadratic = np.sum(
                (
                    diff ** 2
                )
                /
                variances[state],
                axis=1,
            )

            log_probs[:, state] = (
                -0.5
                * (
                    constant
                    + log_det
                    + quadratic
                )
            )

        return log_probs

    raise ValueError(
        "Unsupported covariance type: "
        f"{model.covariance_type}"
    )


# ============================================================
# LOG-SUM-EXP
# ============================================================

def logsumexp(
    values: np.ndarray,
) -> float:

    maximum = np.max(
        values
    )

    return (
        maximum
        +
        np.log(
            np.sum(
                np.exp(
                    values - maximum
                )
            )
        )
    )


# ============================================================
# TRANSITION SHRINKAGE
# ============================================================

def shrink_transition_matrix(
    transition_matrix: np.ndarray,
    shrinkage: float = TRANSITION_SHRINKAGE,
) -> np.ndarray:

    transition_matrix = np.asarray(
        transition_matrix,
        dtype=np.float64,
    )

    transition_matrix = np.maximum(
        transition_matrix,
        0.0,
    )

    transition_matrix /= (
        transition_matrix.sum(
            axis=1,
            keepdims=True,
        )
    )

    uniform = np.full(
        transition_matrix.shape,
        1.0 / transition_matrix.shape[1],
    )

    shrunk = (
        (1.0 - shrinkage)
        * transition_matrix
        +
        shrinkage
        * uniform
    )

    shrunk /= (
        shrunk.sum(
            axis=1,
            keepdims=True,
        )
    )

    return shrunk


# ============================================================
# CAUSAL FORWARD FILTER
# ============================================================

def causal_forward_filter(
    X: np.ndarray,
    model: GaussianHMM,
    transition_matrix: np.ndarray,
) -> np.ndarray:

    """
    Calculates:

        P(S_t | X_1 ... X_t)

    using only information available through time t.
    """

    log_emissions = (
        gaussian_log_emission_probabilities(
            X,
            model,
        )
    )

    start_prob = np.asarray(
        model.startprob_,
        dtype=np.float64,
    )

    start_prob = np.maximum(
        start_prob,
        1e-12,
    )

    start_prob /= (
        start_prob.sum()
    )

    log_transition = np.log(
        np.maximum(
            transition_matrix,
            1e-12,
        )
    )

    n_observations = X.shape[0]

    n_states = transition_matrix.shape[0]

    filtered = np.zeros(
        (
            n_observations,
            n_states,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # First observation
    # --------------------------------------------------------

    log_alpha = (
        np.log(start_prob)
        +
        log_emissions[0]
    )

    normalization = logsumexp(
        log_alpha
    )

    log_alpha -= normalization

    filtered[0] = np.exp(
        log_alpha
    )

    # --------------------------------------------------------
    # Forward recursion
    # --------------------------------------------------------

    for t in range(
        1,
        n_observations,
    ):

        next_log_alpha = np.zeros(
            n_states,
            dtype=np.float64,
        )

        for state in range(
            n_states
        ):

            transition_terms = (
                log_alpha
                +
                log_transition[
                    :,
                    state,
                ]
            )

            next_log_alpha[state] = (
                logsumexp(
                    transition_terms
                )
                +
                log_emissions[
                    t,
                    state,
                ]
            )

        normalization = logsumexp(
            next_log_alpha
        )

        log_alpha = (
            next_log_alpha
            -
            normalization
        )

        filtered[t] = np.exp(
            log_alpha
        )

    return filtered


# ============================================================
# TEMPERATURE CALIBRATION
# ============================================================

def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.maximum(
        probabilities,
        1e-12,
    )

    probabilities /= (
        probabilities.sum()
    )

    logits = np.log(
        probabilities
    )

    logits /= temperature

    logits -= np.max(
        logits
    )

    calibrated = np.exp(
        logits
    )

    calibrated /= (
        calibrated.sum()
    )

    return calibrated


# ============================================================
# STATE MAPPING
# ============================================================

def map_states_to_regimes(
    model: GaussianHMM,
    scaler: RobustScaler,
) -> dict[int, str]:

    """
    Map latent HMM states according to estimated return.

    Lowest return  -> BEAR
    Middle return  -> SIDE
    Highest return -> BULL
    """

    scaled_return_mean = (
        model.means_[:, 0]
    )

    # RobustScaler transformation:
    #
    # x_scaled = (x - center) / scale

    return_mean = (
        scaled_return_mean
        * scaler.scale_[0]
        +
        scaler.center_[0]
    )

    ordering = np.argsort(
        return_mean
    )

    return {
        int(ordering[0]): "BEAR",
        int(ordering[1]): "SIDE",
        int(ordering[2]): "BULL",
    }


# ============================================================
# ENTROPY
# ============================================================

def probability_entropy(
    probabilities: np.ndarray,
) -> float:

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.maximum(
        probabilities,
        1e-12,
    )

    probabilities /= (
        probabilities.sum()
    )

    return float(
        -np.sum(
            probabilities
            *
            np.log(
                probabilities
            )
        )
    )


# ============================================================
# SINGLE HMM SIGNAL
# ============================================================

def calculate_hmm_signal(
    feature_data: pd.DataFrame,
    signal_date: pd.Timestamp,
    window: int = HMM_WINDOW,
) -> dict | None:

    if signal_date not in feature_data.index:

        return None

    end_position = (
        feature_data.index.get_loc(
            signal_date
        )
    )

    if not isinstance(
        end_position,
        (int, np.integer),
    ):

        raise ValueError(
            "Signal date does not map "
            "to a unique index position."
        )

    if end_position < (
        window - 1
    ):

        return None

    start_position = (
        end_position
        -
        window
        +
        1
    )

    training_data = (
        feature_data
        .iloc[
            start_position:
            end_position + 1
        ]
        .dropna()
    )

    if len(training_data) < (
        window * 0.90
    ):

        return None

    X = training_data.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            "Non-finite HMM input."
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    (
        model,
        scaler,
        log_likelihood,
    ) = fit_best_hmm(
        X
    )

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    raw_transition_matrix = (
        np.asarray(
            model.transmat_,
            dtype=np.float64,
        )
    )

    transition_matrix = (
        shrink_transition_matrix(
            raw_transition_matrix
        )
    )

    # --------------------------------------------------------
    # CAUSAL FILTER
    # --------------------------------------------------------

    filtered = (
        causal_forward_filter(
            X_scaled,
            model,
            transition_matrix,
        )
    )

    raw_current_probability = (
        filtered[-1]
    )

    # --------------------------------------------------------
    # CALIBRATE CURRENT PROBABILITY
    # --------------------------------------------------------

    calibrated_current_probability = (
        apply_temperature(
            raw_current_probability,
            PROBABILITY_TEMPERATURE,
        )
    )

    # --------------------------------------------------------
    # NEXT STATE
    # --------------------------------------------------------

    raw_next_probability = (
        calibrated_current_probability
        @ transition_matrix
    )

    calibrated_next_probability = (
        apply_temperature(
            raw_next_probability,
            PROBABILITY_TEMPERATURE,
        )
    )

    # --------------------------------------------------------
    # STATE MAPPING
    # --------------------------------------------------------

    state_mapping = (
        map_states_to_regimes(
            model,
            scaler,
        )
    )

    current_probabilities = {
        state_mapping[state]:
        float(
            calibrated_current_probability[
                state
            ]
        )
        for state in range(
            N_STATES
        )
    }

    next_probabilities = {
        state_mapping[state]:
        float(
            calibrated_next_probability[
                state
            ]
        )
        for state in range(
            N_STATES
        )
    }

    current_regime = max(
        current_probabilities,
        key=current_probabilities.get,
    )

    next_regime = max(
        next_probabilities,
        key=next_probabilities.get,
    )

    return {
        "date":
            signal_date,

        "current_regime":
            current_regime,

        "next_regime":
            next_regime,

        "bull_probability":
            current_probabilities[
                "BULL"
            ],

        "side_probability":
            current_probabilities[
                "SIDE"
            ],

        "bear_probability":
            current_probabilities[
                "BEAR"
            ],

        "next_bull_probability":
            next_probabilities[
                "BULL"
            ],

        "next_side_probability":
            next_probabilities[
                "SIDE"
            ],

        "next_bear_probability":
            next_probabilities[
                "BEAR"
            ],

        "next_regime_probability":
            float(
                max(
                    next_probabilities.values()
                )
            ),

        "raw_max_probability":
            float(
                np.max(
                    raw_current_probability
                )
            ),

        "probability_entropy":
            probability_entropy(
                calibrated_current_probability
            ),

        "training_observations":
            len(training_data),

        "log_likelihood":
            float(
                log_likelihood
            ),

        "max_raw_transition_probability":
            float(
                np.max(
                    raw_transition_matrix
                )
            ),

        "max_shrunk_transition_probability":
            float(
                np.max(
                    transition_matrix
                )
            ),

        "state_mapping":
            str(
                state_mapping
            ),
    }


# ============================================================
# SIGNAL DATE SELECTION
# ============================================================

def build_signal_dates(
    feature_data: pd.DataFrame,
    every: int = REBALANCE_EVERY,
) -> pd.DatetimeIndex:

    dates = feature_data.index

    if len(dates) < HMM_WINDOW:

        return pd.DatetimeIndex([])

    usable_dates = dates[
        HMM_WINDOW - 1:
    ]

    selected = list(
        usable_dates[::every]
    )

    final_date = usable_dates[-1]

    if selected[-1] != final_date:

        selected.append(
            final_date
        )

    return pd.DatetimeIndex(
        selected
    )


# ============================================================
# BUILD SIGNAL TABLE
# ============================================================

def build_long_history_hmm_signals(
    rebalance_every: int = REBALANCE_EVERY,
) -> pd.DataFrame:

    from .long_history_universe import (
        build_long_history_returns,
    )

    returns = (
        build_long_history_returns()
    )

    if "NIFTY_50" not in returns.columns:

        raise ValueError(
            "NIFTY_50 not found."
        )

    nifty_returns = (
        returns[
            "NIFTY_50"
        ]
        .dropna()
        .sort_index()
    )

    features = build_hmm_features(
        nifty_returns
    )

    feature_data = (
        features
        .dropna()
    )

    signal_dates = (
        build_signal_dates(
            feature_data,
            every=rebalance_every,
        )
    )

    if len(signal_dates) == 0:

        raise RuntimeError(
            "No HMM signal dates available."
        )

    print("=" * 80)
    print(
        "QUANTOS LONG-HISTORY "
        "CAUSAL HMM V3"
    )
    print("=" * 80)

    print(
        "\nMarket:\nNIFTY_50"
    )

    print(
        "\nFeatures:"
    )

    print(
        features.columns.tolist()
    )

    print(
        "\nHMM states:"
    )

    print(
        "3 states -> BEAR / SIDE / BULL"
    )

    print(
        "\nTraining window:"
    )

    print(
        HMM_WINDOW
    )

    print(
        "\nSignal frequency:"
    )

    print(
        rebalance_every,
        "observations",
    )

    print(
        "\nTransition shrinkage:"
    )

    print(
        TRANSITION_SHRINKAGE
    )

    print(
        "\nProbability temperature:"
    )

    print(
        PROBABILITY_TEMPERATURE
    )

    print(
        "\nSignal dates:"
    )

    print(
        signal_dates[0].date(),
        "->",
        signal_dates[-1].date(),
    )

    print(
        "\nGenerating causal signals..."
    )

    results = []

    for count, signal_date in enumerate(
        signal_dates,
        start=1,
    ):

        result = calculate_hmm_signal(
            feature_data,
            signal_date,
            window=HMM_WINDOW,
        )

        if result is not None:

            results.append(
                result
            )

        if count % 25 == 0:

            print(
                f"Processed {count} / "
                f"{len(signal_dates)}"
            )

    if not results:

        raise RuntimeError(
            "No valid HMM signals generated."
        )

    signals = pd.DataFrame(
        results
    )

    signals = (
        signals
        .sort_values("date")
        .reset_index(drop=True)
    )

    return signals


# ============================================================
# EMPIRICAL CALIBRATION
# ============================================================

def calculate_calibration_diagnostic(
    signals: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for i in range(
        len(signals) - 1
    ):

        current = signals.iloc[i]

        actual_next = (
            signals.iloc[
                i + 1
            ]["current_regime"]
        )

        predicted = (
            current["next_regime"]
        )

        probability = (
            current[
                "next_regime_probability"
            ]
        )

        rows.append(
            {
                "date":
                    current["date"],

                "predicted_regime":
                    predicted,

                "actual_regime":
                    actual_next,

                "predicted_probability":
                    probability,

                "correct":
                    int(
                        predicted
                        == actual_next
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def calibration_table(
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:

    if diagnostic.empty:

        return pd.DataFrame()

    bins = [
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    labels = [
        "<50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    diagnostic = diagnostic.copy()

    diagnostic["bin"] = pd.cut(
        diagnostic[
            "predicted_probability"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    table = (
        diagnostic
        .groupby(
            "bin",
            observed=False,
        )
        .agg(
            observations=(
                "correct",
                "size",
            ),

            empirical_accuracy=(
                "correct",
                "mean",
            ),

            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
        )
    )

    return table


# ============================================================
# VALIDATION
# ============================================================

def validate_long_history_hmm():

    signals = (
        build_long_history_hmm_signals()
    )

    print("\n")
    print("=" * 80)
    print(
        "HMM V3 SIGNAL VALIDATION"
    )
    print("=" * 80)

    print(
        "\nSignal shape:"
    )

    print(
        signals.shape
    )

    print(
        "\nDate range:"
    )

    print(
        signals["date"].min().date(),
        "->",
        signals["date"].max().date(),
    )

    print(
        "\nRegime counts:"
    )

    print(
        signals[
            "current_regime"
        ].value_counts()
    )

    # --------------------------------------------------------
    # LATEST
    # --------------------------------------------------------

    latest = signals.iloc[-1]

    print(
        "\nLatest signal:"
    )

    print(
        "Date:",
        latest["date"].date(),
    )

    print(
        "Current regime:",
        latest["current_regime"],
    )

    print(
        "Next regime:",
        latest["next_regime"],
    )

    print(
        "\nCurrent probabilities:"
    )

    print(
        "  BULL:",
        f"{latest['bull_probability']:.4f}",
    )

    print(
        "  SIDE:",
        f"{latest['side_probability']:.4f}",
    )

    print(
        "  BEAR:",
        f"{latest['bear_probability']:.4f}",
    )

    print(
        "\nNext probabilities:"
    )

    print(
        "  BULL:",
        f"{latest['next_bull_probability']:.4f}",
    )

    print(
        "  SIDE:",
        f"{latest['next_side_probability']:.4f}",
    )

    print(
        "  BEAR:",
        f"{latest['next_bear_probability']:.4f}",
    )

    print(
        "\nNext regime probability:",
        f"{latest['next_regime_probability']:.4f}",
    )

    print(
        "\nRaw maximum probability:",
        f"{latest['raw_max_probability']:.4f}",
    )

    print(
        "Probability entropy:",
        f"{latest['probability_entropy']:.4f}",
    )

    print(
        "\nRaw maximum transition probability:",
        f"{latest['max_raw_transition_probability']:.4f}",
    )

    print(
        "Shrunk maximum transition probability:",
        f"{latest['max_shrunk_transition_probability']:.4f}",
    )

    print(
        "Training observations:",
        int(
            latest[
                "training_observations"
            ]
        ),
    )

    # --------------------------------------------------------
    # PROBABILITY SUM
    # --------------------------------------------------------

    current_sum = (
        signals[
            [
                "bull_probability",
                "side_probability",
                "bear_probability",
            ]
        ]
        .sum(axis=1)
    )

    next_sum = (
        signals[
            [
                "next_bull_probability",
                "next_side_probability",
                "next_bear_probability",
            ]
        ]
        .sum(axis=1)
    )

    print(
        "\nProbability validation:"
    )

    print(
        "Current probabilities sum to 1:",
        np.allclose(
            current_sum,
            1.0,
            atol=1e-6,
        ),
    )

    print(
        "Next probabilities sum to 1:",
        np.allclose(
            next_sum,
            1.0,
            atol=1e-6,
        ),
    )

    print(
        "All probabilities finite:",
        np.isfinite(
            signals[
                [
                    "bull_probability",
                    "side_probability",
                    "bear_probability",
                    "next_bull_probability",
                    "next_side_probability",
                    "next_bear_probability",
                ]
            ].to_numpy()
        ).all(),
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    diagnostic = (
        calculate_calibration_diagnostic(
            signals
        )
    )

    print(
        "\nEmpirical probability calibration:"
    )

    table = calibration_table(
        diagnostic
    )

    if table.empty:

        print(
            "No calibration observations."
        )

    else:

        print(
            table.round(4)
            .to_string()
        )

    # --------------------------------------------------------
    # OVERALL EMPIRICAL ACCURACY
    # --------------------------------------------------------

    if not diagnostic.empty:

        accuracy = (
            diagnostic[
                "correct"
            ].mean()
        )

        mean_probability = (
            diagnostic[
                "predicted_probability"
            ].mean()
        )

        print(
            "\nOverall empirical accuracy:",
            f"{accuracy:.4f}",
        )

        print(
            "Mean predicted probability:",
            f"{mean_probability:.4f}",
        )

        print(
            "Calibration gap:",
            f"{mean_probability - accuracy:.4f}",
        )

    # --------------------------------------------------------
    # LAST 10
    # --------------------------------------------------------

    print(
        "\nLast 10 signals:"
    )

    print(
        signals[
            [
                "date",
                "current_regime",
                "next_regime",
                "next_regime_probability",
                "raw_max_probability",
                "probability_entropy",
                "max_raw_transition_probability",
                "max_shrunk_transition_probability",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    validate_long_history_hmm()