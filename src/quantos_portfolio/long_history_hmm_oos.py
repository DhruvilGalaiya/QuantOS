"""
QuantOS HMM V4 - Causal Out-of-Sample Evaluation
=================================================

Purpose
-------
Evaluate whether the NIFTY 50 HMM provides useful forward regime
information out-of-sample.

V4 changes from V3
------------------
1. Proper 21-day probabilistic propagation using the transition matrix.
2. Reports:
   - one-step probability
   - terminal t+21 probability
   - average 21-day state occupancy
3. Robust convergence diagnostics.
4. Multiple random seeds.
5. Causal walk-forward fitting.
6. Transition shrinkage toward uniform distribution.
7. Probability temperature scaling.
8. Persistence baseline.
9. Accuracy, balanced accuracy, macro F1, Brier score and log loss.
10. Calibration diagnostics.

Important
---------
This script does NOT train one final model on the complete dataset.

Every OOS forecast fits a new HMM using only the previous HMM_WINDOW
observations. Future observations are never used in model fitting.

The 21-day realized regime is defined independently from future
NIFTY 50 returns:

    cumulative 21-day log return >= +5%  -> BULL
    cumulative 21-day log return <= -5%  -> BEAR
    otherwise                             -> SIDE

This target is intentionally simple and transparent.

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
    f1_score,
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

RETURNS_FILE = PORTFOLIO_DIR / "nifty50_returns.csv"
PRICE_FILE = PORTFOLIO_DIR / "nifty50_prices.csv"

OUTPUT_FILE = PORTFOLIO_DIR / "nifty50_hmm_v4_oos_results.csv"


# HMM
HMM_WINDOW = 504
N_STATES = 3

# Forward forecast horizon
FORECAST_HORIZON = 21

# Refit / evaluation frequency
SIGNAL_EVERY = 21

# Seeds for robustness
SEEDS = [7, 17, 27, 37, 47]

# HMM fitting
N_ITER = 300
TOL = 1e-3

# Feature configuration
FEATURE_COUNT = 8

# Transition shrinkage
#
# 0.00 = use raw HMM transition matrix
# 1.00 = completely uniform transitions
TRANSITION_SHRINKAGE = 0.35

# Probability temperature
#
# >1.0 makes probabilities less extreme.
# This is retained from V3 for comparability.
#
# IMPORTANT:
# This is a diagnostic calibration parameter, not an OOS-tuned
# production parameter.
PROBABILITY_TEMPERATURE = 2.5

# Regime thresholds
BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05

STATE_NAMES = ["BULL", "SIDE", "BEAR"]


# Numerical tolerance used only for convergence diagnostics.
#
# If the final EM likelihood decreases by an absolutely tiny amount
# relative to the likelihood scale, we treat it as numerical noise.
CONVERGENCE_RELATIVE_NOISE = 1e-8
CONVERGENCE_ABSOLUTE_NOISE = 1e-5


# ============================================================
# PATH / DATA HELPERS
# ============================================================
def find_input_file() -> Path:
    """
    Locate the LONG-HISTORY NIFTY 50 dataset.

    V4 requires at least:
        HMM_WINDOW + FORECAST_HORIZON

    observations, and preferably several thousand observations.

    The function therefore searches candidate files and selects
    the largest valid NIFTY 50 dataset instead of simply taking
    the first filename match.
    """

    # --------------------------------------------------------
    # Search locations
    # --------------------------------------------------------

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

    # Remove duplicates
    candidates = list(
        dict.fromkeys(candidates)
    )

    # --------------------------------------------------------
    # Remove generated HMM output files
    # --------------------------------------------------------

    excluded_names = {
        "nifty50_hmm_oos_results.csv",
        "nifty50_hmm_v4_oos_results.csv",
    }

    candidates = [
        path
        for path in candidates
        if path.name not in excluded_names
    ]

    if not candidates:

        raise FileNotFoundError(
            "\nCould not find any NIFTY 50 dataset.\n"
            f"Searched:\n"
            f"  {DATA_DIR}\n"
            f"  {PROJECT_ROOT / 'src'}\n"
        )

    # --------------------------------------------------------
    # Inspect every candidate
    # --------------------------------------------------------

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

            # Find date column
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

            # Number of usable observations
            observation_count = len(dates)

            inspected.append(
                (
                    observation_count,
                    dates.min(),
                    dates.max(),
                    path,
                    list(df.columns),
                )
            )

        except Exception:
            continue

    if not inspected:

        raise FileNotFoundError(
            "\nNIFTY 50 files were found, but none could be "
            "read as a valid historical dataset.\n"
        )

    # --------------------------------------------------------
    # Sort by number of observations.
    #
    # The largest dataset is preferred.
    # --------------------------------------------------------

    inspected.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Print what was discovered
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NIFTY 50 DATASET DISCOVERY")
    print("=" * 70)

    for (
        count,
        start_date,
        end_date,
        path,
        columns,
    ) in inspected:

        print()
        print(
            f"Observations : {count}"
        )

        print(
            f"Period       : "
            f"{start_date.date()} -> "
            f"{end_date.date()}"
        )

        print(
            f"File         : {path}"
        )

    # --------------------------------------------------------
    # Select largest dataset
    # --------------------------------------------------------

    # --------------------------------------------------------
# Select the correct NIFTY 50 dataset
#
# Prefer the exact NIFTY 50 file over NIFTY Next 50,
# even when both have the same number of observations.
# --------------------------------------------------------

preferred_names = [
    "nifty_50.parquet",
    "nifty50.parquet",
    "nifty_50.csv",
    "nifty50.csv",
]

preferred = [
    item
    for item in inspected
    if item[3].name.lower() in preferred_names
]

if preferred:
    preferred.sort(
        key=lambda x: x[0],
        reverse=True
    )

    (
        best_count,
        best_start,
        best_end,
        best_path,
        best_columns,
    ) = preferred[0]

else:
    # Fallback to the largest dataset
    (
        best_count,
        best_start,
        best_end,
        best_path,
        best_columns,
    ) = inspected[0]
    # Require enough data for V4
    minimum_required = (
        HMM_WINDOW
        +
        FORECAST_HORIZON
        +
        100
    )

    if best_count < minimum_required:

        raise ValueError(
            "\nThe largest NIFTY 50 dataset found contains only "
            f"{best_count} observations.\n\n"
            f"V4 requires at least approximately "
            f"{minimum_required} observations.\n\n"
            "The available files were:\n"
            +
            "\n".join(
                [
                    f"  {x[3]} -> {x[0]} observations"
                    for x in inspected
                ]
            )
        )

    print()
    print(
        "SELECTED LONG-HISTORY DATASET:"
    )

    print(
        f"  {best_path}"
    )

    print(
        f"  Observations: {best_count}"
    )

    print(
        f"  Period: "
        f"{best_start.date()} -> "
        f"{best_end.date()}"
    )

    print("=" * 70)

    return best_path
def load_nifty50_returns() -> pd.Series:
    """
    Load NIFTY 50 returns from an existing QuantOS CSV or Parquet
    dataset and return a clean daily log-return Series.
    """

    input_file = find_input_file()

    print()
    print(
        f"Loading data: {input_file}"
    )

    # --------------------------------------------------------
    # Read CSV or Parquet
    # --------------------------------------------------------

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

    if df.empty:
        raise ValueError(
            "Input dataset is empty."
        )

    print(
        f"Columns found: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Normalize date column
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

        # Try first column
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
    # Detect return column
    # --------------------------------------------------------

    return_candidates = [
        "log_return",
        "Log_Return",
        "log_returns",
        "Log Returns",
        "return",
        "Return",
        "returns",
        "Returns",
    ]

    return_col = None

    for col in return_candidates:

        if col in df.columns:

            return_col = col
            break

    # --------------------------------------------------------
    # If an explicit return column exists
    # --------------------------------------------------------

    if return_col is not None:

        print(
            f"Using return column: "
            f"{return_col}"
        )

        returns = pd.to_numeric(
            df[return_col],
            errors="coerce"
        )

        # Explicit log return
        if "log" in return_col.lower():

            log_returns = returns

        # Otherwise assume simple return
        else:

            log_returns = np.log1p(
                returns
            )

    # --------------------------------------------------------
    # Otherwise find a price column
    # --------------------------------------------------------

    else:

        price_candidates = [
            "close",
            "Close",
            "close_price",
            "Close Price",
            "adj_close",
            "Adj Close",
            "adjusted_close",
            "Adjusted Close",
            "price",
            "Price",
        ]

        price_col = None

        for col in price_candidates:

            if col in df.columns:

                price_col = col
                break

        # ----------------------------------------------------
        # If no standard price name exists, inspect numerics
        # ----------------------------------------------------

        if price_col is None:

            numeric_cols = (
                df.select_dtypes(
                    include=[np.number]
                )
                .columns
                .tolist()
            )

            # Remove obvious non-price columns
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
                    "\nCould not identify a NIFTY 50 "
                    "return or price column.\n\n"
                    f"Available columns:\n"
                    f"{list(df.columns)}"
                )

            price_col = numeric_cols[0]

        print(
            f"Using price column: "
            f"{price_col}"
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
                "Price series contains "
                "non-positive values."
            )

        log_returns = np.log(
            prices
        ).diff()

    # --------------------------------------------------------
    # Final cleaning
    # --------------------------------------------------------

    log_returns = pd.Series(
        log_returns,
        index=df.index,
        name="log_return"
    )

    log_returns = log_returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    log_returns = log_returns.dropna()

    log_returns.index = pd.to_datetime(
        log_returns.index
    )

    # Remove timezone if present
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

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

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

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    return (series - mean) / std.replace(0, np.nan)


def build_hmm_features(
    returns: pd.Series
) -> pd.DataFrame:
    """
    Build the 8-dimensional HMM observation vector.

    Features
    --------
    1. return_1d
    2. momentum_5d
    3. momentum_20d
    4. vol_5d
    5. vol_20d
    6. vol_60d
    7. return_zscore
    8. vol_ratio
    """

    r = returns.copy()

    features = pd.DataFrame(index=r.index)

    # 1
    features["return_1d"] = r

    # 2
    features["momentum_5d"] = r.rolling(5).sum()

    # 3
    features["momentum_20d"] = r.rolling(20).sum()

    # 4
    features["vol_5d"] = r.rolling(5).std()

    # 5
    features["vol_20d"] = r.rolling(20).std()

    # 6
    features["vol_60d"] = r.rolling(60).std()

    # 7
    features["return_zscore"] = rolling_zscore(
        r,
        60
    )

    # 8
    short_vol = r.rolling(20).std()
    long_vol = r.rolling(60).std()

    features["vol_ratio"] = (
        short_vol /
        long_vol.replace(0, np.nan)
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.dropna()

    return features


# ============================================================
# STATE MAPPING
# ============================================================

def map_states_to_regimes(
    model: GaussianHMM,
    scaler: RobustScaler,
    X_raw: np.ndarray,
) -> dict:
    """
    Map arbitrary HMM state IDs to:

        BULL
        SIDE
        BEAR

    using the mean of the 20-day momentum feature.

    Feature ordering:

        0 return_1d
        1 momentum_5d
        2 momentum_20d
        3 vol_5d
        4 vol_20d
        5 vol_60d
        6 return_zscore
        7 vol_ratio

    Mapping is based on the fitted training window only.
    """

    means_scaled = model.means_

    means_raw = scaler.inverse_transform(
        means_scaled
    )

    momentum_20 = means_raw[:, 2]

    order = np.argsort(momentum_20)

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
    """
    Convert arbitrary HMM state ordering into:

        [BULL, SIDE, BEAR]
    """

    output = np.zeros(3)

    for state_id, regime in mapping.items():

        target_index = STATE_NAMES.index(regime)

        output[target_index] = probabilities[state_id]

    return output


# ============================================================
# TRANSITION MATRIX
# ============================================================

def shrink_transition_matrix(
    transition_matrix: np.ndarray,
    shrinkage: float
) -> np.ndarray:
    """
    Shrink the HMM transition matrix toward a uniform matrix.

    This reduces excessive persistence caused by finite-sample
    estimation of highly sticky transition probabilities.
    """

    transition_matrix = np.asarray(
        transition_matrix,
        dtype=float
    )

    n = transition_matrix.shape[0]

    uniform = np.ones((n, n)) / n

    shrunk = (
        (1.0 - shrinkage) * transition_matrix
        +
        shrinkage * uniform
    )

    # Numerical normalization
    shrunk = np.clip(
        shrunk,
        1e-12,
        None
    )

    shrunk /= shrunk.sum(
        axis=1,
        keepdims=True
    )

    return shrunk


# ============================================================
# PROBABILITY TEMPERATURE
# ============================================================

def temperature_scale(
    probabilities: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Flatten probability distributions using temperature.

    temperature > 1:
        less confident

    temperature < 1:
        more confident
    """

    p = np.asarray(
        probabilities,
        dtype=float
    )

    p = np.clip(
        p,
        1e-12,
        1.0
    )

    if temperature == 1.0:
        p /= p.sum()
        return p

    log_p = np.log(p)

    scaled = log_p / temperature

    scaled -= np.max(scaled)

    exp_p = np.exp(scaled)

    exp_p /= exp_p.sum()

    return exp_p


# ============================================================
# CAUSAL FORWARD FILTER
# ============================================================

def causal_filter(
    model: GaussianHMM,
    X_scaled: np.ndarray
) -> np.ndarray:
    """
    Explicit causal forward filter.

    Computes:

        P(S_t | X_1 ... X_t)

    recursively.

    No future observations are used.
    """

    means = model.means_

    covars = model.covars_

    # --------------------------------------------------------
    # hmmlearn can expose covariance arrays differently
    # depending on covariance type.
    # --------------------------------------------------------

    if model.covariance_type == "diag":

        if covars.ndim == 3:
            variances = np.diagonal(
                covars,
                axis1=1,
                axis2=2
            )

        elif covars.ndim == 2:
            variances = covars

        else:
            raise ValueError(
                f"Unexpected covariance shape: {covars.shape}"
            )

    else:
        raise ValueError(
            "This implementation expects diag covariance."
        )

    variances = np.clip(
        variances,
        1e-8,
        None
    )

    n_states = model.n_components

    # --------------------------------------------------------
    # Initial state probabilities
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Gaussian log emission
    # --------------------------------------------------------

    def log_emission(x):

        diff = x[None, :] - means

        log_det = np.sum(
            np.log(variances),
            axis=1
        )

        mahalanobis = np.sum(
            diff ** 2 / variances,
            axis=1
        )

        dimension = x.shape[0]

        return -0.5 * (
            dimension * np.log(2.0 * np.pi)
            +
            log_det
            +
            mahalanobis
        )

    # --------------------------------------------------------
    # First observation
    # --------------------------------------------------------

    log_prob = log_emission(
        X_scaled[0]
    )

    log_alpha = (
        np.log(
            np.clip(alpha, 1e-300, None)
        )
        +
        log_prob
    )

    log_alpha -= np.max(log_alpha)

    alpha = np.exp(log_alpha)
    alpha /= alpha.sum()

    # --------------------------------------------------------
    # Remaining observations
    # --------------------------------------------------------

    transition = model.transmat_

    for t in range(1, len(X_scaled)):

        predicted = alpha @ transition

        predicted = np.clip(
            predicted,
            1e-300,
            None
        )

        predicted /= predicted.sum()

        emission = log_emission(
            X_scaled[t]
        )

        log_alpha = (
            np.log(predicted)
            +
            emission
        )

        log_alpha -= np.max(log_alpha)

        alpha = np.exp(log_alpha)

        alpha /= alpha.sum()

    return alpha


# ============================================================
# 21-DAY PROBABILITY FORECAST
# ============================================================

def propagate_probabilities(
    current_probability: np.ndarray,
    transition_matrix: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate the current filtered probability through the
    transition matrix for the complete forecast horizon.

    Returns
    -------
    terminal_probability
        P(S_{t+horizon})

    average_occupancy
        Average expected state occupancy across days 1..horizon.
    """

    p = np.asarray(
        current_probability,
        dtype=float
    )

    p /= p.sum()

    occupancy = np.zeros(
        len(p)
    )

    for _ in range(horizon):

        p = p @ transition_matrix

        p = np.clip(
            p,
            1e-12,
            None
        )

        p /= p.sum()

        occupancy += p

    average_occupancy = (
        occupancy / horizon
    )

    terminal_probability = p.copy()

    return (
        terminal_probability,
        average_occupancy
    )


# ============================================================
# REALIZED 21-DAY REGIME
# ============================================================

def realized_forward_regime(
    returns: pd.Series,
    start_position: int,
    horizon: int
) -> tuple[str | None, float | None]:

    end_position = (
        start_position + horizon
    )

    if end_position > len(returns):
        return None, None

    future_returns = returns.iloc[
        start_position:end_position
    ]

    cumulative_return = future_returns.sum()

    if cumulative_return >= BULL_THRESHOLD:

        regime = "BULL"

    elif cumulative_return <= BEAR_THRESHOLD:

        regime = "BEAR"

    else:

        regime = "SIDE"

    return (
        regime,
        float(cumulative_return)
    )


# ============================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================

def convergence_diagnostics(
    model: GaussianHMM
) -> dict:
    """
    Inspect hmmlearn's EM likelihood history.

    A tiny negative final likelihood change can occur because of
    floating-point/numerical precision.

    We distinguish:

        TRUE_CONVERGENCE
        NUMERICAL_NOISE
        NOT_CONVERGED
        UNKNOWN
    """

    history = list(
        getattr(
            model.monitor_,
            "history",
            []
        )
    )

    iterations = len(history)

    converged_flag = bool(
        getattr(
            model.monitor_,
            "converged",
            False
        )
    )

    if iterations == 0:

        return {
            "iterations": 0,
            "final_log_likelihood": np.nan,
            "final_delta": np.nan,
            "relative_delta": np.nan,
            "monitor_converged": converged_flag,
            "convergence_status": "UNKNOWN",
        }

    final_ll = float(
        history[-1]
    )

    if iterations >= 2:

        previous_ll = float(
            history[-2]
        )

        delta = final_ll - previous_ll

        scale = max(
            abs(previous_ll),
            1.0
        )

        relative_delta = abs(delta) / scale

        # Tiny negative likelihood change:
        # treat as numerical noise rather than a model failure.
        if (
            delta < 0
            and
            abs(delta) <= CONVERGENCE_ABSOLUTE_NOISE
            and
            relative_delta <= CONVERGENCE_RELATIVE_NOISE
        ):

            status = "NUMERICAL_NOISE"

        elif delta >= 0:

            status = (
                "TRUE_CONVERGENCE"
                if converged_flag
                else "LIKELIHOOD_STABLE"
            )

        else:

            status = "NOT_CONVERGED"

    else:

        delta = np.nan
        relative_delta = np.nan

        status = (
            "TRUE_CONVERGENCE"
            if converged_flag
            else "UNKNOWN"
        )

    return {
        "iterations": iterations,
        "final_log_likelihood": final_ll,
        "final_delta": delta,
        "relative_delta": relative_delta,
        "monitor_converged": converged_flag,
        "convergence_status": status,
    }


# ============================================================
# FIT SINGLE HMM
# ============================================================

def fit_single_model(
    X_train: np.ndarray,
    seed: int
):
    """
    Fit one Gaussian HMM.

    Returns
    -------
    model
    scaler
    diagnostics
    """

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

    # --------------------------------------------------------
    # hmmlearn can print convergence information directly.
    #
    # Redirect stdout/stderr so the OOS evaluator remains clean.
    # We still inspect monitor_ afterwards.
    # --------------------------------------------------------

    sink_out = io.StringIO()
    sink_err = io.StringIO()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        with redirect_stdout(sink_out), redirect_stderr(sink_err):

            model.fit(
                X_scaled
            )

    diagnostics = convergence_diagnostics(
        model
    )

    diagnostics["seed"] = seed

    diagnostics["train_score"] = float(
        model.score(X_scaled)
    )

    return (
        model,
        scaler,
        diagnostics,
        X_scaled
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

def fit_best_model(
    X_train: np.ndarray
):
    """
    Fit multiple seeds and select the model with the highest
    finite training log-likelihood.

    A model with tiny numerical convergence noise is not rejected.

    A model with a meaningful negative final EM step is still allowed
    into diagnostics but receives lower priority.
    """

    candidates = []

    for seed in SEEDS:

        try:

            (
                model,
                scaler,
                diagnostics,
                X_scaled
            ) = fit_single_model(
                X_train,
                seed
            )

            if not np.isfinite(
                diagnostics["train_score"]
            ):
                continue

            status = diagnostics[
                "convergence_status"
            ]

            # Priority:
            #
            # 2 = stable / converged
            # 1 = numerical noise
            # 0 = meaningful non-convergence
            #
            if status in (
                "TRUE_CONVERGENCE",
                "LIKELIHOOD_STABLE",
            ):

                quality = 2

            elif status == "NUMERICAL_NOISE":

                quality = 1

            else:

                quality = 0

            candidates.append(
                (
                    quality,
                    diagnostics["train_score"],
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

    # Highest convergence quality first,
    # then highest training likelihood.
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    (
        quality,
        score,
        model,
        scaler,
        diagnostics,
        X_scaled
    ) = candidates[0]

    diagnostics["candidate_count"] = len(
        candidates
    )

    diagnostics["selected_quality"] = quality

    return (
        model,
        scaler,
        diagnostics,
        X_scaled
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibration_report(
    probabilities: np.ndarray,
    actual_indices: np.ndarray,
    bins: list[tuple[float, float]]
):
    """
    Produce calibration statistics using the probability assigned
    to the actually realized class.
    """

    predicted_probability = np.max(
        probabilities,
        axis=1
    )

    predicted_class = np.argmax(
        probabilities,
        axis=1
    )

    correct = (
        predicted_class ==
        actual_indices
    )

    rows = []

    for low, high in bins:

        mask = (
            (predicted_probability >= low)
            &
            (predicted_probability < high)
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        rows.append(
            {
                "bin": f"{low:.0%}-{high:.0%}",
                "observations": count,
                "accuracy": float(
                    correct[mask].mean()
                ),
                "mean_predicted_probability": float(
                    predicted_probability[mask].mean()
                ),
            }
        )

    return rows


# ============================================================
# PROBABILITY METRICS
# ============================================================

def multiclass_brier_score(
    probabilities: np.ndarray,
    actual_indices: np.ndarray
) -> float:
    """
    Multiclass Brier score:

        mean(sum((p_k - y_k)^2))
    """

    y = np.zeros_like(
        probabilities
    )

    y[
        np.arange(len(actual_indices)),
        actual_indices
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - y) ** 2,
                axis=1
            )
        )
    )


def evaluate_predictions(
    probabilities: np.ndarray,
    actual_labels: list[str],
    title: str
) -> dict:
    """
    Evaluate probabilistic and categorical forecasts.
    """

    actual_indices = np.array(
        [
            STATE_NAMES.index(x)
            for x in actual_labels
        ]
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_labels = [
        STATE_NAMES[i]
        for i in predicted_indices
    ]

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            actual_labels,
            predicted_labels
        )
    )

    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        labels=STATE_NAMES,
        average="macro",
        zero_division=0
    )

    brier = multiclass_brier_score(
        probabilities,
        actual_indices
    )

    ll = log_loss(
        actual_indices,
        probabilities,
        labels=np.arange(
            N_STATES
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual_labels,
            predicted_labels,
            labels=STATE_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=STATE_NAMES
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Accuracy            : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy   : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1            : {macro_f1:.4f}"
    )

    print(
        f"Multiclass Brier    : {brier:.4f}"
    )

    print(
        f"Multiclass Log Loss : {ll:.4f}"
    )

    print()
    print("Per-class metrics:")

    for i, regime in enumerate(
        STATE_NAMES
    ):

        print(
            f"{regime:<6} "
            f"Precision={precision[i]:.4f} "
            f"Recall={recall[i]:.4f} "
            f"F1={f1[i]:.4f} "
            f"Support={support[i]}"
        )

    print()
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=STATE_NAMES,
            columns=STATE_NAMES
        )
    )

    return {
        "title": title,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "brier": float(
            brier
        ),
        "log_loss": float(
            ll
        ),
        "confusion_matrix": cm,
    }


# ============================================================
# MAIN OOS ENGINE
# ============================================================

def run_oos_evaluation():

    print()
    print("=" * 70)
    print("QuantOS HMM V4 - OUT-OF-SAMPLE TEST")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "This test does NOT train a final model on future data."
    )
    print(
        "Every forecast uses only the previous "
        f"{HMM_WINDOW} observations."
    )
    print(
        f"Forecast horizon: {FORECAST_HORIZON} trading days."
    )
    print(
        f"Signal frequency: every {SIGNAL_EVERY} observations."
    )

    # --------------------------------------------------------
    # Load returns
    # --------------------------------------------------------

    returns = load_nifty50_returns()

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = build_hmm_features(
        returns
    )

    # Align returns to feature dates
    aligned_returns = returns.reindex(
        features.index
    )

    valid = aligned_returns.notna()

    features = features.loc[valid]
    aligned_returns = aligned_returns.loc[valid]

    print()
    print(
        f"HMM feature observations: {len(features)}"
    )

    print(
        f"HMM features: {features.shape[1]}"
    )

    print(
        f"Feature period: "
        f"{features.index.min().date()} -> "
        f"{features.index.max().date()}"
    )

    if len(features) < (
        HMM_WINDOW + FORECAST_HORIZON
    ):
        raise ValueError(
            "Not enough observations for the requested "
            "HMM window and forecast horizon."
        )

    # --------------------------------------------------------
    # OOS storage
    # --------------------------------------------------------

    results = []

    # Start after a complete training window.
    #
    # Stop early enough that the full 21-day future target
    # exists.
    start_position = HMM_WINDOW

    final_position = (
        len(features)
        - FORECAST_HORIZON
    )

    signal_positions = range(
        start_position,
        final_position,
        SIGNAL_EVERY
    )

    # --------------------------------------------------------
    # Walk-forward OOS
    # --------------------------------------------------------

    for counter, position in enumerate(
        signal_positions,
        start=1
    ):

        signal_date = features.index[
            position
        ]

        train_start = (
            position - HMM_WINDOW
        )

        train_end = position

        X_train = features.iloc[
            train_start:train_end
        ].values

        # ----------------------------------------------------
        # Fit using only historical data
        # ----------------------------------------------------

        try:

            (
                model,
                scaler,
                diagnostics,
                X_scaled
            ) = fit_best_model(
                X_train
            )

        except Exception as exc:

            print(
                f"Skipping {signal_date.date()} "
                f"due to fitting failure: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Causal filtering
        #
        # We filter the complete training window.
        # The final alpha corresponds to:
        #
        # P(S_t | X_1 ... X_t)
        #
        # where t is the signal date.
        # ----------------------------------------------------

        current_state_prob_raw = (
            causal_filter(
                model,
                X_scaled
            )
        )

        # ----------------------------------------------------
        # Transition matrix
        # ----------------------------------------------------

        raw_transition = np.asarray(
            model.transmat_,
            dtype=float
        )

        shrunk_transition = (
            shrink_transition_matrix(
                raw_transition,
                TRANSITION_SHRINKAGE
            )
        )

        # ----------------------------------------------------
        # Reorder current probabilities
        # ----------------------------------------------------

        mapping = map_states_to_regimes(
            model,
            scaler,
            X_train
        )

        current_prob = (
            reorder_probabilities(
                current_state_prob_raw,
                mapping
            )
        )

        # ----------------------------------------------------
        # Reorder transition matrix
        #
        # Convert arbitrary HMM state IDs to:
        #
        # BULL, SIDE, BEAR
        # ----------------------------------------------------

        reordered_transition = np.zeros(
            (N_STATES, N_STATES)
        )

        for old_i, regime_i in mapping.items():

            new_i = STATE_NAMES.index(
                regime_i
            )

            for old_j, regime_j in mapping.items():

                new_j = STATE_NAMES.index(
                    regime_j
                )

                reordered_transition[
                    new_i,
                    new_j
                ] = shrunk_transition[
                    old_i,
                    old_j
                ]

        # Normalize again after reordering.
        reordered_transition /= (
            reordered_transition.sum(
                axis=1,
                keepdims=True
            )
        )

        # ----------------------------------------------------
        # 21-day propagation
        # ----------------------------------------------------

        (
            terminal_probability,
            average_occupancy
        ) = propagate_probabilities(
            current_prob,
            reordered_transition,
            FORECAST_HORIZON
        )

        # ----------------------------------------------------
        # Temperature scaling
        #
        # Applied after propagation.
        # ----------------------------------------------------

        terminal_probability_scaled = (
            temperature_scale(
                terminal_probability,
                PROBABILITY_TEMPERATURE
            )
        )

        average_occupancy_scaled = (
            temperature_scale(
                average_occupancy,
                PROBABILITY_TEMPERATURE
            )
        )

        # ----------------------------------------------------
        # One-step forecast
        #
        # Retained for comparison with V3.
        # ----------------------------------------------------

        one_step_probability = (
            current_prob
            @
            reordered_transition
        )

        one_step_probability = (
            temperature_scale(
                one_step_probability,
                PROBABILITY_TEMPERATURE
            )
        )

        # ----------------------------------------------------
        # Realized future regime
        # ----------------------------------------------------

        actual_regime, future_return = (
            realized_forward_regime(
                aligned_returns,
                position,
                FORECAST_HORIZON
            )
        )

        if actual_regime is None:
            continue

        # ----------------------------------------------------
        # Forecast regime labels
        # ----------------------------------------------------

        one_step_regime = STATE_NAMES[
            int(
                np.argmax(
                    one_step_probability
                )
            )
        ]

        terminal_regime = STATE_NAMES[
            int(
                np.argmax(
                    terminal_probability_scaled
                )
            )
        ]

        occupancy_regime = STATE_NAMES[
            int(
                np.argmax(
                    average_occupancy_scaled
                )
            )
        ]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        row = {
            "signal_date": signal_date,

            "actual_regime": actual_regime,

            "future_21d_log_return": future_return,

            # One-step
            "one_step_regime": one_step_regime,
            "one_step_prob_bull": one_step_probability[0],
            "one_step_prob_side": one_step_probability[1],
            "one_step_prob_bear": one_step_probability[2],

            # Terminal t+21
            "terminal_21d_regime": terminal_regime,
            "terminal_21d_prob_bull": (
                terminal_probability_scaled[0]
            ),
            "terminal_21d_prob_side": (
                terminal_probability_scaled[1]
            ),
            "terminal_21d_prob_bear": (
                terminal_probability_scaled[2]
            ),

            # Average occupancy
            "occupancy_regime": occupancy_regime,
            "occupancy_prob_bull": (
                average_occupancy_scaled[0]
            ),
            "occupancy_prob_side": (
                average_occupancy_scaled[1]
            ),
            "occupancy_prob_bear": (
                average_occupancy_scaled[2]
            ),

            # Current filtered state
            "current_prob_bull": current_prob[0],
            "current_prob_side": current_prob[1],
            "current_prob_bear": current_prob[2],

            # Convergence
            "selected_seed": diagnostics["seed"],
            "candidate_count": diagnostics[
                "candidate_count"
            ],
            "iterations": diagnostics[
                "iterations"
            ],
            "train_log_likelihood": diagnostics[
                "train_score"
            ],
            "final_log_likelihood": diagnostics[
                "final_log_likelihood"
            ],
            "final_delta": diagnostics[
                "final_delta"
            ],
            "relative_delta": diagnostics[
                "relative_delta"
            ],
            "monitor_converged": diagnostics[
                "monitor_converged"
            ],
            "convergence_status": diagnostics[
                "convergence_status"
            ],

            # Transition diagnostics
            "raw_max_transition": float(
                np.max(raw_transition)
            ),
            "shrunk_max_transition": float(
                np.max(reordered_transition)
            ),
        }

        results.append(
            row
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            counter <= 5
            or counter % 10 == 0
        ):

            print(
                f"[{counter:03d}] "
                f"{signal_date.date()} | "
                f"Actual={actual_regime:<4} | "
                f"1D={one_step_regime:<4} | "
                f"21D={terminal_regime:<4} | "
                f"Occ={occupancy_regime:<4} | "
                f"P21={terminal_probability_scaled.max():.3f} | "
                f"{diagnostics['convergence_status']}"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise RuntimeError(
            "No OOS forecasts were generated."
        )

    result_df["signal_date"] = pd.to_datetime(
        result_df["signal_date"]
    )

    result_df = result_df.sort_values(
        "signal_date"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
# REPORTING
# ============================================================

def print_summary(
    result_df: pd.DataFrame
):

    print()
    print()
    print("=" * 70)
    print("HMM V4 OOS SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Forecast observations: "
        f"{len(result_df)}"
    )

    print(
        f"OOS date range: "
        f"{result_df['signal_date'].min().date()} -> "
        f"{result_df['signal_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Actual distribution
    # --------------------------------------------------------

    print()
    print("Actual regime distribution:")

    actual_counts = (
        result_df[
            "actual_regime"
        ]
        .value_counts()
        .reindex(
            STATE_NAMES,
            fill_value=0
        )
    )

    print(
        actual_counts.to_string()
    )

    # --------------------------------------------------------
    # Forecast distributions
    # --------------------------------------------------------

    print()
    print("Terminal 21-day forecast distribution:")

    terminal_counts = (
        result_df[
            "terminal_21d_regime"
        ]
        .value_counts()
        .reindex(
            STATE_NAMES,
            fill_value=0
        )
    )

    print(
        terminal_counts.to_string()
    )

    print()
    print("Average occupancy forecast distribution:")

    occupancy_counts = (
        result_df[
            "occupancy_regime"
        ]
        .value_counts()
        .reindex(
            STATE_NAMES,
            fill_value=0
        )
    )

    print(
        occupancy_counts.to_string()
    )

    # --------------------------------------------------------
    # Evaluation datasets
    # --------------------------------------------------------

    actual_labels = result_df[
        "actual_regime"
    ].tolist()

    terminal_probabilities = (
        result_df[
            [
                "terminal_21d_prob_bull",
                "terminal_21d_prob_side",
                "terminal_21d_prob_bear",
            ]
        ]
        .values
    )

    occupancy_probabilities = (
        result_df[
            [
                "occupancy_prob_bull",
                "occupancy_prob_side",
                "occupancy_prob_bear",
            ]
        ]
        .values
    )

    one_step_probabilities = (
        result_df[
            [
                "one_step_prob_bull",
                "one_step_prob_side",
                "one_step_prob_bear",
            ]
        ]
        .values
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    one_step_metrics = evaluate_predictions(
        one_step_probabilities,
        actual_labels,
        "ONE-STEP FORECAST vs 21-DAY REALIZED REGIME"
    )

    terminal_metrics = evaluate_predictions(
        terminal_probabilities,
        actual_labels,
        "TERMINAL 21-DAY FORECAST"
    )

    occupancy_metrics = evaluate_predictions(
        occupancy_probabilities,
        actual_labels,
        "AVERAGE 21-DAY OCCUPANCY FORECAST"
    )

    # --------------------------------------------------------
    # Persistence baseline
    #
    # Previous realized regime is used as the forecast.
    #
    # Since signals are every 21 observations, this corresponds
    # to asking whether the previous observed 21-day regime
    # persists into the next 21-day window.
    # --------------------------------------------------------

    persistence_labels = []

    for i in range(
        len(result_df)
    ):

        if i == 0:

            persistence_labels.append(
                None
            )

        else:

            persistence_labels.append(
                result_df.iloc[
                    i - 1
                ][
                    "actual_regime"
                ]
            )

    persistence_mask = np.array(
        [
            x is not None
            for x in persistence_labels
        ]
    )

    persistence_actual = np.array(
        actual_labels,
        dtype=object
    )[persistence_mask]

    persistence_predicted = np.array(
        persistence_labels,
        dtype=object
    )[persistence_mask]

    persistence_accuracy = (
        accuracy_score(
            persistence_actual,
            persistence_predicted
        )
    )

    persistence_balanced = (
        balanced_accuracy_score(
            persistence_actual,
            persistence_predicted
        )
    )

    persistence_f1 = (
        f1_score(
            persistence_actual,
            persistence_predicted,
            labels=STATE_NAMES,
            average="macro",
            zero_division=0
        )
    )

    print()
    print("=" * 70)
    print("PERSISTENCE BASELINE")
    print("=" * 70)

    print(
        f"Accuracy            : "
        f"{persistence_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{persistence_balanced:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{persistence_f1:.4f}"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    bins = [
        (0.0, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]

    actual_indices = np.array(
        [
            STATE_NAMES.index(x)
            for x in actual_labels
        ]
    )

    print()
    print("=" * 70)
    print("TERMINAL 21-DAY CALIBRATION")
    print("=" * 70)

    terminal_calibration = (
        calibration_report(
            terminal_probabilities,
            actual_indices,
            bins
        )
    )

    if terminal_calibration:

        print(
            pd.DataFrame(
                terminal_calibration
            ).to_string(
                index=False
            )
        )

    else:

        print(
            "No calibration bins available."
        )

    print()
    print("=" * 70)
    print("AVERAGE 21-DAY OCCUPANCY CALIBRATION")
    print("=" * 70)

    occupancy_calibration = (
        calibration_report(
            occupancy_probabilities,
            actual_indices,
            bins
        )
    )

    if occupancy_calibration:

        print(
            pd.DataFrame(
                occupancy_calibration
            ).to_string(
                index=False
            )
        )

    else:

        print(
            "No calibration bins available."
        )

    # --------------------------------------------------------
    # Convergence summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)

    convergence_counts = (
        result_df[
            "convergence_status"
        ]
        .value_counts()
    )

    print(
        convergence_counts.to_string()
    )

    print()
    print(
        "Mean iterations: "
        f"{result_df['iterations'].mean():.2f}"
    )

    print(
        "Maximum iterations: "
        f"{result_df['iterations'].max()}"
    )

    finite_delta = result_df[
        "final_delta"
    ].dropna()

    if len(finite_delta):

        print(
            "Median final likelihood delta: "
            f"{finite_delta.median():.8f}"
        )

        print(
            "Minimum final likelihood delta: "
            f"{finite_delta.min():.8f}"
        )

        print(
            "Maximum final likelihood delta: "
            f"{finite_delta.max():.8f}"
        )

    # --------------------------------------------------------
    # Probability diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROBABILITY DIAGNOSTICS")
    print("=" * 70)

    terminal_max = terminal_probabilities.max(
        axis=1
    )

    occupancy_max = occupancy_probabilities.max(
        axis=1
    )

    one_step_max = one_step_probabilities.max(
        axis=1
    )

    print(
        "One-step mean max probability: "
        f"{one_step_max.mean():.4f}"
    )

    print(
        "Terminal 21D mean max probability: "
        f"{terminal_max.mean():.4f}"
    )

    print(
        "Terminal 21D max probability: "
        f"{terminal_max.max():.4f}"
    )

    print(
        "Occupancy mean max probability: "
        f"{occupancy_max.mean():.4f}"
    )

    print(
        "Occupancy max probability: "
        f"{occupancy_max.max():.4f}"
    )

    print(
        "Mean raw max transition: "
        f"{result_df['raw_max_transition'].mean():.4f}"
    )

    print(
        "Mean shrunk max transition: "
        f"{result_df['shrunk_max_transition'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Last 10 observations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LAST 10 OOS FORECASTS")
    print("=" * 70)

    last = result_df.tail(
        10
    ).copy()

    display_columns = [
        "signal_date",
        "actual_regime",
        "future_21d_log_return",
        "one_step_regime",
        "terminal_21d_regime",
        "occupancy_regime",
        "terminal_21d_prob_bull",
        "terminal_21d_prob_side",
        "terminal_21d_prob_bear",
    ]

    print(
        last[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final conclusion block
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"OOS results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("END HMM V4 OOS TEST")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    result_df = run_oos_evaluation()

    print_summary(
        result_df
    )