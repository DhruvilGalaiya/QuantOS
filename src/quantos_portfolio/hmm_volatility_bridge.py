"""
QuantOS HMM ↔ Volatility Bridge
================================

Purpose
-------
1. Build the same 8-feature HMM regime representation used by QuantOS.
2. Perform causal walk-forward HMM inference.
3. Evaluate forward realized volatility conditional on regime.
4. Calculate the CURRENT market regime using the latest available
   observations without requiring future data.
5. Save:
      - hmm_volatility_bridge_timeseries.csv
      - hmm_volatility_bridge_summary.csv
      - current_quantos_market_state.csv

Methodology
-----------
- 3-state Gaussian HMM
- 504-observation rolling training window
- RobustScaler
- diagonal covariance
- causal forward filtering
- transition-matrix shrinkage toward uniform
- probability temperature
- semantic state mapping using 20-day momentum
- no future data used in current-state calculation
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM


# ================================================================
# CONFIGURATION
# ================================================================

HMM_WINDOW = 504
N_STATES = 3

RANDOM_SEEDS = [7, 17, 27, 37, 47]

TRANSITION_SHRINKAGE = 0.35
PROBABILITY_TEMPERATURE = 2.5

FORECAST_5D = 5
FORECAST_21D = 21

FEATURE_COLUMNS = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "return_zscore",
    "vol_ratio",
]

OUTPUT_DIR = Path(
    "data/portfolio/hmm_volatility_bridge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# DATA LOCATION
# ================================================================

POSSIBLE_FILES = [
    Path("data/regime/daily/nifty_50.parquet"),
    Path("data/regime/daily/NIFTY_50.parquet"),
    Path("data/regime/nifty_50.parquet"),
    Path("data/regime/NIFTY_50.parquet"),
    Path("data/nifty_50.parquet"),
    Path("data/NIFTY_50.parquet"),
]


# ================================================================
# HELPERS
# ================================================================

def find_nifty50_file():
    """
    Locate the exact NIFTY 50 daily dataset.
    """

    for path in POSSIBLE_FILES:

        if path.exists():

            print(
                f"NIFTY 50 data file : {path}"
            )

            return path

    # Recursive fallback, but explicitly prefer filenames
    # containing nifty_50 rather than accidentally selecting
    # another index such as NIFTY Next 50.

    candidates = []

    for root in [
        Path("data"),
        Path("."),
    ]:

        if not root.exists():
            continue

        for path in root.rglob("*.parquet"):

            name = path.name.lower()

            if (
                "nifty_50" in name
                or "nifty50" in name
            ):

                candidates.append(path)

    if candidates:

        candidates = sorted(
            candidates,
            key=lambda x: len(str(x))
        )

        print(
            f"NIFTY 50 data file : {candidates[0]}"
        )

        return candidates[0]

    raise FileNotFoundError(
        "Could not locate NIFTY 50 parquet data. "
        "Expected something such as "
        "data/regime/daily/nifty_50.parquet"
    )


def normalize_datetime_index(df):
    """
    Normalize all timestamps to timezone-naive daily timestamps.
    """

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):

        for column in [
            "date",
            "Date",
            "datetime",
            "timestamp",
            "Timestamp",
        ]:

            if column in df.columns:

                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                df = df.set_index(
                    column
                )

                break

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):

        raise ValueError(
            "Could not identify a datetime index."
        )

    index = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    if getattr(
        index,
        "tz",
        None
    ) is not None:

        index = index.tz_localize(
            None
        )

    df.index = index

    df = df[
        ~df.index.isna()
    ]

    df = df.sort_index()

    # Daily data only.
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


def find_close_column(df):
    """
    Identify the closing-price column.
    """

    candidates = [
        "close",
        "Close",
        "close_price",
        "Close Price",
        "ltp",
        "LTP",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in [
        "close",
        "close_price",
        "ltp",
    ]:

        if candidate in lower_map:

            return lower_map[
                candidate
            ]

    raise ValueError(
        "Could not identify a close-price column. "
        f"Available columns: {list(df.columns)}"
    )


def load_nifty50():
    """
    Load and clean NIFTY 50 daily close data.
    """

    path = find_nifty50_file()

    df = pd.read_parquet(
        path
    )

    df = normalize_datetime_index(
        df
    )

    close_column = find_close_column(
        df
    )

    prices = pd.to_numeric(
        df[close_column],
        errors="coerce"
    ).dropna()

    prices = prices[
        prices > 0
    ]

    prices = prices[
        ~prices.index.duplicated(
            keep="last"
        )
    ]

    prices = prices.sort_index()

    print()
    print("=" * 70)
    print("NIFTY 50 DATA")
    print("=" * 70)

    print(
        f"Rows              : {len(prices)}"
    )

    print(
        f"Period             : "
        f"{prices.index.min().date()} → "
        f"{prices.index.max().date()}"
    )

    print(
        f"Close column       : {close_column}"
    )

    return prices


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_features(prices):
    """
    Construct the eight HMM features.
    """

    log_returns = np.log(
        prices / prices.shift(1)
    )

    features = pd.DataFrame(
        index=prices.index
    )

    features["return_1d"] = (
        log_returns
    )

    features["momentum_5d"] = (
        log_returns
        .rolling(5)
        .sum()
    )

    features["momentum_20d"] = (
        log_returns
        .rolling(20)
        .sum()
    )

    features["vol_5d"] = (
        log_returns
        .rolling(5)
        .std()
        * np.sqrt(252)
    )

    features["vol_20d"] = (
        log_returns
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    features["vol_60d"] = (
        log_returns
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    rolling_mean = (
        log_returns
        .rolling(60)
        .mean()
    )

    rolling_std = (
        log_returns
        .rolling(60)
        .std()
    )

    features["return_zscore"] = (
        log_returns
        - rolling_mean
    ) / rolling_std.replace(
        0,
        np.nan
    )

    features["vol_ratio"] = (
        features["vol_5d"]
        /
        features["vol_60d"]
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.dropna()

    return features


# ================================================================
# HMM FITTING
# ================================================================

def fit_best_hmm(X):
    """
    Fit several deterministic HMM seeds and retain the model
    with the highest training log likelihood.
    """

    best_model = None
    best_score = -np.inf
    best_seed = None

    for seed in RANDOM_SEEDS:

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                model = GaussianHMM(
                    n_components=N_STATES,
                    covariance_type="diag",
                    n_iter=300,
                    tol=1e-5,
                    random_state=seed,
                )

                model.fit(X)

            score = model.score(
                X
            )

            if (
                np.isfinite(score)
                and score > best_score
            ):

                best_score = score
                best_model = model
                best_seed = seed

        except Exception:

            continue

    if best_model is None:

        raise RuntimeError(
            "All HMM seeds failed."
        )

    return (
        best_model,
        best_seed,
        best_score,
    )


# ================================================================
# COVARIANCE HANDLING
# ================================================================

def extract_diagonal_covariance(model):
    """
    hmmlearn can expose diagonal covariance in different shapes
    depending on version/internal representation.
    """

    covars = np.asarray(
        model.covars_,
        dtype=float
    )

    if covars.ndim == 3:

        covars = np.diagonal(
            covars,
            axis1=1,
            axis2=2
        )

    elif covars.ndim != 2:

        raise ValueError(
            f"Unexpected covariance shape: "
            f"{covars.shape}"
        )

    return np.maximum(
        covars,
        1e-8
    )


# ================================================================
# CAUSAL FILTER
# ================================================================

def causal_filter(
    model,
    X,
):
    """
    Causal forward filtering.

    No future observations are used to estimate the posterior
    at each observation.
    """

    means = np.asarray(
        model.means_,
        dtype=float
    )

    covars = extract_diagonal_covariance(
        model
    )

    transition = np.asarray(
        model.transmat_,
        dtype=float
    )

    transition = (
        (1.0 - TRANSITION_SHRINKAGE)
        * transition
        +
        TRANSITION_SHRINKAGE
        / N_STATES
    )

    transition = (
        transition
        /
        transition.sum(
            axis=1,
            keepdims=True
        )
    )

    alpha = np.asarray(
        model.startprob_,
        dtype=float
    )

    alpha = (
        alpha
        /
        alpha.sum()
    )

    posteriors = []

    for observation in X:

        predicted = (
            alpha
            @
            transition
        )

        log_probability = np.zeros(
            N_STATES
        )

        for state in range(N_STATES):

            variance = covars[state]

            difference = (
                observation
                -
                means[state]
            )

            log_det = np.sum(
                np.log(
                    2.0
                    * np.pi
                    * variance
                )
            )

            mahalanobis = np.sum(
                (
                    difference ** 2
                )
                /
                variance
            )

            log_probability[state] = (
                -0.5
                *
                (
                    log_det
                    +
                    mahalanobis
                )
            )

        log_probability -= (
            np.max(
                log_probability
            )
        )

        likelihood = np.exp(
            log_probability
        )

        posterior = (
            predicted
            *
            likelihood
        )

        total = posterior.sum()

        if (
            total <= 0
            or
            not np.isfinite(total)
        ):

            posterior = (
                np.ones(N_STATES)
                /
                N_STATES
            )

        else:

            posterior /= total

        # Probability temperature.
        posterior = np.power(
            np.maximum(
                posterior,
                1e-12
            ),
            1.0
            /
            PROBABILITY_TEMPERATURE
        )

        posterior /= posterior.sum()

        alpha = posterior

        posteriors.append(
            posterior.copy()
        )

    return np.asarray(
        posteriors
    )


# ================================================================
# STATE MAPPING
# ================================================================

def map_states(
    model,
    X,
    feature_frame,
):
    """
    Map arbitrary HMM state IDs to:

        lowest momentum  -> BEAR
        middle momentum  -> SIDE
        highest momentum -> BULL
    """

    hidden_states = model.predict(
        X
    )

    scores = {}

    momentum = feature_frame[
        "momentum_20d"
    ].to_numpy()

    for state in range(N_STATES):

        mask = (
            hidden_states == state
        )

        if np.any(mask):

            scores[state] = float(
                np.mean(
                    momentum[mask]
                )
            )

        else:

            scores[state] = -np.inf

    ordered = sorted(
        scores,
        key=scores.get
    )

    return {
        ordered[0]: "BEAR",
        ordered[1]: "SIDE",
        ordered[2]: "BULL",
    }


def named_probabilities(
    posterior,
    state_mapping,
):
    """
    Convert numerical HMM posterior into semantic probabilities.
    """

    result = {
        "BULL": 0.0,
        "SIDE": 0.0,
        "BEAR": 0.0,
    }

    for state in range(N_STATES):

        name = state_mapping[
            state
        ]

        result[name] = float(
            posterior[state]
        )

    return result


# ================================================================
# OOS REGIME ANALYSIS
# ================================================================

def build_oos_regime_series(
    prices,
    features,
):
    """
    Walk-forward causal HMM.

    At each point:
        - use only the preceding 504 observations
        - fit HMM
        - causally filter through the window
        - assign current regime
        - calculate future realized volatility separately

    This is computationally expensive, but it is the research-grade
    OOS calculation.
    """

    print()
    print("=" * 70)
    print("CALCULATING OOS HMM REGIME SERIES")
    print("=" * 70)

    records = []

    feature_dates = features.index

    # Start only after a complete 504-observation window.
    for i in range(
        HMM_WINDOW,
        len(features)
    ):

        window = features.iloc[
            i - HMM_WINDOW:
            i
        ].copy()

        current_date = (
            feature_dates[i]
        )

        scaler = RobustScaler()

        X = scaler.fit_transform(
            window[
                FEATURE_COLUMNS
            ]
        )

        try:

            model, seed, score = (
                fit_best_hmm(X)
            )

            posterior_series = (
                causal_filter(
                    model,
                    X
                )
            )

            mapping = map_states(
                model,
                X,
                window
            )

            latest_posterior = (
                posterior_series[-1]
            )

            probabilities = (
                named_probabilities(
                    latest_posterior,
                    mapping
                )
            )

            regime = max(
                probabilities,
                key=probabilities.get
            )

            row = features.loc[
                current_date
            ]

            records.append(
                {
                    "date": current_date,
                    "regime": regime,
                    "p_bull":
                        probabilities["BULL"],
                    "p_side":
                        probabilities["SIDE"],
                    "p_bear":
                        probabilities["BEAR"],
                    "current_vol_5d":
                        row["vol_5d"],
                    "current_vol_20d":
                        row["vol_20d"],
                    "current_vol_60d":
                        row["vol_60d"],
                }
            )

        except Exception:

            continue

    regime = pd.DataFrame(
        records
    )

    if regime.empty:

        raise RuntimeError(
            "No OOS HMM observations were generated."
        )

    regime["date"] = pd.to_datetime(
        regime["date"]
    )

    regime = regime.set_index(
        "date"
    )

    # ------------------------------------------------------------
    # Forward realized volatility and returns
    # ------------------------------------------------------------

    log_returns = np.log(
        prices / prices.shift(1)
    )

    forward_vol_5d = (
        log_returns
        .rolling(
            FORECAST_5D
        )
        .std()
        .shift(
            -(FORECAST_5D - 1)
        )
        * np.sqrt(252)
    )

    forward_vol_21d = (
        log_returns
        .rolling(
            FORECAST_21D
        )
        .std()
        .shift(
            -(FORECAST_21D - 1)
        )
        * np.sqrt(252)
    )

    forward_return_5d = (
        np.log(
            prices.shift(
                -FORECAST_5D
            )
            /
            prices
        )
    )

    forward_return_21d = (
        np.log(
            prices.shift(
                -FORECAST_21D
            )
            /
            prices
        )
    )

    regime["realized_vol_5d"] = (
        forward_vol_5d.reindex(
            regime.index
        )
    )

    regime["realized_vol_21d"] = (
        forward_vol_21d.reindex(
            regime.index
        )
    )

    regime["forward_return_5d"] = (
        forward_return_5d.reindex(
            regime.index
        )
    )

    regime["forward_return_21d"] = (
        forward_return_21d.reindex(
            regime.index
        )
    )

    return regime


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("QUANTOS HMM ↔ VOLATILITY BRIDGE")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    prices = load_nifty50()

    # ------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING HMM FEATURES")
    print("=" * 70)

    features = build_features(
        prices
    )

    print(
        f"Feature observations : "
        f"{len(features)}"
    )

    print(
        f"Feature period       : "
        f"{features.index.min().date()} → "
        f"{features.index.max().date()}"
    )

    # ------------------------------------------------------------
    # OOS HMM
    # ------------------------------------------------------------

    regime = (
        build_oos_regime_series(
            prices,
            features
        )
    )

    # ------------------------------------------------------------
    # Forward-volatility evaluation
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CALCULATING FORWARD VOLATILITY")
    print("=" * 70)

    evaluation = regime.dropna(
        subset=[
            "realized_vol_5d",
            "realized_vol_21d",
            "forward_return_5d",
            "forward_return_21d",
        ]
    ).copy()

    # ------------------------------------------------------------
    # Regime summary
    # ------------------------------------------------------------

    summary = (
        evaluation
        .groupby("regime")
        .agg(
            observations=(
                "regime",
                "size"
            ),

            probability_bull=(
                "p_bull",
                "mean"
            ),

            probability_side=(
                "p_side",
                "mean"
            ),

            probability_bear=(
                "p_bear",
                "mean"
            ),

            current_vol_5d=(
                "current_vol_5d",
                "mean"
            ),

            current_vol_20d=(
                "current_vol_20d",
                "mean"
            ),

            current_vol_60d=(
                "current_vol_60d",
                "mean"
            ),

            forward_vol_5d=(
                "realized_vol_5d",
                "mean"
            ),

            forward_vol_21d=(
                "realized_vol_21d",
                "mean"
            ),

            forward_return_5d=(
                "forward_return_5d",
                "mean"
            ),

            forward_return_21d=(
                "forward_return_21d",
                "mean"
            ),
        )
    )

    print()
    print("=" * 70)
    print("REGIME ↔ VOLATILITY RESULTS")
    print("=" * 70)

    print(
        summary.round(
            4
        ).to_string()
    )

    # ------------------------------------------------------------
    # Save OOS outputs
    # ------------------------------------------------------------

    timeseries_path = (
        OUTPUT_DIR
        /
        "hmm_volatility_bridge_timeseries.csv"
    )

    summary_path = (
        OUTPUT_DIR
        /
        "hmm_volatility_bridge_summary.csv"
    )

    regime.to_csv(
        timeseries_path
    )

    summary.to_csv(
        summary_path
    )

    # ============================================================
    # CURRENT MARKET STATE
    # ============================================================

    print()
    print("=" * 70)
    print("CALCULATING CURRENT QUANTOS MARKET STATE")
    print("=" * 70)

    # ------------------------------------------------------------
    # IMPORTANT:
    # Use the latest available feature observations.
    #
    # We do NOT drop rows based on future volatility here.
    # Therefore the current state reaches the latest feature date.
    # ------------------------------------------------------------

    current_features = (
        features[
            FEATURE_COLUMNS
        ]
        .dropna()
        .tail(
            HMM_WINDOW
        )
        .copy()
    )

    if len(
        current_features
    ) < HMM_WINDOW:

        raise RuntimeError(
            "Insufficient observations for current HMM."
        )

    current_date = (
        current_features.index[-1]
    )

    print(
        f"Latest feature date : "
        f"{current_date.date()}"
    )

    print(
        f"Training observations: "
        f"{len(current_features)}"
    )

    # ------------------------------------------------------------
    # Current HMM fit
    # ------------------------------------------------------------

    current_scaler = (
        RobustScaler()
    )

    current_X = (
        current_scaler.fit_transform(
            current_features[
                FEATURE_COLUMNS
            ]
        )
    )

    current_model, current_seed, _ = (
        fit_best_hmm(
            current_X
        )
    )

    # ------------------------------------------------------------
    # Current causal posterior
    # ------------------------------------------------------------

    current_posteriors = (
        causal_filter(
            current_model,
            current_X
        )
    )

    current_posterior = (
        current_posteriors[-1]
    )

    # ------------------------------------------------------------
    # Semantic mapping
    # ------------------------------------------------------------

    current_mapping = (
        map_states(
            current_model,
            current_X,
            current_features
        )
    )

    current_probabilities = (
        named_probabilities(
            current_posterior,
            current_mapping
        )
    )

    current_regime = max(
        current_probabilities,
        key=current_probabilities.get
    )

    # ------------------------------------------------------------
    # Current volatility
    # ------------------------------------------------------------

    latest = (
        current_features.iloc[-1]
    )

    current_vol_5d = float(
        latest["vol_5d"]
    )

    current_vol_20d = float(
        latest["vol_20d"]
    )

    current_vol_60d = float(
        latest["vol_60d"]
    )

    # ------------------------------------------------------------
    # Historical regime-conditioned forward volatility
    # ------------------------------------------------------------

    historical_forward_vol = {}

    for name in [
        "BULL",
        "SIDE",
        "BEAR",
    ]:

        if name in summary.index:

            historical_forward_vol[
                name
            ] = float(
                summary.loc[
                    name,
                    "forward_vol_21d"
                ]
            )

        else:

            historical_forward_vol[
                name
            ] = np.nan

    # ------------------------------------------------------------
    # Posterior-weighted conditional estimate
    # ------------------------------------------------------------

    expected_forward_21d_vol = (
        current_probabilities["BULL"]
        *
        historical_forward_vol["BULL"]
        +
        current_probabilities["SIDE"]
        *
        historical_forward_vol["SIDE"]
        +
        current_probabilities["BEAR"]
        *
        historical_forward_vol["BEAR"]
    )

    # ------------------------------------------------------------
    # Current state table
    # ------------------------------------------------------------

    current_state = pd.DataFrame(
        [
            {
                "date": current_date,

                "regime":
                    current_regime,

                "p_bull":
                    current_probabilities[
                        "BULL"
                    ],

                "p_side":
                    current_probabilities[
                        "SIDE"
                    ],

                "p_bear":
                    current_probabilities[
                        "BEAR"
                    ],

                "current_vol_5d":
                    current_vol_5d,

                "current_vol_20d":
                    current_vol_20d,

                "current_vol_60d":
                    current_vol_60d,

                "expected_forward_21d_vol":
                    expected_forward_21d_vol,

                "historical_bull_forward_21d_vol":
                    historical_forward_vol[
                        "BULL"
                    ],

                "historical_side_forward_21d_vol":
                    historical_forward_vol[
                        "SIDE"
                    ],

                "historical_bear_forward_21d_vol":
                    historical_forward_vol[
                        "BEAR"
                    ],

                "hmm_training_window":
                    HMM_WINDOW,

                "hmm_seed":
                    current_seed,

                "transition_shrinkage":
                    TRANSITION_SHRINKAGE,

                "probability_temperature":
                    PROBABILITY_TEMPERATURE,
            }
        ]
    )

    current_state_path = (
        OUTPUT_DIR
        /
        "current_quantos_market_state.csv"
    )

    current_state.to_csv(
        current_state_path,
        index=False
    )

    # ============================================================
    # PRINT FINAL RESULT
    # ============================================================

    print()
    print("=" * 70)
    print("CURRENT QUANTOS MARKET STATE")
    print("=" * 70)

    print(
        f"Date              : "
        f"{current_date.date()}"
    )

    print(
        f"Regime            : "
        f"{current_regime}"
    )

    print(
        f"P(BULL)           : "
        f"{current_probabilities['BULL']:.2%}"
    )

    print(
        f"P(SIDE)           : "
        f"{current_probabilities['SIDE']:.2%}"
    )

    print(
        f"P(BEAR)           : "
        f"{current_probabilities['BEAR']:.2%}"
    )

    print(
        f"Current 5D Vol    : "
        f"{current_vol_5d:.2%}"
    )

    print(
        f"Current 20D Vol   : "
        f"{current_vol_20d:.2%}"
    )

    print(
        f"Current 60D Vol   : "
        f"{current_vol_60d:.2%}"
    )

    print(
        f"Expected 21D Vol  : "
        f"{expected_forward_21d_vol:.2%}"
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        timeseries_path
    )

    print(
        summary_path
    )

    print(
        current_state_path
    )

    print()
    print("=" * 70)
    print("HMM ↔ VOLATILITY BRIDGE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()