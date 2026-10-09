"""
QuantOS
HMM-Conditional Portfolio Allocation Backtest

Purpose
-------
Compare standard portfolio allocation methods against an
HMM-conditioned Risk Parity strategy.

Strategies
----------
1. Equal Weight
2. Minimum Variance
3. Constrained Minimum Variance
4. Risk Parity
5. HMM-Conditional Risk Parity

Design
------
- 13-asset long-history universe
- 504-observation covariance/HMM window
- 21-observation rebalance frequency
- 10 bps transaction cost
- Long-only portfolios
- Maximum 25% weight for constrained minimum variance
- No future information used in portfolio construction

The HMM is used ONLY as a regime-conditioned covariance model.
It is NOT used as a directional return predictor.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

N_STATES = 3

HMM_WINDOW = 504

REBALANCE_EVERY = 21

TRANSACTION_COST = 0.001   # 10 bps

MAX_WEIGHT = 0.25

MIN_STATE_EFFECTIVE_N = 30

SEEDS = [
    7,
    17,
    27,
    37,
    47,
]

STATE_NAMES = [
    "BULL",
    "SIDE",
    "BEAR",
]

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

ASSETS = [
    "NIFTY_50",
    "NIFTY_BANK",
    "NIFTY_MIDCAP_100",
    "NIFTY_NEXT_50",
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "NFLX",
]

DATA_ROOT = Path("data")

OUTPUT_DIR = (
    DATA_ROOT
    / "portfolio"
    / "hmm_allocation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA DISCOVERY + LOADING
# ============================================================

def find_asset_files():
    """
    Locate the historical price file for each asset.

    The QuantOS data is stored as individual asset files,
    so the backtest constructs the unified matrix itself.
    """

    candidates = []

    for extension in ("*.parquet", "*.csv"):
        candidates.extend(
            DATA_ROOT.rglob(extension)
        )

    asset_files = {}

    for asset in ASSETS:

        asset_lower = asset.lower()

        # Exact filename candidates
        filename_variants = {
            asset_lower,
            asset_lower.replace("_", ""),
            asset_lower.replace("_", "-"),
        }

        matches = []

        for path in candidates:

            stem = path.stem.lower()

            if stem in filename_variants:
                matches.append(path)

        # If filename matching failed, inspect columns.
        if not matches:

            for path in candidates:

                try:

                    if path.suffix.lower() == ".parquet":
                        sample = pd.read_parquet(
                            path
                        ).head(2)
                    else:
                        sample = pd.read_csv(
                            path,
                            nrows=2,
                        )

                except Exception:
                    continue

                columns = {
                    str(c).lower()
                    for c in sample.columns
                }

                normalized_asset = (
                    asset_lower
                    .replace("_", "")
                    .replace("-", "")
                )

                normalized_columns = {
                    c.replace("_", "")
                    .replace("-", "")
                    for c in columns
                }

                if (
                    normalized_asset
                    in normalized_columns
                ):
                    matches.append(path)

        if matches:

            # Prefer regime/daily files.
            matches.sort(
                key=lambda p: (
                    "regime" not in str(p).lower(),
                    "daily" not in str(p).lower(),
                    len(str(p)),
                )
            )

            asset_files[asset] = matches[0]

    print("\n" + "=" * 70)
    print("ASSET FILE DISCOVERY")
    print("=" * 70)

    for asset in ASSETS:

        if asset in asset_files:

            print(
                f"{asset:<20} -> "
                f"{asset_files[asset]}"
            )

        else:

            print(
                f"{asset:<20} -> NOT FOUND"
            )

    missing = [
        asset
        for asset in ASSETS
        if asset not in asset_files
    ]

    if missing:

        raise FileNotFoundError(
            "\nMissing asset files:\n"
            + "\n".join(
                f"  {asset}"
                for asset in missing
            )
        )

    return asset_files


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_datetime_index(df):
    """
    Convert any date-like index to a timezone-naive
    normalized DatetimeIndex.
    """

    df = df.copy()

    idx = pd.to_datetime(
        df.index,
        errors="coerce",
        utc=True,
    )

    idx = pd.DatetimeIndex(
        idx
    )

    valid = ~idx.isna()

    df = df.loc[
        valid
    ].copy()

    idx = idx[
        valid
    ]

    idx = idx.tz_localize(
        None
    )

    idx = idx.normalize()

    df.index = idx

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.sort_index()

    return df


# ============================================================
# LOAD ONE ASSET
# ============================================================

def load_single_asset(path, asset):
    """
    Load one asset's close-price series and normalize its dates.
    """

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    else:
        raise ValueError(
            f"Unsupported file type: {path}"
        )

    if df.empty:
        raise ValueError(
            f"{asset}: file is empty: {path}"
        )

    # ------------------------------------------------------------
    # Detect date column if dates are not already the index.
    # ------------------------------------------------------------

    date_candidates = [
        "date",
        "Date",
        "DATE",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp",
        "time",
        "Time",
    ]

    date_column = None

    for column in date_candidates:
        if column in df.columns:
            date_column = column
            break

    if date_column is not None:

        dates = pd.to_datetime(
            df[date_column],
            errors="coerce",
            utc=True,
        )

        df = df.copy()

        df.index = pd.DatetimeIndex(
            dates
        )

    # ------------------------------------------------------------
    # Normalize dates.
    # ------------------------------------------------------------

    df = normalize_datetime_index(
        df
    )

    # ------------------------------------------------------------
    # Find close column.
    # ------------------------------------------------------------

    close_candidates = [
        "close",
        "Close",
        "CLOSE",
        "close_price",
        "Close Price",
        "closing_price",
        "adj_close",
        "Adj Close",
        "adjusted_close",
        "Adjusted Close",
    ]

    close_column = None

    for column in close_candidates:

        if column in df.columns:

            close_column = column
            break

    # Case-insensitive fallback.
    if close_column is None:

        for column in df.columns:

            normalized = (
                str(column)
                .strip()
                .lower()
                .replace("_", " ")
            )

            if normalized in {
                "close",
                "close price",
                "closing price",
                "adj close",
                "adjusted close",
            }:

                close_column = column
                break

    if close_column is None:

        raise ValueError(
            f"{asset}: could not find close column. "
            f"Columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------
    # Clean price series.
    # ------------------------------------------------------------

    series = pd.to_numeric(
        df[close_column],
        errors="coerce",
    )

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    series = series.dropna()

    series.name = asset

    series = series[
        ~series.index.duplicated(
            keep="last"
        )
    ]

    series = series.sort_index()

    if series.empty:

        raise RuntimeError(
            f"{asset}: no valid close prices after cleaning."
        )

    # Final sanity checks.
    if not isinstance(
        series.index,
        pd.DatetimeIndex,
    ):

        raise TypeError(
            f"{asset}: index is not DatetimeIndex."
        )

    if series.index.tz is not None:

        raise TypeError(
            f"{asset}: index remains timezone-aware."
        )

    return series


# ============================================================
# BUILD UNIFIED ASSET RETURN MATRIX
# ============================================================

def load_asset_matrix():
    """
    Build the unified multi-asset return matrix.

    IMPORTANT:
    Returns are calculated separately for every asset BEFORE calendars
    are aligned. Indian and US markets have different holidays, so
    aligning raw prices and then dropping rows containing NaN can
    accidentally destroy the entire dataset.
    """

    print("=" * 70)
    print("ASSET FILE DISCOVERY")
    print("=" * 70)

    asset_files = find_asset_files()

    for asset in ASSETS:
        print(f"{asset:20s} -> {asset_files[asset]}")

    print()
    print("=" * 70)
    print("LOADING ASSET PRICES")
    print("=" * 70)

    prices = {}

    for asset in ASSETS:

        print(f"Loading {asset}...")

        series = load_single_asset(
            asset_files[asset],
            asset,
        )

        if series.empty:
            raise RuntimeError(
                f"{asset}: price series is empty"
            )

        prices[asset] = series

        print(
            f"  {len(series):5d} rows | "
            f"{series.index.min().date()} -> "
            f"{series.index.max().date()}"
        )

    # ============================================================
    # CALCULATE RETURNS PER ASSET FIRST
    # ============================================================

    print()
    print("=" * 70)
    print("CALCULATING PER-ASSET RETURNS")
    print("=" * 70)

    asset_returns = {}

    for asset in ASSETS:

        price = prices[asset]

        ret = price.pct_change(
            fill_method=None
        )

        ret = ret.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        ret = ret.dropna()

        ret.name = asset

        if ret.empty:
            raise RuntimeError(
                f"{asset}: return series became empty"
            )

        asset_returns[asset] = ret

        print(
            f"{asset:20s} "
            f"{len(ret):5d} returns | "
            f"{ret.index.min().date()} -> "
            f"{ret.index.max().date()}"
        )

    # ============================================================
    # FIND COMMON HISTORICAL PERIOD
    # ============================================================

    first_dates = [
        asset_returns[asset].index.min()
        for asset in ASSETS
    ]

    last_dates = [
        asset_returns[asset].index.max()
        for asset in ASSETS
    ]

    common_start = max(first_dates)
    common_end = min(last_dates)

    print()
    print("=" * 70)
    print("COMMON PORTFOLIO PERIOD")
    print("=" * 70)

    print(
        f"Start : {common_start.date()}"
    )

    print(
        f"End   : {common_end.date()}"
    )

    if common_start >= common_end:
        raise RuntimeError(
            "No overlapping historical period exists between "
            "the requested assets."
        )

    # ============================================================
    # ALIGN RETURNS, NOT PRICES
    # ============================================================

    print()
    print("=" * 70)
    print("ALIGNING RETURN SERIES")
    print("=" * 70)

    returns = pd.concat(
        [
            asset_returns[asset]
            for asset in ASSETS
        ],
        axis=1,
        join="outer",
    )

    returns.columns = ASSETS

    returns = returns.sort_index()

    # Keep only the period where every asset has actually existed.
    returns = returns.loc[
        common_start:common_end
    ]

    print(
        f"Before holiday handling: "
        f"{returns.shape}"
    )

    # ============================================================
    # HANDLE DIFFERENT MARKET HOLIDAYS
    # ============================================================
    #
    # Example:
    # India trades on Monday but US is closed.
    #
    # There is no US return for that date.
    # For a daily portfolio return matrix, the US contribution
    # on that date is 0%.
    #
    # This is NOT the same as inventing a US price before inception.
    # We already restricted the dataset to common_start/common_end.
    #

    missing_before = returns.isna().sum()

    print()
    print("Missing returns before holiday handling:")

    for asset in ASSETS:
        print(
            f"  {asset:20s}: "
            f"{int(missing_before[asset])}"
        )

    returns = returns.fillna(0.0)

    # ============================================================
    # FINAL SANITY CHECKS
    # ============================================================

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if returns.isna().any().any():

        print()
        print(
            "WARNING: unresolved NaN values remain:"
        )

        print(
            returns.isna().sum()
        )

        returns = returns.dropna(
            how="any"
        )

    if returns.empty:
        raise RuntimeError(
            "Unified return matrix is empty after "
            "calendar alignment."
        )

    if len(returns.columns) != len(ASSETS):
        raise RuntimeError(
            f"Expected {len(ASSETS)} assets, "
            f"got {len(returns.columns)}"
        )

    if not isinstance(
        returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Return matrix index is not a DatetimeIndex."
        )

    if returns.index.tz is not None:
        raise TypeError(
            "Return matrix index is still timezone-aware."
        )

    returns = returns.astype(float)

    print()
    print("=" * 70)
    print("UNIFIED RETURN MATRIX")
    print("=" * 70)

    print(
        f"Rows       : {len(returns)}"
    )

    print(
        f"Columns    : {len(returns.columns)}"
    )

    print(
        f"Period     : "
        f"{returns.index.min().date()} -> "
        f"{returns.index.max().date()}"
    )

    print(
        f"NaN count  : "
        f"{int(returns.isna().sum().sum())}"
    )

    print()
    print(
        "Assets:"
    )

    for asset in returns.columns:
        print(
            f"  {asset}"
        )

    print()
    print("=" * 70)
    print("ASSET DATA LOADING COMPLETE")
    print("=" * 70)

    return returns

# ============================================================
# NIFTY HMM FEATURES
# ============================================================

def build_hmm_features(
    nifty_returns: pd.Series,
):

    log_returns = np.log1p(
        nifty_returns
    )

    log_price = (
        log_returns
        .cumsum()
    )

    features = pd.DataFrame(
        index=nifty_returns.index
    )

    features[
        "return_1d"
    ] = log_returns

    features[
        "momentum_5d"
    ] = (
        log_price
        - log_price.shift(5)
    )

    features[
        "momentum_20d"
    ] = (
        log_price
        - log_price.shift(20)
    )

    features[
        "vol_5d"
    ] = (
        log_returns
        .rolling(5)
        .std()
    )

    features[
        "vol_20d"
    ] = (
        log_returns
        .rolling(20)
        .std()
    )

    features[
        "vol_60d"
    ] = (
        log_returns
        .rolling(60)
        .std()
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

    features[
        "return_zscore"
    ] = (
        (
            log_returns
            - rolling_mean
        )
        /
        rolling_std.replace(
            0,
            np.nan,
        )
    )

    features[
        "vol_ratio"
    ] = (
        features["vol_20d"]
        /
        features[
            "vol_60d"
        ].replace(
            0,
            np.nan,
        )
    )

    features = features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return features.dropna()


# ============================================================
# HMM FIT
# ============================================================

def fit_best_hmm(
    train_features: pd.DataFrame,
):

    scaler = RobustScaler()

    X = scaler.fit_transform(
        train_features[
            FEATURES
        ]
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
            "All HMM initializations failed."
        )

    return (
        best_model,
        scaler,
        best_seed,
    )


# ============================================================
# CAUSAL HMM FILTER
# ============================================================

def causal_filter(
    model,
    X_scaled,
):

    n_obs = X_scaled.shape[0]

    n_states = (
        model.n_components
    )

    n_features = (
        X_scaled.shape[1]
    )

    alpha = np.zeros(
        (
            n_obs,
            n_states,
        )
    )

    log_start = np.log(
        np.clip(
            model.startprob_,
            1e-300,
            None,
        )
    )

    log_transition = np.log(
        np.clip(
            model.transmat_,
            1e-300,
            None,
        )
    )

    covars = np.asarray(
        model.covars_,
        dtype=float,
    )

    # --------------------------------------------------------
    # hmmlearn may represent diagonal covariance as:
    #
    # (states, features)
    #
    # OR:
    #
    # (states, features, features)
    #
    # depending on version.
    # --------------------------------------------------------

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
        1e-8,
    )

    log_det = np.sum(
        np.log(
            variances
        ),
        axis=1,
    )

    log_2pi = np.log(
        2.0 * np.pi
    )

    def emission_log_prob(x):

        output = np.empty(
            n_states
        )

        for state in range(
            n_states
        ):

            diff = (
                x
                - model.means_[state]
            )

            mahalanobis = np.sum(
                (
                    diff ** 2
                )
                /
                variances[state]
            )

            output[state] = (
                -0.5
                * (
                    n_features
                    * log_2pi
                    + log_det[state]
                    + mahalanobis
                )
            )

        return output

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

    # Remaining observations
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
            n_states
        )

        for state in range(
            n_states
        ):

            next_log_alpha[state] = (
                np.logaddexp.reduce(
                    log_alpha
                    + log_transition[
                        :,
                        state,
                    ]
                )
                + emission[state]
            )

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
# STATE MAPPING
# ============================================================

def map_hmm_states(
    model,
):

    momentum_index = (
        FEATURES.index(
            "momentum_20d"
        )
    )

    state_scores = (
        model.means_[
            :,
            momentum_index,
        ]
    )

    ordered_states = np.argsort(
        state_scores
    )

    mapping = {
        int(ordered_states[2]):
            "BULL",

        int(ordered_states[1]):
            "SIDE",

        int(ordered_states[0]):
            "BEAR",
    }

    return mapping


# ============================================================
# WEIGHTED COVARIANCE
# ============================================================

def weighted_covariance(
    returns: pd.DataFrame,
    probabilities: np.ndarray,
):

    X = returns.values

    w = np.asarray(
        probabilities,
        dtype=float,
    )

    valid = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(w)
        & (w > 0)
    )

    X = X[valid]
    w = w[valid]

    if len(X) < 2:

        return None, 0.0

    weight_sum = w.sum()

    if weight_sum <= 0:

        return None, 0.0

    effective_n = (
        weight_sum ** 2
        /
        np.sum(w ** 2)
    )

    mean = (
        (
            X * w[:, None]
        ).sum(axis=0)
        /
        weight_sum
    )

    centered = (
        X - mean
    )

    covariance = (
        (
            centered.T
            * w
        )
        @ centered
    )

    denominator = (
        weight_sum
        - 1.0
    )

    if denominator <= 0:

        return None, effective_n

    covariance /= denominator

    covariance = (
        covariance
        + covariance.T
    ) / 2.0

    return (
        covariance,
        effective_n,
    )


# ============================================================
# COVARIANCE STABILIZATION
# ============================================================

def stabilize_covariance(
    covariance: np.ndarray,
):

    covariance = (
        covariance
        + covariance.T
    ) / 2.0

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    minimum_eigenvalue = (
        eigenvalues.min()
    )

    if minimum_eigenvalue <= 1e-10:

        covariance += (
            np.eye(
                covariance.shape[0]
            )
            * (
                -minimum_eigenvalue
                + 1e-8
            )
        )

    return covariance


# ============================================================
# MINIMUM VARIANCE
# ============================================================

def minimum_variance_weights(
    covariance,
    max_weight=1.0,
):

    n_assets = covariance.shape[0]

    x0 = np.ones(
        n_assets
    ) / n_assets

    def objective(w):

        return float(
            w
            @ covariance
            @ w
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        }
    ]

    bounds = [
        (
            0.0,
            max_weight,
        )
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:

        return x0

    weights = np.asarray(
        result.x
    )

    weights = np.maximum(
        weights,
        0,
    )

    weights /= weights.sum()

    return weights


# ============================================================
# RISK PARITY
# ============================================================

def risk_parity_weights(
    covariance,
):

    n_assets = covariance.shape[0]

    x0 = np.ones(
        n_assets
    ) / n_assets

    def risk_contributions(w):

        portfolio_variance = (
            w
            @ covariance
            @ w
        )

        portfolio_vol = np.sqrt(
            max(
                portfolio_variance,
                1e-16,
            )
        )

        marginal = (
            covariance
            @ w
        )

        contribution = (
            w
            * marginal
            / portfolio_vol
        )

        return contribution

    def objective(w):

        rc = risk_contributions(
            w
        )

        total = rc.sum()

        if total <= 0:

            return 1e6

        rc_fraction = (
            rc / total
        )

        target = (
            1.0
            / n_assets
        )

        return np.sum(
            (
                rc_fraction
                - target
            ) ** 2
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        }
    ]

    bounds = [
        (
            1e-8,
            1.0,
        )
        for _ in range(n_assets)
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:

        # Stable fallback
        volatility = np.sqrt(
            np.maximum(
                np.diag(
                    covariance
                ),
                1e-12,
            )
        )

        weights = (
            1.0
            / volatility
        )

        weights /= weights.sum()

        return weights

    weights = np.maximum(
        result.x,
        0,
    )

    weights /= weights.sum()

    return weights


# ============================================================
# PORTFOLIO STRATEGIES
# ============================================================

def equal_weight(
    covariance,
):

    n_assets = covariance.shape[0]

    return (
        np.ones(
            n_assets
        )
        / n_assets
    )


# ============================================================
# HMM CONDITIONAL COVARIANCE
# ============================================================

def build_hmm_conditional_covariance(
    asset_returns: pd.DataFrame,
    train_features: pd.DataFrame,
    model,
    scaler,
):

    X_scaled = scaler.transform(
        train_features[
            FEATURES
        ]
    )

    filtered = causal_filter(
        model,
        X_scaled,
    )

    mapping = map_hmm_states(
        model
    )

    # --------------------------------------------------------
    # Reorder filtered posterior into:
    #
    # BULL, SIDE, BEAR
    # --------------------------------------------------------

    posterior = np.zeros(
        (
            len(filtered),
            N_STATES,
        )
    )

    for original_state, name in (
        mapping.items()
    ):

        target_index = (
            STATE_NAMES.index(
                name
            )
        )

        posterior[
            :,
            target_index,
        ] = filtered[
            :,
            original_state,
        ]

    # --------------------------------------------------------
    # Align asset returns with HMM training dates
    # --------------------------------------------------------

    aligned_returns = (
        asset_returns.reindex(
            train_features.index
        )
    )

    valid_rows = (
        aligned_returns
        .notna()
        .all(axis=1)
    )

    aligned_returns = (
        aligned_returns[
            valid_rows
        ]
    )

    posterior = posterior[
        valid_rows.values
    ]

    if len(
        aligned_returns
    ) < 100:

        raise RuntimeError(
            "Too few aligned observations "
            "for state-conditioned covariance."
        )

    # --------------------------------------------------------
    # Overall covariance
    # --------------------------------------------------------

    overall_covariance = (
        aligned_returns
        .cov()
        .values
    )

    overall_covariance = (
        stabilize_covariance(
            overall_covariance
        )
    )

    # --------------------------------------------------------
    # State-conditioned covariance
    # --------------------------------------------------------

    state_covariances = []

    effective_sample_sizes = []

    for state in range(
        N_STATES
    ):

        covariance, effective_n = (
            weighted_covariance(
                aligned_returns,
                posterior[
                    :,
                    state,
                ],
            )
        )

        effective_sample_sizes.append(
            effective_n
        )

        if (
            covariance is None
            or effective_n
            < MIN_STATE_EFFECTIVE_N
        ):

            covariance = (
                overall_covariance.copy()
            )

        covariance = (
            stabilize_covariance(
                covariance
            )
        )

        state_covariances.append(
            covariance
        )

    # --------------------------------------------------------
    # Current filtered posterior
    # --------------------------------------------------------

    current_posterior = posterior[
        -1
    ]

    # --------------------------------------------------------
    # Posterior-weighted covariance
    #
    # Sigma_HMM =
    #
    # sum P(state | current data)
    #     * Sigma_state
    # --------------------------------------------------------

    conditional_covariance = (
        np.zeros_like(
            overall_covariance
        )
    )

    for state in range(
        N_STATES
    ):

        conditional_covariance += (
            current_posterior[state]
            * state_covariances[state]
        )

    conditional_covariance = (
        stabilize_covariance(
            conditional_covariance
        )
    )

    current_state = (
        STATE_NAMES[
            int(
                np.argmax(
                    current_posterior
                )
            )
        ]
    )

    return {
        "covariance":
            conditional_covariance,

        "posterior":
            current_posterior,

        "state":
            current_state,

        "effective_sample_sizes":
            effective_sample_sizes,
    }


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    daily_returns: pd.Series,
):

    daily_returns = (
        daily_returns
        .dropna()
    )

    if len(
        daily_returns
    ) == 0:

        return {}

    wealth = (
        1.0
        + daily_returns
    ).cumprod()

    years = (
        len(daily_returns)
        / 252.0
    )

    final_wealth = (
        wealth.iloc[-1]
    )

    cagr = (
        final_wealth
        ** (1.0 / years)
        - 1.0
    )

    volatility = (
        daily_returns.std()
        * np.sqrt(252)
    )

    mean_daily = (
        daily_returns.mean()
    )

    sharpe = (
        mean_daily
        / daily_returns.std()
        * np.sqrt(252)
        if daily_returns.std() > 0
        else np.nan
    )

    downside = (
        daily_returns[
            daily_returns < 0
        ]
    )

    downside_std = (
        downside.std()
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        mean_daily
        / downside_std
        * np.sqrt(252)
        if (
            downside_std
            and downside_std > 0
        )
        else np.nan
    )

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_max
        - 1.0
    )

    max_drawdown = (
        drawdown.min()
    )

    calmar = (
        cagr
        / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    var_95 = (
        daily_returns
        .quantile(0.05)
    )

    cvar_95 = (
        daily_returns[
            daily_returns
            <= var_95
        ].mean()
    )

    return {
        "CAGR":
            cagr,

        "Volatility":
            volatility,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Max Drawdown":
            max_drawdown,

        "Calmar":
            calmar,

        "Final Wealth":
            final_wealth,

        "VaR 95":
            var_95,

        "CVaR 95":
            cvar_95,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    returns: pd.DataFrame,
):

    features = build_hmm_features(
        returns[
            "NIFTY_50"
        ]
    )

    # --------------------------------------------------------
    # Use only dates for which both HMM features and all
    # portfolio assets are available.
    # --------------------------------------------------------

    common_dates = (
        features.index
        .intersection(
            returns.index
        )
    )

    features = features.loc[
        common_dates
    ]

    returns = returns.loc[
        common_dates
    ]

    print("\n" + "=" * 70)
    print("COMMON BACKTEST DATA")
    print("=" * 70)

    print(
        "Rows:",
        len(returns),
    )

    print(
        "Period:",
        returns.index.min(),
        "->",
        returns.index.max(),
    )

    if len(returns) <= HMM_WINDOW:

        raise RuntimeError(
            "Not enough observations for "
            "the 504-observation HMM window."
        )

    strategy_names = [
        "Equal Weight",
        "Minimum Variance",
        "Constrained Min Variance",
        "Risk Parity",
        "HMM Conditional Risk Parity",
    ]

    n_assets = len(
        ASSETS
    )

    # Daily strategy returns
    strategy_returns = {
        name:
            pd.Series(
                0.0,
                index=returns.index,
            )
        for name in strategy_names
    }

    # Current weights
    current_weights = {
        name:
            None
        for name in strategy_names
    }

    # Portfolio wealth
    wealth = {
        name:
            1.0
        for name in strategy_names
    }

    turnover = {
        name:
            0.0
        for name in strategy_names
    }

    transaction_costs = {
        name:
            0.0
        for name in strategy_names
    }

    rebalance_records = []

    # --------------------------------------------------------
    # Rebalance schedule
    # --------------------------------------------------------

    rebalance_positions = range(
        HMM_WINDOW,
        len(returns),
        REBALANCE_EVERY,
    )

    for position in rebalance_positions:

        rebalance_date = (
            returns.index[
                position
            ]
        )

        train_start = (
            position
            - HMM_WINDOW
        )

        train_end = position

        train_features = (
            features.iloc[
                train_start:
                train_end
            ]
        )

        train_returns = (
            returns.iloc[
                train_start:
                train_end
            ]
        )

        # ----------------------------------------------------
        # HMM
        # ----------------------------------------------------

        (
            model,
            scaler,
            seed,
        ) = fit_best_hmm(
            train_features
        )

        hmm_info = (
            build_hmm_conditional_covariance(
                train_returns,
                train_features,
                model,
                scaler,
            )
        )

        covariance = (
            train_returns.cov().values
        )

        covariance = (
            stabilize_covariance(
                covariance
            )
        )

        # ----------------------------------------------------
        # Strategy weights
        # ----------------------------------------------------

        weights = {}

        weights[
            "Equal Weight"
        ] = equal_weight(
            covariance
        )

        weights[
            "Minimum Variance"
        ] = minimum_variance_weights(
            covariance,
            max_weight=1.0,
        )

        weights[
            "Constrained Min Variance"
        ] = minimum_variance_weights(
            covariance,
            max_weight=MAX_WEIGHT,
        )

        weights[
            "Risk Parity"
        ] = risk_parity_weights(
            covariance
        )

        weights[
            "HMM Conditional Risk Parity"
        ] = risk_parity_weights(
            hmm_info[
                "covariance"
            ]
        )

        # ----------------------------------------------------
        # Apply transaction costs
        # ----------------------------------------------------

        for strategy in strategy_names:

            new_weights = (
                weights[strategy]
            )

            old_weights = (
                current_weights[
                    strategy
                ]
            )

            if old_weights is None:

                strategy_turnover = 0.0
                cost = 0.0

            else:

                strategy_turnover = (
                    np.abs(
                        new_weights
                        - old_weights
                    ).sum()
                )

                cost = (
                    strategy_turnover
                    * TRANSACTION_COST
                )

            turnover[
                strategy
            ] += strategy_turnover

            transaction_costs[
                strategy
            ] += cost

            current_weights[
                strategy
            ] = new_weights

        # ----------------------------------------------------
        # Save rebalance information
        # ----------------------------------------------------

        record = {
            "date":
                rebalance_date,

            "hmm_state":
                hmm_info[
                    "state"
                ],

            "hmm_p_bull":
                hmm_info[
                    "posterior"
                ][0],

            "hmm_p_side":
                hmm_info[
                    "posterior"
                ][1],

            "hmm_p_bear":
                hmm_info[
                    "posterior"
                ][2],

            "hmm_seed":
                seed,

            "state_effective_n_bull":
                hmm_info[
                    "effective_sample_sizes"
                ][0],

            "state_effective_n_side":
                hmm_info[
                    "effective_sample_sizes"
                ][1],

            "state_effective_n_bear":
                hmm_info[
                    "effective_sample_sizes"
                ][2],
        }

        for strategy in strategy_names:

            for asset_index, asset in enumerate(
                ASSETS
            ):

                record[
                    f"{strategy}__{asset}"
                ] = weights[
                    strategy
                ][asset_index]

        rebalance_records.append(
            record
        )

        # ----------------------------------------------------
        # Apply weights to the NEXT 21 observations
        #
        # This is critical:
        #
        # The portfolio cannot earn the return used to
        # determine today's weights.
        # ----------------------------------------------------

        next_start = (
            position + 1
        )

        if next_start >= len(
            returns
        ):
            continue

        next_end = min(
            position
            + REBALANCE_EVERY
            + 1,
            len(returns),
        )

        holding_returns = (
            returns.iloc[
                next_start:
                next_end
            ]
        )

        for strategy in strategy_names:

            w = (
                current_weights[
                    strategy
                ]
            )

            portfolio_daily = (
                holding_returns
                @ w
            )

            # Apply transaction cost at the first
            # day of the new holding period.
            cost = (
                transaction_costs[
                    strategy
                ]
                - (
                    transaction_costs[
                        strategy
                    ]
                    - (
                        np.abs(
                            w
                            - (
                                weights[
                                    strategy
                                ]
                            )
                        ).sum()
                        * TRANSACTION_COST
                    )
                )
            )

            # The above expression is deliberately not
            # used for return modification because cost
            # was already accumulated separately.
            #
            # Instead, apply the incremental cost directly.
            old = (
                0.0
                if current_weights[
                    strategy
                ] is None
                else 0.0
            )

            strategy_returns[
                strategy
            ].loc[
                holding_returns.index
            ] = (
                portfolio_daily
            )

    # --------------------------------------------------------
    # Correct transaction-cost treatment
    #
    # Reconstruct wealth from gross strategy returns and
    # deduct each rebalance cost once.
    # --------------------------------------------------------

    rebalance_df = pd.DataFrame(
        rebalance_records
    )

    rebalance_df = (
        rebalance_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Build clean daily returns again.
    daily_strategy_returns = {
        strategy:
            pd.Series(
                np.nan,
                index=returns.index,
            )
        for strategy in strategy_names
    }

    for i in range(
        len(rebalance_df)
    ):

        date = pd.Timestamp(
            rebalance_df.loc[
                i,
                "date"
            ]
        )

        position = returns.index.get_loc(
            date
        )

        next_start = (
            position + 1
        )

        if i + 1 < len(
            rebalance_df
        ):

            next_rebalance_date = (
                pd.Timestamp(
                    rebalance_df.loc[
                        i + 1,
                        "date"
                    ]
                )
            )

            next_position = (
                returns.index.get_loc(
                    next_rebalance_date
                )
            )

        else:

            next_position = (
                len(returns)
            )

        holding_index = (
            returns.index[
                next_start:
                next_position
            ]
        )

        for strategy in strategy_names:

            weight_columns = [
                f"{strategy}__{asset}"
                for asset in ASSETS
            ]

            w = (
                rebalance_df.loc[
                    i,
                    weight_columns
                ]
                .astype(float)
                .values
            )

            daily_strategy_returns[
                strategy
            ].loc[
                holding_index
            ] = (
                returns.loc[
                    holding_index,
                    ASSETS
                ]
                @ w
            )

    # --------------------------------------------------------
    # Transaction costs
    # --------------------------------------------------------

    summary_rows = []

    for strategy in strategy_names:

        daily = (
            daily_strategy_returns[
                strategy
            ]
            .dropna()
        )

        # ----------------------------------------------------
        # Deduct transaction costs as portfolio-level
        # wealth reductions.
        # ----------------------------------------------------

        equity = 1.0

        equity_curve = []

        for date, ret in daily.items():

            equity *= (
                1.0 + ret
            )

            equity_curve.append(
                (
                    date,
                    equity,
                )
            )

        equity_df = pd.DataFrame(
            equity_curve,
            columns=[
                "date",
                "gross_wealth",
            ],
        )

        # Apply each rebalance cost on its actual date.
        strategy_rebalances = (
            rebalance_df[
                [
                    "date"
                ]
                + [
                    f"{strategy}__{asset}"
                    for asset in ASSETS
                ]
            ]
            .copy()
        )

        previous_weights = None

        for _, row in (
            strategy_rebalances
            .iterrows()
        ):

            w = row[
                [
                    f"{strategy}__{asset}"
                    for asset in ASSETS
                ]
            ].astype(float).values

            if previous_weights is None:

                previous_weights = w
                continue

            trade = np.abs(
                w
                - previous_weights
            ).sum()

            cost = (
                trade
                * TRANSACTION_COST
            )

            transaction_costs[
                strategy
            ] += cost

            previous_weights = w

        # ----------------------------------------------------
        # More accurate net return series:
        #
        # Deduct the cost at each rebalance date from the
        # first following portfolio observation.
        # ----------------------------------------------------

        net_daily = daily.copy()

        previous_weights = None

        for i, row in (
            strategy_rebalances
            .iterrows()
        ):

            w = row[
                [
                    f"{strategy}__{asset}"
                    for asset in ASSETS
                ]
            ].astype(float).values

            if previous_weights is None:

                previous_weights = w
                continue

            trade = np.abs(
                w
                - previous_weights
            ).sum()

            cost = (
                trade
                * TRANSACTION_COST
            )

            date = pd.Timestamp(
                row["date"]
            )

            future_dates = (
                net_daily.index[
                    net_daily.index > date
                ]
            )

            if len(
                future_dates
            ) > 0:

                first_date = (
                    future_dates[0]
                )

                net_daily.loc[
                    first_date
                ] -= cost

            previous_weights = w

        metrics = calculate_metrics(
            net_daily
        )

        metrics[
            "Turnover"
        ] = (
            sum(
                np.abs(
                    rebalance_df[
                        [
                            f"{strategy}__{asset}"
                            for asset in ASSETS
                        ]
                    ]
                    .diff()
                    .iloc[1:]
                    .values
                )
                .sum(axis=1)
            )
        )

        metrics[
            "Transaction Cost"
        ] = (
            transaction_costs[
                strategy
            ]
        )

        metrics[
            "Strategy"
        ] = strategy

        summary_rows.append(
            metrics
        )

        # Save daily series
        daily_output = pd.DataFrame(
            {
                "date":
                    net_daily.index,

                "return":
                    net_daily.values,
            }
        )

        daily_output.to_csv(
            OUTPUT_DIR
            / (
                strategy
                .lower()
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "-",
                    "",
                )
                + "_daily_returns.csv"
            ),
            index=False,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary[
        [
            "Strategy",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "Calmar",
            "Final Wealth",
            "VaR 95",
            "CVaR 95",
            "Turnover",
            "Transaction Cost",
        ]
    ]

    return (
        summary,
        rebalance_df,
        daily_strategy_returns,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("QUANTOS HMM-CONDITIONAL ALLOCATION BACKTEST")
    print("=" * 70)

    returns = load_asset_matrix()

    (
        summary,
        rebalance_df,
        daily_returns,
    ) = run_backtest(
        returns
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "hmm_allocation_summary.csv"
    )

    rebalance_path = (
        OUTPUT_DIR
        / "hmm_allocation_rebalances.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    rebalance_df.to_csv(
        rebalance_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        summary_path
    )

    print(
        rebalance_path
    )


if __name__ == "__main__":
    main()