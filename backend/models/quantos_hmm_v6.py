"""
QuantOS HMM V6
Robust, causal market-regime detection.

Key properties:
- 3-state vs 4-state model selection
- diagonal Gaussian covariance
- covariance floor
- multi-start fitting
- validation-based selection, not training likelihood
- causal forward filtering
- walk-forward validation
- untouched 2025-2026 final test
- state-balance and persistence diagnostics
- same model logic can later be imported by the dashboard
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://dhruvil@localhost:5432/quantos",
)

SYMBOLS = [
    "NIFTY50",
    "NIFTYBANK",
    "NIFTYIT",
    "NIFTYAUTO",
    "NIFTYPHARMA",
    "NIFTYNEXT50",
    "NASDAQ100",
    "SP500",
]

# Keep the dashboard/research feature set compact.
FEATURES = [
    "return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]

# We test whether 4 states are actually justified.
STATE_COUNTS = [3, 4]

N_RESTARTS = 10
RANDOM_SEED = 42

MIN_TRAIN_ROWS = 500
MIN_STATE_FRACTION = 0.05
MAX_SELF_TRANSITION = 0.995

# Because features are standardized, this is a reasonable minimum
# variance per feature/state. It prevents ultra-narrow Gaussians.
COVAR_FLOOR = 0.05

# Walk-forward validation folds.
# Parameters are fitted only before each validation period.
VALIDATION_FOLDS = [
    ("VAL_2021", "2021-01-01", "2021-12-31", "2020-12-31"),
    ("VAL_2023_2024", "2023-01-01", "2024-12-31", "2022-12-31"),
]

# NEVER use this period for model selection.
FINAL_TEST_START = "2025-01-01"
FINAL_TEST_END = "2026-12-31"

OUTPUT_DIR = Path("v6_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL)


# ============================================================
# DATA LOADING
# ============================================================

def load_symbol_data(symbol: str) -> pd.DataFrame:
    query = text("""
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
        WHERE symbol = :symbol
        ORDER BY timestamp
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"symbol": symbol},
    )

    if df.empty:
        raise ValueError(
            f"No PostgreSQL data found for {symbol}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for col in FEATURES + ["close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES + ["close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"{symbol}: only {len(df)} usable rows; "
            f"need at least {MIN_TRAIN_ROWS}"
        )

    return df


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def logsumexp(
    x: np.ndarray,
    axis=None,
    keepdims=False,
):
    x = np.asarray(x, dtype=float)

    m = np.max(
        x,
        axis=axis,
        keepdims=True,
    )

    out = m + np.log(
        np.sum(
            np.exp(x - m),
            axis=axis,
            keepdims=True,
        )
    )

    if not keepdims and axis is not None:
        out = np.squeeze(
            out,
            axis=axis,
        )

    return out


# ============================================================
# CAUSAL FORWARD FILTER
# ============================================================

def gaussian_log_emission(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Calculate log P(X_t | state=k) for diagonal Gaussians.

    Implemented explicitly so the code does not depend on the
    exact covariance-array behavior of a particular hmmlearn version.
    """

    X = np.asarray(X, dtype=float)

    means = np.asarray(
        model.means_,
        dtype=float,
    )

    covars = np.asarray(
        model.covars_,
        dtype=float,
    )

    n_samples, n_features = X.shape
    n_states = means.shape[0]

    out = np.empty(
        (n_samples, n_states),
        dtype=float,
    )

    for state in range(n_states):

        mean = means[state]

        if covars.ndim == 3:
            variance = np.diag(
                covars[state]
            )
        else:
            variance = covars[state]

        variance = np.asarray(
            variance,
            dtype=float,
        ).reshape(-1)

        variance = np.maximum(
            variance,
            COVAR_FLOOR,
        )

        if len(variance) != n_features:
            raise ValueError(
                f"State {state}: covariance has "
                f"{len(variance)} values, expected "
                f"{n_features}"
            )

        diff = X - mean

        quadratic = np.sum(
            (diff * diff) / variance,
            axis=1,
        )

        log_det = np.sum(
            np.log(variance)
        )

        out[:, state] = -0.5 * (
            n_features * np.log(2.0 * np.pi)
            + log_det
            + quadratic
        )

    return out


def filtered_posterior(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Causal forward filter:

        P(z_t | x_1:t)

    No backward pass is used.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    if len(X) == 0:
        return np.empty(
            (0, model.n_components)
        )

    log_B = gaussian_log_emission(
        model,
        X,
    )

    log_A = np.log(
        np.maximum(
            model.transmat_,
            1e-300,
        )
    )

    log_pi = np.log(
        np.maximum(
            model.startprob_,
            1e-300,
        )
    )

    alpha = (
        log_pi
        + log_B[0]
    )

    alpha -= logsumexp(alpha)

    posterior = [
        np.exp(alpha)
    ]

    for t in range(1, len(X)):

        alpha = (
            log_B[t]
            + logsumexp(
                alpha[:, None]
                + log_A,
                axis=0,
            )
        )

        alpha -= logsumexp(alpha)

        posterior.append(
            np.exp(alpha)
        )

    return np.asarray(
        posterior
    )


# ============================================================
# MODEL FITTING
# ============================================================

def parameter_count(
    n_states: int,
    n_features: int,
) -> int:
    """
    Parameters for:
      start probabilities
      transition matrix
      diagonal Gaussian means
      diagonal Gaussian variances
    """

    return (
        (n_states - 1)
        + n_states * (n_states - 1)
        + n_states * n_features
        + n_states * n_features
    )


def fit_best_hmm(
    X: np.ndarray,
    n_states: int,
    n_restarts: int = N_RESTARTS,
):
    """
    Fit multiple random initialisations for one state count.

    IMPORTANT:
    This function only selects the best converged restart on the
    training set. The 3-vs-4 state decision happens later using
    validation data.
    """

    best_model = None
    best_train_ll = -np.inf

    rows = []

    for restart in range(
        n_restarts
    ):

        seed = (
            RANDOM_SEED
            + restart
        )

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=300,
            tol=1e-3,
            min_covar=COVAR_FLOOR,
            random_state=seed,
            init_params="stmc",
            params="stmc",
            verbose=False,
        )

        try:
            model.fit(X)

            if not model.monitor_.converged:
                continue

            ll = float(
                model.score(X)
            )

            if not np.isfinite(ll):
                continue

            rows.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "train_loglik": ll,
                    "iterations": int(
                        model.monitor_.iter
                    ),
                    "converged": True,
                }
            )

            if ll > best_train_ll:
                best_train_ll = ll
                best_model = model

        except Exception as exc:

            rows.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "train_loglik": np.nan,
                    "iterations": 0,
                    "converged": False,
                    "error": repr(exc),
                }
            )

    if best_model is None:
        raise RuntimeError(
            f"No converged {n_states}-state HMM found."
        )

    return (
        best_model,
        pd.DataFrame(rows),
    )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

def model_diagnostics(
    model: GaussianHMM,
    X: np.ndarray,
) -> dict:

    posterior = filtered_posterior(
        model,
        X,
    )

    occupancy = posterior.mean(
        axis=0
    )

    self_transitions = np.diag(
        model.transmat_
    )

    low_state_count = int(
        np.sum(
            occupancy
            < MIN_STATE_FRACTION
        )
    )

    pathological_count = int(
        np.sum(
            self_transitions
            >= MAX_SELF_TRANSITION
        )
    )

    # Normalised posterior entropy.
    # 0% = nearly certain all the time.
    # 100% = maximum uncertainty.
    entropy = -np.sum(
        posterior
        * np.log(
            np.maximum(
                posterior,
                1e-12,
            )
        ),
        axis=1,
    )

    max_entropy = np.log(
        model.n_components
    )

    return {
        "mean_entropy": float(
            np.mean(entropy)
        ),
        "mean_entropy_pct": float(
            np.mean(entropy)
            / max_entropy
        )
        if max_entropy > 0
        else 0.0,
        "min_occupancy": float(
            np.min(occupancy)
        ),
        "max_occupancy": float(
            np.max(occupancy)
        ),
        "low_state_count": low_state_count,
        "max_self_transition": float(
            np.max(self_transitions)
        ),
        "pathological_count": pathological_count,
        "occupancy": occupancy,
        "self_transitions": self_transitions,
    }


def selection_score(
    validation_loglik_per_obs: float,
    n_states: int,
    n_features: int,
    n_validation: int,
    diagnostics: dict,
) -> float:
    """
    Higher is better.

    Validation likelihood is the main signal.

    Additional modest penalties:
      - model complexity
      - unused states
      - pathological persistence
      - near-total posterior concentration
    """

    p = parameter_count(
        n_states,
        n_features,
    )

    complexity_penalty = (
        0.5
        * p
        * np.log(
            max(n_validation, 2)
        )
        / max(n_validation, 1)
    )

    quality_penalty = (
        0.10
        * diagnostics[
            "low_state_count"
        ]
        + 0.50
        * diagnostics[
            "pathological_count"
        ]
    )

    concentration_penalty = max(
        0.0,
        0.25
        - diagnostics[
            "mean_entropy_pct"
        ],
    )

    return float(
        validation_loglik_per_obs
        - complexity_penalty
        - quality_penalty
        - concentration_penalty
    )


# ============================================================
# REGIME MAPPING
# ============================================================

def build_state_profiles(
    df: pd.DataFrame,
    posterior: np.ndarray,
) -> pd.DataFrame:
    states = np.argmax(
        posterior,
        axis=1,
    )

    rows = []

    for state in range(
        posterior.shape[1]
    ):

        subset = df.loc[
            states == state
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "state": state,
                "observations": len(
                    subset
                ),
                "mean_return":
                    subset[
                        "return_1d"
                    ].mean(),
                "mean_volatility":
                    subset[
                        "volatility_20"
                    ].mean(),
                "mean_rsi":
                    subset[
                        "rsi_14"
                    ].mean(),
                "mean_volume_ratio":
                    subset[
                        "volume_ratio"
                    ].mean(),
                "mean_close_vs_sma20":
                    subset[
                        "close_vs_sma20"
                    ].mean(),
                "mean_close_vs_sma50":
                    subset[
                        "close_vs_sma50"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


def map_states_to_regimes(
    profile: pd.DataFrame,
) -> dict[int, str]:
    """
    4 states:
        highest mean return -> BULL
        lowest mean return  -> BEAR
        highest volatility among remaining -> STRESS
        remaining -> SIDE

    3 states:
        highest mean return -> BULL
        lowest mean return  -> BEAR
        remaining -> SIDE
    """

    states = (
        profile["state"]
        .astype(int)
        .tolist()
    )

    if len(states) < 3:
        raise ValueError(
            "Fewer than 3 populated states; "
            "model is not usable."
        )

    mapping = {}

    bull = int(
        profile.sort_values(
            [
                "mean_return",
                "mean_rsi",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]["state"]
    )

    bear = int(
        profile.sort_values(
            [
                "mean_return",
                "mean_rsi",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .iloc[0]["state"]
    )

    if bull == bear:
        raise ValueError(
            "Unable to distinguish BULL and BEAR states."
        )

    mapping[bull] = "BULL"
    mapping[bear] = "BEAR"

    remaining = [
        s
        for s in states
        if s not in mapping
    ]

    if len(states) == 4:

        stress = int(
            profile[
                profile["state"].isin(
                    remaining
                )
            ]
            .sort_values(
                "mean_volatility",
                ascending=False,
            )
            .iloc[0]["state"]
        )

        mapping[stress] = "STRESS"

        for s in remaining:
            if s != stress:
                mapping[s] = "SIDE"

    else:
        mapping[
            remaining[0]
        ] = "SIDE"

    return mapping


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def evaluate_candidate_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    n_states: int,
):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[
            FEATURES
        ].astype(float)
    )

    X_validation = scaler.transform(
        validation[
            FEATURES
        ].astype(float)
    )

    model, restarts = fit_best_hmm(
        X_train,
        n_states=n_states,
    )

    val_ll = float(
        model.score(
            X_validation
        )
    )

    val_ll_per_obs = (
        val_ll
        / len(X_validation)
    )

    diagnostics = model_diagnostics(
        model,
        X_train,
    )

    score = selection_score(
        validation_loglik_per_obs=
            val_ll_per_obs,
        n_states=n_states,
        n_features=len(FEATURES),
        n_validation=len(
            X_validation
        ),
        diagnostics=diagnostics,
    )

    return {
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
        "validation_loglik": val_ll,
        "validation_loglik_per_obs":
            val_ll_per_obs,
        "selection_score": score,
        "diagnostics": diagnostics,
    }


def select_state_count(
    df: pd.DataFrame,
):

    fold_rows = []

    for (
        fold_name,
        val_start,
        val_end,
        train_end,
    ) in VALIDATION_FOLDS:

        train = df[
            df["timestamp"]
            <= pd.Timestamp(
                train_end
            )
        ].copy()

        validation = df[
            (
                df["timestamp"]
                >= pd.Timestamp(
                    val_start
                )
            )
            & (
                df["timestamp"]
                <= pd.Timestamp(
                    val_end
                )
            )
        ].copy()

        if len(train) < MIN_TRAIN_ROWS:
            print(
                f"  {fold_name}: skipped, "
                f"only {len(train)} train rows"
            )
            continue

        if len(validation) < 30:
            print(
                f"  {fold_name}: skipped, "
                f"only {len(validation)} validation rows"
            )
            continue

        for n_states in STATE_COUNTS:

            result = (
                evaluate_candidate_fold(
                    train,
                    validation,
                    n_states,
                )
            )

            d = result[
                "diagnostics"
            ]

            fold_rows.append(
                {
                    "fold": fold_name,
                    "n_states": n_states,
                    "train_rows": len(
                        train
                    ),
                    "validation_rows": len(
                        validation
                    ),
                    "validation_loglik":
                        result[
                            "validation_loglik"
                        ],
                    "validation_loglik_per_obs":
                        result[
                            "validation_loglik_per_obs"
                        ],
                    "selection_score":
                        result[
                            "selection_score"
                        ],
                    "mean_entropy_pct":
                        d[
                            "mean_entropy_pct"
                        ],
                    "min_occupancy":
                        d[
                            "min_occupancy"
                        ],
                    "max_occupancy":
                        d[
                            "max_occupancy"
                        ],
                    "max_self_transition":
                        d[
                            "max_self_transition"
                        ],
                    "low_state_count":
                        d[
                            "low_state_count"
                        ],
                    "pathological_count":
                        d[
                            "pathological_count"
                        ],
                }
            )

            print(
                f"    {fold_name} | "
                f"{n_states} states | "
                f"val LL/obs="
                f"{result['validation_loglik_per_obs']:.4f} | "
                f"score="
                f"{result['selection_score']:.4f} | "
                f"entropy="
                f"{d['mean_entropy_pct']:.2%}"
            )

    if not fold_rows:
        raise RuntimeError(
            "No usable validation folds were produced."
        )

    folds = pd.DataFrame(
        fold_rows
    )

    summary = (
        folds.groupby(
            "n_states"
        )
        .agg(
            mean_selection_score=(
                "selection_score",
                "mean",
            ),
            mean_validation_ll_per_obs=(
                "validation_loglik_per_obs",
                "mean",
            ),
            mean_entropy_pct=(
                "mean_entropy_pct",
                "mean",
            ),
            worst_min_occupancy=(
                "min_occupancy",
                "min",
            ),
            worst_max_self_transition=(
                "max_self_transition",
                "max",
            ),
            total_pathological=(
                "pathological_count",
                "sum",
            ),
        )
        .reset_index()
    )

    summary = (
        summary.sort_values(
            "mean_selection_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    selected_states = int(
        summary.iloc[0][
            "n_states"
        ]
    )

    return (
        selected_states,
        folds,
        summary,
    )


# ============================================================
# FINAL MODEL
# ============================================================

def fit_final_model(
    df: pd.DataFrame,
    n_states: int,
):

    train = df[
        df["timestamp"]
        < pd.Timestamp(
            FINAL_TEST_START
        )
    ].copy()

    final_test = df[
        (
            df["timestamp"]
            >= pd.Timestamp(
                FINAL_TEST_START
            )
        )
        & (
            df["timestamp"]
            <= pd.Timestamp(
                FINAL_TEST_END
            )
        )
    ].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Final training set too small: "
            f"{len(train)}"
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[
            FEATURES
        ].astype(float)
    )

    model, restarts = fit_best_hmm(
        X_train,
        n_states=n_states,
    )

    train_posterior = (
        filtered_posterior(
            model,
            X_train,
        )
    )

    profile = (
        build_state_profiles(
            train,
            train_posterior,
        )
    )

    mapping = (
        map_states_to_regimes(
            profile
        )
    )

    diagnostics = (
        model_diagnostics(
            model,
            X_train,
        )
    )

    current = None
    final_metrics = None

    if not final_test.empty:

        # Apply the training-only scaler to the full historical
        # sequence. The model parameters themselves were fitted
        # only before 2025.
        X_all = scaler.transform(
            df[
                FEATURES
            ].astype(float)
        )

        posterior_all = (
            filtered_posterior(
                model,
                X_all,
            )
        )

        current_probs = (
            posterior_all[-1]
        )

        current_state = int(
            np.argmax(
                current_probs
            )
        )

        current_regime = (
            mapping[
                current_state
            ]
        )

        # P(z_t+1) =
        # P(z_t | x_1:t) * A
        next_probs = (
            current_probs
            @ model.transmat_
        )

        current_regime_probs = {
            mapping[i]: float(
                current_probs[i]
            )
            for i in range(
                n_states
            )
        }

        next_regime_probs = {
            mapping[i]: float(
                next_probs[i]
            )
            for i in range(
                n_states
            )
        }

        final_X = scaler.transform(
            final_test[
                FEATURES
            ].astype(float)
        )

        final_ll = float(
            model.score(
                final_X
            )
        )

        final_metrics = {
            "test_rows": len(
                final_test
            ),
            "test_loglik": final_ll,
            "test_loglik_per_obs":
                final_ll
                / len(final_test),
        }

        current = {
            "date":
                df[
                    "timestamp"
                ].iloc[-1],
            "state":
                current_state,
            "regime":
                current_regime,
            "confidence":
                float(
                    current_probs[
                        current_state
                    ]
                ),
            "probabilities":
                current_regime_probs,
            "next_probabilities":
                next_regime_probs,
            "next_regime":
                max(
                    next_regime_probs,
                    key=next_regime_probs.get,
                ),
        }

    return {
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
        "profile": profile,
        "mapping": mapping,
        "diagnostics": diagnostics,
        "current": current,
        "final_metrics": final_metrics,
        "train_rows": len(train),
        "final_test_rows":
            len(final_test),
    }


# ============================================================
# OUTPUT
# ============================================================

def save_symbol_outputs(
    symbol: str,
    selected_states: int,
    folds: pd.DataFrame,
    state_summary: pd.DataFrame,
    final: dict,
):

    prefix = symbol.lower()

    folds.to_csv(
        OUTPUT_DIR
        / f"{prefix}_v6_validation_folds.csv",
        index=False,
    )

    state_summary.to_csv(
        OUTPUT_DIR
        / f"{prefix}_v6_model_comparison.csv",
        index=False,
    )

    profile = (
        final["profile"]
        .copy()
    )

    profile["symbol"] = symbol
    profile["selected_states"] = (
        selected_states
    )
    profile["regime"] = (
        profile["state"].map(
            final["mapping"]
        )
    )

    profile.to_csv(
        OUTPUT_DIR
        / f"{prefix}_v6_state_profiles.csv",
        index=False,
    )

    transition = pd.DataFrame(
        final["model"].transmat_,
        index=[
            final["mapping"][i]
            for i in range(
                selected_states
            )
        ],
        columns=[
            final["mapping"][i]
            for i in range(
                selected_states
            )
        ],
    )

    transition.to_csv(
        OUTPUT_DIR
        / f"{prefix}_v6_transition_matrix.csv"
    )

    diagnostics = (
        final["diagnostics"]
    )

    current = final["current"]

    summary_row = {
        "symbol": symbol,
        "selected_states":
            selected_states,
        "train_rows":
            final["train_rows"],
        "final_test_rows":
            final["final_test_rows"],
        "mean_entropy_pct":
            diagnostics[
                "mean_entropy_pct"
            ],
        "min_occupancy":
            diagnostics[
                "min_occupancy"
            ],
        "max_occupancy":
            diagnostics[
                "max_occupancy"
            ],
        "max_self_transition":
            diagnostics[
                "max_self_transition"
            ],
        "pathological_count":
            diagnostics[
                "pathological_count"
            ],
        "current_regime":
            current["regime"]
            if current
            else None,
        "current_confidence":
            current["confidence"]
            if current
            else None,
        "next_regime":
            current["next_regime"]
            if current
            else None,
    }

    if final[
        "final_metrics"
    ]:
        summary_row.update(
            final[
                "final_metrics"
            ]
        )

    return summary_row


# ============================================================
# RUN ONE SYMBOL
# ============================================================

def run_symbol(
    symbol: str,
):

    print("\n" + "=" * 78)
    print(
        f"QUANTOS HMM V6 | {symbol}"
    )
    print("=" * 78)

    df = load_symbol_data(
        symbol
    )

    print(
        f"Rows: {len(df):,} | "
        f"{df['timestamp'].min().date()} -> "
        f"{df['timestamp'].max().date()}"
    )

    print(
        "\nMODEL SELECTION: "
        "3 vs 4 STATES"
    )

    (
        selected_states,
        folds,
        state_summary,
    ) = select_state_count(
        df
    )

    print("\nMODEL COMPARISON")

    print(
        state_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}",
        )
    )

    print(
        f"\nSELECTED MODEL: "
        f"{selected_states} states"
    )

    final = fit_final_model(
        df,
        selected_states,
    )

    print(
        "\nFINAL MODEL DIAGNOSTICS"
    )

    d = final[
        "diagnostics"
    ]

    print(
        "Mean posterior entropy: "
        f"{d['mean_entropy_pct']:.2%} "
        "of maximum"
    )

    print(
        "State occupancy range: "
        f"{d['min_occupancy']:.2%} -> "
        f"{d['max_occupancy']:.2%}"
    )

    print(
        "Maximum self-transition: "
        f"{d['max_self_transition']:.2%}"
    )

    print(
        "Pathological states: "
        f"{d['pathological_count']}"
    )

    print(
        "\nSTATE PROFILES"
    )

    profile_display = (
        final["profile"].copy()
    )

    profile_display["regime"] = (
        profile_display[
            "state"
        ].map(
            final["mapping"]
        )
    )

    print(
        profile_display[
            [
                "state",
                "regime",
                "observations",
                "mean_return",
                "mean_volatility",
                "mean_rsi",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\nTRANSITION MATRIX"
    )

    transition = pd.DataFrame(
        final["model"].transmat_,
        index=[
            final["mapping"][i]
            for i in range(
                selected_states
            )
        ],
        columns=[
            final["mapping"][i]
            for i in range(
                selected_states
            )
        ],
    )

    print(
        transition.round(4)
        .to_string()
    )

    if final["current"]:

        current = final[
            "current"
        ]

        print(
            "\nCURRENT CAUSAL REGIME"
        )

        print(
            f"Date:       "
            f"{current['date'].date()}"
        )

        print(
            f"Regime:     "
            f"{current['regime']}"
        )

        print(
            f"Confidence: "
            f"{current['confidence']:.2%}"
        )

        print(
            f"Next state: "
            f"{current['next_regime']}"
        )

        print(
            "\nCURRENT FILTERED "
            "PROBABILITIES"
        )

        for (
            regime,
            probability,
        ) in sorted(
            current[
                "probabilities"
            ].items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            print(
                f"{regime:<8} "
                f"{probability:.4f}"
            )

        print(
            "\n1-STEP FORECAST "
            "PROBABILITIES"
        )

        for (
            regime,
            probability,
        ) in sorted(
            current[
                "next_probabilities"
            ].items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            print(
                f"{regime:<8} "
                f"{probability:.4f}"
            )

    if final[
        "final_metrics"
    ]:

        print(
            "\nFINAL TEST: 2025-2026"
        )

        print(
            "Rows:        "
            f"{final['final_metrics']['test_rows']:,}"
        )

        print(
            "LogLik/obs:  "
            f"{final['final_metrics']['test_loglik_per_obs']:.5f}"
        )

    return save_symbol_outputs(
        symbol=symbol,
        selected_states=selected_states,
        folds=folds,
        state_summary=state_summary,
        final=final,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print("QUANTOS HMM V6")
    print("=" * 78)
    print("Robust causal regime model")
    print("3-state vs 4-state selection")
    print("Diagonal Gaussian covariance")
    print("Walk-forward validation")
    print("Untouched final test: 2025-2026")
    print("=" * 78)

    summary_rows = []

    for symbol in SYMBOLS:

        try:

            summary_rows.append(
                run_symbol(
                    symbol
                )
            )

        except Exception as exc:

            print(
                f"\n{symbol} FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "v6_summary.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("V6 COMPLETE")
    print("=" * 78)
    print(
        "Summary: "
        f"{OUTPUT_DIR / 'v6_summary.csv'}"
    )
