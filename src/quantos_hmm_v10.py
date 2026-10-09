
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hmmlearn.hmm import GaussianHMM
from sqlalchemy import create_engine, text
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ============================================================
# QUANTOS HMM V10
# ============================================================
# Final 3-state daily regime model:
#     BULL / SIDE / BEAR
#
# WHY V9 EXISTS
# ----------------
# V8 exposed two different quantities as though they were the
# same thing:
#
#   1. HMM filtered posterior:
#        "How strongly does the Gaussian HMM believe today's
#         observation belongs to a latent state?"
#
#   2. Next-day regime probability:
#        "Given today's state, how likely is each state tomorrow?"
#
# A Gaussian HMM can legitimately become extremely concentrated
# when several correlated technical features separate the states.
# V8 therefore produced raw posteriors around 99-100%.
#
# V9 keeps the HMM for CURRENT causal regime inference, but uses
# a transparent empirical Markov transition layer for TOMORROW.
#
# The transition layer follows the reference TradingView method:
#
#     20-day log return > +5%  -> BULL
#     20-day log return < -5%  -> BEAR
#     otherwise                -> SIDE
#
# The transition matrix is then counted from those observable
# regimes, Laplace-smoothed, and used with the HMM's current
# filtered probability vector.
#
# Finally, a temperature calibration is learned ONLY on the
# 2021-2024 validation periods and frozen before the 2025-2026
# final OOS test.
#
# There is deliberately NO artificial "89% cap". If the honest
# calibrated result exceeds 90%, the model reports it rather than
# hiding it.
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://dhruvil@localhost:5432/quantos",
)

ENGINE = create_engine(DATABASE_URL)

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

DISPLAY_NAMES = {
    "NIFTY50": "NIFTY 50",
    "NIFTYBANK": "NIFTY BANK",
    "NIFTYIT": "NIFTY IT",
    "NIFTYAUTO": "NIFTY AUTO",
    "NIFTYPHARMA": "NIFTY PHARMA",
    "NIFTYNEXT50": "NIFTY NEXT 50",
    "NASDAQ100": "NASDAQ 100",
    "SP500": "S&P 500",
}

OUT = Path("./v10_outputs")
OUT.mkdir(parents=True, exist_ok=True)

N_STATES = 3
STATE_NAMES = ["BULL", "SIDE", "BEAR"]

LOOKBACK = 20
BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05

# Small-count smoothing prevents exact 0% / 100% transition cells.
LAPLACE_ALPHA = 2.0

# Only these causal daily features are used by the HMM.
# Redundant trend/momentum composites from earlier versions are
# intentionally removed.
HMM_FEATURES = [
    "log_return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]

N_RESTARTS = 12
SEEDS = [42 + i * 7 for i in range(N_RESTARTS)]

# V10 robustness settings.
# A trailing window reduces regime drift from very old market behaviour.
TRAINING_YEARS = 5
TRANSITION_LOOKBACK = 756  # roughly 3 trading years

CALIBRATION_PERIODS = [
    ("VAL_2021_2022", "2021-01-01", "2022-12-31"),
    ("VAL_2023_2024", "2023-01-01", "2024-12-31"),
]

FINAL_PERIOD = (
    "FINAL_TEST_2025_2026",
    "2025-01-01",
    "2026-12-31",
)


# ============================================================
# DATA
# ============================================================

def load_symbol_data(symbol: str) -> pd.DataFrame:
    query = text("""
        SELECT
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume,
            return_1d,
            log_return_1d,
            volatility_20,
            rsi_14,
            volume_ratio,
            atr_14,
            close_vs_sma20,
            close_vs_sma50
        FROM market_features
        WHERE symbol = :symbol
        ORDER BY timestamp
    """)

    df = pd.read_sql(query, ENGINE, params={"symbol": symbol})

    if df.empty:
        raise ValueError(f"No PostgreSQL data found for {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Recompute the core features from raw OHLCV so V9 is not
    # dependent on a stale feature implementation.
    df["log_return_1d"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    df["volatility_20"] = (
        df["log_return_1d"]
        .rolling(20, min_periods=20)
        .std()
    )

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    volume_ma = (
        df["volume"]
        .rolling(20, min_periods=20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] / volume_ma.replace(0, np.nan)
    )

    sma20 = (
        df["close"]
        .rolling(20, min_periods=20)
        .mean()
    )

    sma50 = (
        df["close"]
        .rolling(50, min_periods=50)
        .mean()
    )

    df["close_vs_sma20"] = df["close"] / sma20 - 1.0
    df["close_vs_sma50"] = df["close"] / sma50 - 1.0

    # Mild feature clipping protects Gaussian estimation from
    # pathological observations without clipping ordinary data.
    df["volume_ratio"] = df["volume_ratio"].clip(0.25, 4.0)
    df["rsi_14"] = df["rsi_14"].clip(1.0, 99.0)
    df["volatility_20"] = df["volatility_20"].clip(lower=1e-6)
    df["close_vs_sma20"] = df["close_vs_sma20"].clip(-0.50, 0.50)
    df["close_vs_sma50"] = df["close_vs_sma50"].clip(-0.75, 0.75)

    # ========================================================
    # REFERENCE OBSERVABLE REGIME
    # ========================================================
    # This intentionally mirrors the supplied TradingView method.
    rolling_log_return = np.log(
        df["close"] / df["close"].shift(LOOKBACK)
    )

    df["rule_regime"] = np.select(
        [
            rolling_log_return > BULL_THRESHOLD,
            rolling_log_return < BEAR_THRESHOLD,
        ],
        [
            0,  # BULL
            2,  # BEAR
        ],
        default=1,  # SIDE
    ).astype(float)

    return df


def clean_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "timestamp",
        "close",
        "rule_regime",
        *HMM_FEATURES,
    ]

    out = df[needed].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    out = (
        out.dropna(subset=HMM_FEATURES + ["rule_regime"])
        .reset_index(drop=True)
    )

    out["rule_regime"] = out["rule_regime"].astype(int)

    return out


# ============================================================
# SCALER
# ============================================================

def fit_scaler(X: np.ndarray):
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return mean, std


def transform(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    # Standardisation is fitted only on the relevant training window.
    # A final robust clip prevents one abnormal market observation from
    # making a Gaussian state covariance numerically pathological.
    Z = (X - mean) / std
    return np.clip(Z, -6.0, 6.0)


# ============================================================
# HMM FITTING
# ============================================================

def fit_best_hmm(X: np.ndarray):
    """
    Fit multiple 3-state HMM restarts and select the best viable model.

    V9.0 had an overly aggressive health gate:
        max(covariance) < 25
        min(covariance) > 1e-5
        every state >= 5% occupancy

    That is not a sound convergence criterion. A scaled Gaussian state
    can legitimately have variance >25, especially during volatile
    historical periods. The old gate could therefore reject every
    otherwise usable restart and crash before the actual model ran.

    V10 uses:
      - finite covariance parameters
      - finite transition probabilities
      - convergence
      - at least 1% state occupancy as a pathological-state check

    Persistence and concentration are diagnostics, not reasons to make
    the training process fail. The next-day probability is separately
    calibrated on validation data.
    """

    best = None
    results = []

    for seed in SEEDS:
        try:
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=500,
                tol=1e-4,
                min_covar=0.05,
                random_state=seed,
                init_params="stmc",
                params="stmc",
            )

            model.fit(X)

            ll = float(model.score(X))
            states = model.predict(X)

            counts = np.bincount(
                states,
                minlength=N_STATES,
            )

            occupancy = (
                counts / max(len(states), 1)
            )

            max_self = float(
                np.max(np.diag(model.transmat_))
            )

            covars = np.asarray(
                model.covars_,
                dtype=float,
            )

            finite_parameters = (
                np.all(np.isfinite(model.means_))
                and np.all(np.isfinite(covars))
                and np.all(np.isfinite(model.transmat_))
                and np.all(np.isfinite(model.startprob_))
            )

            converged = bool(
                model.monitor_.converged
            )

            min_occupancy = float(
                np.min(occupancy)
            )

            # 1% is only a pathological-state guard.
            # We do NOT require an arbitrary 5% occupancy.
            viable = (
                converged
                and finite_parameters
                and min_occupancy >= 0.01
                and np.all(covars > 1e-8)
            )

            ll_per_obs = ll / len(X)

            results.append(
                {
                    "seed": seed,
                    "ll_per_obs": ll_per_obs,
                    "max_self": max_self,
                    "min_occupancy": min_occupancy,
                    "max_covariance": float(np.max(covars)),
                    "min_covariance": float(np.min(covars)),
                    "converged": converged,
                    "finite_parameters": finite_parameters,
                    "viable": viable,
                    "iterations": int(model.monitor_.iter),
                }
            )

            if viable and (
                best is None
                or ll_per_obs > best[0]
            ):
                best = (
                    ll_per_obs,
                    model,
                    seed,
                )

        except Exception as exc:
            results.append(
                {
                    "seed": seed,
                    "error": str(exc),
                }
            )

    # If every restart failed the 1% occupancy guard, use the best
    # converged finite model rather than crashing. The diagnostics will
    # make the pathological fit visible to us.
    if best is None:
        candidates = [
            r for r in results
            if r.get("converged")
            and r.get("finite_parameters")
        ]

        if candidates:
            fallback_seed = max(
                candidates,
                key=lambda r: r["ll_per_obs"],
            )["seed"]

            # Refit the selected fallback seed deterministically.
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=500,
                tol=1e-4,
                min_covar=0.05,
                random_state=int(fallback_seed),
                init_params="stmc",
                params="stmc",
            )
            model.fit(X)

            best = (
                float(model.score(X) / len(X)),
                model,
                int(fallback_seed),
            )

    if best is None:
        raise RuntimeError(
            "No finite/converged 3-state HMM could be fitted. "
            "Inspect v10_restart_diagnostics.csv."
        )

    return (
        best[1],
        best[2],
        pd.DataFrame(results),
    )


# ============================================================
# CAUSAL FILTER
# ============================================================

def logsumexp(
    x: np.ndarray,
    axis=None,
    keepdims=False,
):
    x = np.asarray(x)

    m = np.max(
        x,
        axis=axis,
        keepdims=True,
    )

    out = (
        m
        + np.log(
            np.sum(
                np.exp(x - m),
                axis=axis,
                keepdims=True,
            )
        )
    )

    if not keepdims and axis is not None:
        out = np.squeeze(
            out,
            axis=axis,
        )

    return out


def gaussian_log_emission(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
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

    for k in range(n_states):
        mean = means[k]

        if covars.ndim == 3:
            variance = np.diag(covars[k])
        else:
            variance = covars[k]

        variance = (
            np.asarray(variance)
            .reshape(-1)
        )

        variance = np.maximum(
            variance,
            1e-8,
        )

        diff = X - mean

        quadratic = np.sum(
            (diff * diff) / variance,
            axis=1,
        )

        log_det = np.sum(
            np.log(variance)
        )

        out[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi)
            + log_det
            + quadratic
        )

    return out


def filtered_posterior_continuation(
    model: GaussianHMM,
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Causal forward filter.

    First process all training observations to obtain the posterior
    at the end of training. Then process evaluation observations
    sequentially. No future evaluation observation is used.
    """

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

    log_B_train = gaussian_log_emission(
        model,
        X_train,
    )

    alpha = (
        log_pi
        + log_B_train[0]
    )

    alpha -= logsumexp(alpha)

    for t in range(1, len(X_train)):
        alpha = (
            log_B_train[t]
            + logsumexp(
                alpha[:, None] + log_A,
                axis=0,
            )
        )

        alpha -= logsumexp(alpha)

    log_B_eval = gaussian_log_emission(
        model,
        X_eval,
    )

    posteriors = []

    for t in range(len(X_eval)):
        alpha = (
            log_B_eval[t]
            + logsumexp(
                alpha[:, None] + log_A,
                axis=0,
            )
        )

        alpha -= logsumexp(alpha)

        posteriors.append(
            np.exp(alpha)
        )

    return np.asarray(posteriors)


def filtered_posterior_all(
    model: GaussianHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Full causal filtering over X.

    This is deliberately separate from hmmlearn.predict_proba(),
    because predict_proba() uses the backward pass and therefore
    is not appropriate for a live causal dashboard.
    """

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

    log_B = gaussian_log_emission(
        model,
        X,
    )

    alpha = (
        log_pi
        + log_B[0]
    )

    alpha -= logsumexp(alpha)

    posteriors = [
        np.exp(alpha)
    ]

    for t in range(1, len(X)):
        alpha = (
            log_B[t]
            + logsumexp(
                alpha[:, None] + log_A,
                axis=0,
            )
        )

        alpha -= logsumexp(alpha)

        posteriors.append(
            np.exp(alpha)
        )

    return np.asarray(posteriors)


# ============================================================
# HMM STATE -> CANONICAL REGIME
# ============================================================

def state_mapping(
    train_df: pd.DataFrame,
    states: np.ndarray,
) -> dict[int, int]:
    """
    Align unsupervised HMM states to the observable BULL/SIDE/BEAR
    reference regimes using TRAINING DATA ONLY.

    V9 mapped states only by sorting mean daily return. That is a
    reasonable descriptive convention, but it can produce a poor
    correspondence with the actual 20-day BULL/SIDE/BEAR labels used
    for evaluation. V10 fixes that label-switching problem explicitly.

    We build a 3x3 overlap matrix between HMM states and the observable
    rule regimes, then use a one-to-one assignment. No validation or
    final-test observations are used.
    """
    states = np.asarray(states, dtype=int)
    labels = train_df["rule_regime"].to_numpy(dtype=int)

    overlap = np.zeros((N_STATES, N_STATES), dtype=float)
    for h, y in zip(states, labels):
        if 0 <= h < N_STATES and 0 <= y < N_STATES:
            overlap[h, y] += 1.0

    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}

    if len(mapping) != N_STATES:
        raise RuntimeError("Could not create a unique HMM regime mapping.")

    return mapping


def remap_probabilities(
    p: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros(
        N_STATES,
        dtype=float,
    )

    for hmm_state, canonical_state in mapping.items():
        out[canonical_state] = p[hmm_state]

    return out


def remap_transition(
    A: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros(
        (N_STATES, N_STATES),
        dtype=float,
    )

    for i in range(N_STATES):
        for j in range(N_STATES):
            out[
                mapping[i],
                mapping[j],
            ] = A[i, j]

    return out


# ============================================================
# OBSERVABLE MARKOV TRANSITION LAYER
# ============================================================

def recent_transition_matrix(
    labels: np.ndarray,
    lookback: int = TRANSITION_LOOKBACK,
) -> np.ndarray:
    """Transition matrix from the most recent observable regimes.

    This is used for next-day forecasting so that the model can adapt
    to changing market persistence instead of treating 15-year-old
    transition behaviour as equally relevant to today's market.
    """
    labels = np.asarray(labels, dtype=int)
    if len(labels) > lookback:
        labels = labels[-lookback:]
    return empirical_transition_matrix(labels)


def empirical_transition_matrix(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Count transitions between the observable 20-day regimes.

    Laplace smoothing avoids exact 0% and 100% cells.
    """

    counts = np.full(
        (N_STATES, N_STATES),
        LAPLACE_ALPHA,
        dtype=float,
    )

    for a, b in zip(
        labels[:-1],
        labels[1:],
    ):
        if (
            0 <= a < N_STATES
            and 0 <= b < N_STATES
        ):
            counts[
                int(a),
                int(b),
            ] += 1.0

    return (
        counts
        / counts.sum(
            axis=1,
            keepdims=True,
        )
    )


def stationary_distribution(
    P: np.ndarray,
    iterations: int = 200,
) -> np.ndarray:
    v = np.full(
        N_STATES,
        1.0 / N_STATES,
    )

    for _ in range(iterations):
        v = v @ P

    return v / v.sum()


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================

def temperature_scale(
    p: np.ndarray,
    temperature: float,
) -> np.ndarray:
    p = np.asarray(
        p,
        dtype=float,
    )

    p = np.clip(
        p,
        1e-12,
        1.0,
    )

    q = np.exp(
        np.log(p)
        / temperature
    )

    return (
        q
        / q.sum(
            axis=-1,
            keepdims=True,
        )
    )


def multiclass_nll(
    probs: np.ndarray,
    y: np.ndarray,
) -> float:
    probs = np.clip(
        probs,
        1e-12,
        1.0,
    )

    return float(
        -np.mean(
            np.log(
                probs[
                    np.arange(len(y)),
                    y,
                ]
            )
        )
    )


def fit_temperature(
    raw_probs: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Learn the probability-softening temperature from validation.

    T=1.0 means no calibration.
    Larger T means a less concentrated distribution.

    The temperature is never learned from the final 2025-2026 test.
    """

    candidates = np.linspace(
        1.0,
        4.0,
        121,
    )

    losses = []

    for T in candidates:
        p = temperature_scale(
            raw_probs,
            T,
        )

        losses.append(
            multiclass_nll(
                p,
                y,
            )
        )

    return float(
        candidates[
            int(np.argmin(losses))
        ]
    )


# ============================================================
# STICKINESS
# ============================================================

def recent_consistency(
    labels: np.ndarray,
    lookback: int = 10,
) -> float:
    tail = labels[-lookback:]

    if len(tail) == 0:
        return 0.0

    current = tail[-1]

    return float(
        np.mean(
            tail == current
        )
    )


def current_streak(
    labels: np.ndarray,
) -> int:
    if len(labels) == 0:
        return 0

    current = labels[-1]
    streak = 0

    for x in labels[::-1]:
        if x != current:
            break

        streak += 1

    return streak


def stickiness_score(
    stay_probability: float,
    consistency: float,
    streak: int,
) -> float:
    """
    Composite explanatory score.

    It is intentionally NOT the same thing as next-day probability.
    """

    streak_component = min(
        streak / 10.0,
        1.0,
    )

    score = (
        0.60 * stay_probability
        + 0.25 * consistency
        + 0.15 * streak_component
    )

    return float(
        np.clip(
            score * 100.0,
            0.0,
            100.0,
        )
    )


# ============================================================
# WALK-FORWARD HELPERS
# ============================================================

def period_split(
    df: pd.DataFrame,
    start: str,
    end: str,
):
    train = df[
        df["timestamp"]
        < pd.Timestamp(start)
    ].copy()

    evaluation = df[
        (df["timestamp"] >= pd.Timestamp(start))
        & (df["timestamp"] <= pd.Timestamp(end))
    ].copy()

    return (
        train.reset_index(drop=True),
        evaluation.reset_index(drop=True),
    )


def fit_period_model(
    train: pd.DataFrame,
):
    # Use only the most recent five years available before the forecast
    # period. This keeps the HMM responsive to regime drift.
    cutoff = train["timestamp"].max() - pd.DateOffset(years=TRAINING_YEARS)
    fit_df = train[train["timestamp"] >= cutoff].copy().reset_index(drop=True)

    if len(fit_df) < 500:
        fit_df = train.copy().reset_index(drop=True)

    X_raw = fit_df[
        HMM_FEATURES
    ].to_numpy(float)

    mean, std = fit_scaler(X_raw)

    X_train = transform(
        X_raw,
        mean,
        std,
    )

    model, seed, restart_df = fit_best_hmm(
        X_train
    )

    mapping = state_mapping(
        fit_df,
        model.predict(X_train),
    )

    return (
        model,
        mapping,
        mean,
        std,
        restart_df,
        seed,
        fit_df,
    )


def validation_forecasts(
    full_df: pd.DataFrame,
    start: str,
    end: str,
):
    train, evaluation = period_split(
        full_df,
        start,
        end,
    )

    if (
        len(train) < 500
        or len(evaluation) < 50
    ):
        raise ValueError(
            f"Insufficient data for "
            f"{start} -> {end}"
        )

    (
        model,
        mapping,
        mean,
        std,
        restart_df,
        seed,
        fit_df,
    ) = fit_period_model(train)

    X_train = transform(
        fit_df[
            HMM_FEATURES
        ].to_numpy(float),
        mean,
        std,
    )

    X_eval = transform(
        evaluation[
            HMM_FEATURES
        ].to_numpy(float),
        mean,
        std,
    )

    posterior = filtered_posterior_continuation(
        model,
        X_train,
        X_eval,
    )

    # IMPORTANT:
    # The transition layer is built ONLY from information available
    # before the validation period.
    P_reference = recent_transition_matrix(
        fit_df[
            "rule_regime"
        ].to_numpy(int)
    )

    raw_next = np.asarray(
        [
            remap_probabilities(
                p,
                mapping,
            ) @ P_reference
            for p in posterior
        ]
    )

    # Row t predicts the regime at t+1.
    y = evaluation[
        "rule_regime"
    ].to_numpy(int)

    usable = len(y) - 1

    return {
        "raw_probs": raw_next[:usable],
        "y": y[1:],
        "evaluation": evaluation.iloc[:usable].copy(),
        "model": model,
        "mapping": mapping,
        "P_reference": P_reference,
        "seed": seed,
        "restarts": restart_df,
    }


# ============================================================
# PRODUCTION MODEL
# ============================================================

def production_output(
    df: pd.DataFrame,
    temperature: float,
):
    clean = clean_model_frame(df)

    cutoff = clean["timestamp"].max() - pd.DateOffset(years=TRAINING_YEARS)
    fit_df = clean[clean["timestamp"] >= cutoff].copy().reset_index(drop=True)
    if len(fit_df) < 500:
        fit_df = clean.copy().reset_index(drop=True)

    X_raw = fit_df[
        HMM_FEATURES
    ].to_numpy(float)

    mean, std = fit_scaler(X_raw)

    X = transform(
        X_raw,
        mean,
        std,
    )

    model, seed, restarts = fit_best_hmm(X)

    train_states = model.predict(X)

    mapping = state_mapping(
        fit_df,
        train_states,
    )

    # HMM transition matrix retained as a diagnostic.
    P_hmm = remap_transition(
        model.transmat_,
        mapping,
    )

    # V10 production transition layer.
    P_reference = recent_transition_matrix(
        fit_df[
            "rule_regime"
        ].to_numpy(int)
    )

    # Full-history transition matrix is retained for the long-run mix.
    P_longrun = empirical_transition_matrix(
        clean[
            "rule_regime"
        ].to_numpy(int)
    )

    # Causal HMM posterior over the complete available history.
#
# The model is trained on the recent training window (fit_df),
# but the dashboard needs a posterior for every observation in
# `clean`. Use the production scaler learned from fit_df and
# apply it to the complete history before causal filtering.
    X_all_raw = clean[
    HMM_FEATURES
].to_numpy(float)

    X_all = transform(
    X_all_raw,
    mean,
    std,
)

    posterior = filtered_posterior_all(
    model,
    X_all,
)

    current_raw = remap_probabilities(
        posterior[-1],
        mapping,
    )

    # Tomorrow's raw Markov forecast.
    raw_next = (
        current_raw
        @ P_reference
    )

    # Calibrated probability shown by the application.
    calibrated_next = temperature_scale(
        raw_next.reshape(1, -1),
        temperature,
    )[0]

    current_regime = int(
        np.argmax(current_raw)
    )

    next_regime = int(
        np.argmax(calibrated_next)
    )

    labels = clean[
        "rule_regime"
    ].to_numpy(int)

    consistency = recent_consistency(
        labels,
        10,
    )

    streak = current_streak(
        labels
    )

    stay_probability = float(
        calibrated_next[
            current_regime
        ]
    )

    score = stickiness_score(
        stay_probability,
        consistency,
        streak,
    )

    summary = {
        "current_regime": STATE_NAMES[current_regime],
        "current_raw_hmm_confidence": float(
            current_raw[current_regime]
        ),
        "next_regime": STATE_NAMES[next_regime],
        "next_day_probability": float(
            calibrated_next[next_regime]
        ),
        "temperature": float(
            temperature
        ),
        "stickiness_score": float(score),
        "stickiness_level": (
            "HIGH"
            if score >= 75
            else "MODERATE"
            if score >= 50
            else "LOW"
        ),
        "recent_consistency": float(
            consistency
        ),
        "current_regime_streak": int(
            streak
        ),
        "stay_probability": float(
            stay_probability
        ),
        "hmm_max_self_transition": float(
            np.max(
                np.diag(P_hmm)
            )
        ),
        "reference_max_self_transition": float(
            np.max(
                np.diag(P_reference)
            )
        ),
        "hmm_seed": int(seed),
        "latest_date": clean[
            "timestamp"
        ].iloc[-1].date().isoformat(),
        "latest_close": float(
            clean["close"].iloc[-1]
        ),
        "hmm_training_rows": int(len(fit_df)),
        "transition_lookback_rows": int(min(len(fit_df), TRANSITION_LOOKBACK)),
    }

    probability_df = pd.DataFrame(
        [
            {
                "regime": state,
                "next_day_probability": float(p),
            }
            for state, p in zip(
                STATE_NAMES,
                calibrated_next,
            )
        ]
    )

    matrix_df = pd.DataFrame(
        [
            {
                "current_regime": STATE_NAMES[r],
                "next_regime": STATE_NAMES[c],
                "probability": float(
                    P_reference[r, c]
                ),
            }
            for r in range(N_STATES)
            for c in range(N_STATES)
        ]
    )

    stationary = stationary_distribution(
        P_longrun
    )

    stationary_df = pd.DataFrame(
        [
            {
                "regime": state,
                "long_run_probability": float(p),
            }
            for state, p in zip(
                STATE_NAMES,
                stationary,
            )
        ]
    )

    # Historical daily output for the dashboard.
    daily_rows = []

    for i in range(len(clean)):
        if i == 0:
            continue

        p_current = remap_probabilities(
            posterior[i],
            mapping,
        )

        p_next_raw = (
            p_current
            @ P_reference
        )

        p_next = temperature_scale(
            p_next_raw.reshape(1, -1),
            temperature,
        )[0]

        current = int(
            np.argmax(p_current)
        )

        nxt = int(
            np.argmax(p_next)
        )

        recent = labels[
            max(0, i - 9): i + 1
        ]

        cons = float(
            np.mean(
                recent == labels[i]
            )
        )

        streak_i = current_streak(
            labels[: i + 1]
        )

        stay_i = float(
            p_next[current]
        )

        daily_rows.append(
            {
                "date": clean[
                    "timestamp"
                ].iloc[i].date().isoformat(),
                "current_regime": STATE_NAMES[current],
                "current_raw_hmm_confidence": float(
                    p_current[current]
                ),
                "next_regime": STATE_NAMES[nxt],
                "next_day_probability": float(
                    p_next[nxt]
                ),
                "bull_probability": float(
                    p_next[0]
                ),
                "side_probability": float(
                    p_next[1]
                ),
                "bear_probability": float(
                    p_next[2]
                ),
                "stay_probability": stay_i,
                "recent_consistency": cons,
                "current_regime_streak": int(
                    streak_i
                ),
                "stickiness_score": stickiness_score(
                    stay_i,
                    cons,
                    streak_i,
                ),
            }
        )

    return (
        summary,
        probability_df,
        matrix_df,
        stationary_df,
        pd.DataFrame(daily_rows),
        P_hmm,
        P_reference,
        restarts,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("QUANTOS HMM V10")
    print("=" * 78)
    print("States: BULL / SIDE / BEAR")
    print("Forecast: NEXT TRADING DAY")
    print("HMM: causal Gaussian latent-state inference")
    print("Transition layer: recent 3-year observable Markov rule")
    print("Calibration: validation-only temperature scaling")
    print("Final test: 2025-2026")
    print("=" * 78)

    loaded = {}

    validation_probability_blocks = []
    validation_target_blocks = []

    # --------------------------------------------------------
    # 1. Load all data and learn calibration data from 2021-2024.
    # --------------------------------------------------------
    for symbol in SYMBOLS:
        print(
            f"\nLoading {DISPLAY_NAMES[symbol]}..."
        )

        raw = load_symbol_data(
            symbol
        )

        clean = clean_model_frame(
            raw
        )

        raw["symbol"] = symbol
        clean["symbol"] = symbol

        loaded[symbol] = (
            raw,
            clean,
        )

        print(
            f"Rows: {len(clean):,} | "
            f"{clean['timestamp'].iloc[0].date()} -> "
            f"{clean['timestamp'].iloc[-1].date()}"
        )

        for period_name, start, end in CALIBRATION_PERIODS:
            try:
                result = validation_forecasts(
                    clean,
                    start,
                    end,
                )
            except ValueError as exc:
                print(
                    f"{period_name}: SKIPPED | {exc}"
                )
                continue

            probs = result[
                "raw_probs"
            ]

            y = result["y"]

            validation_probability_blocks.append(
                probs
            )

            validation_target_blocks.append(
                y
            )

            raw_accuracy = float(
                np.mean(
                    np.argmax(probs, axis=1)
                    == y
                )
            )

            print(
                f"{period_name}: "
                f"n={len(y):,} | "
                f"raw hit={raw_accuracy * 100:.2f}%"
            )

    if not validation_probability_blocks:
        raise RuntimeError(
            "No valid validation period is available for probability "
            "calibration. The database needs enough history before "
            "2025 for at least one validation window."
        )

    calibration_probs = np.vstack(
        validation_probability_blocks
    )

    calibration_y = np.concatenate(
        validation_target_blocks
    )

    # --------------------------------------------------------
    # 2. Learn probability temperature ONLY from validation.
    # --------------------------------------------------------
    temperature = fit_temperature(
        calibration_probs,
        calibration_y,
    )

    calibrated_validation = temperature_scale(
        calibration_probs,
        temperature,
    )

    print("\n" + "-" * 78)
    print("V10 PROBABILITY CALIBRATION")
    print("-" * 78)
    print(
        f"Temperature: {temperature:.2f}"
    )
    print(
        "Validation NLL before: "
        f"{multiclass_nll(calibration_probs, calibration_y):.5f}"
    )
    print(
        "Validation NLL after:  "
        f"{multiclass_nll(calibrated_validation, calibration_y):.5f}"
    )
    print(
        "Mean max probability before: "
        f"{np.mean(np.max(calibration_probs, axis=1)) * 100:.2f}%"
    )
    print(
        "Mean max probability after:  "
        f"{np.mean(np.max(calibrated_validation, axis=1)) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 3. Untouched final OOS test.
    # --------------------------------------------------------
    final_rows = []

    for symbol in SYMBOLS:
        raw, clean = loaded[symbol]

        result = validation_forecasts(
            clean,
            FINAL_PERIOD[1],
            FINAL_PERIOD[2],
        )

        probs = result["raw_probs"]
        y = result["y"]

        calibrated = temperature_scale(
            probs,
            temperature,
        )

        raw_pred = np.argmax(
            probs,
            axis=1,
        )

        calibrated_pred = np.argmax(
            calibrated,
            axis=1,
        )

        raw_accuracy = float(
            np.mean(
                raw_pred == y
            )
        )

        calibrated_accuracy = float(
            np.mean(
                calibrated_pred == y
            )
        )

        final_rows.append(
            {
                "symbol": symbol,
                "period": FINAL_PERIOD[0],
                "test_rows": len(y),
                "raw_next_state_hit_rate": raw_accuracy,
                "calibrated_next_state_hit_rate": calibrated_accuracy,
                "raw_mean_max_probability": float(
                    np.mean(
                        np.max(
                            probs,
                            axis=1,
                        )
                    )
                ),
                "calibrated_mean_max_probability": float(
                    np.mean(
                        np.max(
                            calibrated,
                            axis=1,
                        )
                    )
                ),
                "raw_nll": multiclass_nll(
                    probs,
                    y,
                ),
                "calibrated_nll": multiclass_nll(
                    calibrated,
                    y,
                ),
            }
        )

        print(
            f"{symbol} FINAL OOS | "
            f"raw hit={raw_accuracy * 100:.2f}% | "
            f"calibrated hit={calibrated_accuracy * 100:.2f}% | "
            f"calibrated mean max="
            f"{np.mean(np.max(calibrated, axis=1)) * 100:.2f}%"
        )

    pd.DataFrame(final_rows).to_csv(
        OUT / "v10_validation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Production model and dashboard files.
    # --------------------------------------------------------
    summaries = []
    all_probability = []
    all_matrices = []
    all_stationary = []
    all_daily = []
    all_restarts = []

    for symbol in SYMBOLS:
        _, clean = loaded[symbol]

        (
            summary,
            probability_df,
            matrix_df,
            stationary_df,
            daily_df,
            P_hmm,
            P_reference,
            restarts,
        ) = production_output(
            clean,
            temperature,
        )

        summary["symbol"] = symbol

        probability_df["symbol"] = symbol
        matrix_df["symbol"] = symbol
        stationary_df["symbol"] = symbol
        daily_df["symbol"] = symbol

        restarts = restarts.copy()
        restarts["symbol"] = symbol

        summaries.append(summary)
        all_probability.append(
            probability_df
        )
        all_matrices.append(
            matrix_df
        )
        all_stationary.append(
            stationary_df
        )
        all_daily.append(
            daily_df
        )
        all_restarts.append(
            restarts
        )

        print("\n" + "=" * 78)
        print(
            f"QUANTOS HMM V10 | "
            f"{DISPLAY_NAMES[symbol]}"
        )
        print("=" * 78)

        print(
            f"Current regime:       "
            f"{summary['current_regime']}"
        )

        print(
            f"Next trading day:     "
            f"{summary['next_regime']}"
        )

        print(
            f"Next-day probability: "
            f"{summary['next_day_probability'] * 100:.2f}%"
        )

        print(
            f"Calibration T:        "
            f"{temperature:.2f}"
        )

        print(
            f"Stickiness:           "
            f"{summary['stickiness_score']:.1f}/100 "
            f"({summary['stickiness_level']})"
        )

        print(
            f"Reference max stay:   "
            f"{summary['reference_max_self_transition'] * 100:.2f}%"
        )

        print("\nRECENT 3x3 TRANSITION MATRIX")

        print(
            pd.DataFrame(
                P_reference,
                index=STATE_NAMES,
                columns=STATE_NAMES,
            )
            .round(4)
            .to_string()
        )

        stationary = stationary_distribution(
            P_reference
        )

        print("\nLONG-RUN MIX")

        for state, p in zip(
            STATE_NAMES,
            stationary,
        ):
            print(
                f"  {state:<5} "
                f"{p * 100:6.2f}%"
            )

    pd.DataFrame(
        summaries
    ).to_csv(
        OUT / "v10_summary.csv",
        index=False,
    )

    pd.concat(
        all_probability,
        ignore_index=True,
    ).to_csv(
        OUT / "v10_daily_regime_output_all_indices.csv",
        index=False,
    )

    pd.concat(
        all_matrices,
        ignore_index=True,
    ).to_csv(
        OUT / "v10_transition_matrices.csv",
        index=False,
    )

    pd.concat(
        all_stationary,
        ignore_index=True,
    ).to_csv(
        OUT / "v10_long_run_mix.csv",
        index=False,
    )

    pd.concat(
        all_daily,
        ignore_index=True,
    ).to_csv(
        OUT / "v10_daily_stickiness_all_indices.csv",
        index=False,
    )

    pd.concat(
        all_restarts,
        ignore_index=True,
    ).to_csv(
        OUT / "v10_restart_diagnostics.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("QUANTOS HMM V10 COMPLETE")
    print("=" * 78)
    print(
        f"Calibration temperature: "
        f"{temperature:.2f}"
    )
    print("States: BULL / SIDE / BEAR")
    print("Forecast horizon: NEXT TRADING DAY")
    print(
        "Raw HMM posterior is retained only as a diagnostic; the dashboard uses calibrated next-day probabilities."
    )
    print(
        f"Outputs: {OUT.resolve()}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
