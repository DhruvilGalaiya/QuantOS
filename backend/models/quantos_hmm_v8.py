# ================================================================
# QUANTOS HMM V8
# Final 3-state causal daily regime model
#
# STATES:
#   BULL / SIDE / BEAR
#
# V8 objectives:
#   - Independently model each index
#   - Fixed 3-state economic interpretation
#   - Daily data only; forecast horizon = NEXT TRADING DAY
#   - Causal forward filtering: P(z_t | x_1:t)
#   - No future observations used for current-regime probabilities
#   - Training-only StandardScaler for OOS evaluation
#   - Multi-start Gaussian HMM with diagonal covariance
#   - Time-based OOS validation and final test (2025-2026)
#   - No artificial probability floor or forced 50-80% confidence
#   - Separate decision-confidence metric based on posterior concentration
#   - Next-day probability forecast from current filtered posterior @ A
#   - Daily TREND STICKINESS score using:
#         1) learned probability of remaining in the current state
#         2) recent causal regime consistency
#   - Outputs suitable for the QuantOS UI
#
# IMPORTANT:
#   HMM posterior probabilities are model probabilities, not guarantees.
#   V8 deliberately keeps raw probabilities auditable. A high posterior
#   is not automatically treated as proof of predictive accuracy.
# ================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://dhruvil@localhost:5432/quantos",
)

# Each index is modelled independently.
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

N_STATES = 3
RANDOM_SEED = 42
N_RESTARTS = 12
HMM_ITERATIONS = 500
HMM_TOL = 1e-4
MIN_COVAR = 1e-3

MIN_TRAIN_ROWS = 500
MIN_STATE_FRACTION = 0.03
MAX_SELF_TRANSITION = 0.995

# Daily data => one-step forecast means one NEXT TRADING DAY.
FORECAST_HORIZON_DAYS = 1

# OOS validation periods.
VALIDATION_PERIODS = [
    ("VAL_2021_2022", "2021-01-01", "2022-12-31"),
    ("VAL_2023_2024", "2023-01-01", "2024-12-31"),
]

FINAL_TEST_START = "2025-01-01"
FINAL_TEST_END = "2026-12-31"

OUTPUT_DIR = Path("v8_outputs")

# Only information available at the end of day t.
FEATURES = [
    "return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]

# Number of recent causal days used for trend consistency.
STICKINESS_LOOKBACK = 10

# Weights are explicit and intentionally simple.
# Persistence comes from the learned Markov model; consistency comes
# from the recent causal regime path.
STICKINESS_TRANSITION_WEIGHT = 0.70
STICKINESS_RECENT_WEIGHT = 0.30

# ================================================================
# NUMERICAL HELPERS
# ================================================================

def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(values - m))))


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        return np.ones(len(p), dtype=float) / len(p)
    return p / total


def normalized_entropy(p: np.ndarray) -> float:
    p = normalize_probabilities(p)
    k = len(p)
    if k <= 1:
        return 0.0
    safe = np.clip(p, 1e-12, 1.0)
    h = -np.sum(safe * np.log(safe))
    return float(h / np.log(k))


def posterior_concentration(p: np.ndarray) -> float:
    """
    0..1 concentration score for a 3-state posterior.

    0 = completely ambiguous/equal probabilities (33/33/33)
    1 = one state has probability 100%

    This is a descriptive concentration metric, NOT calibrated accuracy.
    """
    p = normalize_probabilities(p)
    k = len(p)
    if k <= 1:
        return 1.0
    return float((np.max(p) - 1.0 / k) / (1.0 - 1.0 / k))


def decision_confidence(p: np.ndarray, transition: np.ndarray) -> float:
    """
    Conservative decision-confidence score for the UI (0..100).

    It is based on the *next-day forecast*, not the raw same-day HMM
    posterior. This avoids presenting a near-100% filtered state posterior
    as though it were a near-100% prediction of tomorrow.

    The score is descriptive, not a calibrated probability of correctness.
    """
    next_p = next_day_forecast(p, transition)
    top = np.max(next_p)
    # Convert a 1/3-to-1 range into a 0-100 concentration scale.
    return 100.0 * posterior_concentration(next_p)

# ================================================================
# DATA LOADING / PREPARATION
# ================================================================

def load_symbol_data(symbol: str) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)

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
            sma_10,
            sma_20,
            sma_50,
            ema_20,
            ema_50,
            volatility_20,
            rsi_14,
            volume_sma_20,
            volume_ratio,
            high_low_range,
            high_low_range_pct,
            atr_14,
            close_vs_sma20,
            close_vs_sma50
        FROM market_features
        WHERE symbol = :symbol
        ORDER BY timestamp
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol})

    if df.empty:
        raise ValueError(f"No PostgreSQL data found for {symbol}")

    df["Date"] = pd.to_datetime(df["timestamp"])
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Evaluation-only forward return. Never supplied to the HMM.
    df["forward_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["forward_5d"] = df["Close"].shift(-5) / df["Close"] - 1.0

    return df.sort_values("Date").reset_index(drop=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for col in FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=FEATURES).copy()
    work = work.sort_values("Date").reset_index(drop=True)

    if len(work) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(work)} usable rows after feature cleaning; "
            f"need at least {MIN_TRAIN_ROWS}."
        )

    return work

# ================================================================
# HMM FITTING
# ================================================================

def fit_best_hmm(X: np.ndarray, n_states: int = N_STATES):
    best_model = None
    best_score = -np.inf
    best_seed = None
    restart_rows = []

    for restart in range(N_RESTARTS):
        seed = RANDOM_SEED + restart

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=HMM_ITERATIONS,
            tol=HMM_TOL,
            min_covar=MIN_COVAR,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )

        try:
            model.fit(X)
            score = float(model.score(X))
            converged = bool(model.monitor_.converged)
            self_transition = float(np.max(np.diag(model.transmat_)))

            restart_rows.append({
                "restart": restart,
                "seed": seed,
                "train_loglik": score,
                "converged": converged,
                "iterations": int(model.monitor_.iter),
                "max_self_transition": self_transition,
            })

            if (
                converged
                and np.isfinite(score)
                and self_transition < MAX_SELF_TRANSITION
                and score > best_score
            ):
                best_model = model
                best_score = score
                best_seed = seed

        except Exception as exc:
            restart_rows.append({
                "restart": restart,
                "seed": seed,
                "train_loglik": np.nan,
                "converged": False,
                "iterations": 0,
                "max_self_transition": np.nan,
                "error": repr(exc),
            })

    if best_model is None:
        raise RuntimeError(
            f"No valid converged {n_states}-state HMM was found."
        )

    return best_model, pd.DataFrame(restart_rows), best_seed

# ================================================================
# CAUSAL FORWARD FILTER
# ================================================================

def causal_filtered_posterior(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute P(z_t | x_1:t) using forward recursion only.

    model.predict_proba() is deliberately NOT used because it performs
    forward-backward smoothing and therefore allows later observations
    to influence earlier state probabilities.
    """
    log_emission = model._compute_log_likelihood(X)
    log_start = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-300, 1.0))

    n_obs, n_states = log_emission.shape
    filtered = np.zeros((n_obs, n_states), dtype=float)

    alpha = log_start + log_emission[0]
    alpha -= logsumexp(alpha)
    filtered[0] = np.exp(alpha)

    for t in range(1, n_obs):
        next_alpha = np.empty(n_states, dtype=float)

        for j in range(n_states):
            next_alpha[j] = (
                logsumexp(alpha + log_trans[:, j])
                + log_emission[t, j]
            )

        next_alpha -= logsumexp(next_alpha)
        alpha = next_alpha
        filtered[t] = np.exp(alpha)

    return filtered


def causal_filter_continuation(
    model: GaussianHMM,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Causally filter train + test as one chronological sequence and return
    only the test portion. This preserves the filtered state distribution
    carried forward from the training history without fitting on test data.
    """
    combined = np.vstack([X_train, X_test])
    filtered = causal_filtered_posterior(model, combined)
    return filtered[len(X_train):]

# ================================================================
# STATE INTERPRETATION
# ================================================================

def state_profile(df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    work = df.copy()
    work["state"] = states

    rows = []
    for state in sorted(np.unique(states)):
        subset = work[work["state"] == state]
        rows.append({
            "state": int(state),
            "observations": int(len(subset)),
            "mean_return": float(subset["return_1d"].mean()),
            "median_return": float(subset["return_1d"].median()),
            "mean_volatility": float(subset["volatility_20"].mean()),
            "mean_rsi": float(subset["rsi_14"].mean()),
            "mean_volume_ratio": float(subset["volume_ratio"].mean()),
            "mean_close_vs_sma20": float(subset["close_vs_sma20"].mean()),
            "mean_close_vs_sma50": float(subset["close_vs_sma50"].mean()),
        })

    return pd.DataFrame(rows).set_index("state")


def map_three_states(profile: pd.DataFrame) -> dict:
    """Map latent states to BULL / SIDE / BEAR using return ordering."""
    bull = int(profile["mean_return"].idxmax())
    bear = int(profile["mean_return"].idxmin())
    side_candidates = [s for s in profile.index if s not in (bull, bear)]

    if len(side_candidates) != 1:
        raise RuntimeError("Could not uniquely identify SIDE state.")

    return {
        bull: "BULL",
        bear: "BEAR",
        int(side_candidates[0]): "SIDE",
    }

# ================================================================
# QUALITY / STICKINESS
# ================================================================

def quality_report(model: GaussianHMM, states: np.ndarray) -> pd.DataFrame:
    total = len(states)
    rows = []

    for state in range(model.n_components):
        count = int(np.sum(states == state))
        fraction = count / total if total else 0.0
        persistence = float(model.transmat_[state, state])

        flags = []
        if fraction < MIN_STATE_FRACTION:
            flags.append("LOW_FRACTION")
        if persistence >= MAX_SELF_TRANSITION:
            flags.append("PATHOLOGICAL_PERSISTENCE")

        status = "PATHOLOGICAL" if "PATHOLOGICAL_PERSISTENCE" in flags else (
            "WEAK" if flags else "PASS"
        )

        rows.append({
            "state": state,
            "observations": count,
            "fraction": fraction,
            "self_transition": persistence,
            "status": status,
            "flags": ",".join(flags),
        })

    return pd.DataFrame(rows)


def calculate_stickiness(
    filtered: np.ndarray,
    transition: np.ndarray,
    mapping: dict,
    dates: pd.Series,
    lookback: int = STICKINESS_LOOKBACK,
) -> pd.DataFrame:
    """
    Daily trend-stickiness score.

    transition_component:
        Probability of remaining in today's inferred state tomorrow,
        averaged using today's full causal posterior.

        p_stay = sum_i P(z_t=i | x_1:t) * A[i,i]

    recent_component:
        Fraction of the last N causal daily state decisions matching
        today's most likely state. This captures "trend is your friend".

    score:
        70% transition persistence + 30% recent consistency.

    This is NOT a forecast probability. It is a 0-100 persistence score.
    """
    transition = np.asarray(transition, dtype=float)
    causal_states = np.argmax(filtered, axis=1)
    rows = []

    for t in range(len(filtered)):
        current_state = int(causal_states[t])
        current_regime = mapping[current_state]

        p_stay = float(
            np.sum(filtered[t] * np.diag(transition))
        )

        start = max(0, t - lookback + 1)
        recent_states = causal_states[start:t + 1]
        recent_consistency = float(
            np.mean(recent_states == current_state)
        )

        score = 100.0 * (
            STICKINESS_TRANSITION_WEIGHT * p_stay
            + STICKINESS_RECENT_WEIGHT * recent_consistency
        )

        # Consecutive causal days in the current state.
        streak = 1
        for j in range(t - 1, -1, -1):
            if causal_states[j] == current_state:
                streak += 1
            else:
                break

        if score >= 75:
            label = "HIGH"
        elif score >= 55:
            label = "MODERATE"
        else:
            label = "LOW"

        rows.append({
            "Date": pd.Timestamp(dates.iloc[t]),
            "state": current_state,
            "regime": current_regime,
            "stickiness_score": score,
            "stickiness_level": label,
            "stay_probability": p_stay,
            "recent_consistency": recent_consistency,
            "regime_streak_days": streak,
        })

    return pd.DataFrame(rows)

# ================================================================
# NEXT-DAY FORECAST
# ================================================================

def next_day_forecast(
    current_posterior: np.ndarray,
    transition: np.ndarray,
):
    raw_next = normalize_probabilities(
        np.asarray(current_posterior) @ np.asarray(transition)
    )
    return raw_next

# ================================================================
# OOS EVALUATION
# ================================================================

def evaluate_oos_regime_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: GaussianHMM,
    scaler: StandardScaler,
    mapping: dict,
):
    """
    Evaluate the 1-day regime forecast without refitting on the test set.

    We use the training mapping and the model's causal filtered posterior.
    The forecast for day t+1 is produced from information available at t.

    Metrics:
      - next-state hit rate against the model's causal state on t+1
      - mean BULL/BEAR/SIDE directional agreement using next-day return

    The first metric checks latent-state persistence/transition forecasting;
    the second checks whether regime direction has an economic relationship
    with the realized next-day return.
    """
    X_train = scaler.transform(train[FEATURES].astype(float))
    X_test = scaler.transform(test[FEATURES].astype(float))
    filtered = causal_filter_continuation(model, X_train, X_test)
    transition = model.transmat_

    predicted_next_states = []
    predicted_next_probs = []

    for t in range(len(test) - 1):
        p_next = next_day_forecast(filtered[t], transition)
        predicted_next_probs.append(p_next)
        predicted_next_states.append(int(np.argmax(p_next)))

    # Causal state on the actual following observation.
    actual_next_states = np.argmax(filtered[1:], axis=1)

    if predicted_next_states:
        latent_hit_rate = float(
            np.mean(
                np.asarray(predicted_next_states)
                == actual_next_states
            )
        )
    else:
        latent_hit_rate = np.nan

    # Directional return agreement.
    realized_returns = test["forward_1d"].iloc[:-1].to_numpy(dtype=float)
    pred_regimes = [mapping[s] for s in predicted_next_states]

    direction_correct = []
    for regime, ret in zip(pred_regimes, realized_returns):
        if not np.isfinite(ret):
            continue
        if regime == "BULL":
            direction_correct.append(ret > 0)
        elif regime == "BEAR":
            direction_correct.append(ret < 0)
        else:
            # SIDE means no strong directional expectation. For evaluation,
            # count a day as agreement when the absolute return is below
            # the historical test median absolute return.
            direction_correct.append(
                abs(ret)
                <= float(np.nanmedian(np.abs(realized_returns)))
            )

    directional_agreement = (
        float(np.mean(direction_correct))
        if direction_correct
        else np.nan
    )

    return {
        "oos_rows": len(test),
        "next_state_hit_rate": latent_hit_rate,
        "directional_agreement": directional_agreement,
        "mean_test_abs_return": float(
            np.nanmean(np.abs(realized_returns))
        ),
    }


def final_oos_test(df: pd.DataFrame):
    train = df[df["Date"] < pd.Timestamp(FINAL_TEST_START)].copy()
    test = df[
        (df["Date"] >= pd.Timestamp(FINAL_TEST_START))
        & (df["Date"] <= pd.Timestamp(FINAL_TEST_END))
    ].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Final-test training set too small: {len(train)}"
        )
    if len(test) < 2:
        raise RuntimeError("Not enough final-test observations.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].astype(float))
    X_test = scaler.transform(test[FEATURES].astype(float))

    model, restarts, seed = fit_best_hmm(X_train, N_STATES)

    test_ll = float(model.score(X_test))
    filtered_test = causal_filter_continuation(model, X_train, X_test)

    train_states = model.predict(X_train)
    profile = state_profile(train, train_states)
    mapping = map_three_states(profile)

    raw_conf = np.max(filtered_test, axis=1)
    concentration = np.array([
        posterior_concentration(p) for p in filtered_test
    ])

    oos = evaluate_oos_regime_forecast(
        train,
        test,
        model,
        scaler,
        mapping,
    )

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "loglik": test_ll,
        "loglik_per_obs": test_ll / len(X_test),
        "mean_raw_confidence": float(np.mean(raw_conf)),
        "median_raw_confidence": float(np.median(raw_conf)),
        "mean_posterior_concentration": float(np.mean(concentration)),
        "next_state_hit_rate": oos["next_state_hit_rate"],
        "directional_agreement": oos["directional_agreement"],
        "mean_test_abs_return": oos["mean_test_abs_return"],
        "best_seed": seed,
        "mapping": mapping,
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
    }

# ================================================================
# TIME-BASED VALIDATION
# ================================================================

def validation_test(df: pd.DataFrame, start: str, end: str):
    train = df[df["Date"] < pd.Timestamp(start)].copy()
    test = df[
        (df["Date"] >= pd.Timestamp(start))
        & (df["Date"] <= pd.Timestamp(end))
    ].copy()

    if len(train) < MIN_TRAIN_ROWS or len(test) < 2:
        raise RuntimeError(
            f"Insufficient validation data: train={len(train)}, test={len(test)}"
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].astype(float))
    X_test = scaler.transform(test[FEATURES].astype(float))

    model, restarts, seed = fit_best_hmm(X_train, N_STATES)
    test_ll = float(model.score(X_test))
    filtered = causal_filter_continuation(model, X_train, X_test)

    train_states = model.predict(X_train)
    profile = state_profile(train, train_states)
    mapping = map_three_states(profile)

    quality = quality_report(model, train_states)

    concentration = float(
        np.mean([posterior_concentration(p) for p in filtered])
    )
    entropy = float(
        np.mean([normalized_entropy(p) for p in filtered])
    )

    oos = evaluate_oos_regime_forecast(
        train,
        test,
        model,
        scaler,
        mapping,
    )

    return {
        "period": f"{start[:4]}_{end[:4]}",
        "train_rows": len(train),
        "test_rows": len(test),
        "loglik_per_obs": test_ll / len(X_test),
        "mean_entropy": entropy,
        "mean_concentration": concentration,
        "max_self_transition": float(np.max(np.diag(model.transmat_))),
        "pathological": int(np.sum(quality["status"] == "PATHOLOGICAL")),
        "next_state_hit_rate": oos["next_state_hit_rate"],
        "directional_agreement": oos["directional_agreement"],
        "best_seed": seed,
        "mapping": mapping,
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
    }

# ================================================================
# PRODUCTION MODEL
# ================================================================

def fit_production_model(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].astype(float))

    model, restarts, seed = fit_best_hmm(X, N_STATES)
    filtered = causal_filtered_posterior(model, X)

    train_states = np.argmax(filtered, axis=1)
    profile = state_profile(df, train_states)
    mapping = map_three_states(profile)

    transition = model.transmat_.copy()

    current_raw = normalize_probabilities(filtered[-1])
    next_raw = next_day_forecast(current_raw, transition)

    current_state = int(np.argmax(current_raw))
    next_state = int(np.argmax(next_raw))

    stickiness = calculate_stickiness(
        filtered=filtered,
        transition=transition,
        mapping=mapping,
        dates=df["Date"],
    )

    return {
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
        "best_seed": seed,
        "filtered": filtered,
        "profile": profile,
        "mapping": mapping,
        "transition": transition,
        "current_raw_probs": current_raw,
        "next_raw_probs": next_raw,
        "decision_confidence": decision_confidence(current_raw, transition),
        "current_state": current_state,
        "current_regime": mapping[current_state],
        "next_state": next_state,
        "next_regime": mapping[next_state],
        "stickiness": stickiness,
        "quality": quality_report(model, train_states),
    }

# ================================================================
# PRINTING / OUTPUT
# ================================================================

def transition_dataframe(transition: np.ndarray, mapping: dict):
    labels = [mapping[i] for i in range(N_STATES)]
    return pd.DataFrame(
        transition,
        index=labels,
        columns=labels,
    )


def print_probabilities(title: str, probs: np.ndarray, mapping: dict):
    print(f"\n{title}")
    order = np.argsort(probs)[::-1]
    for state in order:
        print(
            f"  {mapping[int(state)]:<5} "
            f"{probs[int(state)]:.4f} "
            f"({probs[int(state)]:.2%})"
        )


def build_daily_regime_output(production: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Create one UI-ready row per trading day."""
    filtered = production["filtered"]
    transition = production["transition"]
    mapping = production["mapping"]
    stickiness = production["stickiness"].copy()

    rows = []
    for i in range(len(df)):
        current = normalize_probabilities(filtered[i])
        next_probs = next_day_forecast(current, transition)
        current_state = int(np.argmax(current))
        next_state = int(np.argmax(next_probs))
        st = stickiness.iloc[i]

        rows.append({
            "Date": df["Date"].iloc[i],
            "regime": mapping[current_state],
            "bull_probability": float(current[[s for s, r in mapping.items() if r == "BULL"][0]]),
            "side_probability": float(current[[s for s, r in mapping.items() if r == "SIDE"][0]]),
            "bear_probability": float(current[[s for s, r in mapping.items() if r == "BEAR"][0]]),
            "next_bull_probability": float(next_probs[[s for s, r in mapping.items() if r == "BULL"][0]]),
            "next_side_probability": float(next_probs[[s for s, r in mapping.items() if r == "SIDE"][0]]),
            "next_bear_probability": float(next_probs[[s for s, r in mapping.items() if r == "BEAR"][0]]),
            "current_confidence": float(np.max(current)),
            "decision_confidence": float(decision_confidence(current, transition)),
            "posterior_concentration": float(posterior_concentration(current)),
            "next_regime": mapping[next_state],
            "next_confidence": float(np.max(next_probs)),
            "stickiness_score": float(st["stickiness_score"]),
            "stickiness_level": st["stickiness_level"],
            "stay_probability": float(st["stay_probability"]),
            "recent_consistency": float(st["recent_consistency"]),
            "regime_streak_days": int(st["regime_streak_days"]),
        })

    return pd.DataFrame(rows)


def save_symbol_outputs(
    symbol: str,
    production: dict,
    df: pd.DataFrame,
    final_test: dict,
    validation_rows: list,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.lower()

    mapping = production["mapping"]
    transition = production["transition"]
    profile = production["profile"].copy()
    profile["regime"] = [mapping[int(s)] for s in profile.index]
    profile = profile.reset_index()
    profile["symbol"] = symbol

    transition_df = transition_dataframe(transition, mapping)
    transition_df.insert(0, "symbol", symbol)
    transition_df.to_csv(
        OUTPUT_DIR / f"{safe_symbol}_transition_matrix.csv"
    )

    profile.to_csv(
        OUTPUT_DIR / f"{safe_symbol}_state_profile.csv",
        index=False,
    )

    stickiness = production["stickiness"].copy()
    stickiness.insert(0, "symbol", symbol)
    stickiness.to_csv(
        OUTPUT_DIR / f"{safe_symbol}_daily_stickiness.csv",
        index=False,
    )

    daily_output = build_daily_regime_output(production, df)
    daily_output.insert(0, "symbol", symbol)
    daily_output.to_csv(
        OUTPUT_DIR / f"{safe_symbol}_daily_regime_output.csv",
        index=False,
    )

    current = production["current_raw_probs"]
    nxt = production["next_raw_probs"]
    current_stick = stickiness.iloc[-1]

    current_rows = []
    for state in range(N_STATES):
        current_rows.append({
            "symbol": symbol,
            "Date": stickiness.iloc[-1]["Date"],
            "regime": mapping[state],
            "current_probability": float(current[state]),
            "next_day_probability": float(nxt[state]),
        })

    pd.DataFrame(current_rows).to_csv(
        OUTPUT_DIR / f"{safe_symbol}_current_forecast.csv",
        index=False,
    )

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        OUTPUT_DIR / f"{safe_symbol}_validation.csv",
        index=False,
    )

    return {
        "symbol": symbol,
        "rows": len(production["filtered"]),
        "date_start": production["stickiness"]["Date"].min(),
        "date_end": production["stickiness"]["Date"].max(),
        "current_regime": production["current_regime"],
        "current_probability": float(np.max(current)),
        "decision_confidence": float(production["decision_confidence"]),
        "next_regime": production["next_regime"],
        "next_probability": float(np.max(nxt)),
        "current_stickiness": float(
            current_stick["stickiness_score"]
        ),
        "stickiness_level": current_stick["stickiness_level"],
        "stay_probability": float(
            current_stick["stay_probability"]
        ),
        "recent_consistency": float(
            current_stick["recent_consistency"]
        ),
        "regime_streak_days": int(
            current_stick["regime_streak_days"]
        ),
        "max_self_transition": float(
            np.max(np.diag(transition))
        ),
        "pathological_states": int(
            np.sum(
                production["quality"]["status"] == "PATHOLOGICAL"
            )
        ),
        "final_test_rows": final_test["test_rows"],
        "final_test_loglik_per_obs": final_test["loglik_per_obs"],
        "final_test_median_raw_confidence": final_test[
            "median_raw_confidence"
        ],
        "final_test_next_state_hit_rate": final_test[
            "next_state_hit_rate"
        ],
        "final_test_directional_agreement": final_test[
            "directional_agreement"
        ],
    }

# ================================================================
# ONE SYMBOL
# ================================================================

def run_symbol(symbol: str):
    print("\n" + "=" * 78)
    print(f"QUANTOS HMM V8 | {symbol}")
    print("=" * 78)
    print("3-state Gaussian HMM: BULL / SIDE / BEAR")
    print("Forecast horizon: NEXT TRADING DAY")

    raw = load_symbol_data(symbol)
    df = prepare_data(raw)

    print(
        f"Rows: {len(df):,} | "
        f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    )

    # ------------------------------------------------------------
    # Rolling/time validation. No random split.
    # ------------------------------------------------------------
    print("\nTIME-BASED OOS VALIDATION")
    validation_rows = []

    for name, start, end in VALIDATION_PERIODS:
        try:
            result = validation_test(df, start, end)
            validation_rows.append({
                "period": name,
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "loglik_per_obs": result["loglik_per_obs"],
                "mean_entropy": result["mean_entropy"],
                "mean_concentration": result["mean_concentration"],
                "max_self_transition": result["max_self_transition"],
                "pathological": result["pathological"],
                "next_state_hit_rate": result["next_state_hit_rate"],
                "directional_agreement": result["directional_agreement"],
                "best_seed": result["best_seed"],
            })

            print(
                f"  {name}: train={result['train_rows']:,} "
                f"test={result['test_rows']:,} | "
                f"LL/obs={result['loglik_per_obs']:.5f} | "
                f"entropy={result['mean_entropy']:.2%} | "
                f"next-state hit={result['next_state_hit_rate']:.2%} | "
                f"direction agreement={result['directional_agreement']:.2%}"
            )
        except Exception as exc:
            print(f"  {name} FAILED: {exc}")
            validation_rows.append({
                "period": name,
                "error": repr(exc),
            })

    # ------------------------------------------------------------
    # Final OOS test: 2025-2026, fit only before 2025.
    # ------------------------------------------------------------
    print("\nFINAL OOS TEST: 2025-2026")
    final_test = final_oos_test(df)
    print(f"  Train rows: {final_test['train_rows']:,}")
    print(f"  Test rows:  {final_test['test_rows']:,}")
    print(f"  LL/obs:     {final_test['loglik_per_obs']:.5f}")
    print(
        f"  Median raw posterior max: "
        f"{final_test['median_raw_confidence']:.2%}"
    )
    print(
        f"  Next-state hit rate: "
        f"{final_test['next_state_hit_rate']:.2%}"
    )
    print(
        f"  Direction agreement: "
        f"{final_test['directional_agreement']:.2%}"
    )

    # ------------------------------------------------------------
    # Production model on all currently available data.
    # ------------------------------------------------------------
    print("\nPRODUCTION MODEL")
    production = fit_production_model(df)
    model = production["model"]
    mapping = production["mapping"]
    transition = production["transition"]

    print(f"  Converged:            {model.monitor_.converged}")
    print(f"  Iterations:           {model.monitor_.iter}")
    print(f"  Best seed:            {production['best_seed']}")
    print(
        f"  Max self-transition:  "
        f"{np.max(np.diag(transition)):.2%}"
    )

    print("\nSTATE PROFILES")
    profile_display = production["profile"].copy()
    profile_display["regime"] = [
        mapping[int(s)] for s in profile_display.index
    ]
    print(
        profile_display[
            [
                "regime",
                "observations",
                "mean_return",
                "mean_volatility",
                "mean_rsi",
                "mean_volume_ratio",
            ]
        ].round(6).to_string()
    )

    print("\nTRANSITION MATRIX")
    print(
        transition_dataframe(transition, mapping)
        .round(4)
        .to_string()
    )

    print("\nCURRENT CAUSAL REGIME")
    print(f"  Date:                 {df['Date'].iloc[-1].date()}")
    print(f"  Regime:               {production['current_regime']}")
    print(
        f"  Posterior max:        "
        f"{np.max(production['current_raw_probs']):.2%}"
    )
    print(
        f"  Posterior concentration: "
        f"{posterior_concentration(production['current_raw_probs']):.2%}"
    )
    print(
        f"  Decision confidence (next-day basis): "
        f"{production['decision_confidence']:.2f}%"
    )

    print_probabilities(
        "CURRENT REGIME PROBABILITIES",
        production["current_raw_probs"],
        mapping,
    )

    print("\nNEXT TRADING DAY FORECAST")
    print("  Horizon: 1 trading day")
    print(f"  Most likely state: {production['next_regime']}")
    print_probabilities(
        "NEXT-DAY PROBABILITIES",
        production["next_raw_probs"],
        mapping,
    )

    current_stick = production["stickiness"].iloc[-1]
    print("\nTREND STICKINESS")
    print(
        f"  Score:                 "
        f"{current_stick['stickiness_score']:.2f}/100"
    )
    print(f"  Level:                 {current_stick['stickiness_level']}")
    print(
        f"  Stay probability:      "
        f"{current_stick['stay_probability']:.2%}"
    )
    print(
        f"  Recent consistency:    "
        f"{current_stick['recent_consistency']:.2%}"
    )
    print(
        f"  Current regime streak: "
        f"{int(current_stick['regime_streak_days'])} days"
    )

    summary = save_symbol_outputs(
        symbol=symbol,
        production=production,
        df=df,
        final_test=final_test,
        validation_rows=validation_rows,
    )

    return summary, validation_rows

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("QUANTOS HMM V8")
    print("=" * 78)
    print("FINAL 3-state causal daily regime model")
    print("States: BULL / SIDE / BEAR")
    print("Indices: independently modelled")
    print("Forecast: NEXT TRADING DAY")
    print(f"Multi-starts: {N_RESTARTS}")
    print(f"Stickiness lookback: {STICKINESS_LOOKBACK} trading days")
    print("=" * 78)

    summaries = []
    all_validation = []

    for symbol in SYMBOLS:
        try:
            summary, validation_rows = run_symbol(symbol)
            summaries.append(summary)
            all_validation.extend(
                [
                    {**row, "symbol": symbol}
                    for row in validation_rows
                ]
            )
        except Exception as exc:
            print("\n" + "!" * 78)
            print(f"{symbol} FAILED")
            print(repr(exc))
            print("!" * 78)
            summaries.append({
                "symbol": symbol,
                "error": repr(exc),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "v8_summary.csv",
        index=False,
    )

    pd.DataFrame(all_validation).to_csv(
        OUTPUT_DIR / "v8_validation_summary.csv",
        index=False,
    )

    # Combined daily stickiness file for the future UI.
    combined_stickiness = []
    for symbol in SYMBOLS:
        path = OUTPUT_DIR / f"{symbol.lower()}_daily_stickiness.csv"
        if path.exists():
            combined_stickiness.append(pd.read_csv(path))

    if combined_stickiness:
        pd.concat(combined_stickiness, ignore_index=True).to_csv(
            OUTPUT_DIR / "v8_daily_stickiness_all_indices.csv",
            index=False,
        )

    combined_daily = []
    for symbol in SYMBOLS:
        path = OUTPUT_DIR / f"{symbol.lower()}_daily_regime_output.csv"
        if path.exists():
            combined_daily.append(pd.read_csv(path))

    if combined_daily:
        pd.concat(combined_daily, ignore_index=True).to_csv(
            OUTPUT_DIR / "v8_daily_regime_output_all_indices.csv",
            index=False,
        )

    print("\n" + "=" * 78)
    print("QUANTOS HMM V8 COMPLETE")
    print("=" * 78)
    print(f"Summary: {OUTPUT_DIR / 'v8_summary.csv'}")
    print(
        f"Validation: {OUTPUT_DIR / 'v8_validation_summary.csv'}"
    )
    print(
        f"Daily stickiness: "
        f"{OUTPUT_DIR / 'v8_daily_stickiness_all_indices.csv'}"
    )
    print(
        f"Daily UI output: "
        f"{OUTPUT_DIR / 'v8_daily_regime_output_all_indices.csv'}"
    )
    print("States: BULL / SIDE / BEAR")
    print("Forecast horizon: NEXT TRADING DAY")
    print("Probability display: RAW CAUSAL POSTERIOR / MARKOV FORECAST")
    print("UI confidence: NEXT-DAY posterior concentration")
    print("=" * 78)
