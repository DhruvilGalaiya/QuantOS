# ================================================================
# QUANTOS HMM V7
# Conservative causal daily-regime model
#
# V7 goals:
#   - Daily / next-trading-day forecast made explicit
#   - 3 vs 4 state model selection
#   - Diagonal Gaussian covariance
#   - Multi-start fitting with convergence checks
#   - Training-only scaling (no leakage)
#   - Causal forward filtering: P(z_t | x_1:t)
#   - Conservative posterior stabilization
#   - Separate OOS final test: 2025-2026
#   - Production model refit on all available data for current state
#   - Raw probabilities retained for auditability
#
# IMPORTANT:
#   The stabilization layer is NOT a cosmetic hard cap.
#   It applies a probability floor and temperature transform.
#   This deliberately prevents a single observation from producing
#   absurd 99.9-100% live confidence.
#
# Forecast horizon:
#   1 observation = 1 trading day because market_features is daily.
#   Therefore "next state" means NEXT TRADING DAY, not next hour/week.
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

RANDOM_SEED = 42
N_RESTARTS = 12

CANDIDATE_STATES = (3, 4)

MIN_TRAIN_ROWS = 500
MIN_STATE_FRACTION = 0.05
MAX_SELF_TRANSITION = 0.995

# Conservative posterior stabilization.
# Higher T = flatter probability distribution.
POSTERIOR_TEMPERATURE = 2.20
PROBABILITY_FLOOR = 0.01

# One row in the database represents one trading day.
FORECAST_HORIZON_DAYS = 1

# Validation / final test periods.
VALIDATION_PERIODS = [
    ("VAL_2021", "2021-01-01", "2022-12-31"),
    ("VAL_2023_2024", "2023-01-01", "2024-12-31"),
]

FINAL_TEST_NAME = "FINAL_TEST"
FINAL_TEST_START = "2025-01-01"
FINAL_TEST_END = "2026-12-31"

OUTPUT_DIR = Path("v7_outputs")

# Use features available at time t only.
# No forward return is ever used as an HMM input.
FEATURES = [
    "return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]


# ================================================================
# DATA
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

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Canonical names used by the rest of this script.
    df["Date"] = df["timestamp"]
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]

    # These are evaluation-only fields.
    df["forward_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["forward_5d"] = df["Close"].shift(-5) / df["Close"] - 1.0
    df["forward_20d"] = df["Close"].shift(-20) / df["Close"] - 1.0

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for col in FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=FEATURES).copy()
    work = work.sort_values("Date").reset_index(drop=True)

    return work


# ================================================================
# MODEL HELPERS
# ================================================================

def fit_best_hmm(X: np.ndarray, n_states: int):
    best_model = None
    best_score = -np.inf
    best_seed = None
    rows = []

    for restart in range(N_RESTARTS):
        seed = RANDOM_SEED + restart

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=400,
            tol=1e-4,
            min_covar=1e-3,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )

        try:
            model.fit(X)

            score = float(model.score(X))
            converged = bool(model.monitor_.converged)

            self_max = float(np.max(np.diag(model.transmat_)))

            rows.append({
                "restart": restart,
                "seed": seed,
                "train_loglik": score,
                "converged": converged,
                "iterations": int(model.monitor_.iter),
                "max_self_transition": self_max,
            })

            # Only accept converged, finite models.
            if (
                converged
                and np.isfinite(score)
                and self_max < MAX_SELF_TRANSITION
                and score > best_score
            ):
                best_model = model
                best_score = score
                best_seed = seed

        except Exception as exc:
            rows.append({
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

    return best_model, pd.DataFrame(rows), best_seed


def parameter_count(n_states: int, n_features: int) -> int:
    # Initial probabilities + transition probabilities +
    # diagonal Gaussian means + diagonal variances.
    return (
        (n_states - 1)
        + n_states * (n_states - 1)
        + n_states * n_features
        + n_states * n_features
    )


def bic(loglik: float, n_params: int, n_obs: int) -> float:
    return -2.0 * loglik + n_params * np.log(max(n_obs, 1))


# ================================================================
# CAUSAL FILTER
# ================================================================

def causal_filtered_posterior(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Forward-only filtering.

    Returns P(z_t | x_1:t) for every t.

    This deliberately does NOT use model.predict_proba(), because
    predict_proba() uses the forward-backward smoother and can use
    future observations relative to t.
    """
    log_emission = model._compute_log_likelihood(X)

    log_start = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-300, 1.0))

    n_obs, n_states = log_emission.shape
    filtered = np.zeros((n_obs, n_states), dtype=float)

    alpha = log_start + log_emission[0]
    alpha -= _logsumexp(alpha)
    filtered[0] = np.exp(alpha)

    for t in range(1, n_obs):
        next_alpha = np.empty(n_states)

        for j in range(n_states):
            next_alpha[j] = (
                _logsumexp(alpha + log_trans[:, j])
                + log_emission[t, j]
            )

        next_alpha -= _logsumexp(next_alpha)
        alpha = next_alpha
        filtered[t] = np.exp(alpha)

    return filtered


def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    m = np.max(values)

    if not np.isfinite(m):
        return m

    return m + np.log(np.sum(np.exp(values - m)))


# ================================================================
# PROBABILITY STABILIZATION
# ================================================================

def stabilize_probabilities(
    probabilities: np.ndarray,
    temperature: float = POSTERIOR_TEMPERATURE,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    """
    Conservative probability stabilization.

    Step 1:
        Put a small floor under every state so exact zeros cannot occur.

    Step 2:
        Apply temperature > 1 to flatten overconfident distributions.

    This is intentionally transparent and deterministic. It is NOT
    presented as statistical calibration unless validated calibration
    data is added later.
    """
    p = np.asarray(probabilities, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    p = np.clip(p, floor, None)
    p = p / p.sum()

    log_p = np.log(p)
    scaled = log_p / float(temperature)

    scaled -= np.max(scaled)
    q = np.exp(scaled)
    q /= q.sum()

    return q


def forecast_next_day(
    current_probs: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """
    One-step Markov forecast.

    P(z_(t+1) | x_1:t) =
        P(z_t | x_1:t) @ A
    """
    raw = np.asarray(current_probs) @ np.asarray(transition_matrix)
    raw = raw / raw.sum()

    # Apply the same conservative treatment to the displayed forecast.
    return stabilize_probabilities(raw)


# ================================================================
# STATE MAPPING
# ================================================================

def map_states(profile: pd.DataFrame) -> dict:
    """
    Map latent HMM states to human-readable regimes.

    3-state:
        BULL / BEAR / SIDE

    4-state:
        BULL / BEAR / SIDE / STRESS
    """
    states = list(profile.index)

    if len(states) == 3:
        bull = profile["mean_return"].idxmax()
        bear = profile["mean_return"].idxmin()

        remaining = [s for s in states if s not in (bull, bear)]
        mapping = {
            bull: "BULL",
            bear: "BEAR",
            remaining[0]: "SIDE",
        }
        return mapping

    # Four states:
    # BULL = strongest return with healthy trend/RSI
    # BEAR = weakest return
    # STRESS = highest volatility among remaining states
    # SIDE = remaining state
    bull = profile["mean_return"].idxmax()
    bear = profile["mean_return"].idxmin()

    remaining = [s for s in states if s not in (bull, bear)]

    if len(remaining) == 2:
        stress = profile.loc[remaining, "mean_volatility"].idxmax()
        side = [s for s in remaining if s != stress][0]

        return {
            bull: "BULL",
            bear: "BEAR",
            stress: "STRESS",
            side: "SIDE",
        }

    return {s: "SIDE" for s in states}


def state_profile(
    df: pd.DataFrame,
    states: np.ndarray,
) -> pd.DataFrame:
    work = df.copy()
    work["state"] = states

    rows = []

    for state in sorted(np.unique(states)):
        subset = work[work["state"] == state]

        rows.append({
            "state": int(state),
            "observations": len(subset),
            "mean_return": subset["return_1d"].mean(),
            "mean_volatility": subset["volatility_20"].mean(),
            "mean_rsi": subset["rsi_14"].mean(),
            "mean_volume_ratio": subset["volume_ratio"].mean(),
            "mean_close_vs_sma20": subset["close_vs_sma20"].mean(),
            "mean_close_vs_sma50": subset["close_vs_sma50"].mean(),
        })

    return pd.DataFrame(rows).set_index("state")


# ================================================================
# DIAGNOSTICS
# ================================================================

def quality_report(
    model: GaussianHMM,
    states: np.ndarray,
) -> pd.DataFrame:
    n_states = model.n_components
    total = len(states)

    rows = []

    for state in range(n_states):
        count = int(np.sum(states == state))
        fraction = count / total if total else 0.0
        persistence = float(model.transmat_[state, state])

        flags = []

        if fraction < MIN_STATE_FRACTION:
            flags.append("LOW_FRACTION")

        if persistence >= MAX_SELF_TRANSITION:
            flags.append("PATHOLOGICAL_PERSISTENCE")

        rows.append({
            "state": state,
            "observations": count,
            "fraction": fraction,
            "self_transition": persistence,
            "status": (
                "PATHOLOGICAL"
                if "PATHOLOGICAL_PERSISTENCE" in flags
                else "WEAK"
                if flags
                else "PASS"
            ),
            "flags": ",".join(flags),
        })

    return pd.DataFrame(rows)


def entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(probabilities: np.ndarray) -> float:
    k = len(probabilities)

    if k <= 1:
        return 0.0

    return entropy(probabilities) / np.log(k)


# ================================================================
# MODEL SELECTION
# ================================================================

def evaluate_candidate(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    n_states: int,
):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[FEATURES].astype(float))
    X_eval = scaler.transform(evaluation[FEATURES].astype(float))

    model, restarts, seed = fit_best_hmm(
        X_train,
        n_states=n_states,
    )

    eval_ll = float(model.score(X_eval))
    eval_ll_per_obs = eval_ll / max(len(X_eval), 1)

    train_states = model.predict(X_train)
    profile = state_profile(train, train_states)
    mapping = map_states(profile)

    quality = quality_report(model, train_states)

    filtered_eval = causal_filtered_posterior(model, X_eval)

    mean_entropy = float(
        np.mean(
            [
                normalized_entropy(p)
                for p in filtered_eval
            ]
        )
    )

    n_params = parameter_count(
        n_states,
        len(FEATURES),
    )

    validation_bic = bic(
        eval_ll,
        n_params,
        len(X_eval),
    )

    # Selection score rewards OOS likelihood and mildly penalizes
    # model complexity and pathological persistence.
    score = (
        eval_ll_per_obs
        - 0.02 * np.log(max(n_params, 1))
        - 0.50 * max(
            float(quality["self_transition"].max())
            - 0.95,
            0.0,
        )
    )

    return {
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
        "best_seed": seed,
        "profile": profile,
        "mapping": mapping,
        "quality": quality,
        "eval_ll_per_obs": eval_ll_per_obs,
        "validation_bic": validation_bic,
        "selection_score": score,
        "mean_entropy": mean_entropy,
    }


def select_model(
    df: pd.DataFrame,
    validation_periods=VALIDATION_PERIODS,
):
    candidate_records = []

    for period_name, eval_start, eval_end in validation_periods:
        eval_start_ts = pd.Timestamp(eval_start)

        train = df[df["Date"] < eval_start_ts].copy()
        evaluation = df[
            (df["Date"] >= pd.Timestamp(eval_start))
            & (df["Date"] <= pd.Timestamp(eval_end))
        ].copy()

        if len(train) < MIN_TRAIN_ROWS or len(evaluation) == 0:
            continue

        print()
        print(
            f"{period_name} | train={len(train):,} "
            f"| eval={len(evaluation):,}"
        )

        for n_states in CANDIDATE_STATES:
            try:
                result = evaluate_candidate(
                    train,
                    evaluation,
                    n_states,
                )

                candidate_records.append({
                    "period": period_name,
                    "n_states": n_states,
                    "eval_ll_per_obs":
                        result["eval_ll_per_obs"],
                    "selection_score":
                        result["selection_score"],
                    "entropy":
                        result["mean_entropy"],
                    "max_self_transition":
                        float(
                            result["quality"][
                                "self_transition"
                            ].max()
                        ),
                    "pathological":
                        int(
                            np.sum(
                                result["quality"]["status"]
                                == "PATHOLOGICAL"
                            )
                        ),
                })

                print(
                    f"  {n_states} states | "
                    f"OOS LL/obs={result['eval_ll_per_obs']:.5f} | "
                    f"score={result['selection_score']:.5f} | "
                    f"entropy={result['mean_entropy']:.2%}"
                )

            except Exception as exc:
                print(
                    f"  {n_states} states FAILED: {exc}"
                )

    if not candidate_records:
        raise RuntimeError(
            "No valid validation model was produced."
        )

    comparison = pd.DataFrame(candidate_records)

    grouped = (
        comparison
        .groupby("n_states")
        .agg(
            mean_selection_score=("selection_score", "mean"),
            mean_validation_ll_per_obs=(
                "eval_ll_per_obs",
                "mean",
            ),
            mean_entropy=("entropy", "mean"),
            worst_max_self_transition=(
                "max_self_transition",
                "max",
            ),
            total_pathological=(
                "pathological",
                "sum",
            ),
        )
        .reset_index()
    )

    valid = grouped[
        grouped["total_pathological"] == 0
    ].copy()

    if valid.empty:
        valid = grouped.copy()

    selected = int(
        valid.sort_values(
            "mean_selection_score",
            ascending=False,
        ).iloc[0]["n_states"]
    )

    return selected, comparison, grouped


# ================================================================
# PRODUCTION MODEL
# ================================================================

def fit_production_model(df: pd.DataFrame, n_states: int):
    """
    Production/current-state model.

    This model is intentionally refit on all currently available
    observations. The separate 2025-2026 final-test score remains
    untouched and is calculated from a pre-2025 training cutoff.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(
        df[FEATURES].astype(float)
    )

    model, restarts, seed = fit_best_hmm(
        X,
        n_states=n_states,
    )

    causal = causal_filtered_posterior(
        model,
        X,
    )

    raw_current = causal[-1]
    stabilized_current = stabilize_probabilities(
        raw_current
    )

    profile_states = np.argmax(causal, axis=1)
    profile = state_profile(
        df,
        profile_states,
    )
    mapping = map_states(profile)

    # Transition probabilities are learned by the HMM itself.
    transition = model.transmat_.copy()

    raw_next = raw_current @ transition
    raw_next /= raw_next.sum()

    stabilized_next = forecast_next_day(
        stabilized_current,
        transition,
    )

    current_state = int(np.argmax(stabilized_current))
    next_state = int(np.argmax(stabilized_next))

    return {
        "model": model,
        "scaler": scaler,
        "restarts": restarts,
        "best_seed": seed,
        "causal": causal,
        "raw_current_probs": raw_current,
        "current_probs": stabilized_current,
        "raw_next_probs": raw_next,
        "next_probs": stabilized_next,
        "profile": profile,
        "mapping": mapping,
        "transition": transition,
        "quality": quality_report(
            model,
            profile_states,
        ),
        "current_state": current_state,
        "current_regime": mapping[current_state],
        "next_state": next_state,
        "next_regime": mapping[next_state],
    }


# ================================================================
# FINAL OOS TEST
# ================================================================

def final_test(df: pd.DataFrame, n_states: int):
    """
    Truly out-of-sample final test.

    Model is fitted ONLY on observations before 2025.
    2025-2026 is never used for fitting this test model.
    """
    train = df[
        df["Date"] < pd.Timestamp(FINAL_TEST_START)
    ].copy()

    test = df[
        (df["Date"] >= pd.Timestamp(FINAL_TEST_START))
        & (df["Date"] <= pd.Timestamp(FINAL_TEST_END))
    ].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Final-test training set too small: {len(train)}"
        )

    if test.empty:
        raise RuntimeError(
            "No 2025-2026 final-test rows found."
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES].astype(float)
    )
    X_test = scaler.transform(
        test[FEATURES].astype(float)
    )

    model, _, seed = fit_best_hmm(
        X_train,
        n_states,
    )

    test_ll = float(model.score(X_test))
    test_ll_per_obs = test_ll / len(X_test)

    test_causal = causal_filtered_posterior(
        model,
        X_test,
    )

    test_conf = np.max(test_causal, axis=1)

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "loglik": test_ll,
        "loglik_per_obs": test_ll_per_obs,
        "mean_raw_confidence": float(np.mean(test_conf)),
        "median_raw_confidence": float(np.median(test_conf)),
        "best_seed": seed,
    }


# ================================================================
# OUTPUT / REPORTING
# ================================================================

def print_probabilities(title, probs, mapping):
    print(f"\n{title}")

    ordered = sorted(
        range(len(probs)),
        key=lambda i: probs[i],
        reverse=True,
    )

    for state in ordered:
        print(
            f"  {mapping[state]:<8} "
            f"{probs[state]:.4f} "
            f"({probs[state]:.2%})"
        )


def run_symbol(symbol: str):
    print()
    print("=" * 78)
    print(f"QUANTOS HMM V7 | {symbol}")
    print("=" * 78)

    raw = load_symbol_data(symbol)
    df = prepare_data(raw)

    print(
        f"Rows: {len(df):,} | "
        f"{df['Date'].min().date()} -> "
        f"{df['Date'].max().date()}"
    )

    print()
    print("MODEL SELECTION: 3 vs 4 STATES")

    selected_states, comparison, grouped = select_model(df)

    print()
    print("MODEL COMPARISON")
    print(
        grouped.round(5).to_string(index=False)
    )

    print()
    print(
        f"SELECTED MODEL: {selected_states} states"
    )

    # Honest final OOS test.
    test = final_test(
        df,
        selected_states,
    )

    print()
    print("FINAL TEST: 2025-2026")
    print(
        f"Rows:             {test['test_rows']:,}"
    )
    print(
        f"Train rows:       {test['train_rows']:,}"
    )
    print(
        f"LogLik/obs:       {test['loglik_per_obs']:.5f}"
    )
    print(
        f"Median raw conf:  {test['median_raw_confidence']:.2%}"
    )

    # Production model for current live regime.
    production = fit_production_model(
        df,
        selected_states,
    )

    model = production["model"]
    profile = production["profile"]
    mapping = production["mapping"]
    quality = production["quality"]
    transition = production["transition"]

    print()
    print("PRODUCTION MODEL DIAGNOSTICS")
    print(
        f"Converged:        {model.monitor_.converged}"
    )
    print(
        f"Iterations:       {model.monitor_.iter}"
    )
    print(
        f"Best seed:        {production['best_seed']}"
    )
    print(
        f"Max self-transition: "
        f"{np.max(np.diag(transition)):.2%}"
    )

    print()
    print("STATE PROFILES")

    display_profile = profile.copy()
    display_profile["regime"] = [
        mapping[s]
        for s in display_profile.index
    ]

    print(
        display_profile[
            [
                "regime",
                "observations",
                "mean_return",
                "mean_volatility",
                "mean_rsi",
            ]
        ].round(6).to_string()
    )

    print()
    print("TRANSITION MATRIX")

    transition_df = pd.DataFrame(
        transition,
        index=[
            mapping[s]
            for s in range(selected_states)
        ],
        columns=[
            mapping[s]
            for s in range(selected_states)
        ],
    )

    print(
        transition_df.round(4).to_string()
    )

    print()
    print("CURRENT CAUSAL REGIME")
    print(
        f"Date:              "
        f"{df['Date'].iloc[-1].date()}"
    )
    print(
        f"Regime:            "
        f"{production['current_regime']}"
    )
    print(
        f"Raw probability:   "
        f"{np.max(production['raw_current_probs']):.2%}"
    )
    print(
        f"Stabilized conf.:  "
        f"{np.max(production['current_probs']):.2%}"
    )

    print()
    print(
        "CURRENT STABILIZED PROBABILITIES"
    )

    print_probabilities(
        "",
        production["current_probs"],
        mapping,
    )

    print()
    print(
        "NEXT TRADING DAY FORECAST"
    )
    print(
        "Horizon: 1 trading day"
    )
    print(
        f"Most likely next state: "
        f"{production['next_regime']}"
    )

    print_probabilities(
        "",
        production["next_probs"],
        mapping,
    )

    return {
        "symbol": symbol,
        "rows": len(df),
        "date_start": df["Date"].min(),
        "date_end": df["Date"].max(),
        "selected_states": selected_states,
        "current_regime": production["current_regime"],
        "current_confidence": float(
            np.max(production["current_probs"])
        ),
        "raw_current_confidence": float(
            np.max(production["raw_current_probs"])
        ),
        "next_regime": production["next_regime"],
        "next_confidence": float(
            np.max(production["next_probs"])
        ),
        "final_test_rows": test["test_rows"],
        "final_test_loglik_per_obs": test[
            "loglik_per_obs"
        ],
        "max_self_transition": float(
            np.max(np.diag(transition))
        ),
        "pathological_states": int(
            np.sum(
                quality["status"]
                == "PATHOLOGICAL"
            )
        ),
        "mean_selection_score": float(
            grouped.loc[
                grouped["n_states"]
                == selected_states,
                "mean_selection_score",
            ].iloc[0]
        ),
    }


def save_outputs(summary_rows):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "v7_summary.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("V7 COMPLETE")
    print("=" * 78)
    print(
        f"Summary saved to: "
        f"{OUTPUT_DIR / 'v7_summary.csv'}"
    )
    print(
        "Forecast horizon: NEXT TRADING DAY"
    )
    print(
        "Probability display: conservative stabilized posterior"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("QUANTOS HMM V7")
    print("=" * 78)
    print(
        "3/4-state Gaussian HMM | causal daily filtering"
    )
    print(
        f"Multi-starts: {N_RESTARTS}"
    )
    print(
        f"Posterior temperature: "
        f"{POSTERIOR_TEMPERATURE}"
    )
    print(
        f"Probability floor: "
        f"{PROBABILITY_FLOOR}"
    )
    print(
        "Final OOS test: 2025-2026"
    )
    print(
        "Forecast: NEXT TRADING DAY"
    )
    print("=" * 78)

    summary_rows = []

    for symbol in SYMBOLS:
        try:
            result = run_symbol(symbol)
            summary_rows.append(result)

        except Exception as exc:
            print()
            print("!" * 78)
            print(f"{symbol} FAILED")
            print(repr(exc))
            print("!" * 78)

            summary_rows.append({
                "symbol": symbol,
                "error": repr(exc),
            })

    save_outputs(summary_rows)
