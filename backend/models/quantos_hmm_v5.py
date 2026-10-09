# ================================================================
# QUANTOS HMM V5
# Stable 4-Regime Market Regime Detection
#
# V5 improvements:
#   - 4-state Gaussian HMM
#   - multi-start fitting
#   - stable economic regime mapping
#   - pathological state detection
#   - posterior regime confidence
#   - transition probabilities
#   - regime persistence diagnostics
#   - small-sample OOS warnings
#   - walk-forward validation
#   - untouched 2025-2026 final test
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================

DATA_DIR = "./data"

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

N_STATES = 4
N_RESTARTS = 10

RANDOM_SEED = 42

MIN_OOS_SAMPLE = 30
MIN_STATE_FRACTION = 0.05

# Pathological persistence threshold
MAX_SELF_TRANSITION = 0.995

# Validation periods
PERIODS = [
    ("2019-2020", "2019-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2024", "2023-01-01", "2024-12-31"),
    ("FINAL_TEST", "2025-01-01", "2026-12-31"),
]

FINAL_TEST_NAME = "FINAL_TEST"

# ================================================================
# FEATURES
# ================================================================

FEATURES = [
    "return_5d",
    "return_20d",
    "volatility",
    "rsi",
    "volume_ratio",
    "trend_score",
    "momentum_score",
]

# ================================================================
# DATA LOADING
# ================================================================

from sqlalchemy import create_engine, text
import os
import pandas as pd


# ============================================================
# POSTGRESQL CONNECTION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://dhruvil@localhost:5432/quantos"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# LOAD SYMBOL DATA
# ============================================================

def load_symbol_data(symbol: str) -> pd.DataFrame:
    """
    Load OHLCV + engineered market features for one symbol
    directly from PostgreSQL.
    """

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

    return df


# ================================================================
# TECHNICAL FEATURES
# ================================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def build_features(df):

    df = df.copy()

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Returns
    df["return_1d"] = close.pct_change()

    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    # Volatility
    daily_returns = close.pct_change()

    df["volatility"] = (
        daily_returns
        .rolling(20)
        .std()
    )

    # RSI
    df["rsi"] = calculate_rsi(close, 14)

    # Volume ratio
    volume_ma = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / volume_ma.replace(0, np.nan)
    )

    # Moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    df["close_vs_sma20"] = (
        close / sma20 - 1
    )

    df["close_vs_sma50"] = (
        close / sma50 - 1
    )

    # Trend score
    df["trend_score"] = (
        0.5 * df["close_vs_sma20"] +
        0.5 * df["close_vs_sma50"]
    )

    # Momentum score
    # RSI centered around 50 + normalized multi-day momentum
    df["momentum_score"] = (
        (df["rsi"] - 50) / 5
        +
        df["return_5d"] * 100
    )

    # Future returns ONLY for evaluation
    df["forward_1d"] = close.shift(-1) / close - 1
    df["forward_5d"] = close.shift(-5) / close - 1
    df["forward_20d"] = close.shift(-20) / close - 1

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=FEATURES
    ).reset_index(drop=True)

    return df


# ================================================================
# STATE QUALITY
# ================================================================

def state_durations(states):

    durations = []

    if len(states) == 0:
        return durations

    current = states[0]
    length = 1

    for s in states[1:]:

        if s == current:
            length += 1

        else:
            durations.append(
                (current, length)
            )

            current = s
            length = 1

    durations.append(
        (current, length)
    )

    return durations


def calculate_persistence(states, n_states):

    episodes = np.zeros(n_states)
    durations = {i: [] for i in range(n_states)}

    for state, duration in state_durations(states):

        episodes[state] += 1
        durations[state].append(duration)

    rows = []

    for state in range(n_states):

        d = durations[state]

        if not d:
            rows.append({
                "state": state,
                "episodes": 0,
                "mean_duration": np.nan,
                "median_duration": np.nan,
                "max_duration": np.nan,
                "one_day_pct": np.nan
            })

            continue

        rows.append({
            "state": state,
            "episodes": int(episodes[state]),
            "mean_duration": np.mean(d),
            "median_duration": np.median(d),
            "max_duration": np.max(d),
            "one_day_pct":
                100 * np.mean(np.array(d) == 1)
        })

    return pd.DataFrame(rows)


def transition_matrix(states, n_states):

    matrix = np.zeros(
        (n_states, n_states)
    )

    for a, b in zip(states[:-1], states[1:]):
        matrix[a, b] += 1

    row_sums = matrix.sum(axis=1)

    for i in range(n_states):

        if row_sums[i] > 0:
            matrix[i] /= row_sums[i]

    return matrix


# ================================================================
# ECONOMIC REGIME MAPPING
# ================================================================

def classify_state(profile, volatility_rank):

    r5 = profile["return_5d"]
    r20 = profile["return_20d"]
    vol = profile["volatility"]
    rsi = profile["rsi"]
    trend = profile["trend_score"]
    momentum = profile["momentum_score"]

    # Strong bullish trend
    if (
        trend > 0.01
        and momentum > 2.0
        and r20 > 0
        and rsi >= 55
    ):
        return "BULL_TREND"

    # Strong bearish trend
    if (
        trend < -0.01
        and momentum < -2.0
        and r20 < 0
        and rsi <= 45
    ):
        return "BEAR_TREND"

    # High volatility state
    if (
        volatility_rank >= 0.75
        and (
            vol > 0
            or abs(momentum) > 3
        )
    ):
        return "HIGH_VOL"

    # Everything between directional extremes
    return "TRANSITION"


def map_states_to_regimes(
    profile_df
):

    profile = profile_df.copy()

    # Relative volatility ranking
    vol_rank = (
        profile["volatility"]
        .rank(pct=True)
    )

    mapping = {}

    for state in profile.index:

        mapping[state] = classify_state(
            profile.loc[state],
            vol_rank.loc[state]
        )

    # ------------------------------------------------------------
    # Prevent duplicate regime labels where possible
    # ------------------------------------------------------------

    # If multiple states become BULL/BEAR/etc,
    # keep the strongest directional state and demote
    # weaker duplicates to TRANSITION.
    #
    # This prevents V4-style:
    # STATE 0 = BULL
    # STATE 2 = BULL
    # STATE 3 = BULL
    #
    # Humans deserve at least one regime per regime.
    # ------------------------------------------------------------

    for regime in [
        "BULL_TREND",
        "BEAR_TREND",
        "HIGH_VOL"
    ]:

        candidates = [
            s for s in mapping
            if mapping[s] == regime
        ]

        if len(candidates) <= 1:
            continue

        if regime == "BULL_TREND":

            winner = max(
                candidates,
                key=lambda s:
                    profile.loc[s, "momentum_score"]
            )

        elif regime == "BEAR_TREND":

            winner = min(
                candidates,
                key=lambda s:
                    profile.loc[s, "momentum_score"]
            )

        else:

            winner = max(
                candidates,
                key=lambda s:
                    profile.loc[s, "volatility"]
            )

        for s in candidates:

            if s != winner:
                mapping[s] = "TRANSITION"

    return mapping


# ================================================================
# HMM FITTING
# ================================================================

def fit_best_hmm(X):

    best_model = None
    best_score = -np.inf
    best_seed = None

    results = []

    for restart in range(N_RESTARTS):

        seed = RANDOM_SEED + restart

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=300,
            tol=1e-4,
            random_state=seed,
            verbose=False
        )

        try:

            model.fit(X)

            score = model.score(X)

            results.append({
                "restart": restart,
                "seed": seed,
                "score": score,
                "converged": model.monitor_.converged,
                "iterations": model.monitor_.iter
            })

            if (
                model.monitor_.converged
                and score > best_score
            ):
                best_score = score
                best_model = model
                best_seed = seed

        except Exception as e:

            print(
                f"Restart {restart} failed: {e}"
            )

    if best_model is None:

        raise RuntimeError(
            "No HMM restart converged successfully."
        )

    return (
        best_model,
        pd.DataFrame(results),
        best_seed
    )


# ================================================================
# STATE PROFILES
# ================================================================

def build_state_profiles(
    df,
    states
):

    work = df.copy()

    work["state"] = states

    rows = []

    for state in range(N_STATES):

        subset = work[
            work["state"] == state
        ]

        if len(subset) == 0:
            continue

        row = {
            "state": state,
            "observations": len(subset)
        }

        for feature in FEATURES:

            row[feature] = subset[
                feature
            ].mean()

        rows.append(row)

    profile = pd.DataFrame(
        rows
    ).set_index("state")

    mapping = map_states_to_regimes(
        profile
    )

    profile["regime"] = profile.index.map(
        mapping
    )

    return profile, mapping


# ================================================================
# PATHOLOGICAL STATE DETECTION
# ================================================================

def state_quality(
    profile,
    states,
    transition
):

    total = len(states)

    rows = []

    for state in range(N_STATES):

        observations = int(
            np.sum(states == state)
        )

        fraction = (
            observations / total
            if total
            else 0
        )

        self_transition = (
            transition[state, state]
        )

        flags = []

        if fraction < MIN_STATE_FRACTION:
            flags.append("LOW_FRACTION")

        if self_transition >= MAX_SELF_TRANSITION:
            flags.append("PATHOLOGICAL_PERSISTENCE")

        status = (
            "PATHOLOGICAL"
            if "PATHOLOGICAL_PERSISTENCE" in flags
            else
            "WEAK"
            if flags
            else
            "PASS"
        )

        rows.append({
            "state": state,
            "observations": observations,
            "fraction": fraction,
            "self_transition":
                self_transition,
            "status": status,
            "flags": ",".join(flags)
        })

    return pd.DataFrame(rows)


# ================================================================
# OOS EVALUATION
# ================================================================

def evaluate_oos(
    df,
    states,
    mapping
):

    work = df.copy()

    work["state"] = states

    work["regime"] = work[
        "state"
    ].map(mapping)

    rows = []

    for regime in [
        "BULL_TREND",
        "BEAR_TREND",
        "HIGH_VOL",
        "TRANSITION"
    ]:

        subset = work[
            work["regime"] == regime
        ]

        n = len(subset)

        if n == 0:

            rows.append({
                "regime": regime,
                "observations": 0,
                "forward_1d": np.nan,
                "forward_5d": np.nan,
                "forward_20d": np.nan,
                "win_rate_5d": np.nan,
                "sample_quality": "NO_DATA"
            })

            continue

        quality = (
            "OK"
            if n >= MIN_OOS_SAMPLE
            else "LOW_SAMPLE"
        )

        rows.append({
            "regime": regime,
            "observations": n,
            "forward_1d":
                subset["forward_1d"].mean(),
            "forward_5d":
                subset["forward_5d"].mean(),
            "forward_20d":
                subset["forward_20d"].mean(),
            "win_rate_5d":
                (
                    subset["forward_5d"] > 0
                ).mean(),
            "volatility":
                subset["volatility"].mean(),
            "sample_quality": quality
        })

    return pd.DataFrame(rows)


# ================================================================
# CURRENT STATE
# ================================================================

def current_state_info(
    model,
    X,
    mapping,
    dates
):

    probabilities = model.predict_proba(X)

    current_probs = probabilities[-1]

    current_state = int(
        np.argmax(current_probs)
    )

    current_regime = mapping[
        current_state
    ]

    confidence = float(
        current_probs[current_state]
    )

    date = dates.iloc[-1]

    return {
        "date": date,
        "state": current_state,
        "regime": current_regime,
        "confidence": confidence,
        "probabilities": current_probs
    }


# ================================================================
# PRINT HELPERS
# ================================================================

def print_matrix(matrix, mapping):

    states = list(range(N_STATES))

    names = [
        mapping[s]
        for s in states
    ]

    print("\nTRANSITION MATRIX")

    print(
        pd.DataFrame(
            matrix,
            index=names,
            columns=names
        ).round(3)
    )


def print_profile(profile):

    cols = [
        "regime",
        "observations",
        "return_5d",
        "return_20d",
        "volatility",
        "rsi",
        "volume_ratio",
        "trend_score",
        "momentum_score"
    ]

    print("\nSTATE / REGIME PROFILE")

    print(
        profile[cols]
        .round(5)
        .to_string()
    )


# ================================================================
# SINGLE WALK-FORWARD PERIOD
# ================================================================

def run_period(
    symbol,
    df,
    period_name,
    train_end,
    eval_start,
    eval_end,
    final_test=False
):

    train = df[
        df["Date"] <= pd.Timestamp(train_end)
    ].copy()

    evaluation = df[
        (df["Date"] >= pd.Timestamp(eval_start))
        &
        (df["Date"] <= pd.Timestamp(eval_end))
    ].copy()

    if len(train) < 500:
        print(
            f"Skipping {symbol} {period_name}: "
            f"only {len(train)} training rows."
        )
        return None

    if len(evaluation) == 0:
        print(
            f"Skipping {symbol} {period_name}: "
            "no evaluation rows."
        )
        return None

    print("\n" + "=" * 70)
    print(
        f"{symbol} | {period_name}"
    )
    print("=" * 70)

    print(
        f"Training rows:   {len(train):,}"
    )

    print(
        f"Evaluation rows: {len(evaluation):,}"
    )

    # ------------------------------------------------------------
    # Fit scaler ONLY on training data
    # ------------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_eval = scaler.transform(
        evaluation[FEATURES]
    )

    # ------------------------------------------------------------
    # Multi-start HMM
    # ------------------------------------------------------------

    model, restart_results, best_seed = (
        fit_best_hmm(X_train)
    )

    print("\nHMM")

    print(
        f"Best seed:       {best_seed}"
    )

    print(
        f"Converged:       {model.monitor_.converged}"
    )

    print(
        f"Iterations:      {model.monitor_.iter}"
    )

    print(
        f"Log likelihood:  "
        f"{model.score(X_train):,.2f}"
    )

    # ------------------------------------------------------------
    # Training states
    # ------------------------------------------------------------

    train_states = model.predict(
        X_train
    )

    profile, mapping = (
        build_state_profiles(
            train,
            train_states
        )
    )

    print_profile(profile)

    # ------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------

    transition = transition_matrix(
        train_states,
        N_STATES
    )

    print_matrix(
        transition,
        mapping
    )

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    persistence = calculate_persistence(
        train_states,
        N_STATES
    )

    print("\nREGIME PERSISTENCE")

    print(
        persistence.round(2).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # State quality
    # ------------------------------------------------------------

    quality = state_quality(
        profile,
        train_states,
        transition
    )

    print("\nSTATE QUALITY")

    print(
        quality.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # OOS prediction
    # ------------------------------------------------------------

    eval_states = model.predict(
        X_eval
    )

    oos = evaluate_oos(
        evaluation,
        eval_states,
        mapping
    )

    print("\nOUT-OF-SAMPLE PERFORMANCE")

    print(
        oos.round(5).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Current state for final test only
    # ------------------------------------------------------------

    current = None

    if final_test:

        # Current state based on latest available data
        X_all = scaler.transform(
            df[FEATURES]
        )

        current = current_state_info(
            model,
            X_all,
            mapping,
            df["Date"]
        )

        print("\nCURRENT REGIME")

        print(
            f"Date:       {current['date']}"
        )

        print(
            f"State:      {current['state']}"
        )

        print(
            f"Regime:     {current['regime']}"
        )

        print(
            f"Confidence: "
            f"{current['confidence']:.2%}"
        )

        print("\nSTATE PROBABILITIES")

        for state, probability in enumerate(
            current["probabilities"]
        ):

            print(
                f"{mapping[state]:<18}"
                f"{probability:.4f}"
            )

    return {
        "symbol": symbol,
        "period": period_name,
        "model": model,
        "scaler": scaler,
        "profile": profile,
        "mapping": mapping,
        "transition": transition,
        "persistence": persistence,
        "quality": quality,
        "oos": oos,
        "restart_results":
            restart_results,
        "current": current
    }


# ================================================================
# MAIN WALK-FORWARD PIPELINE
# ================================================================

def run_symbol(symbol):

    print("\n")
    print("#" * 70)
    print(f"QUANTOS HMM V5: {symbol}")
    print("#" * 70)

    raw = load_symbol_data(symbol)

    print(
        f"Raw rows: {len(raw):,}"
    )

    print(
        f"Date range: "
        f"{raw['Date'].min()} -> "
        f"{raw['Date'].max()}"
    )

    df = build_features(raw)

    print(
        f"Feature rows: {len(df):,}"
    )

    results = []

    # ------------------------------------------------------------
    # Rolling validation
    #
    # 2019-2020:
    #   train = all data before 2019
    #
    # 2021-2022:
    #   train = all data before 2021
    #
    # 2023-2024:
    #   train = all data before 2023
    #
    # 2025-2026:
    #   train = all data before 2025
    #
    # Final test is NEVER used for tuning.
    # ------------------------------------------------------------

    for period_name, start, end in PERIODS:

        start_ts = pd.Timestamp(start)

        train_end = (
            start_ts -
            pd.Timedelta(days=1)
        )

        result = run_period(
            symbol=symbol,
            df=df,
            period_name=period_name,
            train_end=train_end,
            eval_start=start,
            eval_end=end,
            final_test=(
                period_name ==
                FINAL_TEST_NAME
            )
        )

        if result is not None:
            results.append(result)

    return results


# ================================================================
# FINAL SUMMARY
# ================================================================

def build_summary(all_results):

    rows = []

    for symbol, results in all_results.items():

        for result in results:

            oos = result["oos"]

            for _, row in oos.iterrows():

                rows.append({
                    "symbol":
                        symbol,

                    "period":
                        result["period"],

                    "regime":
                        row["regime"],

                    "observations":
                        row["observations"],

                    "forward_1d":
                        row["forward_1d"],

                    "forward_5d":
                        row["forward_5d"],

                    "forward_20d":
                        row["forward_20d"],

                    "win_rate_5d":
                        row["win_rate_5d"],

                    "volatility":
                        row.get(
                            "volatility",
                            np.nan
                        ),

                    "sample_quality":
                        row["sample_quality"]
                })

    return pd.DataFrame(rows)


# ================================================================
# SAVE OUTPUTS
# ================================================================

def save_outputs(
    all_results,
    summary
):

    os.makedirs(
        "v5_outputs",
        exist_ok=True
    )

    summary.to_csv(
        "v5_outputs/v5_summary.csv",
        index=False
    )

    # Save detailed state profiles
    profiles = []

    for symbol, results in all_results.items():

        for result in results:

            profile = result["profile"].copy()

            profile["symbol"] = symbol
            profile["period"] = result["period"]

            profile = profile.reset_index()

            profiles.append(profile)

    if profiles:

        pd.concat(
            profiles,
            ignore_index=True
        ).to_csv(
            "v5_outputs/v5_state_profiles.csv",
            index=False
        )

    # Save state quality
    quality_rows = []

    for symbol, results in all_results.items():

        for result in results:

            q = result["quality"].copy()

            q["symbol"] = symbol
            q["period"] = result["period"]

            quality_rows.append(q)

    if quality_rows:

        pd.concat(
            quality_rows,
            ignore_index=True
        ).to_csv(
            "v5_outputs/v5_state_quality.csv",
            index=False
        )

    print("\n")
    print("=" * 70)
    print("V5 OUTPUTS SAVED")
    print("=" * 70)

    print(
        "v5_outputs/v5_summary.csv"
    )

    print(
        "v5_outputs/v5_state_profiles.csv"
    )

    print(
        "v5_outputs/v5_state_quality.csv"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("QUANTOS HMM V5")
    print("=" * 70)

    print(
        "4-state Gaussian HMM"
    )

    print(
        f"Multi-starts: {N_RESTARTS}"
    )

    print(
        "Final test: 2025-2026"
    )

    print("=" * 70)

    all_results = {}

    for symbol in SYMBOLS:

        try:

            all_results[symbol] = (
                run_symbol(symbol)
            )

        except Exception as e:

            print("\n" + "!" * 70)

            print(
                f"{symbol} FAILED"
            )

            print(
                repr(e)
            )

            print("!" * 70)

    summary = build_summary(
        all_results
    )

    save_outputs(
        all_results,
        summary
    )

    print("\n")
    print("=" * 70)
    print("QUANTOS HMM V5 COMPLETE")
    print("=" * 70)