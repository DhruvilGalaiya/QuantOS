"""
QuantOS - Historical Causal HMM State Reconstruction

Purpose
-------
Build a historical causal HMM state series for NIFTY 50.

This is a lightweight research artifact used by the Monte Carlo
regime-switching model.

Important:
- Each HMM fit only uses data available up to that historical date.
- No future observations are used for the state being generated.
- Uses the same general 3-state BULL / SIDE / BEAR concept as QuantOS.
- This module is intentionally independent from quantos_hmm_v10.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/regime/daily/nifty_50.parquet"
)

OUTPUT_PATH = Path(
    "data/regime/live_predictions/"
    "historical_hmm_states_nifty_50.csv"
)

FEATURE_COLUMNS = [
    "log_return_1d",
    "volatility_20",
    "rsi_14",
    "atr_pct",
    "close_vs_sma20",
    "close_vs_sma50",
]

STATE_NAMES = [
    "BULL",
    "SIDE",
    "BEAR",
]

TRAIN_DAYS = 5 * 252
MIN_TRAIN = 500

REFIT_STEP = 21
N_RESTARTS = 8

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    ).copy()

    required = [
        "timestamp",
        "high",
        "low",
        "close",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df[
            [
                "timestamp",
                "high",
                "low",
                "close",
            ]
        ]
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FEATURES
# ============================================================

def build_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # 1. Daily log return
    # --------------------------------------------------------

    df["log_return_1d"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    # --------------------------------------------------------
    # 2. 20-day annualized realized volatility
    # --------------------------------------------------------

    df["volatility_20"] = (
        df["log_return_1d"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # 3. RSI(14)
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    df["rsi_14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # Neutral fallback for the initial period.
    df["rsi_14"] = (
        df["rsi_14"]
        .fillna(50.0)
    )

    # --------------------------------------------------------
    # 4. ATR as percentage of price
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"]
                - previous_close
            ).abs(),

            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .rolling(14)
        .mean()
    )

    df["atr_pct"] = (
        atr
        / df["close"]
    )

    # --------------------------------------------------------
    # 5. Price relative to SMA20
    # --------------------------------------------------------

    sma20 = (
        df["close"]
        .rolling(20)
        .mean()
    )

    df["close_vs_sma20"] = (
        df["close"]
        / sma20
        - 1.0
    )

    # --------------------------------------------------------
    # 6. Price relative to SMA50
    # --------------------------------------------------------

    sma50 = (
        df["close"]
        .rolling(50)
        .mean()
    )

    df["close_vs_sma50"] = (
        df["close"]
        / sma50
        - 1.0
    )

    # --------------------------------------------------------
    # Clean numerical values.
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FIT ONE HMM
# ============================================================

def fit_hmm(train_df):

    X = train_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    if len(X) < MIN_TRAIN:
        return None

    # Standardize using training data only.
    mean = np.mean(
        X,
        axis=0,
    )

    std = np.std(
        X,
        axis=0,
    )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    )

    X_scaled = (
        X - mean
    ) / std

    best_model = None
    best_score = -np.inf

    for restart in range(
        N_RESTARTS
    ):

        try:

            model = GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=300,
                random_state=(
                    RANDOM_SEED
                    + restart
                ),
            )

            model.fit(
                X_scaled
            )

            score = model.score(
                X_scaled
            )

            if (
                np.isfinite(score)
                and score > best_score
            ):

                best_score = score
                best_model = model

        except Exception:
            continue

    if best_model is None:
        return None

    return {
        "model": best_model,
        "mean": mean,
        "std": std,
    }


# ============================================================
# MAP HMM STATES TO BULL / SIDE / BEAR
# ============================================================

def map_states(
    model,
    train_df,
    mean,
    std,
):

    X = train_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    X_scaled = (
        X - mean
    ) / std

    raw_states = model.predict(
        X_scaled
    )

    temp = train_df[
        [
            "log_return_1d"
        ]
    ].copy()

    temp["state"] = (
        raw_states
    )

    state_returns = (
        temp
        .groupby("state")[
            "log_return_1d"
        ]
        .mean()
    )

    ordered = (
        state_returns
        .sort_values()
        .index
        .tolist()
    )

    mapping = {}

    if len(ordered) == 3:

        mapping[
            ordered[0]
        ] = "BEAR"

        mapping[
            ordered[1]
        ] = "SIDE"

        mapping[
            ordered[2]
        ] = "BULL"

    else:

        # Defensive fallback.
        for i in range(3):
            mapping[i] = (
                STATE_NAMES[i]
            )

    return mapping


# ============================================================
# BUILD HISTORY
# ============================================================

def build_history():

    print("=" * 78)
    print(
        "QUANTOS HISTORICAL CAUSAL HMM STATES"
    )
    print("=" * 78)

    print()

    df = load_data()

    print(
        f"Input observations : "
        f"{len(df):,}"
    )

    print(
        f"Period              : "
        f"{df['timestamp'].iloc[0].date()} "
        f"→ "
        f"{df['timestamp'].iloc[-1].date()}"
    )

    print()

    df = build_features(
        df
    )

    print(
        f"Valid HMM observations: "
        f"{len(df):,}"
    )

    if len(df) < MIN_TRAIN:

        raise RuntimeError(
            "Not enough valid observations "
            "after feature construction."
        )

    print()

    results = []

    cached = None

    last_fit_position = -REFIT_STEP

    total = len(df)

    # --------------------------------------------------------
    # Causal walk-forward reconstruction.
    # --------------------------------------------------------

    for position in range(
        MIN_TRAIN - 1,
        total,
    ):

        # ----------------------------------------------------
        # Refit every 21 observations.
        # ----------------------------------------------------

        if (
            cached is None
            or (
                position
                - last_fit_position
                >= REFIT_STEP
            )
        ):

            train_start = max(
                0,
                position
                - TRAIN_DAYS
                + 1,
            )

            train_df = df.iloc[
                train_start:
                position + 1
            ].copy()

            fitted = fit_hmm(
                train_df
            )

            if fitted is None:

                print(
                    f"WARNING: HMM fit failed "
                    f"at "
                    f"{df.iloc[position]['timestamp'].date()}"
                )

                last_fit_position = (
                    position
                )

                continue

            mapping = map_states(
                fitted["model"],
                train_df,
                fitted["mean"],
                fitted["std"],
            )

            cached = {
                **fitted,
                "mapping": mapping,
            }

            last_fit_position = (
                position
            )

        # ----------------------------------------------------
        # Use only observations available up to current date.
        # ----------------------------------------------------

        train_start = max(
            0,
            position
            - TRAIN_DAYS
            + 1,
        )

        history_df = df.iloc[
            train_start:
            position + 1
        ].copy()

        X_history = (
            history_df[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        X_history_scaled = (
            X_history
            - cached["mean"]
        ) / cached["std"]

        try:

            probabilities = (
                cached["model"]
                .predict_proba(
                    X_history_scaled
                )
            )

        except Exception:

            continue

        current_probability = (
            probabilities[-1]
        )

        raw_state = int(
            np.argmax(
                current_probability
            )
        )

        mapping = cached[
            "mapping"
        ]

        current_regime = mapping.get(
            raw_state,
            STATE_NAMES[
                raw_state
            ],
        )

        named_probabilities = {
            "BULL": 0.0,
            "SIDE": 0.0,
            "BEAR": 0.0,
        }

        for raw_index, probability in enumerate(
            current_probability
        ):

            regime = mapping.get(
                raw_index,
                STATE_NAMES[
                    raw_index
                ],
            )

            named_probabilities[
                regime
            ] = float(
                probability
            )

        row = df.iloc[
            position
        ]

        results.append(
            {
                "timestamp":
                    row["timestamp"],

                "close":
                    float(
                        row["close"]
                    ),

                "current_regime":
                    current_regime,

                "bull_probability":
                    named_probabilities[
                        "BULL"
                    ],

                "side_probability":
                    named_probabilities[
                        "SIDE"
                    ],

                "bear_probability":
                    named_probabilities[
                        "BEAR"
                    ],

                "hmm_confidence":
                    float(
                        np.max(
                            current_probability
                        )
                    ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:

        raise RuntimeError(
            "HMM reconstruction produced "
            "no results."
        )

    result_df = (
        result_df
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    return result_df


# ============================================================
# REPORT
# ============================================================

def print_report(
    result_df,
):

    print()
    print("=" * 78)
    print(
        "HISTORICAL HMM RESULT"
    )
    print("=" * 78)

    print()

    print(
        f"Output observations : "
        f"{len(result_df):,}"
    )

    print(
        f"Period              : "
        f"{result_df['timestamp'].iloc[0].date()} "
        f"→ "
        f"{result_df['timestamp'].iloc[-1].date()}"
    )

    print()

    print(
        "REGIME COUNTS"
    )

    print("-" * 78)

    counts = (
        result_df[
            "current_regime"
        ]
        .value_counts()
        .reindex(
            [
                "BULL",
                "SIDE",
                "BEAR",
            ],
            fill_value=0,
        )
    )

    for regime, count in (
        counts.items()
    ):

        percentage = (
            count
            / len(result_df)
            * 100
        )

        print(
            f"{regime:>5} : "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    print()

    print(
        "LATEST STATE"
    )

    print("-" * 78)

    latest = result_df.iloc[
        -1
    ]

    print(
        f"Date       : "
        f"{latest['timestamp'].date()}"
    )

    print(
        f"Regime     : "
        f"{latest['current_regime']}"
    )

    print(
        f"P(BULL)    : "
        f"{latest['bull_probability']:.4f}"
    )

    print(
        f"P(SIDE)    : "
        f"{latest['side_probability']:.4f}"
    )

    print(
        f"P(BEAR)    : "
        f"{latest['bear_probability']:.4f}"
    )

    print(
        f"Confidence : "
        f"{latest['hmm_confidence']:.4f}"
    )

    print()

    print(
        "FILES SAVED"
    )

    print("-" * 78)

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "HISTORICAL HMM RECONSTRUCTION COMPLETE."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    result = build_history()

    print_report(
        result
    )


if __name__ == "__main__":
    main()