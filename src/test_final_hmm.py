from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "models"
    / "hmm_3_states.joblib"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "test.parquet"
)

RAW_FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "regime_features.parquet"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "models"
    / "final_test_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "NIFTY_50_RET_1D",
    "NIFTY_RET_20D",
    "NIFTY_VOL_20D",
    "NIFTY_DRAWDOWN_60D",
    "NIFTY_BANK_REL_20D",
    "NIFTY_IT_REL_20D",
    "SECTOR_DISPERSION_1D",
    "VIX_REL_20D",
]


# ============================================================
# MODEL EXTRACTION
# ============================================================

def extract_model(obj):

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):

        for key in [
            "model",
            "hmm",
            "best_model",
            "estimator",
        ]:

            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

        for value in obj.values():

            result = extract_model(value)

            if result is not None:
                return result

    if isinstance(obj, (list, tuple)):

        for value in obj:

            result = extract_model(value)

            if result is not None:
                return result

    return None


# ============================================================
# TRANSITION MATRIX
# ============================================================

def transition_matrix(states, n_states):

    matrix = np.zeros(
        (n_states, n_states),
        dtype=float,
    )

    for current_state, next_state in zip(
        states[:-1],
        states[1:],
    ):
        matrix[
            current_state,
            next_state,
        ] += 1

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )


# ============================================================
# FORWARD RETURNS
# ============================================================

def add_forward_returns(df):

    df = df.copy()

    returns = df[
        "NIFTY_50_RET_1D"
    ]

    df["FWD_5D"] = (
        (1 + returns)
        .rolling(5)
        .apply(np.prod, raw=True)
        .shift(-4)
        - 1
    )

    df["FWD_20D"] = (
        (1 + returns)
        .rolling(20)
        .apply(np.prod, raw=True)
        .shift(-19)
        - 1
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QuantOS FINAL HMM OUT-OF-SAMPLE TEST")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("This test does NOT train or modify the HMM.")
    print("The test dataset has not been used for model selection.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)

    model = extract_model(saved)

    if model is None:
        raise RuntimeError(
            "Could not extract HMM model."
        )

    print()
    print(
        f"Model loaded: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    test_scaled = pd.read_parquet(
        TEST_PATH
    )

    test_scaled["date"] = pd.to_datetime(
        test_scaled["date"]
    )

    test_scaled = (
        test_scaled
        .sort_values("date")
        .reset_index(drop=True)
    )

    X_test = test_scaled[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print(
        f"Observations : {len(test_scaled)}"
    )

    print(
        f"Features     : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Period       : "
        f"{test_scaled['date'].min().date()} "
        f"→ "
        f"{test_scaled['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Load raw data for interpretation
    # --------------------------------------------------------

    raw = pd.read_parquet(
        RAW_FEATURE_PATH
    )

    raw["date"] = pd.to_datetime(
        raw["date"]
    )

    raw = (
        raw
        .sort_values("date")
        .reset_index(drop=True)
    )

    raw = add_forward_returns(raw)

    # --------------------------------------------------------
    # FINAL TEST LOG LIKELIHOOD
    # --------------------------------------------------------

    log_likelihood = model.score(
        X_test
    )

    avg_log_likelihood = (
        log_likelihood
        / len(X_test)
    )

    print()
    print("=" * 70)
    print("FINAL OUT-OF-SAMPLE LIKELIHOOD")
    print("=" * 70)

    print(
        f"Test log-likelihood : "
        f"{log_likelihood:,.2f}"
    )

    print(
        f"Average LL/day      : "
        f"{avg_log_likelihood:,.4f}"
    )

    # --------------------------------------------------------
    # STATE PREDICTIONS
    # --------------------------------------------------------

    states = model.predict(
        X_test
    )

    test_states = pd.DataFrame(
        {
            "date": test_scaled["date"],
            "state": states,
        }
    )

    analysis = raw.merge(
        test_states,
        on="date",
        how="inner",
    )

    # --------------------------------------------------------
    # STATE OCCUPANCY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST STATE OCCUPANCY")
    print("=" * 70)

    occupancy = (
        pd.Series(states)
        .value_counts()
        .sort_index()
    )

    for state in range(3):

        count = int(
            occupancy.get(state, 0)
        )

        percentage = (
            count
            / len(states)
            * 100
        )

        print(
            f"State {state}: "
            f"{count:3d} days "
            f"({percentage:5.1f}%)"
        )

    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    matrix = transition_matrix(
        states,
        3,
    )

    print()
    print("=" * 70)
    print("TEST TRANSITION MATRIX")
    print("=" * 70)

    print(
        "         State 0   State 1   State 2"
    )

    for state in range(3):

        print(
            f"State {state}   "
            f"{matrix[state, 0]:.4f}    "
            f"{matrix[state, 1]:.4f}    "
            f"{matrix[state, 2]:.4f}"
        )

    print()
    print("State persistence:")

    persistence = np.diag(matrix)

    for state in range(3):

        print(
            f"State {state}: "
            f"{persistence[state]:.4f}"
        )

    # --------------------------------------------------------
    # STATE CHARACTERISTICS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST STATE CHARACTERISTICS")
    print("=" * 70)

    results = []

    for state in range(3):

        subset = analysis[
            analysis["state"] == state
        ]

        print()
        print("-" * 70)
        print(f"STATE {state}")
        print("-" * 70)

        if len(subset) == 0:

            print("NO OBSERVATIONS")

            continue

        print(
            f"Observations       : "
            f"{len(subset)}"
        )

        print(
            f"Share              : "
            f"{len(subset) / len(analysis) * 100:.2f}%"
        )

        print(
            f"1D return          : "
            f"{subset['NIFTY_50_RET_1D'].mean():.6f}"
        )

        print(
            f"20D return         : "
            f"{subset['NIFTY_RET_20D'].mean():.6f}"
        )

        print(
            f"20D volatility     : "
            f"{subset['NIFTY_VOL_20D'].mean():.6f}"
        )

        print(
            f"Drawdown           : "
            f"{subset['NIFTY_DRAWDOWN_60D'].mean():.6f}"
        )

        print(
            f"Bank relative      : "
            f"{subset['NIFTY_BANK_REL_20D'].mean():.6f}"
        )

        print(
            f"IT relative        : "
            f"{subset['NIFTY_IT_REL_20D'].mean():.6f}"
        )

        print(
            f"Sector dispersion  : "
            f"{subset['SECTOR_DISPERSION_1D'].mean():.6f}"
        )

        print(
            f"VIX relative       : "
            f"{subset['VIX_REL_20D'].mean():.6f}"
        )

        print(
            f"Forward 5D return  : "
            f"{subset['FWD_5D'].mean():.6f}"
        )

        print(
            f"Forward 20D return : "
            f"{subset['FWD_20D'].mean():.6f}"
        )

        results.append(
            {
                "state": state,
                "observations": len(subset),
                "share_pct": (
                    len(subset)
                    / len(analysis)
                    * 100
                ),
                "return_1d": subset[
                    "NIFTY_50_RET_1D"
                ].mean(),
                "return_20d": subset[
                    "NIFTY_RET_20D"
                ].mean(),
                "vol_20d": subset[
                    "NIFTY_VOL_20D"
                ].mean(),
                "drawdown": subset[
                    "NIFTY_DRAWDOWN_60D"
                ].mean(),
                "bank_relative": subset[
                    "NIFTY_BANK_REL_20D"
                ].mean(),
                "it_relative": subset[
                    "NIFTY_IT_REL_20D"
                ].mean(),
                "dispersion": subset[
                    "SECTOR_DISPERSION_1D"
                ].mean(),
                "vix_relative": subset[
                    "VIX_REL_20D"
                ].mean(),
                "forward_5d": subset[
                    "FWD_5D"
                ].mean(),
                "forward_20d": subset[
                    "FWD_20D"
                ].mean(),
            }
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL TEST COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()