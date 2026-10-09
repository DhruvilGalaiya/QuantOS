from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "data" / "regime" / "hmm" / "models"
VALIDATION_PATH = BASE_DIR / "data" / "regime" / "hmm" / "validation.parquet"


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
# ROBUST MODEL EXTRACTION
# ============================================================

def extract_hmm_model(obj):
    """
    Extract the actual hmmlearn model from whatever
    structure was saved with joblib.

    Handles:
        - direct HMM object
        - dictionary containing HMM
        - nested dictionaries/lists/tuples
    """

    # Direct HMM-like object
    if hasattr(obj, "score") and hasattr(obj, "predict"):
        return obj

    # Dictionary
    if isinstance(obj, dict):
        for key, value in obj.items():

            # Prefer obvious model keys
            if key in {"model", "hmm", "best_model", "estimator"}:
                if hasattr(value, "score") and hasattr(value, "predict"):
                    return value

            # Search recursively
            try:
                result = extract_hmm_model(value)
                if result is not None:
                    return result
            except Exception:
                pass

    # List / tuple
    if isinstance(obj, (list, tuple)):
        for value in obj:
            try:
                result = extract_hmm_model(value)
                if result is not None:
                    return result
            except Exception:
                pass

    return None


# ============================================================
# TRANSITION MATRIX
# ============================================================

def calculate_transition_matrix(states, n_states):

    matrix = np.zeros((n_states, n_states), dtype=float)

    for current_state, next_state in zip(states[:-1], states[1:]):
        matrix[current_state, next_state] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)

    matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QuantOS HMM Validation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_PATH}"
        )

    validation = pd.read_parquet(VALIDATION_PATH)

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in validation.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing validation features: {missing_features}"
        )

    validation = validation.sort_values("date").reset_index(drop=True)

    X_val = validation[FEATURE_COLUMNS].to_numpy(dtype=float)

    print(f"Validation observations : {len(validation)}")
    print(f"Features                : {len(FEATURE_COLUMNS)}")
    print(
        f"Period                  : "
        f"{validation['date'].min().date()} → "
        f"{validation['date'].max().date()}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # Validate 2, 3 and 4 state models
    # --------------------------------------------------------

    for n_states in [2, 3, 4]:

        print("=" * 70)
        print(f"VALIDATING {n_states}-STATE HMM")
        print("=" * 70)

        model_path = MODEL_DIR / f"hmm_{n_states}_states.joblib"

        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            continue

        # ----------------------------------------------------
        # Load saved object
        # ----------------------------------------------------

        saved_object = joblib.load(model_path)

        print(
            f"Loaded object type      : "
            f"{type(saved_object).__name__}"
        )

        # Extract actual HMM
        model = extract_hmm_model(saved_object)

        if model is None:
            raise TypeError(
                f"Could not find an HMM model inside:\n"
                f"{model_path}\n\n"
                f"Saved object type: {type(saved_object)}"
            )

        print(
            f"Extracted model type    : "
            f"{type(model).__name__}"
        )

        # ----------------------------------------------------
        # Out-of-sample likelihood
        # ----------------------------------------------------

        log_likelihood = model.score(X_val)

        average_log_likelihood = (
            log_likelihood / len(X_val)
        )

        print(
            f"Validation log-likelihood : "
            f"{log_likelihood:,.2f}"
        )

        print(
            f"Avg log-likelihood/day    : "
            f"{average_log_likelihood:,.4f}"
        )

        # ----------------------------------------------------
        # Predict validation states
        # ----------------------------------------------------

        states = model.predict(X_val)

        # ----------------------------------------------------
        # State occupancy
        # ----------------------------------------------------

        occupancy = pd.Series(states).value_counts().sort_index()

        print()
        print("State occupancy:")

        occupancy_percentages = []

        for state in range(n_states):

            count = int(occupancy.get(state, 0))

            percentage = (
                count / len(states) * 100
            )

            occupancy_percentages.append(percentage)

            print(
                f"  State {state}: "
                f"{count:3d} days "
                f"({percentage:5.1f}%)"
            )

        # ----------------------------------------------------
        # Transition matrix
        # ----------------------------------------------------

        transition = calculate_transition_matrix(
            states,
            n_states,
        )

        print()
        print("Validation transition matrix:")

        header = "         "

        for state in range(n_states):
            header += f"State {state:<8}"

        print(header)

        for state in range(n_states):

            row = transition[state]

            values = ""

            for value in row:
                values += f"{value:<10.4f}"

            print(
                f"State {state:<3} {values}"
            )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence = np.diag(transition)

        print()
        print("Validation state persistence:")

        for state in range(n_states):

            print(
                f"  State {state}: "
                f"{persistence[state]:.4f}"
            )

        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        results.append(
            {
                "states": n_states,
                "validation_log_likelihood": log_likelihood,
                "avg_log_likelihood": average_log_likelihood,
                "min_occupancy_pct": min(
                    occupancy_percentages
                ),
                "max_occupancy_pct": max(
                    occupancy_percentages
                ),
                "avg_persistence": float(
                    np.mean(persistence)
                ),
            }
        )

        print()

    # ========================================================
    # COMPARISON
    # ========================================================

    if not results:
        raise RuntimeError(
            "No HMM models were successfully validated."
        )

    results_df = pd.DataFrame(results)

    output_path = (
        MODEL_DIR / "validation_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"Saved: {output_path}")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()