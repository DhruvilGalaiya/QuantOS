import os
import joblib
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM


TRAIN_PATH = "data/regime/hmm/train.parquet"
MODEL_DIR = "data/regime/hmm/models"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# Load training data
# =========================================================

df = pd.read_parquet(TRAIN_PATH)

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

X = df[FEATURE_COLUMNS].values


print()
print("=" * 70)
print("QuantOS HMM Candidate Models")
print("=" * 70)

print(f"Training observations : {len(X)}")
print(f"Features              : {len(FEATURE_COLUMNS)}")


# =========================================================
# Parameter count
# =========================================================

def count_parameters(n_states, n_features):

    means = n_states * n_features

    covariances = n_states * n_features

    transition_params = (
        n_states * (n_states - 1)
    )

    initial_params = n_states - 1

    return (
        means
        + covariances
        + transition_params
        + initial_params
    )


# =========================================================
# Candidate models
# =========================================================

results = []


for n_states in [2, 3, 4]:

    print()
    print("=" * 70)
    print(f"Testing {n_states}-state HMM")
    print("=" * 70)

    best_model = None
    best_loglik = -np.inf
    best_seed = None


    # -----------------------------------------------------
    # Multiple initializations
    # -----------------------------------------------------

    for seed in range(10):

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
            init_params="stmc"
        )

        try:

            model.fit(X)

            loglik = model.score(X)

            print(
                f"Seed {seed:2d} | "
                f"log-likelihood = {loglik:,.2f}"
            )

            if loglik > best_loglik:

                best_loglik = loglik
                best_model = model
                best_seed = seed

        except Exception as e:

            print(
                f"Seed {seed:2d} FAILED: {e}"
            )


    if best_model is None:

        print("No successful model.")
        continue


    # =====================================================
    # Information criteria
    # =====================================================

    n_parameters = count_parameters(
        n_states,
        len(FEATURE_COLUMNS)
    )

    n_observations = len(X)

    AIC = (
        -2 * best_loglik
        + 2 * n_parameters
    )

    BIC = (
        -2 * best_loglik
        + n_parameters
        * np.log(n_observations)
    )


    # =====================================================
    # State assignment
    # =====================================================

    states = best_model.predict(X)

    state_counts = np.bincount(
        states,
        minlength=n_states
    )

    state_percentages = (
        state_counts
        / len(states)
        * 100
    )


    # =====================================================
    # Transition matrix
    # =====================================================

    transition_matrix = best_model.transmat_

    persistence = np.diag(
        transition_matrix
    )


    # =====================================================
    # Print diagnostics
    # =====================================================

    print()
    print(f"Best seed : {best_seed}")
    print(f"LogLik    : {best_loglik:,.2f}")
    print(f"Parameters: {n_parameters}")
    print(f"AIC       : {AIC:,.2f}")
    print(f"BIC       : {BIC:,.2f}")


    print()
    print("State occupancy:")

    for state in range(n_states):

        print(
            f"  State {state}: "
            f"{state_counts[state]:4d} days "
            f"({state_percentages[state]:5.1f}%)"
        )


    print()
    print("State persistence:")

    for state in range(n_states):

        print(
            f"  State {state}: "
            f"{persistence[state]:.4f}"
        )


    print()
    print("Transition matrix:")

    transition_df = pd.DataFrame(
        transition_matrix,
        index=[
            f"State {i}"
            for i in range(n_states)
        ],
        columns=[
            f"State {i}"
            for i in range(n_states)
        ]
    )

    print(
        transition_df.round(4)
    )


    # =====================================================
    # Save model
    # =====================================================

    model_path = (
        f"{MODEL_DIR}/"
        f"hmm_{n_states}_states.joblib"
    )

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURE_COLUMNS,
            "n_states": n_states,
            "seed": best_seed
        },
        model_path
    )

    print()
    print(f"Saved: {model_path}")


    results.append(
        {
            "states": n_states,
            "seed": best_seed,
            "log_likelihood": best_loglik,
            "parameters": n_parameters,
            "AIC": AIC,
            "BIC": BIC
        }
    )


# =========================================================
# Candidate comparison
# =========================================================

results_df = pd.DataFrame(results)

print()
print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df
    .sort_values("BIC")
    .round(2)
    .to_string(index=False)
)


comparison_path = (
    f"{MODEL_DIR}/model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print()
print(f"Saved: {comparison_path}")