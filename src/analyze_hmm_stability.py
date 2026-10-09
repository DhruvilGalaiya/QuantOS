from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "data" / "regime" / "hmm" / "models"

TRAIN_PATH = BASE_DIR / "data" / "regime" / "hmm" / "train.parquet"
VALIDATION_PATH = BASE_DIR / "data" / "regime" / "hmm" / "validation.parquet"


FEATURE_COLUMNS = [
    "NIFTY_50_RET_1D",
    "NIFTY_RET_20D",
    "NIFTY_VOL_20D",
    "NIFTY_VOL_60D",
    "NIFTY_DRAWDOWN_60D",
    "NIFTY_BANK_REL_20D",
    "NIFTY_IT_REL_20D",
    "NIFTY_PHARMA_REL_20D",
    "NIFTY_AUTO_REL_20D",
    "SECTOR_DISPERSION_1D",
    "VIX_LEVEL",
    "VIX_REL_20D",
]


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


def characterize(df, states):

    data = df.copy()
    data["state"] = states

    rows = []

    for state in sorted(data["state"].unique()):

        subset = data[data["state"] == state]

        rows.append(
            {
                "state": state,
                "observations": len(subset),
                "share_pct": len(subset) / len(data) * 100,
                "return_1d": subset[
                    "NIFTY_50_RET_1D"
                ].mean(),
                "return_20d": subset[
                    "NIFTY_RET_20D"
                ].mean(),
                "vol_20d": subset[
                    "NIFTY_VOL_20D"
                ].mean(),
                "vol_60d": subset[
                    "NIFTY_VOL_60D"
                ].mean(),
                "drawdown": subset[
                    "NIFTY_DRAWDOWN_60D"
                ].mean(),
                "dispersion": subset[
                    "SECTOR_DISPERSION_1D"
                ].mean(),
                "vix": subset[
                    "VIX_LEVEL"
                ].mean(),
                "vix_relative": subset[
                    "VIX_REL_20D"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("QuantOS HMM Regime Stability Analysis")
    print("=" * 70)

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    train = train.sort_values("date").reset_index(drop=True)
    validation = validation.sort_values("date").reset_index(drop=True)

    X_train = train[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    X_validation = validation[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    results = []

    for n_states in [2, 3, 4]:

        print()
        print("=" * 70)
        print(f"{n_states}-STATE MODEL")
        print("=" * 70)

        model_path = (
            MODEL_DIR
            / f"hmm_{n_states}_states.joblib"
        )

        saved = joblib.load(model_path)

        model = extract_model(saved)

        if model is None:
            raise RuntimeError(
                f"Could not extract model from {model_path}"
            )

        # ----------------------------------------------------
        # Predict both periods
        # ----------------------------------------------------

        train_states = model.predict(X_train)
        validation_states = model.predict(X_validation)

        train_characteristics = characterize(
            train,
            train_states,
        )

        validation_characteristics = characterize(
            validation,
            validation_states,
        )

        print()
        print("TRAIN STATE CHARACTERISTICS")
        print("-" * 70)

        print(
            train_characteristics.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        print()
        print("VALIDATION STATE CHARACTERISTICS")
        print("-" * 70)

        print(
            validation_characteristics.to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        # ----------------------------------------------------
        # Model likelihoods
        # ----------------------------------------------------

        train_ll = model.score(X_train)
        validation_ll = model.score(X_validation)

        print()
        print(
            f"Train log-likelihood       : "
            f"{train_ll:,.2f}"
        )

        print(
            f"Validation log-likelihood  : "
            f"{validation_ll:,.2f}"
        )

        print(
            f"Train LL/day               : "
            f"{train_ll / len(train):,.4f}"
        )

        print(
            f"Validation LL/day          : "
            f"{validation_ll / len(validation):,.4f}"
        )

        results.append(
            {
                "states": n_states,
                "train_ll_per_day": (
                    train_ll / len(train)
                ),
                "validation_ll_per_day": (
                    validation_ll
                    / len(validation)
                ),
            }
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = (
        MODEL_DIR
        / "stability_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()