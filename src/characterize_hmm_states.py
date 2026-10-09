from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "data" / "regime" / "hmm" / "models"
VALIDATION_PATH = BASE_DIR / "data" / "regime" / "hmm" / "validation.parquet"
RAW_FEATURE_PATH = BASE_DIR / "data" / "regime" / "regime_features.parquet"


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

        for key in ["model", "hmm", "best_model", "estimator"]:

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


def main():

    print("=" * 70)
    print("QuantOS HMM State Characterization")
    print("=" * 70)

    validation = pd.read_parquet(VALIDATION_PATH)
    raw = pd.read_parquet(RAW_FEATURE_PATH)

    validation["date"] = pd.to_datetime(validation["date"])
    raw["date"] = pd.to_datetime(raw["date"])

    validation = validation.sort_values("date").reset_index(drop=True)
    raw = raw.sort_values("date").reset_index(drop=True)

    X_val = validation[FEATURE_COLUMNS].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Future returns for economic interpretation
    # --------------------------------------------------------

    raw = raw.sort_values("date").reset_index(drop=True)

    raw["NIFTY_FWD_5D"] = (
        (1 + raw["NIFTY_50_RET_1D"])
        .rolling(5)
        .apply(np.prod, raw=False)
        .shift(-4)
        - 1
    )

    raw["NIFTY_FWD_20D"] = (
        (1 + raw["NIFTY_50_RET_1D"])
        .rolling(20)
        .apply(np.prod, raw=False)
        .shift(-19)
        - 1
    )

    # --------------------------------------------------------
    # Match raw features to validation dates
    # --------------------------------------------------------

    analysis = validation[
        ["date"]
    ].merge(
        raw,
        on="date",
        how="left",
        suffixes=("", "_raw"),
    )

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    all_results = []

    for n_states in [2, 3, 4]:

        print()
        print("=" * 70)
        print(f"{n_states}-STATE HMM")
        print("=" * 70)

        model_path = (
            MODEL_DIR / f"hmm_{n_states}_states.joblib"
        )

        saved = joblib.load(model_path)
        model = extract_model(saved)

        if model is None:
            raise RuntimeError(
                f"Could not extract model from {model_path}"
            )

        states = model.predict(X_val)

        analysis_model = analysis.copy()
        analysis_model["state"] = states

        print()

        for state in range(n_states):

            subset = analysis_model[
                analysis_model["state"] == state
            ]

            if len(subset) == 0:

                print(
                    f"State {state}: NO OBSERVATIONS"
                )

                continue

            print("-" * 70)
            print(f"STATE {state}")
            print("-" * 70)

            print(f"Observations : {len(subset)}")
            print(
                f"Share        : "
                f"{len(subset) / len(analysis_model) * 100:.1f}%"
            )

            print(
                f"Avg 1D return       : "
                f"{subset['NIFTY_50_RET_1D'].mean():.5f}"
            )

            print(
                f"Avg 20D return      : "
                f"{subset['NIFTY_RET_20D'].mean():.5f}"
            )

            print(
                f"Avg 20D volatility   : "
                f"{subset['NIFTY_VOL_20D'].mean():.5f}"
            )

            print(
                f"Avg 60D volatility   : "
                f"{subset['NIFTY_VOL_60D'].mean():.5f}"
            )

            print(
                f"Avg drawdown         : "
                f"{subset['NIFTY_DRAWDOWN_60D'].mean():.5f}"
            )

            print(
                f"Avg sector dispersion: "
                f"{subset['SECTOR_DISPERSION_1D'].mean():.5f}"
            )

            print(
                f"Avg VIX level        : "
                f"{subset['VIX_LEVEL'].mean():.5f}"
            )

            print(
                f"Avg VIX relative     : "
                f"{subset['VIX_REL_20D'].mean():.5f}"
            )

            print(
                f"Avg forward 5D ret   : "
                f"{subset['NIFTY_FWD_5D'].mean():.5f}"
            )

            print(
                f"Avg forward 20D ret  : "
                f"{subset['NIFTY_FWD_20D'].mean():.5f}"
            )

            all_results.append(
                {
                    "states": n_states,
                    "state": state,
                    "observations": len(subset),
                    "share_pct": (
                        len(subset)
                        / len(analysis_model)
                        * 100
                    ),
                    "avg_1d_return": subset[
                        "NIFTY_50_RET_1D"
                    ].mean(),
                    "avg_20d_return": subset[
                        "NIFTY_RET_20D"
                    ].mean(),
                    "avg_20d_vol": subset[
                        "NIFTY_VOL_20D"
                    ].mean(),
                    "avg_60d_vol": subset[
                        "NIFTY_VOL_60D"
                    ].mean(),
                    "avg_drawdown": subset[
                        "NIFTY_DRAWDOWN_60D"
                    ].mean(),
                    "avg_dispersion": subset[
                        "SECTOR_DISPERSION_1D"
                    ].mean(),
                    "avg_vix": subset[
                        "VIX_LEVEL"
                    ].mean(),
                    "avg_vix_relative": subset[
                        "VIX_REL_20D"
                    ].mean(),
                    "avg_forward_5d": subset[
                        "NIFTY_FWD_5D"
                    ].mean(),
                    "avg_forward_20d": subset[
                        "NIFTY_FWD_20D"
                    ].mean(),
                }
            )

    # --------------------------------------------------------
    # Save characterization
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    output_path = (
        MODEL_DIR / "state_characterization.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("STATE CHARACTERIZATION SAVED")
    print("=" * 70)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()