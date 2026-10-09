from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "models"
    / "hmm_3_states.joblib"
)

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "train.parquet"
)

VALIDATION_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "hmm"
    / "validation.parquet"
)

RAW_FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "regime"
    / "regime_features.parquet"
)


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


def add_forward_returns(df):

    df = df.copy()

    # These are RAW daily returns, not scaled values.
    returns = df["NIFTY_50_RET_1D"]

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


def analyze_period(
    scaled_df,
    raw_df,
    model,
    name,
):

    scaled_df = (
        scaled_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    raw_df = (
        raw_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # HMM prediction uses SCALED data
    # --------------------------------------------------------

    X = scaled_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    states = model.predict(X)

    state_df = pd.DataFrame(
        {
            "date": scaled_df["date"].values,
            "state": states,
        }
    )

    # --------------------------------------------------------
    # Economic interpretation uses RAW data
    # --------------------------------------------------------

    raw_df = raw_df[
        ["date"] + FEATURE_COLUMNS
    ].copy()

    raw_df = add_forward_returns(raw_df)

    analysis = raw_df.merge(
        state_df,
        on="date",
        how="inner",
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

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
            f"Observations        : {len(subset)}"
        )

        print(
            f"Share               : "
            f"{len(subset) / len(analysis) * 100:.2f}%"
        )

        print(
            f"1D return           : "
            f"{subset['NIFTY_50_RET_1D'].mean():.6f}"
        )

        print(
            f"20D return          : "
            f"{subset['NIFTY_RET_20D'].mean():.6f}"
        )

        print(
            f"20D volatility      : "
            f"{subset['NIFTY_VOL_20D'].mean():.6f}"
        )

        print(
            f"Drawdown            : "
            f"{subset['NIFTY_DRAWDOWN_60D'].mean():.6f}"
        )

        print(
            f"Bank relative       : "
            f"{subset['NIFTY_BANK_REL_20D'].mean():.6f}"
        )

        print(
            f"IT relative         : "
            f"{subset['NIFTY_IT_REL_20D'].mean():.6f}"
        )

        print(
            f"Sector dispersion   : "
            f"{subset['SECTOR_DISPERSION_1D'].mean():.6f}"
        )

        print(
            f"VIX relative        : "
            f"{subset['VIX_REL_20D'].mean():.6f}"
        )

        print(
            f"Forward 5D return   : "
            f"{subset['FWD_5D'].mean():.6f}"
        )

        print(
            f"Forward 20D return  : "
            f"{subset['FWD_20D'].mean():.6f}"
        )


def main():

    print("=" * 70)
    print("QuantOS HMM V2 Characterization")
    print("=" * 70)

    # Scaled data used by the HMM
    train_scaled = pd.read_parquet(
        TRAIN_PATH
    )

    validation_scaled = pd.read_parquet(
        VALIDATION_PATH
    )

    # Original unscaled feature data
    raw = pd.read_parquet(
        RAW_FEATURE_PATH
    )

    train_scaled["date"] = pd.to_datetime(
        train_scaled["date"]
    )

    validation_scaled["date"] = pd.to_datetime(
        validation_scaled["date"]
    )

    raw["date"] = pd.to_datetime(
        raw["date"]
    )

    saved = joblib.load(MODEL_PATH)

    model = extract_model(saved)

    if model is None:
        raise RuntimeError(
            "Could not extract HMM model."
        )

    print("Model: 3-state Gaussian HMM")
    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    analyze_period(
        train_scaled,
        raw,
        model,
        "TRAIN",
    )

    analyze_period(
        validation_scaled,
        raw,
        model,
        "VALIDATION",
    )


if __name__ == "__main__":
    main()