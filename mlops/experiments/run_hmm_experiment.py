
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import quantos_hmm_v10 as v10

FEATURE_FILE = ROOT / "data/regime/features/nifty_50.parquet"
ARTIFACT_DIR = ROOT / "mlops/artifacts"

TEMPERATURE = 1.85
LOOKBACK = 20
THRESHOLD = 0.05

FEATURES = [
    "log_return_1d",
    "volatility_20",
    "rsi_14",
    "volume_ratio",
    "close_vs_sma20",
    "close_vs_sma50",
]

REQUIRED = ["timestamp", "close", *FEATURES]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Feature dataset is empty")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamps found")

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    if (df["close"].isna() | ~np.isfinite(df["close"]) | (df["close"] <= 0)).any():
        raise ValueError("Close prices must be finite and positive")

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("HMM features contain missing or non-finite values")

    if len(df) <= LOOKBACK:
        raise ValueError(f"Need more than {LOOKBACK} rows")

    rolling_return = np.log(df["close"] / df["close"].shift(LOOKBACK))
    df["rule_regime"] = 1
    df.loc[rolling_return > THRESHOLD, "rule_regime"] = 0
    df.loc[rolling_return < -THRESHOLD, "rule_regime"] = 2

    return df


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(FEATURE_FILE)
    dataset_hash = sha256_file(FEATURE_FILE)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("QuantOS-HMM-V10")

    with mlflow.start_run(run_name="nifty50-v10-baseline") as run:
        mlflow.log_params({
            "symbol": "NIFTY_50",
            "model_version": "HMM_V10",
            "temperature": TEMPERATURE,
            "reference_lookback": LOOKBACK,
            "reference_threshold": THRESHOLD,
            "features": ",".join(FEATURES),
            "feature_count": len(FEATURES),
            "input_rows": len(df),
            "dataset_sha256": dataset_hash,
            "dataset_start": str(df["timestamp"].min()),
            "dataset_end": str(df["timestamp"].max()),
        })

        outputs = v10.production_output(df, TEMPERATURE)
        summary = outputs[0]
        probabilities = outputs[1]
        reference_matrix = np.asarray(outputs[6], dtype=float)
        restarts = outputs[7]

        if not isinstance(summary, dict) or not summary:
            raise ValueError("V10 returned an invalid summary")

        for key in ("current_raw_hmm_confidence", "next_day_probability"):
            value = float(summary[key])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid probability: {key}={value}")

        if (
            reference_matrix.shape != (3, 3)
            or not np.isfinite(reference_matrix).all()
            or not np.allclose(reference_matrix.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("Invalid reference transition matrix")

        metrics = {
            "current_raw_hmm_confidence": float(
                summary["current_raw_hmm_confidence"]
            ),
            "next_day_probability": float(summary["next_day_probability"]),
            "input_rows": float(len(df)),
        }
        if "stickiness_score" in summary:
            metrics["stickiness_score"] = float(summary["stickiness_score"])

        mlflow.log_metrics(metrics)

        result = {
            "symbol": "NIFTY_50",
            "model_version": "HMM_V10",
            "dataset_sha256": dataset_hash,
            "dataset_start": str(df["timestamp"].min()),
            "dataset_end": str(df["timestamp"].max()),
            "summary": summary,
            "metrics": metrics,
        }

        json_path = ARTIFACT_DIR / "nifty50_hmm_experiment.json"
        json_path.write_text(json.dumps(result, indent=2, default=str))
        mlflow.log_artifact(str(json_path), artifact_path="results")

        if isinstance(probabilities, pd.DataFrame) and not probabilities.empty:
            csv_path = ARTIFACT_DIR / "nifty50_probabilities.csv"
            probabilities.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="results")

        if isinstance(restarts, pd.DataFrame) and not restarts.empty:
            restart_path = ARTIFACT_DIR / "nifty50_restarts.csv"
            restarts.to_csv(restart_path, index=False)
            mlflow.log_artifact(str(restart_path), artifact_path="diagnostics")

        print("MLOps experiment completed successfully")
        print("Run ID:", run.info.run_id)
        print("Dataset end:", result["dataset_end"])
        print("Current regime:", summary.get("current_regime"))
        print("Next regime:", summary.get("next_regime"))
        print("Next-day probability:", summary.get("next_day_probability"))
        print("Artifacts:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
