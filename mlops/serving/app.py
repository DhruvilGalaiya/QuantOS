
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data/regime/features/nifty_50.parquet"
TEMPERATURE = 1.85
MODEL_VERSION = "HMM_V10"

# The snapshot is calculated once when the API starts.
# Requests reuse it instead of refitting the HMM every time.
_snapshot: dict | None = None


def build_snapshot() -> dict:
    from mlops.experiments.run_hmm_experiment import (
        load_and_validate,
    )
    import quantos_hmm_v10 as v10

    df = load_and_validate(FEATURE_FILE)

    summary, probabilities, transition_matrix, *_ = (
        v10.production_output(df, TEMPERATURE)
    )

    if not isinstance(summary, dict) or not summary:
        raise ValueError("HMM returned an invalid summary")

    if not isinstance(probabilities, pd.DataFrame):
        raise ValueError("HMM returned invalid probabilities")

    required = {"regime", "next_day_probability"}
    if not required.issubset(probabilities.columns):
        raise ValueError("HMM probability output has missing columns")

    probs = probabilities.set_index("regime")[
        "next_day_probability"
    ].to_dict()

    expected = {"BULL", "SIDE", "BEAR"}
    if set(probs) != expected:
        raise ValueError("HMM did not return all three regimes")

    values = [float(probs[name]) for name in ("BULL", "SIDE", "BEAR")]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("HMM returned an invalid probability")

    if abs(sum(values) - 1.0) > 1e-5:
        raise ValueError("Regime probabilities do not sum to one")

    return {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_through": str(summary["latest_date"]),
        "symbol": "NIFTY_50",
        "summary": summary,
        "next_day_probabilities": {
            name: float(probs[name])
            for name in ("BULL", "SIDE", "BEAR")
        },
        "transition_matrix": transition_matrix.to_dict(
            orient="records"
        ),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _snapshot

    # Fail startup if the dataset or model output is invalid.
    _snapshot = build_snapshot()
    yield
    _snapshot = None


app = FastAPI(
    title="QuantOS HMM Inference API",
    version="1.0.0",
    description="Serves the latest validated QuantOS HMM V10 regime snapshot.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if _snapshot is not None else "not_ready",
        "model_version": MODEL_VERSION,
    }


@app.get("/v1/regime")
def get_regime():
    if _snapshot is None:
        raise HTTPException(
            status_code=503,
            detail="HMM snapshot is not ready",
        )

    return _snapshot
