from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FEATURE_FILE = ROOT / "data/regime/features/nifty_50.parquet"
TEMPERATURE = 1.85
MODEL_VERSION = "HMM_V10"
FRESHNESS_MAX_DAYS = int(os.getenv("FRESHNESS_MAX_DAYS", "3"))

_snapshot: dict | None = None


def data_age_days(snapshot: dict) -> int:
    data_date = date.fromisoformat(str(snapshot["data_through"])[:10])
    return (date.today() - data_date).days


def is_data_fresh(snapshot: dict) -> bool:
    age = data_age_days(snapshot)
    return 0 <= age <= FRESHNESS_MAX_DAYS


def build_snapshot() -> dict:
    import quantos_hmm_v10 as v10
    from mlops.experiments.run_hmm_experiment import load_and_validate

    df = load_and_validate(FEATURE_FILE)
    outputs = v10.production_output(df, TEMPERATURE)

    summary = outputs[0]
    probabilities = outputs[1]
    transition_matrix = outputs[2]

    if not isinstance(summary, dict) or not summary:
        raise ValueError("HMM returned an invalid summary")

    if not isinstance(probabilities, pd.DataFrame):
        raise ValueError("HMM returned invalid probabilities")

    required = {"regime", "next_day_probability"}
    if not required.issubset(probabilities.columns):
        raise ValueError("Probability output is missing required columns")

    probs = probabilities.set_index("regime")["next_day_probability"].to_dict()
    expected = {"BULL", "SIDE", "BEAR"}

    if set(probs) != expected:
        raise ValueError("HMM did not return all three regimes")

    probs = {name: float(probs[name]) for name in ("BULL", "SIDE", "BEAR")}

    if any(not 0 <= value <= 1 for value in probs.values()):
        raise ValueError("HMM returned an invalid probability")

    if abs(sum(probs.values()) - 1.0) > 1e-5:
        raise ValueError("Regime probabilities do not sum to one")

    return {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_through": str(summary["latest_date"]),
        "symbol": "NIFTY_50",
        "summary": summary,
        "next_day_probabilities": probs,
        "transition_matrix": transition_matrix.to_dict(orient="records"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _snapshot
    _snapshot = build_snapshot()
    yield
    _snapshot = None


app = FastAPI(
    title="QuantOS HMM Inference API",
    version="1.1.0",
    description="Serves a validated QuantOS HMM V10 regime snapshot with freshness monitoring.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if _snapshot is None:
        return {
            "status": "not_ready",
            "model_version": MODEL_VERSION,
        }

    age = data_age_days(_snapshot)
    fresh = is_data_fresh(_snapshot)

    return {
        "status": "ok" if fresh else "degraded",
        "model_version": MODEL_VERSION,
        "data_through": _snapshot["data_through"],
        "data_age_days": age,
        "freshness_max_days": FRESHNESS_MAX_DAYS,
        "data_fresh": fresh,
    }


@app.get("/v1/regime")
def get_regime():
    if _snapshot is None:
        raise HTTPException(status_code=503, detail="HMM snapshot is not ready")

    if not is_data_fresh(_snapshot):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "stale_market_data",
                "data_through": _snapshot["data_through"],
                "data_age_days": data_age_days(_snapshot),
                "freshness_max_days": FRESHNESS_MAX_DAYS,
            },
        )

    return _snapshot
