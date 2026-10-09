from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app import models


# ============================================================
# CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

DASHBOARD_STATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "portfolio"
    / "dashboard_state"
    / "quantos_dashboard_state.json"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="QuantOS API",
    description="Institutional Quantitative Market Intelligence Platform",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "project": "QuantOS",
        "status": "online",
        "message": "Quantitative research backend is running",
    }


# ============================================================
# DATABASE TEST
# ============================================================

@app.get("/db-test")
def database_test():
    try:
        with engine.connect() as connection:
            return {
                "database": "connected",
                "status": "success",
            }

    except Exception as e:
        return {
            "database": "connection_failed",
            "error": str(e),
        }


# ============================================================
# QUANTOS DASHBOARD STATE
# ============================================================

@app.get("/api/quantos/state")
def get_quantos_state():
    """
    Return the latest unified QuantOS dashboard state.

    This endpoint is read-only.

    It does NOT:
    - train models
    - run backtests
    - modify portfolio allocations
    - modify market data
    """

    if not DASHBOARD_STATE_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Dashboard state file not found",
                "path": str(DASHBOARD_STATE_FILE),
            },
        )

    try:
        with DASHBOARD_STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            state = json.load(f)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Dashboard state JSON is invalid",
                "message": str(e),
            },
        )

    return state