"""
QuantOS settings: one place for paths, environment and run mode.

Environment variables (all optional except the token for data jobs):
    UPSTOX_ANALYTICS_TOKEN   Upstox analytics (read-only market data) token
    QUANTOS_MODE             "local" (default) or "public"
                             local : pages may refresh stale data on launch, refresh buttons shown
                             public: pages are read-only; only scheduled jobs fetch data
    QUANTOS_DATA_DIR         override the data directory (default: <repo>/data)
    QUANTOS_LOG_LEVEL        logging level (default INFO)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

IST = "Asia/Kolkata"
# Streamlit Community Cloud checks repos out under /mount/src; default to read-only there
# so the public site never tries to fetch data even if QUANTOS_MODE was not set.
_HOSTED = str(ROOT).startswith("/mount/src/")
MODE = os.getenv("QUANTOS_MODE", "public" if _HOSTED else "local").strip().lower()
PUBLIC = MODE == "public"
LOG_LEVEL = os.getenv("QUANTOS_LOG_LEVEL", "INFO").upper()

DATA_DIR = Path(os.getenv("QUANTOS_DATA_DIR", ROOT / "data")).resolve()

# --- market data
REGIME_DIR = DATA_DIR / "regime"
DAILY_DIR = REGIME_DIR / "daily"
FEATURE_DIR = REGIME_DIR / "features"
INTRADAY_DIR = REGIME_DIR / "intraday" / "15m"
PREDICTION_DIR = REGIME_DIR / "live_predictions"
V11_DIR = PREDICTION_DIR / "v11"
US_OVERNIGHT_FILE = REGIME_DIR / "us_overnight" / "us_close.parquet"
ASIA_FILE = REGIME_DIR / "global" / "asia_close.parquet"
US_EQUITY_DIR = REGIME_DIR / "portfolio_us"

# --- derivatives / volatility
DERIV_DIR = DATA_DIR / "derivatives"
OPTIONS_DIR = DERIV_DIR / "options"
PLAYBOOK_DIR = DERIV_DIR / "playbook"
OPTIONS_DB = REGIME_DIR / "options_live" / "options_live.sqlite"
SURFACE_DIR = REGIME_DIR / "volatility_surface"

# --- descriptive analytics (Volatility & Edge, Event Lab, Today brief)
ANALYTICS_DIR = DATA_DIR / "analytics"

# --- system (job status, logs, caches)
SYSTEM_DIR = DATA_DIR / "system"
STATUS_DB = SYSTEM_DIR / "status.sqlite"
LOG_DIR = SYSTEM_DIR / "logs"
CACHE_DIR = SYSTEM_DIR / "cache"
LOCK_DIR = SYSTEM_DIR / "locks"

LEGACY_SRC = ROOT / "src"   # research code + the legacy V10 chain


def upstox_token() -> str | None:
    return os.getenv("UPSTOX_ANALYTICS_TOKEN")


def ensure_dirs() -> None:
    for d in (SYSTEM_DIR, LOG_DIR, CACHE_DIR, LOCK_DIR):
        d.mkdir(parents=True, exist_ok=True)
