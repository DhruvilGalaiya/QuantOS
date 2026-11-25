from __future__ import annotations

from fastapi import APIRouter

from quantos_platform.api import API_VERSION
from quantos_platform.api.schemas import Meta
from quantos_platform.core import calendar, settings
from quantos_platform.services.common import freshness

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=Meta)
def get_meta() -> dict:
    """Run mode, whether NSE is open now, and how fresh each dataset is."""
    return {"version": API_VERSION, "mode": settings.MODE, "market_open": calendar.market_open_now(),
            "now": calendar.now_ist().isoformat(), "freshness": freshness()}
