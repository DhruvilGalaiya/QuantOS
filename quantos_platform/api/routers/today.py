from __future__ import annotations

from fastapi import APIRouter, Request

from quantos_platform.api.etag import cached_json
from quantos_platform.api.schemas import Today
from quantos_platform.core import settings
from quantos_platform.services import today

router = APIRouter(tags=["today"])


def _sources() -> list:
    a = settings.ANALYTICS_DIR
    return [a / "today.json", a / "news_calendar.json", a / "risk.json", a / "states_nifty_50.json",
            a / "states_nifty_bank.json", settings.DAILY_DIR / "nifty_50.parquet",
            settings.DAILY_DIR / "nifty_bank.parquet", settings.STATUS_DB]


@router.get("/today", response_model=Today, responses={304: {"description": "Not modified"}})
def get_today(request: Request):
    """The Today brief: index cards, the range options are pricing by expiry, readings, events and news."""
    return cached_json(request, _sources(), today.build, Today)
