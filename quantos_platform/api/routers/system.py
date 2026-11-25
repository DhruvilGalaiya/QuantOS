from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from quantos_platform.api.etag import cached_json
from quantos_platform.api.schemas import System, Ticks
from quantos_platform.core import settings
from quantos_platform.core.upstox import UpstoxAuthError
from quantos_platform.services import system, ticks

router = APIRouter(tags=["system"])


@router.get("/system", response_model=System)
def get_system(request: Request):
    """Last runs of every data job and the freshness of each dataset."""
    return cached_json(request, [settings.STATUS_DB], system.build, System)


@router.get("/ticks", response_model=Ticks)
def get_ticks() -> dict:
    """Live NIFTY 50, NIFTY BANK and India VIX prices. The gateway serves these only to the signed-in owner."""
    if settings.PUBLIC:
        raise HTTPException(status_code=403, detail="live prices are off in public mode")
    try:
        return ticks.quotes()
    except UpstoxAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # provider down or rate limited: the strip keeps its last values
        raise HTTPException(status_code=503, detail=f"live prices unavailable: {type(exc).__name__}") from exc
