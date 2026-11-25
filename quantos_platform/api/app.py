"""
QuantOS internal quant API (FastAPI). Only the Node gateway talks to it; bind it to localhost / a private network.

    uvicorn quantos_platform.api.app:app --port 8001 --reload

Routes live under /internal/v1. Responses carry an ETag built from the artifact files they read, so the
gateway (and browsers behind it) can revalidate cheaply. The service never runs models or jobs; the one live call is /ticks (owner-only prices behind the gateway).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from quantos_platform.api import API_VERSION
from quantos_platform.api.routers import meta, pages, system, today

app = FastAPI(title="QuantOS quant service", version=f"{API_VERSION}.0.0",
              description="Internal analytics API behind the QuantOS gateway. Descriptive measurements and "
                          "historical base rates; no forecasts.",
              docs_url="/internal/docs", openapi_url="/internal/openapi.json", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(meta.router, prefix="/internal/v1")
app.include_router(today.router, prefix="/internal/v1")
app.include_router(pages.router, prefix="/internal/v1")
app.include_router(system.router, prefix="/internal/v1")


@app.get("/internal/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}
