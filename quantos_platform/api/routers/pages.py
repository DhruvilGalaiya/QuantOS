"""Page routes: one instrument, index or ticker per request."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from quantos_platform.analytics import INDICES
from quantos_platform.api.etag import cached_json
from quantos_platform.api.schemas import (
    Events,
    Regimes,
    RiskMonitor,
    States,
    Surface,
    UsCompany,
    UsUniverse,
    Volatility,
)
from quantos_platform.core import settings
from quantos_platform.data import us_snapshot
from quantos_platform.services import events, regimes, risk, states, surface, us_stocks, volatility

router = APIRouter(tags=["pages"])
SYMBOL = Path(pattern=r"^[A-Z0-9_]{2,24}$")
TICKER = Path(pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


@router.get("/regimes/{symbol}", response_model=Regimes, responses={304: {"description": "Not modified"}, 404: {}})
def get_regimes(request: Request, symbol: str = SYMBOL):
    """Trend regime, persistence, transitions, every instrument, and the next-session forecast with its record."""
    if symbol not in regimes.INSTRUMENTS:
        raise HTTPException(404, f"unknown instrument {symbol}")
    p = settings.PREDICTION_DIR
    files = [p / "latest_predictions.json", p / f"prediction_history_{symbol.lower()}.csv",
             settings.V11_DIR / f"metrics_v11_{symbol.lower()}.json", settings.V11_DIR / f"prediction_history_v11_{symbol.lower()}.csv",
             settings.DAILY_DIR / f"{symbol.lower()}.parquet"]
    return cached_json(request, files, lambda: regimes.build(symbol), Regimes)


@router.get("/volatility/{symbol}", response_model=Volatility, responses={304: {"description": "Not modified"}, 404: {}})
def get_volatility(request: Request, symbol: str = SYMBOL):
    """Realized vs implied volatility, the premium sellers collect, the vol cone, GARCH-t and overnight straddle results."""
    if symbol not in INDICES:
        raise HTTPException(404, f"unknown index {symbol}")
    a, s = settings.ANALYTICS_DIR, symbol.lower()
    files = [a / f"volatility_{s}.json", a / f"volatility_{s}_series.parquet", a / f"straddle_decay_{s}.parquet", a / "today.json"]
    return cached_json(request, files, lambda: volatility.build(symbol), Volatility)


@router.get("/us", response_model=UsUniverse, responses={304: {"description": "Not modified"}})
def get_us_universe(request: Request):
    """The US large-cap universe with returns and 3-month sparklines."""
    d = settings.US_EQUITY_DIR
    return cached_json(request, [d / "universe.json", d / "AAPL.parquet", d / "MSFT.parquet"], us_stocks.universe, UsUniverse)


@router.get("/us/{ticker}", response_model=UsCompany, responses={304: {"description": "Not modified"}, 404: {}})
def get_us_company(request: Request, ticker: str = TICKER):
    """One company in plain language: price, returns, explained ratios, business summary and statements."""
    t = ticker.upper()
    d = settings.US_EQUITY_DIR
    files = [d / "universe.json", d / f"{t}.parquet", us_snapshot.path_for(t)]

    def build():
        try:
            return us_stocks.company(t)
        except KeyError as exc:
            raise HTTPException(404, f"{t} is not in the stored US universe") from exc

    return cached_json(request, files, build, UsCompany)


@router.get("/states/{symbol}", response_model=States, responses={304: {"description": "Not modified"}, 404: {}})
def get_states(request: Request, symbol: str = SYMBOL):
    """Calm, Normal or Stressed: behaviour by state, how long states last, the timeline and every major fall."""
    if symbol not in states.SYMBOLS:
        raise HTTPException(404, f"unknown index {symbol}")
    a, s = settings.ANALYTICS_DIR, symbol.lower()
    return cached_json(request, [a / f"states_{s}.json", a / f"states_{s}_series.parquet"], lambda: states.build(symbol), States)


@router.get("/events/{symbol}", response_model=Events, responses={304: {"description": "Not modified"}, 404: {}})
def get_events(request: Request, symbol: str = SYMBOL):
    """What followed each market event, against ordinary days, with confidence intervals and every occurrence."""
    if symbol not in INDICES:
        raise HTTPException(404, f"unknown index {symbol}")
    a, s = settings.ANALYTICS_DIR, symbol.lower()
    return cached_json(request, [a / f"events_summary_{s}.json", a / f"events_{s}.parquet"], lambda: events.build(symbol), Events)


@router.get("/risk", response_model=RiskMonitor, responses={304: {"description": "Not modified"}})
def get_risk(request: Request):
    """Sector co-movement, turbulence, the sector scoreboard, loss levels and past crashes for replay."""
    a = settings.ANALYTICS_DIR
    return cached_json(request, [a / "risk.json", a / "risk_series.parquet", a / "risk_returns.parquet"], risk.build, RiskMonitor)


@router.get("/surface/{symbol}", response_model=Surface, responses={304: {"description": "Not modified"}, 404: {}})
def get_surface(request: Request, symbol: str = SYMBOL):
    """The SVI/SABR implied-volatility surface, smiles against quotes, term structures and calibration checks."""
    if symbol not in surface.SYMBOLS:
        raise HTTPException(404, f"unknown index {symbol}")
    return cached_json(request, [settings.SURFACE_DIR / f"{symbol.lower()}_surface_v2.json"], lambda: surface.build(symbol), Surface)
