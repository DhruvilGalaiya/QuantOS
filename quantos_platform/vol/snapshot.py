"""
QuantOS | Volatility dashboard launch pipeline

Keeps the volatility dashboard current without running the WebSocket feed:

    1. If the latest option snapshot in options_live.sqlite is stale
       (older than MAX_AGE_MIN during market hours, or older than the last
       session's close outside them), pull ONE snapshot from the Upstox
       option-chain REST endpoint (same expiries / strike window as the live
       feed) and store it in the same table with the same schema.
    2. If the surfaces are older than the latest snapshot, rebuild them with
       surface_engine.py (own Black-76 IVs, SVI + SABR fits, arbitrage checks).
       iv_surface.py / sabr_surface.py still work standalone.

The live feed (live_option_feed.py) keeps working as before: while it runs,
its snapshots are always fresh, so step 1 is skipped.

    python -m quantos_platform.vol.snapshot [--force]
"""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

from quantos_platform.core import settings
from quantos_platform.core.calendar import last_close, market_open_now
from quantos_platform.core.upstox import get
from quantos_platform.core.validation import check_option_rows

IST = settings.IST
DB_PATH = settings.OPTIONS_DB
SURFACE_DIR = settings.SURFACE_DIR
STATUS_FILE = SURFACE_DIR / "refresh_status.json"
LOCK_FILE = SURFACE_DIR / ".refresh.lock"

UNDERLYINGS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
}
EXPIRIES_PER_UNDERLYING = 6      # REST chain is cheap: weeklies + monthlies for a real term structure
STRIKE_WINDOW = 0.10             # keep strikes within +/-10% of spot (skew wings)
MAX_AGE_MIN = 10                 # intraday staleness threshold
OPEN, CLOSE = "09:15", "15:30"


@dataclass
class VolRefreshReport:
    snapshot_taken: bool = False
    surfaces_built: bool = False
    snapshot_ts: str | None = None
    rows: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


# ============================================================
# DATABASE
# ============================================================

def latest_snapshot_ts() -> pd.Timestamp | None:
    if not DB_PATH.exists():
        return None
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        row = conn.execute("SELECT MAX(snapshot_ts) FROM option_snapshots").fetchone()
    if not row or row[0] is None:
        return None
    ts = pd.Timestamp(row[0])
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def snapshot_is_stale(latest: pd.Timestamp | None) -> bool:
    if latest is None:
        return True
    if market_open_now():
        return _now() - latest > pd.Timedelta(minutes=MAX_AGE_MIN)
    return latest < last_close()


def _init_db() -> sqlite3.Connection:
    # Reuse the live feed's schema definition so the two never drift apart.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS option_snapshots (
            snapshot_ts TEXT NOT NULL, received_ts TEXT, symbol TEXT NOT NULL,
            instrument_key TEXT NOT NULL, expiry TEXT, strike REAL, option_type TEXT,
            ltp REAL, previous_close REAL, bid REAL, bid_qty REAL, ask REAL, ask_qty REAL,
            volume REAL, oi REAL, iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
            rho REAL, exchange_ts TEXT,
            PRIMARY KEY (snapshot_ts, instrument_key)
        )
        """
    )
    return conn


# ============================================================
# REST SNAPSHOT
# ============================================================

def _expiries(underlying_key: str) -> list[str]:
    data = get("/v2/option/contract", {"instrument_key": underlying_key}).get("data", [])
    now = _now()
    today = now.strftime("%Y-%m-%d")
    after_close = now.strftime("%H:%M") >= CLOSE
    exps = sorted({c["expiry"] for c in data
                   if c["expiry"] > today or (c["expiry"] == today and not after_close)})
    return exps[:EXPIRIES_PER_UNDERLYING]


def _chain_rows(symbol: str, underlying_key: str, snapshot_ts: str) -> list[tuple]:
    rows = []
    for expiry in _expiries(underlying_key):
        chain = get("/v2/option/chain", {"instrument_key": underlying_key,
                                          "expiry_date": expiry}).get("data", []) or []
        if not chain:
            continue
        spot = float(chain[0].get("underlying_spot_price") or 0)
        for e in chain:
            strike = float(e["strike_price"])
            if spot and abs(strike / spot - 1) > STRIKE_WINDOW:
                continue
            for side, opt_type in (("call_options", "CE"), ("put_options", "PE")):
                o = e.get(side) or {}
                md, g = o.get("market_data") or {}, o.get("option_greeks") or {}
                if not o.get("instrument_key"):
                    continue
                iv = g.get("iv")
                rows.append((
                    snapshot_ts, snapshot_ts, symbol, o["instrument_key"], expiry, strike, opt_type,
                    md.get("ltp"), md.get("close_price"), md.get("bid_price"), md.get("bid_qty"),
                    md.get("ask_price"), md.get("ask_qty"), md.get("volume"), md.get("oi"),
                    # REST gives IV in percent; the feed stores decimals
                    (float(iv) / 100.0) if iv not in (None, 0) else None,
                    g.get("delta"), g.get("gamma"), g.get("theta"), g.get("vega"), None, None,
                ))
    return rows


def take_snapshot(report: VolRefreshReport) -> None:
    snapshot_ts = _now().isoformat()
    all_rows = []
    for symbol, key in UNDERLYINGS.items():
        try:
            rows = _chain_rows(symbol, key, snapshot_ts)
            report.rows[symbol] = len(rows)
            all_rows += rows
        except Exception as exc:
            report.errors[symbol] = f"{type(exc).__name__}: {exc}"
    if not all_rows:
        raise RuntimeError("option chain returned no rows")
    problems = check_option_rows(all_rows)
    if problems:
        raise RuntimeError("option snapshot rejected: " + "; ".join(problems))
    conn = _init_db()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO option_snapshots VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            all_rows,
        )
        conn.commit()
    finally:
        conn.close()
    report.snapshot_taken = True
    report.snapshot_ts = snapshot_ts
    report.log.append(f"snapshot {snapshot_ts}: " + ", ".join(f"{k} {v}" for k, v in report.rows.items()))


# ============================================================
# SURFACES
# ============================================================

def _status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {}


def surfaces_are_stale(latest: pd.Timestamp | None) -> bool:
    built_for = _status().get("snapshot_ts")
    if latest is None:
        return False
    return built_for is None or pd.Timestamp(built_for) < latest


def build_surfaces(report: VolRefreshReport) -> None:
    """Surface engine v2 (own Black-76 IVs, SVI + SABR fits, arbitrage checks)."""
    from quantos_platform.vol import surface as surface_engine

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        errors = surface_engine.run(log=lambda *a: None)
    for sym, err in errors.items():
        report.errors[f"surface:{sym}"] = err
    report.surfaces_built = True


# ============================================================
# ENTRY POINT
# ============================================================

def run(force: bool = False) -> VolRefreshReport:
    report = VolRefreshReport()
    SURFACE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime < 300:
        report.log.append("another refresh is running; using current files")
        report.snapshot_ts = _status().get("snapshot_ts")
        return report
    LOCK_FILE.write_text(str(time.time()))
    try:
        latest = latest_snapshot_ts()
        if force or snapshot_is_stale(latest):
            try:
                take_snapshot(report)
                latest = latest_snapshot_ts()
            except Exception as exc:
                report.errors["snapshot"] = f"{type(exc).__name__}: {exc}"
        else:
            report.log.append(f"option snapshot is current ({latest})")
        if latest is not None and (force or report.snapshot_taken or surfaces_are_stale(latest)):
            build_surfaces(report)
            STATUS_FILE.write_text(json.dumps({
                "snapshot_ts": latest.isoformat(),
                "source": "Upstox option-chain REST" if report.snapshot_taken else "options_live.sqlite",
                "built_at": _now().isoformat(),
                "errors": report.errors,
            }, indent=2))
        report.snapshot_ts = latest.isoformat() if latest is not None else None
    finally:
        LOCK_FILE.unlink(missing_ok=True)
    return report


def load_status() -> dict:
    return _status()


if __name__ == "__main__":
    t = time.time()
    r = run(force="--force" in sys.argv)
    print(f"Snapshot     : {r.snapshot_ts}  (new: {r.snapshot_taken})")
    print(f"Rows         : {r.rows or '-'}")
    print(f"Surfaces     : {'rebuilt' if r.surfaces_built else 'already current'}")
    for line in r.log:
        print(line)
    for k, v in r.errors.items():
        print(f"ERROR {k}: {v}")
    print(f"Elapsed      : {time.time() - t:.1f}s")
