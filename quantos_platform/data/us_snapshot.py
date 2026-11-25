"""
Weekly snapshot of US company fundamentals and financial statements, so pages read files instead of calling Yahoo.

    python -m quantos_platform.data.us_snapshot [--force] [TICKER ...]

Writes data/regime/portfolio_us/fundamentals/<TICKER>.json:
    {"ticker", "fetched_at", "fundamentals": fetch_fundamentals(...), "financials": fetch_financials(...)}
A ticker is refetched when its file is older than MAX_AGE_DAYS (ratios and statements change quarterly).
The us_equities job calls refresh() after topping up prices.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from quantos_platform.data.us_equities import OUTPUT_DIR

SNAPSHOT_DIR = OUTPUT_DIR / "fundamentals"
MAX_AGE_DAYS = 7
PAUSE_S = 0.4          # be gentle with Yahoo between tickers

def path_for(ticker: str):
    return SNAPSHOT_DIR / f"{ticker.upper()}.json"


def load(ticker: str) -> dict:
    p = path_for(ticker)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def is_stale(ticker: str, max_age_days: float = MAX_AGE_DAYS) -> bool:
    p = path_for(ticker)
    return not p.exists() or (time.time() - p.stat().st_mtime) > max_age_days * 86400


def snapshot(ticker: str) -> dict:
    from quantos_platform.data.us_fundamentals import fetch_financials, fetch_fundamentals

    payload = {"ticker": ticker.upper(), "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
               "fundamentals": fetch_fundamentals(ticker)}
    try:
        payload["financials"] = fetch_financials(ticker)
    except Exception as exc:  # statements are optional; the profile and ratios still help
        payload["financials"] = {}
        payload["financials_error"] = f"{type(exc).__name__}: {exc}"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path_for(ticker).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(path_for(ticker))
    return payload


def refresh(tickers: list[str], force: bool = False) -> dict:
    """Snapshot every stale ticker; returns {"updated": n, "failed": {ticker: error}}."""
    updated, failed = 0, {}
    for t in tickers:
        if not force and not is_stale(t):
            continue
        try:
            snapshot(t)
            updated += 1
        except Exception as exc:
            failed[t] = f"{type(exc).__name__}: {exc}"
        time.sleep(PAUSE_S)
    return {"updated": updated, "failed": failed}


def main() -> None:
    from quantos_platform.data.us_universe import load_universe

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tickers = args.tickers or load_universe()["ticker"].tolist()
    res = refresh(tickers, force=args.force)
    print(f"updated {res['updated']} of {len(tickers)}; failed {len(res['failed'])}")
    for t, e in list(res["failed"].items())[:10]:
        print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
