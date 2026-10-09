
"""
QuantOS | Live Option Feed Collector
------------------------------------
Builds a persistent 1-minute/5-minute option market-data store for:
    NIFTY 50, NIFTY BANK, SENSEX

Architecture:
    Upstox option contracts REST
        -> select nearest expiries / ATM-centered strikes
        -> Upstox MarketDataStreamerV3 (FULL)
        -> maintain latest tick state
        -> snapshot every N seconds
        -> SQLite database

The database is intentionally used for live ingestion because it is
append-friendly and queryable while the process is running. It can later
be exported to Parquet for research.

Default snapshot interval: 60 seconds.
Set VOL_FEED_INTERVAL_SECONDS=300 for 5-minute snapshots.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
import upstox_client


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("UPSTOX_ANALYTICS_TOKEN")
if not TOKEN:
    raise RuntimeError("UPSTOX_ANALYTICS_TOKEN not found in .env")

SNAPSHOT_SECONDS = int(
    os.getenv("VOL_FEED_INTERVAL_SECONDS", "60")
)

EXPIRIES_PER_UNDERLYING = int(
    os.getenv("VOL_FEED_EXPIRIES", "2")
)

STRIKES_EACH_SIDE = int(
    os.getenv("VOL_FEED_STRIKES_EACH_SIDE", "15")
)

DATA_DIR = ROOT / "data" / "regime" / "options_live"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "options_live.sqlite"

UNDERLYINGS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "NIFTY_BANK": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
}

STOP = False


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print("\nStopping collector...")


signal.signal(signal.SIGINT, stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


def api_get(path: str, params: dict | None = None):
    response = requests.get(
        f"https://api.upstox.com{path}",
        params=params,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


def get_spots():
    keys = ",".join(UNDERLYINGS.values())

    payload = api_get(
        "/v3/market-quote/ltp",
        {"instrument_key": keys},
    )

    data = payload.get("data", {})

    spots = {}

    for symbol, key in UNDERLYINGS.items():
        normalized = key.replace("|", ":")

        item = (
            data.get(normalized)
            or data.get(key)
        )

        if not item:
            # Last fallback: find matching instrument_token.
            for value in data.values():
                if value.get("instrument_token") == key:
                    item = value
                    break

        if not item:
            raise RuntimeError(
                f"No LTP returned for {symbol}"
            )

        spots[symbol] = float(
            item["last_price"]
        )

    return spots


def get_contracts(underlying_key: str):
    payload = api_get(
        "/v2/option/contract",
        {"instrument_key": underlying_key},
    )

    contracts = payload.get("data", [])

    if not contracts:
        raise RuntimeError(
            f"No option contracts for {underlying_key}"
        )

    return contracts


def choose_contracts(
    symbol: str,
    underlying_key: str,
    spot: float,
):
    contracts = get_contracts(underlying_key)

    today = datetime.now().date()

    normalized = []

    for c in contracts:
        try:
            expiry = datetime.strptime(
                c["expiry"],
                "%Y-%m-%d",
            ).date()

            strike = float(c["strike_price"])
            key = c["instrument_key"]
            option_type = c["instrument_type"]

            if expiry < today:
                continue

            normalized.append({
                "symbol": symbol,
                "underlying_key": underlying_key,
                "instrument_key": key,
                "expiry": c["expiry"],
                "strike_price": strike,
                "instrument_type": option_type,
                "trading_symbol": c.get(
                    "trading_symbol"
                ),
                "lot_size": c.get("lot_size"),
                "weekly": c.get("weekly"),
            })

        except Exception:
            continue

    expiries = sorted({
        c["expiry"] for c in normalized
    })[:EXPIRIES_PER_UNDERLYING]

    selected = []

    for expiry in expiries:
        rows = [
            c for c in normalized
            if c["expiry"] == expiry
        ]

        strikes = sorted({
            c["strike_price"] for c in rows
        })

        if not strikes:
            continue

        # Choose the closest strike to spot as ATM.
        atm = min(
            strikes,
            key=lambda x: abs(x - spot),
        )

        atm_index = strikes.index(atm)

        lo = max(
            0,
            atm_index - STRIKES_EACH_SIDE,
        )
        hi = min(
            len(strikes),
            atm_index + STRIKES_EACH_SIDE + 1,
        )

        chosen_strikes = set(
            strikes[lo:hi]
        )

        selected.extend(
            c for c in rows
            if c["strike_price"] in chosen_strikes
        )

    return selected


def build_universe():
    spots = get_spots()

    selected = []

    print("=" * 70)
    print("QuantOS LIVE OPTION UNIVERSE")
    print("=" * 70)

    for symbol, key in UNDERLYINGS.items():
        contracts = choose_contracts(
            symbol,
            key,
            spots[symbol],
        )

        selected.extend(contracts)

        print(
            f"{symbol}: spot={spots[symbol]:,.2f} "
            f"| contracts={len(contracts)}"
        )

    payload = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "snapshot_seconds": SNAPSHOT_SECONDS,
        "expiries_per_underlying": (
            EXPIRIES_PER_UNDERLYING
        ),
        "strikes_each_side": STRIKES_EACH_SIDE,
        "underlyings": UNDERLYINGS,
        "spots": spots,
        "contracts": selected,
    }

    path = DATA_DIR / "option_universe.json"

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        f"Total option contracts: "
        f"{len(selected)}"
    )
    print(f"Saved: {path}")

    return payload


def init_db():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS option_snapshots (
            snapshot_ts TEXT NOT NULL,
            received_ts TEXT,
            symbol TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            expiry TEXT,
            strike REAL,
            option_type TEXT,
            ltp REAL,
            previous_close REAL,
            bid REAL,
            bid_qty REAL,
            ask REAL,
            ask_qty REAL,
            volume REAL,
            oi REAL,
            iv REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            rho REAL,
            exchange_ts TEXT,
            PRIMARY KEY (
                snapshot_ts,
                instrument_key
            )
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_option_snapshots_symbol_ts
        ON option_snapshots(symbol, snapshot_ts)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_option_snapshots_expiry_strike
        ON option_snapshots(expiry, strike, option_type)
        """
    )

    conn.commit()

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(option_snapshots)"
        ).fetchall()
    }

    if "exchange_ts" not in columns:
        conn.execute(
            "ALTER TABLE option_snapshots ADD COLUMN exchange_ts TEXT"
        )
        conn.commit()

    return conn


def recursive_find(obj: Any, names: set[str]):
    if isinstance(obj, dict):
        lowered = {
            str(k).lower(): v
            for k, v in obj.items()
        }

        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]

        for value in obj.values():
            found = recursive_find(value, names)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find(value, names)
            if found is not None:
                return found

    return None


def num(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def extract_feed(feed: dict):
    """
    Supports the JSON shapes returned by the V3 protobuf decoder:
      firstLevelWithGreeks
      fullFeed.marketFF
      ltpc
      firstDepth
      marketLevel
      optionGreeks
    """

    ltpc = recursive_find(
        feed,
        {"ltpc"},
    ) or {}

    depth = recursive_find(
        feed,
        {"firstDepth"},
    ) or {}

    greeks = recursive_find(
        feed,
        {"optionGreeks"},
    ) or {}

    # Full feed may expose an array of depth levels.
    if not depth:
        levels = recursive_find(
            feed,
            {"bidAskQuote"},
        )

        if levels:
            try:
                depth = levels[0]
            except Exception:
                depth = {}

    return {
        "ltp": num(
            ltpc.get("ltp")
        ),
        "previous_close": num(
            ltpc.get("cp")
        ),
        "bid": num(
            depth.get("bidP")
        ),
        "bid_qty": num(
            depth.get("bidQ")
        ),
        "ask": num(
            depth.get("askP")
        ),
        "ask_qty": num(
            depth.get("askQ")
        ),
        "volume": num(
            recursive_find(
                feed,
                {"vtt", "volume"},
            )
        ),
        "oi": num(
            recursive_find(
                feed,
                {"oi"},
            )
        ),
        "iv": num(
            greeks.get("iv")
        ) or num(
            recursive_find(feed, {"iv"})
        ),
        "delta": num(
            greeks.get("delta")
        ),
        "gamma": num(
            greeks.get("gamma")
        ),
        "theta": num(
            greeks.get("theta")
        ),
        "vega": num(
            greeks.get("vega")
        ),
        "rho": num(
            greeks.get("rho")
        ),
        "received_ts": datetime.now(timezone.utc).isoformat(),
        "exchange_ts": recursive_find(
            feed,
            {"ltt", "exchangeTs", "exchange_ts"},
        ),
    }


def connect_and_collect(universe):
    conn = init_db()

    metadata = {
        c["instrument_key"]: c
        for c in universe["contracts"]
    }

    keys = list(metadata.keys())

    # Include underlying spot feeds as well.
    keys.extend(
        universe["underlyings"].values()
    )

    keys = list(dict.fromkeys(keys))

    print(
        f"\nConnecting to Upstox WebSocket with "
        f"{len(keys)} instruments..."
    )

    configuration = upstox_client.Configuration()
    configuration.access_token = TOKEN

    streamer = upstox_client.MarketDataStreamerV3(
        upstox_client.ApiClient(configuration)
    )

    latest = {}
    last_snapshot_bucket = None

    def on_open():
        print("WebSocket connected.")
        streamer.subscribe(
            keys,
            "full",
        )
        print(
            f"Subscribed to {len(keys)} instruments."
        )

    def on_message(message):
        nonlocal last_snapshot_bucket

        if not isinstance(message, dict):
            return

        feeds = message.get("feeds", {})

        if not isinstance(feeds, dict):
            return

        now = datetime.now().astimezone()

        for instrument_key, feed in feeds.items():
            if instrument_key not in metadata:
                continue

            values = extract_feed(feed)

            if values["ltp"] is None:
                continue

            latest[instrument_key] = values

        bucket = int(
            time.time() // SNAPSHOT_SECONDS
        )

        if bucket == last_snapshot_bucket:
            return

        if not latest:
            return

        last_snapshot_bucket = bucket

        snapshot_ts = now.isoformat()

        rows = []

        for instrument_key, values in latest.items():
            meta = metadata.get(
                instrument_key
            )

            if not meta:
                continue

            rows.append((
                snapshot_ts,
                values.get("received_ts"),
                meta["symbol"],
                instrument_key,
                meta["expiry"],
                meta["strike_price"],
                meta["instrument_type"],
                values.get("ltp"),
                values.get("previous_close"),
                values.get("bid"),
                values.get("bid_qty"),
                values.get("ask"),
                values.get("ask_qty"),
                values.get("volume"),
                values.get("oi"),
                values.get("iv"),
                values.get("delta"),
                values.get("gamma"),
                values.get("theta"),
                values.get("vega"),
                values.get("rho"),
                (
                    str(values.get("exchange_ts"))
                    if values.get("exchange_ts") is not None
                    else None
                ),
            ))

        if rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO option_snapshots (
                    snapshot_ts,
                    received_ts,
                    symbol,
                    instrument_key,
                    expiry,
                    strike,
                    option_type,
                    ltp,
                    previous_close,
                    bid,
                    bid_qty,
                    ask,
                    ask_qty,
                    volume,
                    oi,
                    iv,
                    delta,
                    gamma,
                    theta,
                    vega,
                    rho,
                    exchange_ts
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                rows,
            )

            conn.commit()

            print(
                f"[{snapshot_ts}] "
                f"stored {len(rows)} option snapshots"
            )

    def on_error(error):
        print(f"WebSocket error: {error}")

    def on_close(*args):
        print("WebSocket closed.")

    streamer.on("open", on_open)
    streamer.on("message", on_message)
    streamer.on("error", on_error)
    streamer.on("close", on_close)

    streamer.auto_reconnect(
        True,
        5,
        100,
    )

    streamer.connect()

    try:
        while not STOP:
            time.sleep(1)
    finally:
        try:
            streamer.disconnect()
        except Exception:
            pass

        conn.commit()
        conn.close()

        print(f"Database: {DB_PATH}")
        print("Collector stopped cleanly.")


def main():
    universe = build_universe()

    if not universe["contracts"]:
        raise RuntimeError(
            "No option contracts selected."
        )

    connect_and_collect(universe)


if __name__ == "__main__":
    main()
