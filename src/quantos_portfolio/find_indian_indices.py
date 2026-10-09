from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

TOKEN = os.getenv("UPSTOX_ANALYTICS_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "UPSTOX_ANALYTICS_TOKEN not found in .env"
    )


URL = "https://api.upstox.com/v2/instruments/search"

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}


QUERIES = [
    "NIFTY MIDCAP 100",
    "NIFTY NEXT 50",
]


for query in QUERIES:

    print("\n" + "=" * 70)
    print(f"SEARCHING: {query}")
    print("=" * 70)

    params = {
        "query": query,
        "exchanges": "NSE",
        "segments": "INDEX",
        "page_number": 1,
        "records": 30,
    }

    response = requests.get(
        URL,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    payload = response.json()

    data = payload.get("data", [])

    if not data:
        print("No instruments found.")
        continue

    for instrument in data:

        print(
            f"Name:             {instrument.get('name')}\n"
            f"Instrument Key:   {instrument.get('instrument_key')}\n"
            f"Trading Symbol:   {instrument.get('trading_symbol')}\n"
            f"Segment:          {instrument.get('segment')}\n"
            f"Instrument Type:  {instrument.get('instrument_type')}\n"
            f"Exchange Token:   {instrument.get('exchange_token')}\n"
        )