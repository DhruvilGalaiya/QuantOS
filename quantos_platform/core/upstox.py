"""
Upstox REST client (read-only market data).

    get(endpoint, params) -> parsed JSON

Retries transient failures (network errors, 429 rate limits, 5xx) with backoff,
and raises UpstoxAuthError with a clear message when the token is missing/expired.
"""

from __future__ import annotations

import time

import requests

from quantos_platform.core import settings

BASE_URL = "https://api.upstox.com"
TIMEOUT_S = 30
RETRIES = 3

_session = requests.Session()


class UpstoxError(RuntimeError):
    pass


class UpstoxAuthError(UpstoxError):
    pass


def get(endpoint: str, params: dict | None = None) -> dict:
    token = settings.upstox_token()
    if not token:
        raise UpstoxAuthError("UPSTOX_ANALYTICS_TOKEN is not set (add it to .env or the server environment)")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = _session.get(BASE_URL + endpoint, headers=headers, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as exc:  # network blip
            last_exc = exc
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise UpstoxAuthError(f"Upstox rejected the token ({r.status_code}): it is invalid or has expired")
        if r.status_code == 429 or r.status_code >= 500:
            last_exc = UpstoxError(f"Upstox {r.status_code}: {r.text[:200]}")
            time.sleep((30 if r.status_code == 429 else 3) * (attempt + 1))
            continue
        raise UpstoxError(f"Upstox {r.status_code} for {endpoint}: {r.text[:300]}")
    raise UpstoxError(f"Upstox request failed after {RETRIES} attempts: {last_exc}")
