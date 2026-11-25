"""Shared helpers for page services: plain-language wording for percentiles and dataset freshness."""

from __future__ import annotations

from quantos_platform.core import status

STATE_LABEL = {"CALM": "Calm", "NORMAL": "Normal", "STRESSED": "Stressed"}
REGIME_LABEL = {"BULL": "Bull", "SIDEWAYS": "Sideways", "BEAR": "Bear"}

# datasets the platform reports freshness for (status.sqlite `freshness` table)
DATASETS = {
    "india_daily": "Index prices (daily)",
    "analytics": "Analytics (volatility, events, states, risk)",
    "options_snapshot": "Option chain snapshot",
    "v11": "Next-session model",
    "playbook": "Expiry playbook",
    "us_equities": "US stocks",
}


def level_word(p: float | None) -> str | None:
    """Percentile (0-1) in words: very low / low / normal / high / very high."""
    if p is None:
        return None
    if p < 0.1:
        return "very low"
    if p < 0.35:
        return "low"
    if p <= 0.65:
        return "normal"
    return "high" if p <= 0.9 else "very high"


def ordinal(p: float | None) -> str | None:
    """Percentile (0-1) as an ordinal: 0.81 -> '81st percentile'."""
    if p is None:
        return None
    n = int(round(p * 100))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} percentile"


def corr_word(p: float | None) -> str | None:
    if p is None:
        return None
    return "low" if p < 0.35 else "high" if p > 0.65 else "normal"


def freshness() -> list[dict]:
    """As-of time and last update of every tracked dataset (missing ones are skipped)."""
    out = []
    for key, label in DATASETS.items():
        f = status.freshness(key)
        if f:
            out.append({"dataset": key, "label": label, "as_of": f["as_of"], "updated": f["updated"]})
    return out
