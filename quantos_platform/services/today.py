"""
Today page service: one JSON payload for the market brief (NIFTY 50 and NIFTY BANK).

Reads analytics/today.json (built by the analytics job), the Market States and Risk artifacts, the news
calendar and the daily price history. Adds the page-level wording the Streamlit Today page used to compute
(percentile words, the "unusual" rule for event base rates, the plain-language readings).
Text fields may contain **bold** markers; the frontend renders them as emphasis.
"""

from __future__ import annotations

from quantos_platform.analytics import load_daily, read_json
from quantos_platform.analytics.risk import MC_DAYS, MC_PATHS
from quantos_platform.services.common import REGIME_LABEL, STATE_LABEL, corr_word, freshness, level_word, ordinal

HISTORY_SESSIONS = 260       # about one year of daily closes for the price chart
UNUSUAL_P = 0.05             # an event's base rate counts as unusual below this p-value ...
UNUSUAL_MIN_N = 10           # ... and with at least this many past cases


def _history(symbol: str, as_of: str | None) -> list[dict]:
    try:
        c = load_daily(symbol)["close"]
    except (OSError, KeyError, ValueError):
        return []
    if as_of:
        c = c.loc[:as_of]
    c = c.iloc[-HISTORY_SESSIONS:]
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in c.items()]


def _state(symbol: str) -> dict | None:
    cur = read_json(f"states_{symbol.lower()}.json").get("current")
    if not cur:
        return None
    return {"state": cur["state"], "label": STATE_LABEL.get(cur["state"], cur["state"]), "streak": cur.get("streak"),
            "since": cur.get("since"), "still_after_5": cur.get("still_after_5"),
            "still_after_21": cur.get("still_after_21")}


def _vol(v: dict) -> dict:
    out = {**v, "rv21_level": level_word(v.get("rv21_percentile")), "rv21_ordinal": ordinal(v.get("rv21_percentile"))}
    if v.get("vix") is not None:
        out.update(vix_level=level_word(v.get("vix_percentile_1y")), vix_ordinal=ordinal(v.get("vix_percentile_1y")),
                   vrp_ordinal=ordinal(v.get("vrp_percentile")))
    return out


def _index(symbol: str, d: dict) -> dict:
    m, r = d["market"], d.get("regime") or {}
    regime = r.get("regime")
    imp = d.get("implied") or {}
    return {
        "symbol": symbol, "name": d["name"], "market": m,
        "regime": {"regime": regime, "label": REGIME_LABEL.get(regime, regime), "streak": r.get("streak"),
                   "stickiness": r.get("stickiness"), "as_of": r.get("as_of")} if regime else None,
        "state": _state(symbol),
        "vol": _vol(d.get("vol") or {}),
        "implied": imp if imp.get("move_points") else None,
        "history": _history(symbol, m.get("as_of")),
    }


def _events(symbols: dict) -> list[dict]:
    rows = []
    for d in symbols.values():
        for e in d.get("events", []):
            unusual = e.get("p_value") is not None and e["p_value"] < UNUSUAL_P and (e.get("n") or 0) >= UNUSUAL_MIN_N
            rows.append({"id": e["id"], "name": e["name"], "index": d["name"], "date": e["date"], "n": e.get("n"),
                         "mean": e.get("mean"), "baseline_mean": e.get("baseline_mean"), "hit_rate": e.get("hit_rate"),
                         "ci_low": e.get("ci_low"), "ci_high": e.get("ci_high"), "p_value": e.get("p_value"),
                         "unusual": unusual})
    return sorted(rows, key=lambda x: x["date"], reverse=True)


def _readings(symbols: dict) -> list[str]:
    v = (symbols.get("NIFTY_50") or {}).get("vol") or {}
    lines = []
    if v.get("vix") is not None:
        lines.append(
            f"India VIX is **{v['vix'] * 100:.1f}**, {level_word(v['vix_percentile_1y'])} for the past year, while "
            f"NIFTY's realized volatility is **{v['rv21'] * 100:.1f}%**. Option buyers are paying "
            f"**{v['vrp_now'] * 100:+.1f}** volatility points over recent movement, the "
            f"{ordinal(v['vrp_percentile'])} of history. Historically, implied volatility was above what followed on "
            f"{v['vrp_share_positive']:.0%} of days.")
    if v.get("garch_21d"):
        lines.append(f"GARCH puts NIFTY's expected volatility over the next 21 sessions at "
                     f"**{v['garch_21d'] * 100:.1f}%**, against a long-run level of "
                     f"{(v.get('garch_long_run') or 0) * 100:.1f}%.")
    risk = read_json("risk.json")
    if (risk.get("tail") or {}).get("NIFTY_50"):
        f95 = risk["tail"]["NIFTY_50"]["fhs"]["0.95"]
        rn = risk["now"]
        lines.append(f"At today's volatility, NIFTY falls **{f95['var'] * 100:.1f}%** or more on about one day in "
                     f"twenty (₹{f95['var'] * 1e5:,.0f} per ₹1 lakh). Sectors are moving together at a "
                     f"{corr_word(rn['avg_corr_pct'])} level and market turbulence is at the "
                     f"{rn['turbulence_pct'] * 100:.0f}th percentile.")
    return lines


def _risk() -> dict | None:
    """NIFTY tail risk at today's volatility and the 21-session GARCH-t Monte Carlo distribution."""
    risk = read_json("risk.json")
    t = (risk.get("tail") or {}).get("NIFTY_50")
    if not t:
        return None
    f95, mc, now = t["fhs"]["0.95"], t.get("mc") or {}, risk.get("now") or {}
    return {"var95": f95["var"], "cvar95": f95["cvar"], "sigma_next": t.get("sigma_next"),
            "mc_paths": MC_PATHS, "mc_days": MC_DAYS, "mc_p05": mc.get("p05"), "mc_p50": mc.get("p50"),
            "mc_p95": mc.get("p95"), "mc_hist": mc.get("hist") or [], "mc_edges": mc.get("edges") or [],
            "turbulence_pct": now.get("turbulence_pct"), "avg_corr_pct": now.get("avg_corr_pct")}


def build() -> dict:
    """The Today payload. Empty `indices` means the analytics job has not run yet."""
    brief = read_json("today.json")
    symbols = brief.get("symbols", {})
    news = read_json("news_calendar.json")
    as_of = max((d["market"]["as_of"] for d in symbols.values()), default=None)
    return {
        "as_of": as_of,
        "built": brief.get("built"),
        "indices": [_index(sym, d) for sym, d in symbols.items()],
        "readings": _readings(symbols),
        "risk": _risk(),
        "events": _events(symbols),
        "news": {"built_at": news.get("built_at"), "start": news.get("start"), "end": news.get("end"),
                 "events": news.get("events", [])},
        "freshness": freshness(),
    }
