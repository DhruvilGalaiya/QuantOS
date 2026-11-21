"""
Scheduled news events for the coming week (free sources, refreshed with the analytics job).

    Global macro:   Forex Factory weekly calendar JSON (high-impact releases: US CPI, Fed decisions, jobs, …)
    Scheduled:      RBI and US Fed policy decisions (macro_calendar.csv, from the banks' published schedules), Budget
                    (event_calendar.csv), CPI inflation (MoSPI, 12th of each month), GDP (last working day of
                    Feb / May / Aug / Nov)
    Derivatives:    next NIFTY and BANK NIFTY option expiries (from the volatility-surface snapshot)
    Earnings:       NIFTY 50 companies' board meetings for results (NSE event calendar)

Writes analytics/news_calendar.json. Every source is optional: a failed fetch is reported, never fatal.
Forex Factory only publishes the current week, so fetched items are cached and merged across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from quantos_platform.analytics import read_json, write_json
from quantos_platform.core import calendar, settings

HERE = Path(__file__).resolve().parent
WINDOW_DAYS = 7
UA = {"User-Agent": "Mozilla/5.0 (QuantOS research dashboard)", "Accept": "application/json,text/csv,*/*"}
FF_URLS = ["https://nfs.faireconomy.media/ff_calendar_thisweek.json",
           "https://nfs.faireconomy.media/ff_calendar_nextweek.json"]
FF_COUNTRIES = {"USD": "US", "CNY": "China", "EUR": "Euro area", "JPY": "Japan", "GBP": "UK"}
NIFTY50_CSV = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_HOME, NSE_EVENTS = "https://www.nseindia.com/", "https://www.nseindia.com/api/event-calendar"

WHY = [  # (keyword in title, plain-language reason it matters to Indian markets)
    (("fomc", "federal funds", "fed chair", "powell", "fed "), "The US Federal Reserve sets the world's most "
     "important interest rate. Surprises move global stocks, the dollar and foreign flows into India."),
    (("cpi", "inflation", "pce"), "Inflation decides where interest rates go next. A hot number usually hurts "
     "stocks and bonds; a soft one helps."),
    (("non-farm", "employment", "unemployment", "jobless"), "US jobs data shapes Fed expectations. Big "
     "surprises move global markets, and Indian markets open with that reaction."),
    (("gdp",), "Economic growth figures: a big miss or beat can shift views on earnings and interest rates."),
    (("retail sales",), "Consumer spending is the engine of the US economy; surprises move global risk appetite."),
    (("pmi",), "Business surveys are an early read on growth; large surprises can move markets."),
    (("rate", "monetary policy", "boj", "ecb", "boe"), "A central-bank decision: surprises move currencies and "
     "global stocks, with knock-on effects for Indian markets."),
]


def _why(title: str, country: str) -> str:
    t = title.lower()
    for keys, text in WHY:
        if any(k in t for k in keys):
            return text
    return f"A major economic release in {country}; global markets can react overnight."


def _item(day, time_ist, title, country, impact, why, kind, source) -> dict:
    return {"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "time": time_ist, "title": title, "country": country,
            "impact": impact, "why": why, "kind": kind, "source": source}


# ── sources ────────────────────────────────────────────────────────────────────
def _forex_factory(status: dict) -> list[dict]:
    cache_path = settings.ANALYTICS_DIR / "news_cache_ff.json"
    cache = {r["key"]: r for r in (read_json(cache_path.name).get("items", []) if cache_path.exists() else [])}
    for url in FF_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code != 200:
                status.setdefault("forex_factory", f"HTTP {r.status_code}")
                continue
            for e in r.json():
                if e.get("impact") != "High" or e.get("country") not in FF_COUNTRIES:
                    continue
                ts = pd.Timestamp(e["date"]).tz_convert(settings.IST)
                key = f"{e['country']}|{e['title']}|{ts:%Y-%m-%d}"
                cache[key] = {"key": key, **_item(ts, ts.strftime("%H:%M"), e["title"], FF_COUNTRIES[e["country"]],
                                                  "High", _why(e["title"], FF_COUNTRIES[e["country"]]),
                                                  "Global macro", "Forex Factory")}
            status["forex_factory"] = "ok"
        except (requests.RequestException, ValueError, KeyError) as exc:
            status["forex_factory"] = f"failed: {type(exc).__name__}"
    keep = [v for v in cache.values() if pd.Timestamp(v["date"]) >= calendar.today_ist() - pd.Timedelta(days=30)]
    write_json("news_cache_ff.json", {"items": keep})
    return [{k: v for k, v in r.items() if k != "key"} for r in keep]


def _india_macro(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    out = []
    m = pd.read_csv(HERE / "macro_calendar.csv", parse_dates=["date"])
    for r in m.itertuples():
        india = r.country == "INR"
        out.append(_item(r.date, "10:00" if india else "00:30", r.event, "India" if india else "US", r.impact,
                         r.why, "India macro" if india else "Global macro",
                         "RBI schedule" if india else "Federal Reserve schedule"))
    b = pd.read_csv(HERE / "event_calendar.csv", parse_dates=["date"])
    for r in b[b["event"].str.contains("budget", case=False)].itertuples():
        out.append(_item(r.date, "11:00", "Union Budget", "India", "High", "The government's yearly tax and "
                         "spending plan. Index swings on Budget day are often double a normal day.",
                         "India macro", "Budget calendar"))
    for month in pd.date_range(start - pd.offsets.MonthBegin(1), end + pd.offsets.MonthEnd(1), freq="MS"):
        out.append(_item(month + pd.Timedelta(days=11), "16:00", "India CPI inflation", "India", "High",
                         "Retail inflation guides the RBI's next rate move. Released after market hours, so the "
                         "reaction shows up the next session.", "India macro", "MoSPI (12th of each month)"))
        if month.month in (2, 5, 8, 11):
            last = month + pd.offsets.MonthEnd(0)
            while not calendar.is_session(last):
                last -= pd.Timedelta(days=1)
            out.append(_item(last, "16:00", "India GDP growth (quarterly)", "India", "Medium",
                             "Quarterly growth figure; a big surprise can shift views on earnings and rates.",
                             "India macro", "MoSPI (last working day)"))
    return out


def _expiries() -> list[dict]:
    out = []
    for sym, name, why in [("nifty_50", "NIFTY", "Weekly NIFTY options expire. Time decay is fastest and option "
                            "positioning can pin or whip the index near big strikes."),
                           ("nifty_bank", "BANK NIFTY", "Monthly BANK NIFTY options and futures expire; expect "
                            "heavier volumes and position rollovers.")]:
        try:
            s = json.loads((settings.SURFACE_DIR / f"{sym}_surface_v2.json").read_text())
        except (OSError, ValueError):
            continue
        for e in (s.get("expiries") or [])[:2]:
            out.append(_item(e["expiry"], "15:30", f"{name} option expiry", "India", "Medium", why,
                             "Derivatives", "Option chain"))
    return out


def _earnings(start: pd.Timestamp, end: pd.Timestamp, status: dict) -> list[dict]:
    try:
        import io

        names = pd.read_csv(io.StringIO(requests.get(NIFTY50_CSV, headers=UA, timeout=20).text))
        names = dict(zip(names["Symbol"], names["Company Name"]))
        s = requests.Session()
        s.get(NSE_HOME, headers=UA, timeout=20)
        rows = s.get(NSE_EVENTS, headers={**UA, "Referer": NSE_HOME}, timeout=20).json()
    except Exception as exc:   # NSE blocks some networks; earnings are optional
        status["nse_results"] = f"failed: {type(exc).__name__}"
        return []
    status["nse_results"] = "ok"
    out = []
    for r in rows:
        if r.get("symbol") not in names or "result" not in str(r.get("purpose", "")).lower():
            continue
        day = pd.to_datetime(r["date"], format="%d-%b-%Y", errors="coerce")
        if pd.isna(day) or not (start <= day <= end):
            continue
        out.append(_item(day, "", f"{names[r['symbol']]} results", "India", "Medium",
                         "A NIFTY 50 company reports earnings; big index weights can move the index by "
                         "themselves.", "Earnings", "NSE event calendar"))
    return out


def build() -> dict:
    start = calendar.today_ist().normalize()
    end = start + pd.Timedelta(days=WINDOW_DAYS)
    status: dict = {}
    items = _forex_factory(status) + _india_macro(start, end) + _expiries() + _earnings(start, end, status)
    items = [i for i in items if start <= pd.Timestamp(i["date"]) <= end]
    items = sorted({(i["date"], i["title"]): i for i in items}.values(), key=lambda i: (i["date"], i["time"] or "99"))
    out = {"built_at": pd.Timestamp.now(tz=settings.IST).isoformat(), "start": f"{start:%Y-%m-%d}",
           "end": f"{end:%Y-%m-%d}", "sources": status, "events": items}
    write_json("news_calendar.json", out)
    return out


if __name__ == "__main__":
    o = build()
    print(o["sources"])
    for e in o["events"]:
        print(e["date"], e["time"], e["impact"], e["title"])
