"""
US Stocks page service: the large-cap universe (S&P 500 top 100 and Nasdaq-100 top 50) at a glance, and one
company in plain language: price history, returns, key ratios each explained in a sentence, the business
summary, key facts and the financial statements. Reads stored closes (us_equities job) and the weekly
fundamentals snapshot (data/us_snapshot.py); never calls Yahoo itself.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quantos_platform.data import us_snapshot
from quantos_platform.data.us_equities import load_us_asset
from quantos_platform.data.us_universe import load_universe

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
SPARK_POINTS = 32


def _prices(ticker: str) -> pd.Series:
    df = load_us_asset(ticker)
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["close"].astype(float).to_numpy(), index=pd.to_datetime(df["timestamp"])).sort_index().dropna()


def _ret(p: pd.Series, days: int) -> float | None:
    prior = p[p.index <= p.index[-1] - pd.Timedelta(days=days)]
    return float(p.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else None


def _ytd(p: pd.Series) -> float | None:
    base = p[p.index < pd.Timestamp(p.index[-1].year, 1, 1)]
    return float(p.iloc[-1] / base.iloc[-1] - 1) if len(base) else None


def _vol20(p: pd.Series) -> float | None:
    r = p.pct_change().dropna()
    return float(r.tail(20).std() * np.sqrt(252)) if len(r) >= 20 else None


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def universe() -> dict:
    """Every stock in the universe with its returns and a 3-month sparkline."""
    uni = load_universe()
    rows, as_of = [], None
    for u in uni.to_dict("records"):
        p = _prices(u["ticker"])
        if len(p) < 2:
            continue
        as_of = max(as_of or p.index[-1], p.index[-1])
        tail = p.iloc[-66:]
        step = max(1, len(tail) // SPARK_POINTS)
        rows.append({"ticker": u["ticker"], "name": u["name"], "sector": u.get("sector") or "Other",
                     "market_cap": _num(u.get("market_cap")), "sp500_rank": _num(u.get("sp500_rank")),
                     "nasdaq_rank": _num(u.get("nasdaq_rank")), "close": float(p.iloc[-1]),
                     "d1": float(p.iloc[-1] / p.iloc[-2] - 1), "m1": _ret(p, 30), "ytd": _ytd(p), "y1": _ret(p, 365),
                     "vol20": _vol20(p), "spark": [round(float(v), 2) for v in tail.iloc[::step]]})
    rows.sort(key=lambda r: -(r["market_cap"] or 0))
    sectors = sorted({r["sector"] for r in rows})
    return {"as_of": as_of.strftime("%Y-%m-%d") if as_of is not None else None, "sectors": sectors, "stocks": rows}


# ------------------------------------------------------------------ one company
def _money(x: float | None, cur: str = "$") -> str:
    if x is None:
        return "—"
    a = abs(x)
    for unit, div in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6)):
        if a >= div:
            return f"{cur}{x / div:,.2f} {unit}" if a / div < 100 else f"{cur}{x / div:,.0f} {unit}"
    return f"{cur}{x:,.2f}"


def _ratios(f: dict, p: pd.Series) -> list[dict]:
    """Each ratio with its value and one plain sentence that uses the number (no jargon left unexplained)."""
    cur = "$" if (f.get("currency") or "USD") == "USD" else f"{f.get('currency')} "
    price = _num(f.get("price")) or float(p.iloc[-1])
    out: list[dict] = []

    def add(key, label, value, kind, plain, group="Valuation"):
        if value is not None:
            out.append({"key": key, "label": label, "value": value, "kind": kind, "plain": plain, "group": group})

    cap = _num(f.get("market_cap"))
    add("market_cap", "Market value", cap, "money", f"What the whole company is worth on the stock market: {_money(cap, cur)}.")
    pe = _num(f.get("pe"))
    if pe and pe > 0:
        add("pe", "Price to earnings", pe, "x", f"Investors pay {cur}{pe:,.0f} for every {cur}1 of profit the company made over the last year.")
    fpe = _num(f.get("forward_pe"))
    if fpe and fpe > 0:
        add("forward_pe", "Forward P/E", fpe, "x", f"On analysts' profit estimates for next year, the price is {fpe:,.1f} times earnings.")
    pb = _num(f.get("price_to_book"))
    if pb and pb > 0:
        add("price_to_book", "Price to book", pb, "x", f"The share price is {pb:,.1f} times what the company's assets minus its debts are worth per share.")
    dy = _num(f.get("dividend_yield"))
    if dy is not None:
        add("dividend_yield", "Dividend yield", dy, "pct",
            f"Pays about {cur}{dy * 100:,.2f} a year in dividends for every {cur}100 of shares." if dy > 0 else "Pays no regular dividend.",
            "Income")
    nm = _num(f.get("net_margin"))
    add("net_margin", "Net margin", nm, "pct", f"Keeps {cur}{nm * 100:,.0f} as profit from every {cur}100 of sales." if nm is not None else "", "Profitability")
    om = _num(f.get("operating_margin"))
    add("operating_margin", "Operating margin", om, "pct",
        f"Earns {cur}{om * 100:,.0f} from its main business for every {cur}100 of sales, before interest and tax." if om is not None else "", "Profitability")
    roe = _num(f.get("roe"))
    add("roe", "Return on equity", roe, "pct",
        f"Makes {cur}{roe * 100:,.0f} of profit a year for every {cur}100 that shareholders have invested." if roe is not None else "", "Profitability")
    roce = _num(f.get("roce"))
    add("roce", "Return on capital", roce, "pct",
        f"Earns {cur}{roce * 100:,.0f} of operating profit a year on every {cur}100 of capital it uses, debt included." if roce is not None else "", "Profitability")
    rg = _num(f.get("revenue_growth"))
    if rg is not None:
        add("revenue_growth", "Sales growth", rg, "pct_signed",
            f"Sales were {abs(rg) * 100:,.0f}% {'higher' if rg >= 0 else 'lower'} than in the same quarter a year earlier.", "Growth")
    eg = _num(f.get("earnings_growth"))
    if eg is not None:
        add("earnings_growth", "Profit growth", eg, "pct_signed",
            f"Profit was {abs(eg) * 100:,.0f}% {'higher' if eg >= 0 else 'lower'} than in the same quarter a year earlier.", "Growth")
    de = _num(f.get("debt_to_equity"))
    if de is not None:
        add("debt_to_equity", "Debt to equity", de, "x", f"Owes {cur}{de:,.2f} of debt for every {cur}1 of shareholders' money.", "Safety")
    cr = _num(f.get("current_ratio"))
    if cr is not None:
        add("current_ratio", "Current ratio", cr, "x",
            f"Has {cur}{cr:,.2f} of cash and short-term assets for every {cur}1 of bills due within a year.", "Safety")
    beta = _num(f.get("beta"))
    if beta is not None:
        add("beta", "Beta", beta, "x",
            f"Has tended to move {beta:,.1f} times as much as the overall US market." if beta > 0 else "Has tended to move against the market.",
            "Risk")
    v20 = _vol20(p)
    if v20 is not None:
        add("vol20", "Recent volatility", v20, "pct",
            f"Over the last 20 trading days the price swung at a pace of {v20 * 100:,.0f}% a year.", "Risk")
    hi, lo = _num(f.get("high_52w")), _num(f.get("low_52w"))
    if hi and lo and hi > lo:
        pos = (price - lo) / (hi - lo)
        add("range52", "52-week range", pos, "pct",
            f"Trades {abs(price / hi - 1) * 100:,.0f}% below its 52-week high of {cur}{hi:,.2f} and "
            f"{abs(price / lo - 1) * 100:,.0f}% above its low of {cur}{lo:,.2f}.", "Risk")
    return out


def _key_facts(f: dict, info: dict) -> list[dict]:
    facts = []
    ranks = []
    if info.get("sp500_rank"):
        ranks.append(f"number {int(info['sp500_rank'])} in the S&P 500")
    if info.get("nasdaq_rank"):
        ranks.append(f"number {int(info['nasdaq_rank'])} in the Nasdaq-100")
    if ranks:
        facts.append({"title": "Size", "text": f"Ranked {' and '.join(ranks)} by market value."})
    ind = " / ".join(x for x in (f.get("sector"), f.get("industry")) if x)
    if ind:
        facts.append({"title": "Business", "text": f"{ind}."})
    place = ", ".join(x for x in (f.get("city"), f.get("state"), f.get("country")) if x)
    emp = _num(f.get("employees"))
    if emp or place:
        bits = ([f"{emp:,.0f} full-time staff"] if emp else []) + ([f"headquartered in {place}"] if place else [])
        s = ", ".join(bits)
        facts.append({"title": "People", "text": s[0].upper() + s[1:] + "."})
    return facts


def _summary(text: str | None) -> dict:
    """The business description as a two-sentence lead plus the rest, for a 'read more' toggle."""
    if not text:
        return {"lead": None, "rest": None}
    # split after a period that ends a word or a 3+ letter acronym ("NASA."), not after "U.S." or "Inc."
    parts = re.split(r"(?:(?<=[a-z0-9)])|(?<=[A-Z]{3}))\.\s+(?=[A-Z])", text.strip())
    sentences = [p + "." for p in parts[:-1]] + parts[-1:]
    return {"lead": " ".join(sentences[:2]), "rest": " ".join(sentences[2:]) or None}


def _history(p: pd.Series) -> list[dict]:
    recent = p.iloc[-504:]
    older = p.iloc[:-504].iloc[::5]
    s = pd.concat([older, recent])
    return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)} for d, v in s.items()]


def company(ticker: str) -> dict:
    """One company. Raises KeyError when the ticker is not in the stored universe."""
    t = ticker.upper()
    if not TICKER_RE.match(t):
        raise KeyError(ticker)
    uni = load_universe()
    row = uni[uni["ticker"] == t]
    p = _prices(t)
    if row.empty or len(p) < 2:
        raise KeyError(ticker)
    info = {k: _num(v) if k.endswith("rank") or k == "market_cap" else v for k, v in row.iloc[0].to_dict().items()}
    snap = us_snapshot.load(t)
    f = snap.get("fundamentals") or {}
    price = float(p.iloc[-1])
    return {
        "ticker": t, "name": f.get("name") or info.get("name"), "sector": f.get("sector") or info.get("sector"),
        "industry": f.get("industry"), "exchange": f.get("exchange"), "website": f.get("website"),
        "currency": f.get("currency") or "USD", "as_of": p.index[-1].strftime("%Y-%m-%d"),
        "fundamentals_as_of": snap.get("fetched_at"),
        "price": price, "change_pct": float(p.iloc[-1] / p.iloc[-2] - 1),
        "returns": {"m1": _ret(p, 30), "m6": _ret(p, 182), "ytd": _ytd(p), "y1": _ret(p, 365), "y5": _ret(p, 5 * 365)},
        "ratios": _ratios(f, p) if f else [],
        "facts": _key_facts(f, info) if f else [],
        "summary": _summary(f.get("summary")),
        "financials": snap.get("financials") or {},
        "history": _history(p),
    }
