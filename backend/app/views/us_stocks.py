"""
QuantOS | US Stocks page

Searchable US equity price history (Yahoo Finance daily closes, split- and
dividend-adjusted). Independent of the Indian index regime models.
Each selected ticker is topped up to its latest completed session on view.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme  # noqa: E402  (backend/app/theme.py)
import ui  # noqa: E402  (backend/app/ui.py)
from quantos_platform.data.us_equities import load_us_asset, update_us_asset  # noqa: E402
from quantos_platform.data.us_fundamentals import (  # noqa: E402
    fetch_financials,
    fetch_fundamentals,
    fetch_price_history,
    fetch_reported_eps,
    pe_history,
)
from quantos_platform.data.us_universe import UNIVERSE_FILE, load_universe  # noqa: E402

# Fallback list if universe.json has not been built yet.
FALLBACK = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta Platforms", "AVGO": "Broadcom", "TSLA": "Tesla",
    "NFLX": "Netflix", "PLTR": "Palantir", "AMD": "Advanced Micro Devices",
    "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe", "INTC": "Intel",
    "QCOM": "Qualcomm", "JPM": "JPMorgan Chase", "V": "Visa", "MA": "Mastercard",
    "BRK-B": "Berkshire Hathaway", "LLY": "Eli Lilly", "UNH": "UnitedHealth",
    "JNJ": "Johnson & Johnson", "COST": "Costco", "WMT": "Walmart", "XOM": "Exxon Mobil",
    "KO": "Coca-Cola", "DIS": "Disney", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
}
UPDATE_TTL_SECONDS = 6 * 60 * 60
FUNDAMENTALS_TTL_SECONDS = 15 * 60
ABOUT_PREVIEW_CHARS = 300
CHART_RANGES = {"1M": 1, "6M": 6, "1Yr": 12, "3Yr": 36, "5Yr": 60, "10Yr": 120, "Max": None}
DEFAULT_RANGE = "5Yr"
ACCENT = theme.ACCENT
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}

PANEL_CSS = """
<style>
.qs-head { display: flex; align-items: center; flex-wrap: wrap; gap: 8px 16px; }
.qs-logo { position: relative; width: 48px; height: 48px; flex: none; overflow: hidden;
  display: flex; align-items: center; justify-content: center; border-radius: 10px;
  background: var(--q-text); color: #fff; font-size: 22px; font-weight: 700; }
.qs-logo img { position: absolute; inset: 0; width: 100%; height: 100%; padding: 6px;
  box-sizing: border-box; object-fit: contain; background: #fff; }
.qs-logo img[hidden] { display: none; }
.qs-head { animation: q-rise .6s var(--q-ease) both; }
.qs-name { font-family: var(--q-serif); font-size: clamp(2.2rem, 1.5rem + 2vw, 3.1rem); font-weight: 400; line-height: 1.05; letter-spacing: -0.015em; }
.qs-sep { opacity: .35; }
.qs-price { font-size: 1.25rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.qs-change { margin-left: 6px; font-size: .9rem; font-weight: 600; }
.qs-up { color: var(--q-bull); }
.qs-down { color: var(--q-bear); }
.qs-muted { opacity: .62; font-weight: 400; }
.qs-time { font-size: .8rem; opacity: .62; }
.qs-links { display: flex; flex-wrap: wrap; gap: 6px 26px; margin: 12px 0 10px; font-size: .92rem; }
.qs-links a { color: var(--q-muted); text-decoration: none; background: linear-gradient(currentColor, currentColor) 0 100% / 0 1px no-repeat;
  transition: color .16s ease, background-size .25s var(--q-ease); }
.qs-links a:hover { color: var(--q-text); background-size: 100% 1px; }
.qs-links a.qs-site { color: var(--q-accent); font-weight: 600; }
.qs-ratios { container-type: inline-size; }
.qs-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px 20px; }
.qs-cell { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  padding: 11px 12px; border-radius: 8px; }
.qs-cell:nth-child(6n+4), .qs-cell:nth-child(6n+5), .qs-cell:nth-child(6n+6) {
  background: color-mix(in srgb, currentColor 4.5%, transparent); }
.qs-label { color: var(--q-muted); font-size: .9rem; }
.qs-value { font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }
@container (max-width: 760px) {
  .qs-grid { gap: 6px 10px; }
  .qs-cell { flex-direction: column; justify-content: flex-start; gap: 3px; padding: 9px 10px; }
  .qs-label { font-size: .85rem; }
  .qs-value { text-align: left; }
}
@container (max-width: 420px) {
  .qs-grid { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .qs-cell { flex-direction: row; justify-content: space-between; }
  .qs-value { text-align: right; }
  .qs-cell:nth-child(n) { background: none; }
  .qs-cell:nth-child(even) { background: color-mix(in srgb, currentColor 4.5%, transparent); }
}
.qs-return { display: flex; align-items: baseline; gap: 10px; padding-top: 6px; font-weight: 600; }
.qs-return small { font-size: .78rem; color: var(--q-muted); font-weight: 500; }
.qs-title { font-family: var(--q-serif); font-size: 1.6rem; font-weight: 400; letter-spacing: -0.01em; }
.qs-sub { margin: 2px 0 12px; font-size: .9rem; opacity: .62; }
.qs-scroll { overflow-x: auto; }
.qs-table { width: 100%; border-collapse: collapse; font-size: .92rem; font-variant-numeric: tabular-nums; }
.qs-table th, .qs-table td { padding: 8px 12px; text-align: right; white-space: nowrap; border: 0; }
.qs-table th { font-weight: 500; color: var(--q-muted); font-size: .8rem; border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
.qs-table th:first-child, .qs-table td:first-child { text-align: left; }
.qs-table tbody tr:nth-child(odd) { background: color-mix(in srgb, currentColor 3.5%, transparent); }
.qs-table tbody tr { transition: background-color .15s ease; }
.qs-table tbody tr:hover { background: var(--q-raised); }
.qs-table tr.qs-strong td { font-weight: 600; border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
.qs-side h4 { margin: 0 0 8px; padding: 0; font-family: var(--q-serif); font-size: 1.35rem; font-weight: 400; letter-spacing: -0.005em; }
.qs-side h4:not(:first-child) { margin-top: 22px; }
.qs-side p { margin: 0 0 8px; line-height: 1.6; color: var(--q-muted); max-width: 62ch; }
.qs-side b { color: var(--q-text); }
.qs-side b { font-weight: 600; }
.qs-side summary { cursor: pointer; color: var(--q-accent); font-size: .85rem; font-weight: 600;
  list-style: none; transition: color .16s ease; }
.qs-side summary::-webkit-details-marker { display: none; }
.qs-side summary:hover { color: var(--q-text); }
.qs-side details[open] > :not(summary) { animation: q-rise .35s var(--q-ease) both; }
.qs-side details[open] summary { margin-top: 4px; }
</style>
"""


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def get_universe() -> pd.DataFrame:
    uni = load_universe()
    if uni.empty:
        uni = pd.DataFrame({"ticker": list(FALLBACK), "name": list(FALLBACK.values())})
        uni["market_cap"], uni["sp500_rank"], uni["nasdaq_rank"], uni["sector"] = np.nan, np.nan, np.nan, "—"
    return uni


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def overview_table(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Snapshot stats from stored files (no network) for the list view."""
    rows = []
    for t in tickers:
        df = load_us_asset(t)
        if len(df) < 2:
            continue
        p = pd.Series(df["close"].astype(float).to_numpy(), index=pd.to_datetime(df["timestamp"])).sort_index()
        r = p.pct_change().dropna()
        ytd_base = p[p.index < pd.Timestamp(p.index[-1].year, 1, 1)]
        rows.append({
            "Ticker": t,
            "Last close": p.iloc[-1],
            "1D": r.iloc[-1],
            "1M": trailing_return(p, 30),
            "YTD": p.iloc[-1] / ytd_base.iloc[-1] - 1 if len(ytd_base) else np.nan,
            "1Y": trailing_return(p, 365),
            "20D vol": r.tail(20).std() * np.sqrt(252) if len(r) >= 20 else np.nan,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def get_prices(symbol: str) -> tuple[pd.Series, str | None]:
    """Latest closes for a ticker; falls back to stored data if Yahoo fails."""
    error = None
    try:
        stored = load_us_asset(symbol)
        # public mode: the scheduled us_equities job keeps stored tickers current;
        # only tickers we don't have yet (custom input) are downloaded on demand
        df = stored if (ui.PUBLIC and len(stored)) else update_us_asset(symbol)
    except Exception as exc:
        df = load_us_asset(symbol)
        error = f"{type(exc).__name__}: {exc}"
    if df.empty:
        return pd.Series(dtype=float), error
    s = pd.Series(df["close"].astype(float).to_numpy(), index=pd.to_datetime(df["timestamp"]))
    return s.sort_index(), error


@st.cache_data(ttl=FUNDAMENTALS_TTL_SECONDS, show_spinner=False)
def get_fundamentals(symbol: str) -> dict:
    """Raises on failure so that an outage is not cached."""
    return fetch_fundamentals(symbol)


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def get_financials(symbol: str) -> dict:
    return fetch_financials(symbol)


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def get_history(symbol: str) -> pd.DataFrame:
    return fetch_price_history(symbol)


@st.cache_data(ttl=UPDATE_TTL_SECONDS, show_spinner=False)
def get_reported_eps(symbol: str) -> pd.Series:
    return fetch_reported_eps(symbol)


def trailing_return(prices: pd.Series, days: int) -> float:
    cutoff = prices.index[-1] - pd.Timedelta(days=days)
    prior = prices[prices.index <= cutoff]
    return float(prices.iloc[-1] / prior.iloc[-1] - 1) if len(prior) else np.nan


# ============================================================
# COMPANY PANEL
# ============================================================

MISSING = "—"


def present(value) -> bool:
    return value is not None and bool(np.isfinite(value))


def fmt_money(value, symbol: str) -> str:
    return f"{symbol}{value:,.2f}" if present(value) else MISSING


def fmt_cap(value, symbol: str) -> str:
    if not present(value):
        return MISSING
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"{symbol}{value / size:,.2f}{suffix}"
    return f"{symbol}{value:,.0f}"


def fmt_pct(value, signed: bool = False) -> str:
    if not present(value):
        return MISSING
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def fmt_num(value) -> str:
    return f"{value:,.2f}" if present(value) else MISSING


def quote_time_label(epoch) -> str:
    if not present(epoch):
        return ""
    stamp = pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert("America/New_York")
    hour = stamp.hour % 12 or 12
    meridiem = "a.m." if stamp.hour < 12 else "p.m."
    return f"{stamp.day} {stamp:%b} {hour}:{stamp:%M} {meridiem} ET"


def safe_website(url) -> tuple[str, str] | None:
    """(href, bare domain) for an http(s) URL, otherwise None."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    return url, parsed.hostname.removeprefix("www.")


def exchange_label(name) -> str:
    if not name:
        return "Quote"
    return "NASDAQ" if name.lower().startswith("nasdaq") else name


def link(href: str, text: str, css: str = "") -> str:
    return (f'<a class="{css}" href="{html.escape(href, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{html.escape(text)} ↗</a>')


def render_header(symbol: str, name: str, data: dict, price: float, change: float) -> str:
    currency = data.get("currency") or "USD"
    sign = "" if data.get("quote_type") == "INDEX" else CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    site = safe_website(data.get("website"))
    logo = html.escape(name[:1].upper())
    if site:
        favicon = f"https://www.google.com/s2/favicons?domain={quote(site[1])}&sz=128"
        logo += f'<img src="{html.escape(favicon, quote=True)}" alt="">'

    change_html = ""
    if present(change):
        css, arrow = ("qs-up", "▲") if change >= 0 else ("qs-down", "▼")
        change_html = f'<span class="qs-change {css}">{arrow} {change:+.2%}</span>'

    ticker = quote(symbol, safe="")
    yahoo = f"https://finance.yahoo.com/quote/{ticker}"
    links = [link(site[0], site[1], "qs-site")] if site else []
    links.append(link(f"{yahoo}/", f"{exchange_label(data.get('exchange'))}: {symbol}"))
    if data.get("quote_type") in ("EQUITY", "ETF"):
        links.append(link(f"{yahoo}/options/", "Options"))
    if data.get("quote_type") == "EQUITY":
        links.append(link(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
            "SEC Filings",
        ))

    return f"""
    <div class="qs-head">
      <div class="qs-logo">{logo}</div>
      <div class="qs-name">{html.escape(name)}</div>
      <div class="qs-sep">•</div>
      <div>
        <div class="qs-price">{fmt_money(price, sign)}{change_html}</div>
        <div class="qs-time">{quote_time_label(data.get("quote_time"))}</div>
      </div>
    </div>
    <div class="qs-links">{"".join(links)}</div>
    """


def build_ratios(data: dict, prices: pd.Series, price: float, info: dict) -> tuple[dict, dict]:
    """(always-shown ratios, optional ratios), as display strings."""
    currency = data.get("currency") or "USD"
    sign = "" if data.get("quote_type") == "INDEX" else CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    year = prices[prices.index > prices.index[-1] - pd.Timedelta(days=365)]
    high = data.get("high_52w") or float(year.max())
    low = data.get("low_52w") or float(year.min())
    is_etf = data.get("quote_type") == "ETF"
    size = data.get("net_assets") if is_etf else data.get("market_cap") or info.get("market_cap")

    returns = prices.pct_change().dropna()
    vol20 = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else np.nan
    ytd_base = prices[prices.index < pd.Timestamp(prices.index[-1].year, 1, 1)]
    ytd = prices.iloc[-1] / ytd_base.iloc[-1] - 1 if len(ytd_base) else np.nan

    core = {
        "Net Assets" if is_etf else "Market Cap": fmt_cap(size, sign),
        "Current Price": fmt_money(price, sign),
        "High / Low": f"{fmt_money(high, sign)} / {fmt_num(low)}",
        "Stock P/E": fmt_num(data.get("pe")),
        "Book Value": fmt_money(data.get("book_value"), sign),
        "Dividend Yield": fmt_pct(data.get("dividend_yield")),
        "ROCE": fmt_pct(data.get("roce")),
        "ROE": fmt_pct(data.get("roe")),
        "EPS (TTM)": fmt_money(data.get("eps"), sign),
    }
    extra = {
        "Forward P/E": fmt_num(data.get("forward_pe")),
        "Price to Book": fmt_num(data.get("price_to_book")),
        "ROA": fmt_pct(data.get("roa")),
        "Net Margin": fmt_pct(data.get("net_margin")),
        "Operating Margin": fmt_pct(data.get("operating_margin")),
        "Revenue Growth (YoY)": fmt_pct(data.get("revenue_growth"), signed=True),
        "Earnings Growth (YoY)": fmt_pct(data.get("earnings_growth"), signed=True),
        "Debt to Equity": fmt_num(data.get("debt_to_equity")),
        "Current Ratio": fmt_num(data.get("current_ratio")),
        "Beta": fmt_num(data.get("beta")),
        "1M Return": fmt_pct(trailing_return(prices, 30), signed=True),
        "YTD Return": fmt_pct(ytd, signed=True),
        "1Y Return": fmt_pct(trailing_return(prices, 365), signed=True),
        "20D Annualised Vol": fmt_pct(vol20),
        "From 52W High": fmt_pct(price / high - 1, signed=True),
    }
    return core, extra


def render_ratio_grid(ratios: dict) -> str:
    cells = "".join(
        f'<div class="qs-cell"><span class="qs-label">{html.escape(label)}</span>'
        f'<span class="qs-value">{html.escape(value)}</span></div>'
        for label, value in ratios.items()
    )
    return f'<div class="qs-ratios"><div class="qs-grid">{cells}</div></div>'


def fmt_cell(value, kind: str) -> str:
    if value is None:
        return MISSING
    if kind == "money":
        return f"{value / 1e6:,.0f}"
    if kind == "pct":
        return f"{value:.1%}"
    return f"{value:,.2f}"  # eps, times


def render_table(table: dict) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["", *table["columns"]])
    body = []
    for row in table["rows"]:
        cells = "".join(f"<td>{html.escape(fmt_cell(v, row['kind']))}</td>" for v in row["values"])
        css = ' class="qs-strong"' if row["bold"] else ""
        body.append(f"<tr{css}><td>{html.escape(row['label'])}</td>{cells}</tr>")
    return (f'<div class="qs-scroll"><table class="qs-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def statement_section(title: str, subtitle: str, table: dict) -> None:
    if not table["rows"]:
        return
    with theme.card("stmt_" + re.sub(r"\W+", "_", title).lower()):
        st.html(f'<div class="qs-title">{html.escape(title)}</div>'
                f'<div class="qs-sub">{html.escape(subtitle)}</div>{render_table(table)}')


def window(series: pd.Series, months: int | None) -> pd.Series:
    if months is None or series.empty:
        return series
    return series[series.index >= series.index[-1] - pd.DateOffset(months=months)]


def render_return(close: pd.Series) -> str:
    if len(close) < 2:
        return ""
    change = close.iloc[-1] / close.iloc[0] - 1
    css, arrow = ("qs-up", "▲") if change >= 0 else ("qs-down", "▼")
    parts = f'<span class="{css}">{arrow} {change:.1%}</span>'
    years = (close.index[-1] - close.index[0]).days / 365.25
    if years >= 0.98:  # a "1Yr" window can fall a day or two short
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
        parts += f'<span class="qs-muted">|</span><span>{cagr:.1%} <small>CAGR</small></span>'
    return f'<div class="qs-return">{parts}</div>'


def volume_bars(volume: pd.Series) -> tuple[pd.Series, str]:
    """Aggregate long windows so the bars stay readable."""
    if len(volume) > 2600:
        return volume.resample("ME").sum(), "Volume (monthly)"
    if len(volume) > 520:
        return volume.resample("W").sum(), "Volume (weekly)"
    return volume, "Volume"


def base_layout(fig: go.Figure, y_title: str) -> None:
    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=8, b=8),
        hovermode="x unified",
        showlegend=False,
        bargap=0.1,
        xaxis=dict(type="date", showgrid=False),
        yaxis=dict(title=dict(text=y_title), side="right"),
    )


def price_figure(history: pd.DataFrame, months: int | None, show: dict) -> go.Figure:
    frame = history.assign(
        dma50=history["close"].rolling(50).mean(),
        dma200=history["close"].rolling(200).mean(),
    )
    frame = frame.loc[window(frame["close"], months).index]
    fig = go.Figure()
    if show["volume"] and frame["volume"].fillna(0).gt(0).any():
        bars, label = volume_bars(frame["volume"])
        fig.add_bar(x=bars.index, y=bars.values, name=label, yaxis="y2",
                    marker=dict(color=theme.alpha(theme.FAINT, 0.28), line=dict(width=0)))
        # Headroom keeps the bars in the lower part of the chart, under the price line.
        fig.update_layout(yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False,
                                      range=[0, float(bars.max()) * 3.2], tickformat=".2s"))
    if show["price"]:
        fig.add_scatter(x=frame.index, y=frame["close"], mode="lines", name="Price",
                        line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(51,70,211,0.07)")
        lo, hi = float(frame["close"].min()), float(frame["close"].max())
        pad = (hi - lo) * 0.08 or hi * 0.02
        fig.update_layout(yaxis=dict(range=[lo - pad, hi + pad]))
    if show["dma50"]:
        fig.add_scatter(x=frame.index, y=frame["dma50"], mode="lines", name="50 DMA",
                        line=dict(color=theme.SIDE, width=1.4))
    if show["dma200"]:
        fig.add_scatter(x=frame.index, y=frame["dma200"], mode="lines", name="200 DMA",
                        line=dict(color=theme.SERIES[4], width=1.4))
    base_layout(fig, "Price")
    return fig


def pe_figure(pe: pd.Series, months: int | None) -> go.Figure:
    shown = window(pe, months)
    fig = go.Figure()
    fig.add_scatter(x=shown.index, y=shown.values, mode="lines", name="P/E",
                    line=dict(color=ACCENT, width=2))
    fig.add_hline(y=float(shown.median()), line=dict(color=theme.SIDE, width=1.2, dash="dash"),
                  annotation_text=f"Median {shown.median():.1f}", annotation_position="top left",
                  annotation_font_color=theme.SIDE)
    base_layout(fig, "P/E")
    return fig


def key_points(data: dict, info: dict, price: float) -> list[tuple[str, str]]:
    points = []

    position = []
    if pd.notna(info.get("sp500_rank")):
        position.append(f"#{int(info['sp500_rank'])} in the S&P 500")
    if pd.notna(info.get("nasdaq_rank")):
        position.append(f"#{int(info['nasdaq_rank'])} in the Nasdaq-100")
    industry = " · ".join(x for x in (data.get("sector"), data.get("industry")) if x)
    text = f"Ranked {' and '.join(position)} by market cap." if position else ""
    if industry:
        text = f"{text} Operates in {industry}.".strip()
    if text:
        points.append(("Market Position", text))

    place = ", ".join(x for x in (data.get("city"), data.get("state"), data.get("country")) if x)
    scale = []
    if present(data.get("employees")):
        scale.append(f"{data['employees']:,.0f} full-time employees")
    if place:
        scale.append(f"headquartered in {place}")
    if scale:
        sentence = "; ".join(scale)
        points.append(("Scale", f"{sentence[0].upper()}{sentence[1:]}."))

    profit = []
    if present(data.get("net_margin")):
        profit.append(f"net margin of {data['net_margin']:.1%}")
    if present(data.get("revenue_growth")):
        profit.append(f"revenue growth of {data['revenue_growth']:+.1%} year on year")
    if profit:
        sentence = " and ".join(profit)
        points.append(("Profitability", f"{sentence[0].upper()}{sentence[1:]}."))

    high, low = data.get("high_52w"), data.get("low_52w")
    if present(high) and present(low) and high > 0 and low > 0:
        points.append(("Price Range", f"{price / high - 1:+.1%} from the 52-week high and "
                                      f"{price / low - 1:+.1%} from the 52-week low."))
    return points


def split_summary(text: str) -> tuple[str, str]:
    """Break a long description at a sentence end near the preview length."""
    if len(text) <= ABOUT_PREVIEW_CHARS:
        return text, ""
    match = re.search(r"(?<=[a-z0-9)]\.)\s+(?=[A-Z])", text[ABOUT_PREVIEW_CHARS:])
    if not match:
        return text, ""
    cut = ABOUT_PREVIEW_CHARS + match.start()
    return text[:cut], text[cut:].strip()


def read_more(body: str) -> str:
    return f"<details><summary>Read more</summary>{body}</details>" if body else ""


def render_side(data: dict, points: list[tuple[str, str]]) -> str:
    parts = []
    if data.get("summary"):
        head, rest = split_summary(data["summary"])
        parts.append(f"<h4>About</h4><p>{html.escape(head)}</p>")
        parts.append(read_more(f"<p>{html.escape(rest)}</p>" if rest else ""))
    if points:
        rendered = [f"<p><b>{html.escape(t)}:</b> {html.escape(b)}</p>" for t, b in points]
        parts.append(f"<h4>Key points</h4>{rendered[0]}")
        parts.append(read_more("".join(rendered[1:])))
    return f'<div class="qs-side">{"".join(parts)}</div>'


# ============================================================
# PAGE
# ============================================================


universe = get_universe()

with st.sidebar:
    st.subheader("US Stocks")
    lists = {
        "All": universe,
        "Nasdaq-100 · top 50": universe[universe["nasdaq_rank"].notna()].sort_values("nasdaq_rank"),
        "S&P 500 · top 100": universe[universe["sp500_rank"].notna()].sort_values("sp500_rank"),
    }
    list_name = st.radio(
        "List",
        list(lists),
        format_func=lambda k: f"{k} ({len(lists[k])})",
        key="us_list",
    )
    shown = lists[list_name]
    sectors = ["All sectors"] + sorted(shown["sector"].dropna().unique())
    sector = st.selectbox("Sector", sectors, key="us_sector")
    if sector != "All sectors":
        shown = shown[shown["sector"] == sector]
    names = dict(zip(shown["ticker"], shown["name"]))
    choice = st.selectbox(
        "Search or select a stock",
        list(names),
        format_func=lambda t: f"{t} · {names[t]}",
        key="us_stock_choice",
    )
    custom = st.text_input(
        "Or enter any US ticker",
        placeholder="e.g. ASML, SHOP, ^GSPC",
        key="us_stock_custom",
    ).strip().upper()
    if not ui.PUBLIC and st.button("Refresh prices", width="stretch"):
        get_prices.clear()
        get_fundamentals.clear()
        get_financials.clear()
        get_history.clear()
        get_reported_eps.clear()
        st.rerun()

    if UNIVERSE_FILE.exists():
        st.caption(
            "Lists ranked by market cap · built "
            f"{pd.Timestamp(UNIVERSE_FILE.stat().st_mtime, unit='s').date()}"
            + ("." if ui.PUBLIC else ". Rebuild: `python -m quantos_platform.data.us_universe`")
        )

symbol = custom or choice
info = universe.set_index("ticker").to_dict("index").get(symbol, {})
name = info.get("name", symbol)

with st.spinner(f"Loading {symbol}…"):
    prices, error = get_prices(symbol)

if prices.empty:
    st.error(f"No price data found for **{symbol}**. Check the ticker symbol.")
    st.stop()
if error:
    st.warning(f"Could not refresh {symbol} from Yahoo Finance; showing stored data. ({error})")

try:
    fundamentals = get_fundamentals(symbol)
except Exception as exc:
    fundamentals = {}
    st.warning(f"Company details are unavailable for {symbol}; showing price data only. ({exc})")

returns = prices.pct_change().dropna()
price_now = fundamentals.get("price") or float(prices.iloc[-1])
change_pct = fundamentals.get("change_pct")
day_change = change_pct / 100 if present(change_pct) else (float(returns.iloc[-1]) if len(returns) else None)
display_name = fundamentals.get("name") or name
core_ratios, extra_ratios = build_ratios(fundamentals, prices, price_now, info)

st.html(PANEL_CSS)
head_col, export_col = st.columns([6, 1], vertical_alignment="top")
head_col.html(render_header(symbol, display_name, fundamentals, price_now, day_change))
export_col.download_button(
    "Export to CSV",
    pd.DataFrame(
        [("Ticker", symbol), ("Company", display_name), *core_ratios.items(), *extra_ratios.items()],
        columns=["Metric", "Value"],
    ).to_csv(index=False),
    file_name=f"{symbol}_ratios.csv",
    mime="text/csv",
    icon=":material/download:",
    width="stretch",
)

ratio_col, side_col = st.columns([2.1, 1], gap="large")
with theme.card("ratios", ratio_col):
    st.html(render_ratio_grid({**core_ratios, **extra_ratios}))
    if fundamentals.get("quote_type") == "EQUITY" and not present(fundamentals.get("roce")):
        st.caption("ROCE is not shown: Yahoo Finance reports no EBIT for this company, which is typical of banks.")
side_col.html(render_side(fundamentals, key_points(fundamentals, info, price_now)))

try:
    history = get_history(symbol)
except Exception:
    history = pd.DataFrame({"close": prices, "raw_close": prices, "volume": np.nan})

pe_series = pd.Series(dtype=float)
try:
    # Reported EPS is in the filing currency, so it only divides a price in that same currency.
    if fundamentals.get("quote_type") == "EQUITY" and fundamentals.get("financial_currency") == fundamentals.get("currency"):
        pe_series = pe_history(history["raw_close"], get_reported_eps(symbol), fundamentals.get("eps"))
except Exception:
    pass  # the P/E view is simply not offered

with theme.card("price"):
    range_col, return_col, metric_col = st.columns([3.2, 1.6, 1.6], vertical_alignment="center")
    range_name = range_col.segmented_control(
        "Range", list(CHART_RANGES), default=DEFAULT_RANGE, key="us_chart_range",
        label_visibility="collapsed",
    ) or DEFAULT_RANGE
    months = CHART_RANGES[range_name]
    metric = "Price"
    if not pe_series.empty:
        metric = metric_col.segmented_control(
            "Chart", ["Price", "PE Ratio"], default="Price", key="us_chart_metric",
            label_visibility="collapsed",
        ) or "Price"

    if metric == "PE Ratio":
        return_col.html(render_return(window(pe_series, months)))
        theme.chart(pe_figure(pe_series, months))
        st.caption("P/E uses the last four reported quarterly EPS figures from earnings releases.")
    else:
        return_col.html(render_return(window(history["close"], months)))
        chart_slot = st.container()
        toggles = st.container(horizontal=True, horizontal_alignment="center", gap="medium")
        show = {
            "price": toggles.checkbox("Price", value=True, key="us_show_price"),
            "dma50": toggles.checkbox("50 DMA", key="us_show_dma50"),
            "dma200": toggles.checkbox("200 DMA", key="us_show_dma200"),
            "volume": toggles.checkbox("Volume", value=True, key="us_show_volume"),
        }
        with chart_slot:
            theme.chart(price_figure(history, months, show))

if fundamentals.get("quote_type") == "EQUITY":
    try:
        with st.spinner("Loading financial statements…"):
            financials = get_financials(symbol)
    except Exception as exc:
        financials = None
        st.warning(f"Financial statements are unavailable for {symbol}. ({exc})")
    if financials:
        unit = f"Consolidated figures in {fundamentals.get('financial_currency') or 'USD'} millions, except per-share data"
        statement_section("Quarterly Results", unit, financials["quarterly"])
        statement_section("Profit & Loss", unit, financials["annual"])
        statement_section("Balance Sheet", unit, financials["balance"])
        statement_section("Cash Flow", unit, financials["cashflow"])
        statement_section("Ratios", "Calculated from year-end statements", financials["ratios"])


# ============================================================
# LIST OVERVIEW
# ============================================================

theme.section(
    f"{list_name}{'' if sector == 'All sectors' else ', ' + sector}",
    "Daily adjusted closes from Yahoo Finance, updated to the latest completed US session.",
)
table = overview_table(tuple(shown["ticker"]))
if not table.empty:
    meta = shown.set_index("ticker")[["name", "market_cap", "sector"]]
    table = table.join(meta, on="Ticker").rename(
        columns={"name": "Company", "market_cap": "Mkt cap ($B)", "sector": "Sector"})
    table["Mkt cap ($B)"] = table["Mkt cap ($B)"] / 1e9
    for col in ("1D", "1M", "YTD", "1Y", "20D vol"):
        table[col] = table[col] * 100
    table = table[["Ticker", "Company", "Sector", "Mkt cap ($B)", "Last close",
                   "1D", "1M", "YTD", "1Y", "20D vol"]]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(36 * (len(table) + 1), 560),
        column_config={
            "Mkt cap ($B)": st.column_config.NumberColumn(format="%.0f"),
            "Last close": st.column_config.NumberColumn(format="%.2f"),
            "1D": st.column_config.NumberColumn(format="%+.2f%%"),
            "1M": st.column_config.NumberColumn(format="%+.1f%%"),
            "YTD": st.column_config.NumberColumn(format="%+.1f%%"),
            "1Y": st.column_config.NumberColumn(format="%+.1f%%"),
            "20D vol": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption("Click a column header to sort. Figures use stored closes (refreshed when a stock is opened).")
