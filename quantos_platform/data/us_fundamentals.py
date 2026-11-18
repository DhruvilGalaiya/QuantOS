"""
QuantOS | Company profile and fundamentals for the US Stocks page (Yahoo Finance).

Yahoo's per-share book value is unreliable for multi-class shares (BRK-B) and
for ADRs reporting in another currency (ASML, TSM, PDD), so it is cross-checked
against the balance sheet. ROCE is not published and is computed here.
"""

from __future__ import annotations

import math

import pandas as pd
import yfinance as yf

BOOK_VALUE_TOLERANCE = 0.25
MAX_TTM_SPAN_DAYS = 290  # four quarter-ends; rejects half-yearly reporters


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row(frame: pd.DataFrame, name: str) -> pd.Series:
    """One statement line, newest period first."""
    if frame is None or frame.empty or name not in frame.index:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame.loc[name], errors="coerce").sort_index(ascending=False)


def _latest(frame: pd.DataFrame, name: str) -> float | None:
    values = _row(frame, name).dropna()
    return float(values.iloc[0]) if len(values) else None


def ttm_ebit(quarterly: pd.DataFrame, annual: pd.DataFrame) -> float | None:
    """Sum of the last four quarters, else the latest annual figure."""
    recent = _row(quarterly, "EBIT").iloc[:4]
    if len(recent) == 4 and recent.notna().all():
        if (recent.index[0] - recent.index[3]).days <= MAX_TTM_SPAN_DAYS:
            return float(recent.sum())
    return _latest(annual, "EBIT")


def compute_roce(ebit: float | None, balance: pd.DataFrame) -> float | None:
    """EBIT / capital employed (assets - current liabilities, else equity + debt)."""
    if ebit is None:
        return None
    assets = _latest(balance, "Total Assets")
    current = _latest(balance, "Current Liabilities")
    if assets is not None and current is not None:
        capital = assets - current
    else:
        equity = _latest(balance, "Common Stock Equity")
        debt = _latest(balance, "Total Debt")
        capital = equity + debt if equity is not None and debt is not None else None
    return ebit / capital if capital and capital > 0 else None


def compute_book_value(
    balance: pd.DataFrame,
    market_cap: float | None,
    price: float | None,
    fx_rate: float | None,
    yahoo_value: float | None,
) -> float | None:
    """
    Book value per traded share, in the quote currency.

    Market cap / price gives the share count in units of the traded line, which
    handles share classes and ADR ratios. Yahoo's figure is kept when it agrees.
    """
    equity = _latest(balance, "Common Stock Equity")
    if equity is None or not market_cap or not price or not fx_rate:
        return None
    book = equity * fx_rate / (market_cap / price)
    if yahoo_value and abs(yahoo_value / book - 1) <= BOOK_VALUE_TOLERANCE:
        return yahoo_value
    return book


def _fx_rate(source: str | None, target: str | None) -> float | None:
    if not source or not target or source == target:
        return 1.0
    try:
        return _num(yf.Ticker(f"{source}{target}=X").fast_info["last_price"])
    except Exception:
        return None


def _dividend_yield(info: dict, quote_type: str | None) -> float | None:
    trailing = _num(info.get("trailingAnnualDividendYield"))
    if quote_type == "ETF":  # Yahoo rounds dividendYield to a whole percent for ETFs
        return trailing
    percent = _num(info.get("dividendYield"))  # yfinance reports this one in percent
    return percent / 100 if percent is not None else trailing


def fetch_fundamentals(symbol: str) -> dict:
    """Profile, quote and ratios. Raises if Yahoo has no quote for the symbol."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None:
        raise ValueError(f"No quote returned for {symbol}")

    quote_type = info.get("quoteType")
    currency = info.get("currency")
    market_cap = _num(info.get("marketCap"))
    yahoo_book = _num(info.get("bookValue"))
    book_value = yahoo_book if quote_type == "ETF" else None
    roce = None

    if quote_type == "EQUITY":
        try:
            balance = ticker.quarterly_balance_sheet
            ebit = ttm_ebit(ticker.quarterly_income_stmt, ticker.income_stmt)
            roce = compute_roce(ebit, balance)
            reporting = info.get("financialCurrency") or currency
            book_value = compute_book_value(
                balance,
                market_cap,
                price,
                _fx_rate(reporting, currency),
                yahoo_book if reporting == currency else None,
            )
        except Exception:
            pass  # the quote and Yahoo's ratios are still worth showing

    debt_to_equity = _num(info.get("debtToEquity"))  # Yahoo reports a percentage
    return {
        "name": info.get("longName") or info.get("shortName"),
        "quote_type": quote_type,
        "currency": currency,
        "financial_currency": info.get("financialCurrency"),
        "exchange": info.get("fullExchangeName"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "employees": _num(info.get("fullTimeEmployees")),
        "city": info.get("city"),
        "state": info.get("state"),
        "country": info.get("country"),
        "price": price,
        "change_pct": _num(info.get("regularMarketChangePercent")),
        "quote_time": _num(info.get("regularMarketTime")),
        "market_cap": market_cap,
        "net_assets": _num(info.get("totalAssets")),
        "high_52w": _num(info.get("fiftyTwoWeekHigh")),
        "low_52w": _num(info.get("fiftyTwoWeekLow")),
        "pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "eps": _num(info.get("trailingEps")),
        "book_value": book_value,
        "price_to_book": price / book_value if book_value and book_value > 0 else None,
        "dividend_yield": _dividend_yield(info, quote_type),
        "roce": roce,
        "roe": _num(info.get("returnOnEquity")),
        "roa": _num(info.get("returnOnAssets")),
        "net_margin": _num(info.get("profitMargins")),
        "operating_margin": _num(info.get("operatingMargins")),
        "revenue_growth": _num(info.get("revenueGrowth")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        "debt_to_equity": debt_to_equity / 100 if debt_to_equity is not None else None,
        "current_ratio": _num(info.get("currentRatio")),
        "beta": _num(info.get("beta")),
    }


# ============================================================
# FINANCIAL STATEMENT TABLES
# ============================================================

MIN_REPORT_GAP_DAYS = 45  # Yahoo can list one quarter's report twice
MAX_FOUR_REPORT_SPAN_DAYS = 420
PE_EPS_TOLERANCE = 0.25
TTM_LABEL = "TTM"


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Oldest-first periods, without the near-empty ones Yahoo appends."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    filled = frame.notna().sum()
    frame = frame.loc[:, filled >= 0.5 * filled.max()]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame[sorted(frame.columns)]


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [pd.Timestamp(c).strftime("%b %Y") for c in frame.columns]
    return out


def _line(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.index:
        return frame.loc[name]
    return pd.Series(float("nan"), index=frame.columns, dtype=float)


def _ratio(top: pd.Series, bottom: pd.Series) -> pd.Series:
    """top / bottom, blank where the denominator is not positive."""
    return top / bottom.where(bottom > 0)


def _table(columns, rows: list[tuple]) -> dict:
    """rows: (label, series, kind, bold). Lines with no data are dropped."""
    kept = [
        {
            "label": label,
            "kind": kind,
            "bold": bold,
            "values": [None if pd.isna(v) else float(v) for v in series.reindex(columns)],
        }
        for label, series, kind, bold in rows
        if series.notna().any()
    ]
    return {"columns": list(columns), "rows": kept} if kept else {"columns": [], "rows": []}


def income_table(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return _table([], [])
    line = lambda name: _line(frame, name)  # noqa: E731
    revenue, gross, operating = line("Total Revenue"), line("Gross Profit"), line("Operating Income")
    pretax, net = line("Pretax Income"), line("Net Income")
    is_lender = operating.isna().all() and line("Net Interest Income").notna().any()
    lender_rows = [
        ("Interest Income", line("Interest Income"), "money", False),
        ("Interest Expense", line("Interest Expense"), "money", False),
        ("Net Interest Income", line("Net Interest Income"), "money", True),
        ("SG&A", line("Selling General And Administration"), "money", False),
    ]
    operating_rows = [
        ("Cost of Revenue", line("Cost Of Revenue"), "money", False),
        ("Gross Profit", gross, "money", True),
        ("Gross Margin %", _ratio(gross, revenue), "pct", False),
        ("R&D", line("Research And Development"), "money", False),
        ("SG&A", line("Selling General And Administration"), "money", False),
        ("Operating Expenses", line("Operating Expense"), "money", False),
        ("Operating Profit", operating, "money", True),
        ("Operating Margin %", _ratio(operating, revenue), "pct", False),
        ("Other Income", line("Other Income Expense"), "money", False),
        ("Interest", line("Interest Expense"), "money", False),
        ("Depreciation", line("Reconciled Depreciation"), "money", False),
    ]
    return _table(frame.columns, [
        ("Revenue", revenue, "money", False),
        *(lender_rows if is_lender else operating_rows),
        ("Profit before Tax", pretax, "money", True),
        ("Tax %", _ratio(line("Tax Provision"), pretax), "pct", False),
        ("Net Profit", net, "money", True),
        ("Net Margin %", _ratio(net, revenue), "pct", False),
        ("Diluted EPS", line("Diluted EPS"), "eps", False),
    ])


def balance_table(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return _table([], [])
    spec = [
        ("Shareholders' Equity", "Common Stock Equity", False),
        ("Retained Earnings", "Retained Earnings", False),
        ("Total Debt", "Total Debt", False),
        ("Current Liabilities", "Current Liabilities", False),
        ("Total Liabilities", "Total Liabilities Net Minority Interest", True),
        ("Cash & Short-Term Investments", "Cash Cash Equivalents And Short Term Investments", False),
        ("Receivables", "Receivables", False),
        ("Inventory", "Inventory", False),
        ("Current Assets", "Current Assets", False),
        ("Net Fixed Assets", "Net PPE", False),
        ("Goodwill & Intangibles", "Goodwill And Other Intangible Assets", False),
        ("Investments", "Investments And Advances", False),
        ("Total Assets", "Total Assets", True),
        ("Shares Outstanding (M)", "Ordinary Shares Number", False),
    ]
    return _table(frame.columns, [(label, _line(frame, row), "money", bold) for label, row, bold in spec])


def cashflow_table(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return _table([], [])
    spec = [
        ("Cash from Operations", "Operating Cash Flow", True),
        ("Capital Expenditure", "Capital Expenditure", False),
        ("Free Cash Flow", "Free Cash Flow", True),
        ("Cash from Investing", "Investing Cash Flow", False),
        ("Cash from Financing", "Financing Cash Flow", False),
        ("Share Buybacks", "Repurchase Of Capital Stock", False),
        ("Dividends Paid", "Cash Dividends Paid", False),
        ("Net Change in Cash", "Changes In Cash", True),
    ]
    return _table(frame.columns, [(label, _line(frame, row), "money", bold) for label, row, bold in spec])


def ratios_table(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame) -> dict:
    """Year-end ratios for the periods present in both income and balance data."""
    columns = [c for c in income.columns if c in balance.columns]
    if not columns:
        return _table([], [])
    inc, bal = income[columns], balance[columns]
    cash = cashflow.reindex(columns=columns) if not cashflow.empty else pd.DataFrame(columns=columns)
    revenue, net, ebit = _line(inc, "Total Revenue"), _line(inc, "Net Income"), _line(inc, "EBIT")
    equity, assets = _line(bal, "Common Stock Equity"), _line(bal, "Total Assets")
    current = _line(bal, "Current Liabilities")
    operating = _line(inc, "Operating Income")
    return _table(columns, [
        ("Operating Margin %", _ratio(operating, revenue), "pct", False),
        ("Net Margin %", _ratio(net, revenue), "pct", False),
        ("Free Cash Flow Margin %", _ratio(_line(cash, "Free Cash Flow"), revenue).where(operating.notna()), "pct", False),  # not meaningful for lenders
        ("ROE %", _ratio(net, equity), "pct", False),
        ("ROCE %", _ratio(ebit, assets - current), "pct", False),
        ("ROA %", _ratio(net, assets), "pct", False),
        ("Debt to Equity", _ratio(_line(bal, "Total Debt"), equity), "times", False),
        ("Current Ratio", _ratio(_line(bal, "Current Assets"), current), "times", False),
        ("Asset Turnover", _ratio(revenue, assets), "times", False),
    ])


def _with_ttm(annual: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Annual income lines plus a trailing-twelve-month column, when it adds anything."""
    labelled = _labelled(annual)
    if annual.empty or quarterly.shape[1] < 4:
        return labelled
    last_four = quarterly.iloc[:, -4:]
    newest, oldest = last_four.columns[-1], last_four.columns[0]
    if newest <= annual.columns[-1] or (newest - oldest).days > MAX_TTM_SPAN_DAYS:
        return labelled
    labelled[TTM_LABEL] = last_four.sum(axis=1, min_count=4).reindex(labelled.index)
    return labelled


def _fill_quarterly_eps(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Yahoo omits per-share data for fiscal Q4; derive it from the prior quarter's share count."""
    if quarterly.empty or "Diluted EPS" not in quarterly.index:
        return quarterly
    out = quarterly.copy()
    shares = _line(out, "Diluted Average Shares").ffill(limit=1)
    out.loc["Diluted EPS"] = out.loc["Diluted EPS"].fillna(_line(out, "Net Income Common Stockholders") / shares)
    return out


def fetch_financials(symbol: str) -> dict:
    """Statement tables for one company; every table is empty for funds and indices."""
    ticker = yf.Ticker(symbol)
    quarterly = _fill_quarterly_eps(_usable(ticker.quarterly_income_stmt))
    annual = _usable(ticker.income_stmt)
    balance = _usable(ticker.balance_sheet)
    cashflow = _usable(ticker.cashflow)
    return {
        "quarterly": income_table(_labelled(quarterly)),
        "annual": income_table(_with_ttm(annual, quarterly)),
        "balance": balance_table(_labelled(balance)),
        "cashflow": cashflow_table(_labelled(cashflow)),
        "ratios": ratios_table(_labelled(annual), _labelled(balance), _labelled(cashflow)),
    }


# ============================================================
# CHART HISTORY
# ============================================================

def fetch_price_history(symbol: str) -> pd.DataFrame:
    """Full daily history: adjusted close, split-only close and volume."""
    raw = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No price history returned for {symbol}")
    out = pd.DataFrame({
        "close": pd.to_numeric(raw["Adj Close"], errors="coerce"),
        "raw_close": pd.to_numeric(raw["Close"], errors="coerce"),
        "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    })
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.dropna(subset=["close"]).loc[lambda d: ~d.index.duplicated(keep="last")].sort_index()


def trailing_eps_series(reported: pd.Series) -> pd.Series:
    """Rolling four-report EPS, blank wherever a report is missing."""
    reported = reported.dropna().sort_index()
    gaps = reported.index.to_series().diff().dt.days
    reported = reported[~(gaps < MIN_REPORT_GAP_DAYS)]
    dates = reported.index.to_series()
    complete = (dates - dates.shift(3)).dt.days <= MAX_FOUR_REPORT_SPAN_DAYS
    return reported.rolling(4).sum().where(complete)


def pe_history(raw_close: pd.Series, reported_eps: pd.Series, current_eps: float | None) -> pd.Series:
    """
    Daily P/E from reported EPS. Empty if the EPS history does not reconcile with
    the current trailing EPS (other currency, or heavily adjusted earnings).
    """
    empty = pd.Series(dtype=float)
    trailing = trailing_eps_series(reported_eps).dropna()
    if trailing.empty or not current_eps or current_eps <= 0:
        return empty
    if abs(trailing.iloc[-1] / current_eps - 1) > PE_EPS_TOLERANCE:
        return empty
    aligned = trailing.reindex(raw_close.index.union(trailing.index)).ffill().reindex(raw_close.index)
    return (raw_close / aligned.where(aligned > 0)).dropna()


def fetch_reported_eps(symbol: str) -> pd.Series:
    dates = yf.Ticker(symbol).get_earnings_dates(limit=60)
    if dates is None or dates.empty:
        return pd.Series(dtype=float)
    eps = pd.to_numeric(dates["Reported EPS"], errors="coerce")
    eps.index = pd.to_datetime(eps.index).tz_localize(None).normalize()
    return eps.dropna()
