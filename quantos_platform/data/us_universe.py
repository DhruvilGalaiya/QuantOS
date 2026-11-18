"""
QuantOS | US stock universe for the dashboard's US Stocks page

Builds data/regime/portfolio_us/universe.json:
    * top 50 Nasdaq-100 members by market cap   (Nasdaq official list)
    * top 100 S&P 500 members by market cap     (Wikipedia membership +
                                                 Nasdaq screener market caps)
then pre-downloads daily closes for every ticker (one batched Yahoo call).

    python -m src.quantos_portfolio.us_universe          # rebuild + download
"""

from __future__ import annotations

import io
import json

import pandas as pd
import requests
import yfinance as yf

from quantos_platform.data.us_equities import OUTPUT_DIR, START_DATE, update_us_asset

UNIVERSE_FILE = OUTPUT_DIR / "universe.json"
TOP_NASDAQ = 50
TOP_SP500 = 100

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/html",
}


def _yahoo(symbol: str) -> str:
    """BRK.B / BRK/B -> BRK-B (Yahoo Finance convention)."""
    return symbol.strip().upper().replace(".", "-").replace("/", "-")


def _cap(value) -> float:
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return float("nan")


def _clean_name(name: str) -> str:
    for suffix in (" Common Stock", " Class A Common Stock", " Class C Capital Stock",
                   " Class A Ordinary Shares", " Ordinary Shares", " American Depositary Shares"):
        name = name.replace(suffix, "")
    return name.strip()


def nasdaq100() -> pd.DataFrame:
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["data"]["rows"]
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "ticker": df["symbol"].map(_yahoo),
        "name": df["companyName"].map(_clean_name),
        "market_cap": df["marketCap"].map(_cap),
    })


def screener_caps() -> pd.DataFrame:
    r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                     params={"tableonly": "true", "limit": "10000", "download": "true"},
                     headers=_HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["rows"])
    return pd.DataFrame({
        "ticker": df["symbol"].map(_yahoo),
        "market_cap": df["marketCap"].map(_cap),
        "sector": df["sector"].replace("", None),
    }).drop_duplicates("ticker")


def sp500_members() -> pd.DataFrame:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": _HEADERS["User-Agent"]}, timeout=30)
    r.raise_for_status()
    table = next(t for t in pd.read_html(io.StringIO(r.text)) if "Symbol" in t.columns and len(t) > 400)
    return pd.DataFrame({
        "ticker": table["Symbol"].map(_yahoo),
        "name": table["Security"],
        "gics_sector": table["GICS Sector"],
    })


def _fill_missing_caps(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["market_cap"].isna() | (df["market_cap"] <= 0)
    for i in df.index[missing]:
        try:
            df.at[i, "market_cap"] = float(yf.Ticker(df.at[i, "ticker"]).fast_info["market_cap"])
        except Exception:
            pass
    return df


def _drop_share_class_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one share class per company (e.g. GOOGL over GOOG, FOXA over FOX)."""
    key = df["name"].str.replace(r"\b(Class [A-C]|Capital Stock|Inc\.?|Corporation)\b", "", regex=True)
    key = key.str.replace(r"[^A-Za-z0-9]", "", regex=True).str.lower()
    return df.loc[~key.duplicated(keep="first")]


def build_universe() -> pd.DataFrame:
    caps = screener_caps()

    ndx = _fill_missing_caps(nasdaq100())
    ndx = _drop_share_class_duplicates(ndx.sort_values("market_cap", ascending=False))
    ndx = ndx.head(TOP_NASDAQ).assign(nasdaq_rank=range(1, TOP_NASDAQ + 1))

    spx = sp500_members().merge(caps, on="ticker", how="left")
    spx = _fill_missing_caps(spx)
    spx = _drop_share_class_duplicates(spx.sort_values("market_cap", ascending=False))
    spx = spx.head(TOP_SP500).assign(sp500_rank=range(1, TOP_SP500 + 1))

    uni = pd.merge(
        spx[["ticker", "name", "gics_sector", "market_cap", "sp500_rank"]],
        ndx[["ticker", "name", "market_cap", "nasdaq_rank"]],
        on="ticker", how="outer", suffixes=("", "_ndx"),
    )
    uni["name"] = uni["name"].fillna(uni["name_ndx"])
    uni["market_cap"] = uni["market_cap"].fillna(uni["market_cap_ndx"])
    # Nasdaq screener sector names -> GICS names used by the S&P list
    to_gics = {
        "Technology": "Information Technology", "Finance": "Financials",
        "Telecommunications": "Communication Services", "Basic Materials": "Materials",
    }
    sector = caps.set_index("ticker")["sector"].replace(to_gics)
    uni["sector"] = uni["gics_sector"].fillna(uni["ticker"].map(sector)).fillna("—")
    uni = uni.drop(columns=["name_ndx", "market_cap_ndx", "gics_sector"])
    uni = uni.sort_values("market_cap", ascending=False).reset_index(drop=True)
    return uni


def save_universe(uni: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "sources": {
            "nasdaq100": "api.nasdaq.com (official Nasdaq-100 list, market caps)",
            "sp500": "Wikipedia S&P 500 membership + Nasdaq screener market caps",
        },
        "stocks": json.loads(uni.to_json(orient="records")),
    }
    UNIVERSE_FILE.write_text(json.dumps(payload, indent=2))


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(UNIVERSE_FILE.read_text())["stocks"])


def download_all(tickers: list[str]) -> None:
    """Initial full history for tickers without a file (one batched call); top up the rest."""
    missing = [t for t in tickers if not (OUTPUT_DIR / f"{t}.parquet").exists()]
    if missing:
        raw = yf.download(missing, start=START_DATE, auto_adjust=True, progress=False,
                          actions=False, group_by="column", threads=True)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(missing[0])
        for t in missing:
            if t not in raw.columns:
                continue
            s = raw[t].dropna()
            if s.empty:
                continue
            out = pd.DataFrame({"timestamp": pd.to_datetime(s.index).tz_localize(None).normalize(),
                                "close": s.to_numpy(), "symbol": t})
            out.to_parquet(OUTPUT_DIR / f"{t}.parquet", index=False)
    for t in tickers:
        if t not in missing:
            try:
                update_us_asset(t)
            except Exception as exc:
                print(f"update failed {t}: {exc}")


def main():
    uni = build_universe()
    save_universe(uni)
    print(f"Universe: {len(uni)} stocks  "
          f"(Nasdaq top {int(uni['nasdaq_rank'].notna().sum())}, "
          f"S&P top {int(uni['sp500_rank'].notna().sum())}, "
          f"in both {int((uni['nasdaq_rank'].notna() & uni['sp500_rank'].notna()).sum())})")
    print(uni.head(12)[["ticker", "name", "market_cap", "sp500_rank", "nasdaq_rank", "sector"]].to_string())
    download_all(uni["ticker"].tolist())
    have = [t for t in uni["ticker"] if (OUTPUT_DIR / f"{t}.parquet").exists()]
    print(f"Price files: {len(have)}/{len(uni)}")


if __name__ == "__main__":
    main()
