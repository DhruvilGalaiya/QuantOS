"""
QuantOS | India daily index data (Upstox)

Incremental, validated, atomic updates of data/regime/daily/<symbol>.parquet for every
instrument in quantos_platform.core.instruments, plus live LTP quotes for display.
Orchestration (when to run, what runs after) lives in quantos_platform.jobs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from quantos_platform.core import calendar, settings
from quantos_platform.core.instruments import BREADTH_INSTRUMENTS, REGIME_INSTRUMENTS, VALIDATION_INSTRUMENTS
from quantos_platform.core.upstox import get
from quantos_platform.core.validation import check_ohlc, require

PROJECT_ROOT = settings.ROOT
DAILY_DIR = settings.DAILY_DIR
FEATURE_DIR = settings.FEATURE_DIR
PREDICTION_DIR = settings.PREDICTION_DIR
PREDICTION_FILE = PREDICTION_DIR / "latest_predictions.json"

IST = settings.IST

INSTRUMENTS = {**REGIME_INSTRUMENTS, **VALIDATION_INSTRUMENTS, **BREADTH_INSTRUMENTS}

# Re-fetch a few days of overlap so a late-corrected candle is replaced.
OVERLAP_DAYS = 5

CANDLE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "open_interest",
]

@dataclass
class PipelineReport:
    started: pd.Timestamp
    finished: pd.Timestamp | None = None
    updated: dict[str, int] = field(default_factory=dict)   # symbol -> rows added
    errors: dict[str, str] = field(default_factory=dict)    # step/symbol -> message
    latest_dates: dict[str, str] = field(default_factory=dict)
    analysis_ran: bool = False
    v11_ran: bool = False
    log: str = ""

    @property
    def seconds(self) -> float:
        end = self.finished or pd.Timestamp.now(tz=IST)
        return (end - self.started).total_seconds()

    @property
    def data_through(self) -> str | None:
        return max(self.latest_dates.values()) if self.latest_dates else None


# ============================================================
# STEP 1: INCREMENTAL DAILY DATA
# ============================================================

def _to_ist(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(IST)


def _fetch_candles(instrument_key: str, start: str, end: str) -> pd.DataFrame:
    response = get(
        f"/v3/historical-candle/{instrument_key}/days/1/{end}/{start}"
    )
    candles = response.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    rows = [
        (list(c[: len(CANDLE_COLUMNS)]) + [None] * len(CANDLE_COLUMNS))[: len(CANDLE_COLUMNS)]
        for c in candles
    ]
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    for col in CANDLE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "close"])


def _with_todays_candle(fresh: pd.DataFrame, instrument_key: str, end_date) -> pd.DataFrame:
    """The historical endpoint publishes today's daily candle only later (often next
    morning). After the close, take it from the intraday endpoint instead; the next
    run's overlap re-download replaces it with the historical value."""
    today = pd.Timestamp.now(tz=IST).date()
    have = set(_to_ist(fresh["timestamp"]).dt.date) if len(fresh) else set()
    if end_date != today or today in have:
        return fresh
    try:
        response = get(f"/v3/historical-candle/intraday/{instrument_key}/days/1")
        candles = response.get("data", {}).get("candles", [])
    except Exception:
        return fresh
    rows = [(list(c[: len(CANDLE_COLUMNS)]) + [None] * len(CANDLE_COLUMNS))[: len(CANDLE_COLUMNS)]
            for c in candles]
    today_df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    for col in CANDLE_COLUMNS[1:]:
        today_df[col] = pd.to_numeric(today_df[col], errors="coerce")
    today_df = today_df.dropna(subset=["timestamp", "close"])
    today_df = today_df[_to_ist(today_df["timestamp"]).dt.date == today]
    return pd.concat([fresh, today_df], ignore_index=True) if len(today_df) else fresh


def update_symbol(symbol: str, instrument_key: str, end_date) -> tuple[int, str]:
    """Append missing sessions to one daily parquet. Returns (rows_added, latest_date)."""
    path = DAILY_DIR / f"{symbol.lower()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    old = pd.read_parquet(path)
    old_tz = getattr(old["timestamp"].dtype, "tz", None)
    old["timestamp"] = _to_ist(old["timestamp"])
    old = old.dropna(subset=["timestamp", "close"])

    latest = old["timestamp"].max()
    if latest.date() >= end_date:
        return 0, latest.date().isoformat()

    start = (latest.date() - timedelta(days=OVERLAP_DAYS)).isoformat()
    fresh = _fetch_candles(instrument_key, start, end_date.isoformat())
    fresh = _with_todays_candle(fresh, instrument_key, end_date)
    fresh = fresh[_to_ist(fresh["timestamp"]).dt.date <= end_date]
    if fresh.empty:
        return 0, latest.date().isoformat()

    # Align fresh candles to the file's existing time-of-day convention
    # (historical rows are stored at 05:30 IST, Upstox now returns 00:00 IST)
    # so the same session is never stored twice.
    time_of_day = latest - latest.normalize()
    fresh["timestamp"] = _to_ist(fresh["timestamp"]).dt.normalize() + time_of_day

    # Keep the existing file's schema (e.g. the `instrument` column).
    if "instrument" in old.columns:
        known = old["instrument"].dropna()
        fresh["instrument"] = known.iloc[-1] if len(known) else pd.NA
    for col in old.columns:
        if col not in fresh.columns:
            fresh[col] = pd.NA
    fresh = fresh[old.columns]

    merged = (
        pd.concat([old, fresh], ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    merged = merged[~merged["timestamp"].dt.date.duplicated(keep="last")].reset_index(drop=True)

    if merged["close"].isna().any() or merged["timestamp"].duplicated().any():
        raise ValueError("merged data failed validation")
    # stricter checks on the rows being added (old history is trusted as-is)
    tail = merged[merged["timestamp"] >= _to_ist(fresh["timestamp"]).min() - pd.Timedelta(days=10)]
    require(check_ohlc(tail, max_daily_move=1.0 if "VIX" in symbol else 0.20), f"{symbol} daily candles")

    old_days = set(old["timestamp"].dt.date)
    added = sum(d not in old_days for d in merged["timestamp"].dt.date)
    deduped = len(merged) != len(old) + added
    new_latest = merged["timestamp"].max().date().isoformat()

    # Nothing new and nothing cleaned: leave the file untouched so the
    # analysis isn't re-run needlessly.
    if added == 0 and not deduped:
        return 0, new_latest

    if old_tz is None:
        merged["timestamp"] = merged["timestamp"].dt.tz_localize(None)

    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return added, new_latest


def last_completed_session_date():
    """Today once the daily candle is final (15:45 IST), otherwise yesterday."""
    return calendar.last_completed_session_date()


def update_all_daily(report: PipelineReport) -> bool:
    """Fetch all instruments in parallel. Returns True if any file changed."""
    end_date = last_completed_session_date()

    def task(item):
        symbol, key = item
        return symbol, update_symbol(symbol, key, end_date)

    changed = False
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(task, item): item[0] for item in INSTRUMENTS.items()}
        for future, symbol in futures.items():
            try:
                _, (added, latest) = future.result()
                report.latest_dates[symbol] = latest
                if added:
                    report.updated[symbol] = added
                    changed = True
            except Exception as exc:  # keep going; dashboard falls back to cached data
                report.errors[f"fetch:{symbol}"] = f"{type(exc).__name__}: {exc}"
    return changed


# ============================================================
# LIVE QUOTES (intraday, display only; not fed to the HMM)
# ============================================================

def fetch_live_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    Latest traded price + previous close for each symbol via Upstox LTP.

    Returns {symbol: {"ltp": float, "prev_close": float | None}}.
    """
    keys = {INSTRUMENTS[s]: s for s in symbols if s in INSTRUMENTS}
    if not keys:
        return {}

    response = get(
        "/v3/market-quote/ltp",
        params={"instrument_key": ",".join(keys)},
    )

    out = {}
    for entry in response.get("data", {}).values():
        symbol = keys.get(entry.get("instrument_token"))
        if symbol is None or entry.get("last_price") is None:
            continue
        out[symbol] = {
            "ltp": float(entry["last_price"]),
            "prev_close": float(entry["cp"]) if entry.get("cp") else None,
        }
    return out
