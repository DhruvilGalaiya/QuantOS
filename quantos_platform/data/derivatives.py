"""
QuantOS | Upstox expired futures & options (15-minute candles)

Upstox's expired-instruments API covers ~Oct 2024 onward.

    python src/derivatives_data.py futures      # near-month futures, both indices
    python src/derivatives_data.py options      # overnight-strategy option set (resumable)

Option set per session T (index price at 15:15 = S):
    nearest expiry strictly after T,
    ATM call + put  (strike nearest S),
    OTM call / put  (nearest strike to S*(1 +/- OTM_PCT)).
Each contract's candles are fetched once, from its first needed day to the
session after its last needed day.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from quantos_platform.core import settings  # noqa: E402
from quantos_platform.core.upstox import get  # noqa: E402
from quantos_platform.data.intraday import _parse, load_intraday  # noqa: E402

PROJECT_ROOT = settings.ROOT
DERIV_DIR = settings.DATA_DIR / "derivatives"
THROTTLE_S = 0.95  # ~1 req/s keeps us inside 2000 requests / 30 min

UNDERLYINGS = {
    "NIFTY_50": {"key": "NSE_INDEX|Nifty 50", "step": 50},
    "NIFTY_BANK": {"key": "NSE_INDEX|Nifty Bank", "step": 100},
}
OTM_PCT = 0.01


def _get(endpoint: str, params=None, retries: int = 4):
    for attempt in range(retries):
        try:
            out = get(endpoint, params=params)
            time.sleep(THROTTLE_S)
            return out
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "Too Many" in msg:
                time.sleep(60 * (attempt + 1))
                continue
            if attempt == retries - 1:
                raise
            time.sleep(3)
    return {}


def expiries(symbol: str) -> list[str]:
    return _get("/v2/expired-instruments/expiries", {"instrument_key": UNDERLYINGS[symbol]["key"]})["data"]


def candles(instrument_key: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=27), end_ts)
        resp = _get(f"/v2/expired-instruments/historical-candle/{instrument_key}/15minute/"
                    f"{chunk_end.date()}/{cursor.date()}")
        raw = resp.get("data", {}).get("candles", [])
        if raw:
            df = _parse(raw)
            vol = pd.DataFrame(raw).iloc[:, [0, 5, 6]] if len(raw[0]) >= 7 else None
            if vol is not None:
                vol.columns = ["ts", "volume", "oi"]
                df["volume"] = pd.to_numeric(vol["volume"]).to_numpy()
                df["oi"] = pd.to_numeric(vol["oi"]).to_numpy()
            frames.append(df)
        cursor = chunk_end + pd.Timedelta(days=1)
    return pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp") if frames else pd.DataFrame()


# ============================================================
# FUTURES: near-month continuous series
# ============================================================

def fetch_futures(symbol: str) -> pd.DataFrame:
    key = UNDERLYINGS[symbol]["key"]
    rows, prev_expiry = [], None
    for exp in expiries(symbol):
        fc = _get("/v2/expired-instruments/future/contract", {"instrument_key": key, "expiry_date": exp})["data"]
        if not fc:
            continue
        c = fc[0]
        # near-month window: day after previous futures expiry .. this expiry
        start = (pd.Timestamp(prev_expiry) + pd.Timedelta(days=1)) if prev_expiry else pd.Timestamp(exp) - pd.Timedelta(days=35)
        df = candles(c["instrument_key"], str(start.date()), exp)
        if len(df):
            df["expiry"] = pd.Timestamp(exp)
            df["instrument_key"] = c["instrument_key"]
            df["lot_size"] = c["lot_size"]
            rows.append(df)
        print(f"{symbol} futures {exp}: {len(df)} bars")
        prev_expiry = exp
    out = pd.concat(rows).sort_values("timestamp").reset_index(drop=True)
    path = DERIV_DIR / "futures" / f"{symbol.lower()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# ============================================================
# OPTIONS: contracts needed by the overnight strategies
# ============================================================

def _snap_prices(symbol: str) -> pd.Series:
    b = load_intraday(symbol)
    b = b[b["timestamp"].dt.strftime("%H:%M") == "15:00"]
    return pd.Series(b["close"].to_numpy(), index=b["timestamp"].dt.tz_localize(None).dt.normalize())


def option_plan(symbol: str, exp_list: list[str]) -> pd.DataFrame:
    step = UNDERLYINGS[symbol]["step"]
    spot = _snap_prices(symbol)
    exps = pd.to_datetime(pd.Series(sorted(exp_list)))
    sessions = spot.index
    rows = []
    for i, (day, s) in enumerate(spot.items()):
        if day < exps.min() - pd.Timedelta(days=40) or i + 1 >= len(sessions):
            continue
        nxt = exps[exps > day]
        if nxt.empty:
            continue
        exp = nxt.iloc[0]
        next_session = sessions[i + 1]
        if next_session > exp:
            continue
        atm = round(s / step) * step
        otm_c = round(s * (1 + OTM_PCT) / step) * step
        otm_p = round(s * (1 - OTM_PCT) / step) * step
        for leg, strike, typ in [("atm_ce", atm, "CE"), ("atm_pe", atm, "PE"),
                                 ("otm_ce", otm_c, "CE"), ("otm_pe", otm_p, "PE")]:
            rows.append({"date": day, "next_session": next_session, "expiry": exp, "leg": leg,
                         "strike": float(strike), "type": typ, "spot": s})
    return pd.DataFrame(rows)


def fetch_options(symbol: str) -> None:
    key = UNDERLYINGS[symbol]["key"]
    out_dir = DERIV_DIR / "options" / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_list = expiries(symbol)
    plan = option_plan(symbol, exp_list)
    plan.to_parquet(out_dir / "plan.parquet", index=False)
    for exp, grp in plan.groupby("expiry"):
        path = out_dir / f"{exp.date()}.parquet"
        if path.exists():
            continue
        contracts = pd.DataFrame(_get("/v2/expired-instruments/option/contract",
                                      {"instrument_key": key, "expiry_date": str(exp.date())})["data"])
        if contracts.empty:
            continue
        contracts["strike"] = contracts["strike_price"].astype(float)
        frames = []
        for (strike, typ), g in grp.groupby(["strike", "type"]):
            match = contracts[(contracts["strike"] == strike) & (contracts["instrument_type"] == typ)]
            if match.empty:
                continue
            ik = match.iloc[0]["instrument_key"]
            df = candles(ik, str(g["date"].min().date()), str(min(g["next_session"].max(), exp).date()))
            if len(df):
                df["strike"], df["type"], df["expiry"] = strike, typ, exp
                df["lot_size"] = match.iloc[0]["lot_size"]
                frames.append(df)
        if frames:
            pd.concat(frames).to_parquet(path, index=False)
        print(f"{symbol} options {exp.date()}: {len(frames)} contracts", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "futures"
    for sym in UNDERLYINGS:
        if what == "futures":
            fetch_futures(sym)
        else:
            fetch_options(sym)
