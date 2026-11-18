"""
QuantOS | Expiry-week overnight playbook (experimental, paper-tracked)

Rule (from the Oct-2024 -> Nov-2026 backtest on real Upstox option candles):
    On the last 2 sessions before an expiry, at ~15:15 IST:
        model bullish -> SELL the ~1% OTM PUT
        model bearish -> SELL the ~1% OTM CALL
    Buy it back at 15:15 IST on the next session (or 09:30 for a quicker exit).

Model ("B"): logistic regression trained on daily bars since 2010 that
estimates P(next open > price at 15:15). The view is bullish when that
probability beats the historical up-rate (drift), bearish otherwise.

Files (data/derivatives/playbook/):
    signals_<sym>.csv     every signal night since TRACK_START, with outcome
    backtest_<sym>.parquet walk-forward per-night book (all strategies, tdte)

    python -m quantos_platform.strategies.expiry_playbook              # update inputs, signal, track
    python -m quantos_platform.strategies.expiry_playbook --backtest   # rebuild the backtest book
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantos_platform.core import settings  # noqa: E402
from quantos_platform.core.calendar import (  # noqa: E402,F401  (re-exported for callers)
    future_sessions,
    now_ist,
    sessions_to_expiry,
    today_ist,
    trading_holidays,
)
from quantos_platform.core.upstox import get  # noqa: E402

PROJECT_ROOT = settings.ROOT
PLAYBOOK_DIR = settings.PLAYBOOK_DIR
ASIA_FILE = settings.ASIA_FILE
IST = settings.IST

TRACK_START = pd.Timestamp("2026-11-18")   # live paper-tracking starts here
MAX_TDTE = 2                               # signal nights: 1 or 2 sessions before expiry
OTM_PCT = 0.01
FINAL_WINDOW = ("15:10", "15:29")          # signal is final inside this IST window
TRAIN_FROM = "2010-04-01"
LOGIT_C = 0.05
REFIT_EVERY = 21

SYMBOLS = {
    "NIFTY_50": {"key": "NSE_INDEX|Nifty 50", "step": 50, "name": "NIFTY"},
    "NIFTY_BANK": {"key": "NSE_INDEX|Nifty Bank", "step": 100, "name": "BANKNIFTY"},
}
# One feature per idea; all available at 15:15 (no US session needed).
FEATURES = ["i_day_z", "i_range_pos", "vix_day", "b_up", "d_st_dir", "d_ema9_21",
            "d_rsi2", "d_z20", "asia_NIKKEI", "asia_HANGSENG", "days_to_next", "expiry_day"]
ASIA_TICKERS = {"^N225": "NIKKEI", "^HSI": "HANGSENG"}


# ============================================================
# INPUTS
# ============================================================

def update_asia() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(list(ASIA_TICKERS), start="2009-01-01", auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=ASIA_TICKERS)[list(ASIA_TICKERS.values())]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.dropna(how="all")
    if raw.empty:
        raise ValueError("no Asia data returned")
    ASIA_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(ASIA_FILE)
    return raw


def update_inputs(log=print) -> dict[str, str]:
    """15-minute bars for the index + breadth set (incl. today), Asia closes."""
    from quantos_platform.data.intraday import INTRADAY_SYMBOLS, update_symbol

    errors = {}
    for sym in INTRADAY_SYMBOLS:
        try:
            update_symbol(sym, include_today=True)
        except Exception as exc:  # keep going; a missing sector only weakens breadth
            errors[sym] = f"{type(exc).__name__}: {exc}"
    try:
        update_asia()
    except Exception as exc:
        errors["ASIA"] = f"{type(exc).__name__}: {exc}"
    for k, v in errors.items():
        log(f"input update failed {k}: {v}")
    return errors


# ============================================================
# CALENDAR: expiries, holidays, sessions-to-expiry
# ============================================================

def _cache(name: str, fetch, max_age_h: float = 12):
    path = PLAYBOOK_DIR / f".cache_{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_h * 3600:
        return json.loads(path.read_text())
    data = fetch()
    PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def active_contracts(symbol: str) -> pd.DataFrame:
    def fetch():
        return get("/v2/option/contract", {"instrument_key": SYMBOLS[symbol]["key"]}).get("data", [])
    df = pd.DataFrame(_cache(f"contracts_{symbol}", fetch, max_age_h=6))
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["strike_price"] = df["strike_price"].astype(float)
    return df


def find_contract(symbol: str, expiry: pd.Timestamp, strike: float, side: str) -> dict | None:
    """Option contract (instrument_key, trading_symbol, lot_size) for live or expired expiries."""
    if expiry >= today_ist():
        cs = active_contracts(symbol)
        cs = cs[cs["expiry"] == expiry]
    else:
        def fetch():
            return get("/v2/expired-instruments/option/contract",
                       {"instrument_key": SYMBOLS[symbol]["key"], "expiry_date": str(expiry.date())}).get("data", [])
        cs = pd.DataFrame(_cache(f"expired_{symbol}_{expiry.date()}", fetch, max_age_h=24 * 365))
        if cs.empty:
            return None
        cs["strike_price"] = cs["strike_price"].astype(float)
    m = cs[(cs["strike_price"] == float(strike)) & (cs["instrument_type"] == side)]
    if m.empty:
        return None
    c = m.iloc[0]
    return {"instrument_key": c["instrument_key"], "trading_symbol": c["trading_symbol"],
            "lot_size": int(c["lot_size"])}


def all_expiries(symbol: str) -> list[pd.Timestamp]:
    def fetch_expired():
        return get("/v2/expired-instruments/expiries", {"instrument_key": SYMBOLS[symbol]["key"]}).get("data", [])
    expired = pd.to_datetime(pd.Series(_cache(f"expired_{symbol}", fetch_expired), dtype="object"))
    live = active_contracts(symbol)["expiry"]
    return sorted(set(expired.tolist()) | set(live.tolist()))


def next_expiry(symbol: str, day: pd.Timestamp) -> pd.Timestamp | None:
    nxt = [e for e in all_expiries(symbol) if e > day]
    return nxt[0] if nxt else None


# ============================================================
# MODEL B
# ============================================================

def _snapshot_frames(symbol: str, partial_today: bool):
    from quantos_platform.models.snapshot_1515 import build_snapshot_frame

    asia = pd.read_parquet(ASIA_FILE)
    snap = build_snapshot_frame(symbol, asia, None, "intraday", partial_today=partial_today)
    dly = build_snapshot_frame(symbol, asia, None, "daily")
    dly = dly[dly.index >= TRAIN_FROM].dropna(subset=["y_gap"])
    return snap, dly


def _model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=LOGIT_C, max_iter=3000))


def _fit(dly: pd.DataFrame, snap: pd.DataFrame, cutoff: pd.Timestamp):
    """Train on rows whose outcome was known before `cutoff`'s 15:15."""
    tr = dly[dly.index < cutoff].iloc[:-1]  # the session right before cutoff resolves at cutoff's open
    model = _model().fit(tr[FEATURES], (tr["y_gap"] > 0).astype(int))
    known = snap[(snap.index < cutoff)].dropna(subset=["y_gap"]).iloc[:-1]
    drift = float((known["y_gap"] > 0).mean())
    return model, drift


def view_for(model, drift: float, row: pd.DataFrame) -> tuple[float, int]:
    p = float(model.predict_proba(row[FEATURES])[:, 1][0])
    return p, (1 if p > drift else -1)


# ============================================================
# LIVE: tonight's signal
# ============================================================

def ltp(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    data = get("/v3/market-quote/ltp", {"instrument_key": ",".join(keys)}).get("data", {})
    return {v.get("instrument_token"): float(v["last_price"]) for v in data.values() if v.get("last_price")}


def _strike(spot: float, step: int, side: int) -> float:
    return float(round(spot * (1 - OTM_PCT * side) / step) * step)


def tonight(symbol: str) -> dict:
    """Signal for the current session (preview before 15:10, final 15:10-15:29)."""
    cfg = SYMBOLS[symbol]
    now = now_ist()
    today = today_ist()
    hm = now.strftime("%H:%M")
    out = {"symbol": symbol, "as_of": now.strftime("%Y-%m-%d %H:%M"), "date": str(today.date())}

    snap, dly = _snapshot_frames(symbol, partial_today=True)
    is_session = len(snap) and snap.index.max() == today
    day = today if is_session else snap.index.max()
    expiry = next_expiry(symbol, day)
    known = pd.DatetimeIndex(dly.index.union(snap.index))
    tdte = sessions_to_expiry(day, expiry, known) if expiry is not None else None
    out.update({"session": str(day.date()), "expiry": str(expiry.date()) if expiry is not None else None,
                "tdte": tdte, "active": bool(tdte is not None and tdte <= MAX_TDTE)})

    if not is_session:
        out["phase"] = "closed"
    elif hm < "09:15":
        out["phase"] = "pre-open"
    elif hm < FINAL_WINDOW[0]:
        out["phase"] = "preview"
    elif hm <= FINAL_WINDOW[1]:
        out["phase"] = "final"
    else:
        out["phase"] = "after"

    # next signal window when tonight is not one
    if expiry is not None and not out["active"]:
        upcoming = future_sessions(day, expiry)
        windows = [d for d in upcoming if sessions_to_expiry(d, expiry, known) <= MAX_TDTE]
        out["next_window"] = str(windows[0].date()) if windows else None

    if not is_session:
        return out

    # days_to_next for the live row comes from the calendar
    nxt = future_sessions(today, today + pd.Timedelta(days=10))
    if nxt:
        snap.loc[today, "days_to_next"] = (nxt[0] - today).days
    model, drift = _fit(dly, snap, today)
    p, view = view_for(model, drift, snap.loc[[today]])
    spot = float(snap.at[today, "p1515"])
    if out["phase"] in ("preview", "final"):
        try:
            spot = ltp([cfg["key"]]).get(cfg["key"]) or spot
        except Exception:
            pass
    side = "PE" if view > 0 else "CE"
    strike = _strike(spot, cfg["step"], view)
    out.update({"p_up": p, "drift": drift, "view": view, "side": side, "strike": strike, "spot": spot})

    c = find_contract(symbol, expiry, strike, side) if expiry is not None else None
    if c:
        out.update(c)
        try:
            q = get("/v3/market-quote/ltp", {"instrument_key": c["instrument_key"]}).get("data", {})
            prem = next((float(v["last_price"]) for v in q.values() if v.get("last_price")), None)
        except Exception:
            prem = None
        out["premium"] = prem
    return out


# ============================================================
# TRACKING: every signal night since TRACK_START, scored
# ============================================================

SIGNAL_COLS = ["date", "next_session", "expiry", "tdte", "p_up", "drift", "view", "side", "strike",
               "spot_1515", "instrument_key", "trading_symbol", "lot_size", "source",
               "entry", "exit_0930", "exit_1515", "pnl_0930_rs", "pnl_1515_rs"]


def signals_path(symbol: str) -> Path:
    return PLAYBOOK_DIR / f"signals_{symbol.lower()}.csv"


def load_signals(symbol: str) -> pd.DataFrame:
    p = signals_path(symbol)
    if not p.exists():
        return pd.DataFrame(columns=SIGNAL_COLS)
    return pd.read_csv(p, parse_dates=["date", "next_session", "expiry"])


def _option_bars(row: pd.Series) -> pd.DataFrame:
    """15-minute candles for the traded option from entry day to exit day."""
    from quantos_platform.data.intraday import _parse

    key, start, end = str(row["instrument_key"]), row["date"].date(), row["next_session"].date()
    frames = []
    if key.count("|") == 2 or row["expiry"] < today_ist():
        from quantos_platform.data.derivatives import candles  # expired contract

        if key.count("|") != 2:
            sym = next(s for s, c in SYMBOLS.items() if str(row["trading_symbol"]).startswith(c["name"] + " "))
            c = find_contract(sym, row["expiry"], row["strike"], row["side"])
            key = c["instrument_key"] if c else key
        try:
            frames.append(candles(key, str(start), str(end)))
        except Exception:
            pass
    else:
        try:
            r = get(f"/v3/historical-candle/{key}/minutes/15/{end}/{start}")
            frames.append(_parse(r.get("data", {}).get("candles", [])))
        except Exception:
            pass
        if row["next_session"] == today_ist():
            try:
                r = get(f"/v3/historical-candle/intraday/{key}/minutes/15")
                frames.append(_parse(r.get("data", {}).get("candles", [])))
            except Exception:
                pass
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    b = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    b["date"] = b["timestamp"].dt.tz_localize(None).dt.normalize()
    b["hm"] = b["timestamp"].dt.strftime("%H:%M")
    return b


def _score(row: pd.Series, symbol: str) -> pd.Series:
    from quantos_platform.strategies.overnight_options import leg_pnl

    b = _option_bars(row)
    if b.empty:
        return row

    def px(day, hm, col="close"):
        x = b[(b["date"] == day) & (b["hm"] == hm)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    row["entry"] = px(row["date"], "15:00")
    row["exit_0930"] = px(row["next_session"], "09:15")
    row["exit_1515"] = px(row["next_session"], "15:00")
    for col, out in (("exit_0930", "pnl_0930_rs"), ("exit_1515", "pnl_1515_rs")):
        if np.isfinite(row["entry"]) and np.isfinite(row[col]):
            leg = pd.DataFrame({"entry": [row["entry"]], col: [row[col]], "lot_size": [row["lot_size"]]})
            row[out] = float(leg_pnl(leg, symbol, col).iloc[0] * row["lot_size"])
    return row


def track(symbol: str, log=print) -> pd.DataFrame:
    """Add any missing signal nights (rebuilt from the 15:15 snapshot) and score outcomes."""
    cfg = SYMBOLS[symbol]
    sig = load_signals(symbol)
    have = set(sig["date"].dt.normalize()) if len(sig) else set()
    snap, dly = _snapshot_frames(symbol, partial_today=False)
    known = pd.DatetimeIndex(dly.index.union(snap.index))
    sessions = snap.index[snap.index >= TRACK_START]
    new_rows = []
    for day in sessions:
        if day in have:
            continue
        if day == today_ist() and now_ist().strftime("%H:%M") < "15:16":
            continue  # 15:00 bar still forming
        expiry = next_expiry(symbol, day)
        if expiry is None:
            continue
        tdte = sessions_to_expiry(day, expiry, known)
        if tdte > MAX_TDTE:
            continue
        model, drift = _fit(dly, snap, day)
        p, view = view_for(model, drift, snap.loc[[day]])
        spot = float(snap.at[day, "p1515"])
        side = "PE" if view > 0 else "CE"
        strike = _strike(spot, cfg["step"], view)
        # exit on the next REGULAR session (skips e.g. a 1-hour Muhurat session)
        nxt_known = snap.index[snap.index > day]
        next_session = nxt_known[0] if len(nxt_known) else (future_sessions(day, day + pd.Timedelta(days=10)) or [pd.NaT])[0]
        c = find_contract(symbol, expiry, strike, side) or {}
        new_rows.append({
            "date": day, "next_session": next_session, "expiry": expiry, "tdte": tdte,
            "p_up": p, "drift": drift, "view": view, "side": side, "strike": strike, "spot_1515": spot,
            "instrument_key": c.get("instrument_key"),
            "trading_symbol": c.get("trading_symbol", f"{cfg['name']} {strike:.0f} {side}"),
            "lot_size": c.get("lot_size", np.nan),
            "source": "15:15 snapshot",
        })
        log(f"{symbol}: signal night {day.date()} tdte={tdte} -> sell {strike:.0f} {side}")
    if new_rows:
        sig = pd.concat([sig, pd.DataFrame(new_rows)], ignore_index=True)

    # score rows whose exit session has finished (15:15 bar done)
    now = now_ist()
    for i, row in sig.iterrows():
        if pd.notna(row.get("pnl_1515_rs")) or pd.isna(row.get("instrument_key")):
            continue
        done = row["next_session"] < today_ist() or (
            row["next_session"] == today_ist() and now.strftime("%H:%M") >= "15:16")
        if done:
            sig.loc[i] = _score(row.copy(), symbol)
    sig = sig.reindex(columns=SIGNAL_COLS).sort_values("date").reset_index(drop=True)
    PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    sig.to_csv(signals_path(symbol), index=False)
    return sig


# ============================================================
# BACKTEST (walk-forward model view on real option candles)
# ============================================================

def walk_forward_view(symbol: str, start: str = "2024-09-01") -> pd.Series:
    snap, dly = _snapshot_frames(symbol, partial_today=False)
    days = snap.index[snap.index >= start]
    views = {}
    for i in range(0, len(days), REFIT_EVERY):
        block = days[i:i + REFIT_EVERY]
        model, drift = _fit(dly, snap, block[0])
        p = model.predict_proba(snap.loc[block, FEATURES])[:, 1]
        views.update(dict(zip(block, np.where(p > drift, 1, -1))))
    return pd.Series(views).sort_index()


def build_backtest(symbol: str) -> pd.DataFrame:
    from quantos_platform.strategies.overnight_options import strategy_book

    view = walk_forward_view(symbol)
    snap, dly = _snapshot_frames(symbol, partial_today=False)
    known = pd.DatetimeIndex(dly.index.union(snap.index))
    books = {}
    for ex in ("x0930", "x1515"):
        b = strategy_book(symbol, view, ex)
        b["tdte"] = [sessions_to_expiry(t, e, known) for t, e in zip(b.index, b["expiry"])]
        books[ex] = b
    keep = ["dir_otm", "straddle_atm", "strangle_1pct", "put_otm"]
    out = books["x1515"][["tdte", "expiry", "spot"]].copy()
    for ex, b in books.items():
        for k in keep:
            out[f"{k}_{ex}"] = b[f"{k}_rs"]
    out["view"] = view.reindex(out.index)
    PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PLAYBOOK_DIR / f"backtest_{symbol.lower()}.parquet")
    return out


def load_backtest(symbol: str) -> pd.DataFrame:
    p = PLAYBOOK_DIR / f"backtest_{symbol.lower()}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def main():
    if "--backtest" in sys.argv:
        for sym in SYMBOLS:
            bt = build_backtest(sym)
            near = bt[bt["tdte"] <= MAX_TDTE]
            print(f"{sym}: {len(near)} signal nights  "
                  f"avg Rs/lot hold {near['dir_otm_x1515'].mean():+.0f}  exit 9:30 {near['dir_otm_x0930'].mean():+.0f}")
        return
    update_inputs()
    for sym in SYMBOLS:
        sig = tonight(sym)
        print(json.dumps(sig, indent=1, default=str))
        track(sym)


if __name__ == "__main__":
    main()
