"""
Data checks applied BEFORE anything is written. Bad data is rejected, not displayed.

Each check returns a list of problems (empty = valid); `require` raises on problems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    pass


def require(problems: list[str], what: str) -> None:
    if problems:
        raise DataValidationError(f"{what}: " + "; ".join(problems[:5]))


def check_ohlc(df: pd.DataFrame, max_daily_move: float = 0.25, price_cols=("open", "high", "low", "close"),
               time_col: str = "timestamp", allow_zero_volume: bool = True) -> list[str]:
    """Daily/intraday OHLC bars: columns, no duplicates, ordered time, sane prices."""
    p = []
    missing = [c for c in (time_col, *price_cols) if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    if df.empty:
        return ["no rows"]
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if ts.isna().any():
        p.append(f"{int(ts.isna().sum())} unparseable timestamps")
    if ts.duplicated().any():
        p.append(f"{int(ts.duplicated().sum())} duplicate timestamps")
    if not ts.dropna().is_monotonic_increasing:
        p.append("timestamps not in increasing order")
    px = df[list(price_cols)].apply(pd.to_numeric, errors="coerce")
    if px.isna().any().any():
        p.append("missing prices")
    if (px <= 0).any().any():
        p.append("non-positive prices")
    tol = 1e-6 * px["close"].abs().clip(lower=1)
    if ((px["high"] + tol) < px[["open", "close"]].max(axis=1)).any() or \
       ((px["low"] - tol) > px[["open", "close"]].min(axis=1)).any():
        p.append("high/low inconsistent with open/close")
    r = np.log(px["close"]).diff().abs()
    if (r > np.log(1 + max_daily_move)).any():
        p.append(f"close-to-close move above {max_daily_move:.0%} (possible bad tick)")
    return p


def check_option_rows(rows: list[tuple] | pd.DataFrame, min_rows: int = 20) -> list[str]:
    """Option-chain snapshot rows before they go into the options DB."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if len(df) < min_rows:
        return [f"only {len(df)} option rows"]
    return []


def check_surface(out: dict) -> list[str]:
    """Volatility-surface artifact before it is written."""
    p = []
    exps = out.get("expiries") or []
    if not exps:
        return ["no fitted expiries"]
    for e in exps:
        atm = e.get("atm_iv")
        if atm is not None and np.isfinite(atm) and not (0.01 < atm < 2.0):
            p.append(f"{e.get('expiry')}: ATM IV {atm:.3f} outside 1%..200%")
        if e.get("svi_rmse") is not None and e["svi_rmse"] > 0.05:
            p.append(f"{e.get('expiry')}: SVI error {e['svi_rmse'] * 100:.1f} vol pts")
    return p


def check_prediction_history(df: pd.DataFrame) -> list[str]:
    p = []
    for col in ("prediction_date", "target_trading_date", "next_regime"):
        if col not in df.columns:
            p.append(f"missing column {col}")
    if not p and df["prediction_date"].duplicated().any():
        p.append("duplicate prediction dates")
    return p
