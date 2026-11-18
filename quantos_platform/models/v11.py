"""
QuantOS | HMM V11 — pre-open next-session forecaster

WHY V11
-------
V10 forecasts tomorrow's regime as argmax(filtered_posterior @ P), with a
transition matrix that is ~95% diagonal and states defined by a 20-day trend
rule. That is a persistence forecast of a slow trend label; against the actual
next-day close it scored 50.3% direction / 31.6% 3-class over Nov-2024..Nov-2026
(coin flip). Also its `volume_ratio` input is constant (Upstox reports zero
volume for indices) and its history is in-sample (fit once over the window).

Walk-forward research (validation 2016-01..2024-10, n=2189) showed:
  * Indian end-of-day information alone (price, VIX, breadth, HMM regime) does
    not beat the 55% always-up drift.
  * The US session that trades AFTER India's close is the dominant, stable
    signal for NIFTY's next close-to-close move (~61% direction).
  * A causal Gaussian-HMM volatility/return regime improves the 3-class
    (BULL/SIDE/BEAR) probabilities.

MODEL
-----
Target   : next-session close-to-close move of the index;
           BULL > +0.25%, BEAR < -0.25%, otherwise SIDE.
Inputs   : US overnight (S&P 500, Nasdaq, US VIX log-moves between India's
           close T and India's open T+1); Indian EOD compact set; filtered
           posteriors of a 3-state Gaussian HMM fit on training rows only.
Model    : equal-weight ensemble of two L2 multinomial logits
           (US-only; US + EOD + HMM).
Protocol : expanding-window walk-forward, quarterly refits, every prediction
           made only with rows whose outcome was known beforehand.
Timing   : the forecast for session T+1 is FINAL once the US session dated T
           has closed (~02:30 IST); before that it is PROVISIONAL.

Usage
-----
    python src/quantos_hmm_v11.py --rebuild   # full walk-forward + metrics
    python src/quantos_hmm_v11.py             # incremental (used on launch)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantos_platform.models.features_v11 import build_feature_frame  # noqa: E402
from quantos_platform.models.regime_hmm import CausalRegimeHMM  # noqa: E402

warnings.filterwarnings("ignore")

from quantos_platform.core import settings  # noqa: E402

PROJECT_ROOT = settings.ROOT
US_FILE = settings.US_OVERNIGHT_FILE
OUT_DIR = settings.V11_DIR

TARGETS = ["NIFTY_50", "NIFTY_BANK"]
STATE_NAMES = ["BEAR", "SIDE", "BULL"]
BAND_PCT = 0.25
REFIT_EVERY = 63
TRAIN_START = "2010-04-01"
VALIDATION = ("2016-01-01", "2024-10-31")
TEST_START = "2024-11-01"

US_TICKERS = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^VIX": "US_VIX"}
US_COLS = ["us_sp", "us_nq", "us_vix"]
EOD_COLS = ["tgt_z1", "tgt_gap", "breadth_up", "vix_chg1", "tom", "tgt_ibs"]
HMM_COLS = ["hmm_p_bear", "hmm_p_bull"]
MEMBERS = {"us": US_COLS, "us_eod_hmm": US_COLS + EOD_COLS + HMM_COLS}
C_REG = 0.1

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")


# ============================================================
# US OVERNIGHT DATA (yfinance; Upstox has no US instruments)
# ============================================================

def _completed_us_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a US bar whose session has not closed yet (16:15 New York)."""
    now_ny = pd.Timestamp.now(tz=NY)
    cutoff = now_ny.normalize() + pd.Timedelta(hours=16, minutes=15)
    last_complete = now_ny.normalize() if now_ny >= cutoff else now_ny.normalize() - pd.Timedelta(days=1)
    return df[df.index <= last_complete.tz_localize(None)]


def update_us_data() -> pd.DataFrame:
    import yfinance as yf

    existing = pd.read_parquet(US_FILE) if US_FILE.exists() else pd.DataFrame()
    start = "2007-01-01" if existing.empty else (existing.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    raw = yf.download(list(US_TICKERS), start=start, auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=US_TICKERS)[list(US_TICKERS.values())]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = _completed_us_sessions(raw.dropna(how="all"))

    merged = pd.concat([existing, raw]) if not existing.empty else raw
    merged = merged[~merged.index.duplicated(keep="last")].sort_index().dropna()
    US_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(US_FILE)
    return merged


def load_us() -> pd.DataFrame:
    return _completed_us_sessions(pd.read_parquet(US_FILE))


def add_us_overnight(df: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """US log-move between India close on T and India open on next_date."""
    out = df.copy()
    for col, name in zip(["SP500", "NASDAQ", "US_VIX"], US_COLS):
        s = np.log(us[col].dropna())
        d, v = s.index.values, s.values
        before = np.searchsorted(d, out.index.values, side="left") - 1
        after = np.searchsorted(d, out["next_date"].values.astype("datetime64[ns]"), side="left") - 1
        ok = (after > before) & (before >= 0)
        out[name] = np.where(ok, v[np.clip(after, 0, None)] - v[np.clip(before, 0, None)], 0.0)
    # Is the US session right after India's close on T already in?
    us_dates = us.index.values
    pos = np.searchsorted(us_dates, out.index.values, side="left")
    has_after = pos < len(us_dates)
    first_after = np.where(has_after, us_dates[np.clip(pos, 0, len(us_dates) - 1)], np.datetime64("NaT"))
    out["us_final"] = has_after & (first_after < out["next_date"].values.astype("datetime64[ns]"))
    return out


# ============================================================
# DESIGN MATRIX
# ============================================================

def _next_weekday(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = ts + pd.Timedelta(days=1)
    while nxt.dayofweek >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def design_frame(target: str, us: pd.DataFrame) -> pd.DataFrame:
    df = build_feature_frame(target)
    df = df[df.index >= TRAIN_START].copy()
    # Latest row: next session not known yet -> assume next weekday.
    if pd.isna(df["next_date"].iloc[-1]):
        df.loc[df.index[-1], "next_date"] = _next_weekday(df.index[-1])
    df = add_us_overnight(df, us)
    df[EOD_COLS] = df[EOD_COLS].ffill()
    df = df.dropna(subset=EOD_COLS)
    r = df["next_ret_pct"]
    df["label"] = np.where(r > BAND_PCT, 2, np.where(r < -BAND_PCT, 0, 1))
    df.loc[r.isna(), "label"] = -1
    return df


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    probs = []
    for cols in MEMBERS.values():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=3000))
        m.fit(train[cols], train["label"].astype(int))
        p = np.zeros((len(test), 3))
        p[:, m.classes_.astype(int)] = m.predict_proba(test[cols])
        probs.append(p)
    return np.mean(probs, axis=0)


def predict_block(df: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    """Predict rows [start, stop) using only rows < start with known outcomes."""
    train_end = start
    hmm = CausalRegimeHMM().fit(df.iloc[:train_end])
    post = hmm.transform(df.iloc[:stop])
    X = pd.concat([df.iloc[:stop], post], axis=1)
    train = X.iloc[:train_end]
    train = train[train["label"] >= 0]
    test = X.iloc[start:stop]
    p = _fit_predict(train, test)
    hmm_state = post.iloc[start:stop].to_numpy().argmax(1)
    return pd.DataFrame(
        {
            "target_trading_date": test["next_date"].values,
            "current_regime": [STATE_NAMES[i] for i in hmm_state],
            "bear_probability": p[:, 0],
            "side_probability": p[:, 1],
            "bull_probability": p[:, 2],
            "next_regime": [STATE_NAMES[i] for i in p.argmax(1)],
            "next_day_probability": p.max(1),
            # Historical rows (outcome known) are final even across US holidays.
            "status": np.where(
                test["us_final"].values | (test["label"].values >= 0),
                "FINAL",
                "PROVISIONAL",
            ),
        },
        index=test.index.rename("prediction_date"),
    )


# ============================================================
# HISTORY: full walk-forward rebuild / incremental update
# ============================================================

def history_path(target: str) -> Path:
    return OUT_DIR / f"prediction_history_v11_{target.lower()}.csv"


def rebuild_history(target: str, us: pd.DataFrame, start: str = VALIDATION[0]) -> pd.DataFrame:
    df = design_frame(target, us)
    first = int(np.searchsorted(df.index.values, np.datetime64(start)))
    blocks = [
        predict_block(df, s, min(s + REFIT_EVERY, len(df)))
        for s in range(first, len(df), REFIT_EVERY)
    ]
    hist = pd.concat(blocks)
    hist["model_version"] = "V11"
    return hist


def update_history(target: str, us: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append predictions for new rows; refresh rows still PROVISIONAL/unresolved."""
    path = history_path(target)
    if not path.exists():
        hist = rebuild_history(target, us)
        return hist, len(hist)

    old = pd.read_csv(path, parse_dates=["prediction_date", "target_trading_date"]).set_index("prediction_date")
    df = design_frame(target, us)

    realised = df["label"] >= 0
    keep = old[(old["status"] == "FINAL") & old.index.isin(df.index[realised])]
    todo = df.index[~df.index.isin(keep.index) & (df.index > old.index.min())]
    if len(todo) == 0:
        return old, 0

    start = int(df.index.get_loc(todo[0]))
    new = predict_block(df, start, len(df))
    new = new[new.index.isin(todo)]
    new["model_version"] = "V11"
    hist = pd.concat([keep, new]).sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    return hist, len(new)


def save_history(target: str, hist: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = hist.copy()
    out.index = pd.to_datetime(out.index)
    frame = out.reset_index()
    from quantos_platform.core.validation import check_prediction_history, require
    require(check_prediction_history(frame.rename(columns={frame.columns[0]: "prediction_date"})),
            f"V11 history {target}")
    tmp = history_path(target).with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(history_path(target))


# ============================================================
# METRICS
# ============================================================

def _score(pred: pd.Series, ret: pd.Series) -> dict:
    ok = ret.notna()
    pred, ret = pred[ok], ret[ok]
    actual3 = np.where(ret > BAND_PCT, "BULL", np.where(ret < -BAND_PCT, "BEAR", "SIDE"))
    directional = pred != "SIDE"
    hit = ((pred == "BULL") & (ret > 0)) | ((pred == "BEAR") & (ret < 0))
    return {
        "n": int(len(pred)),
        "direction_accuracy": float(hit[directional].mean()) if directional.any() else None,
        "direction_calls": int(directional.sum()),
        "three_class_accuracy": float((pred.values == actual3).mean()),
        "always_up_accuracy": float((ret > 0).mean()),
    }


def compute_metrics(target: str, hist: pd.DataFrame) -> dict:
    df = build_feature_frame(target)
    ret = df["next_ret_pct"]
    h = hist.copy()
    h["ret"] = ret.reindex(h.index)
    metrics = {"target": target, "band_pct": BAND_PCT}
    for name, (a, b) in {
        "validation": VALIDATION,
        "test": (TEST_START, "2100-01-01"),
    }.items():
        w = h[(h.index >= a) & (h.index <= b)]
        metrics[name] = {"start": str(w.index.min().date()) if len(w) else None,
                         "end": str(w.index.max().date()) if len(w) else None,
                         **_score(w["next_regime"], w["ret"])}

    # V10 baseline on the same test rows, from its prediction log.
    v10_file = settings.DATA_DIR / "regime" / "live_predictions" / f"prediction_history_{target.lower()}.csv"
    if v10_file.exists():
        v10 = pd.read_csv(v10_file, parse_dates=["prediction_date"]).set_index("prediction_date")
        common = v10.index.intersection(h.index[h.index >= TEST_START])
        if len(common):
            metrics["v10_same_rows"] = _score(v10.loc[common, "next_regime"], ret.reindex(common))
            metrics["v11_same_rows"] = _score(h.loc[common, "next_regime"], ret.reindex(common))
    metrics["generated_at"] = pd.Timestamp.now(tz=IST).isoformat()
    return metrics


# ============================================================
# ENTRY POINTS
# ============================================================

def needs_update() -> bool:
    """New India session, missing artifacts, or a forecast still waiting for the US close."""
    for target in TARGETS:
        metrics = OUT_DIR / f"metrics_v11_{target.lower()}.json"
        daily = settings.DAILY_DIR / f"{target.lower()}.parquet"
        if not metrics.exists():
            return True
        latest = json.loads(metrics.read_text()).get("latest", {})
        if latest.get("status") != "FINAL":
            return True
        last_session = pd.read_parquet(daily, columns=["timestamp"])["timestamp"].max()
        last_session = pd.Timestamp(last_session)
        last_session = (last_session.tz_convert(settings.IST) if last_session.tzinfo
                        else last_session).date().isoformat()
        if latest.get("prediction_date", "") < last_session:
            return True
    return False


def run(rebuild: bool = False, refresh_us: bool = True) -> dict:
    us = update_us_data() if refresh_us else load_us()
    summary = {}
    for target in TARGETS:
        if rebuild:
            hist, n_new = rebuild_history(target, us), None
        else:
            hist, n_new = update_history(target, us)
        save_history(target, hist)
        metrics = compute_metrics(target, hist)
        latest = hist.iloc[-1]
        metrics["latest"] = {
            "prediction_date": str(hist.index[-1].date()),
            "target_trading_date": str(pd.Timestamp(latest["target_trading_date"]).date()),
            "next_regime": latest["next_regime"],
            "next_day_probability": float(latest["next_day_probability"]),
            "bull_probability": float(latest["bull_probability"]),
            "side_probability": float(latest["side_probability"]),
            "bear_probability": float(latest["bear_probability"]),
            "status": latest["status"],
        }
        (OUT_DIR / f"metrics_v11_{target.lower()}.json").write_text(json.dumps(metrics, indent=2))
        summary[target] = {"new_rows": n_new, **metrics}
    return summary


if __name__ == "__main__":
    result = run(rebuild="--rebuild" in sys.argv)
    for target, m in result.items():
        print("=" * 70)
        print(f"{target}  (new rows: {m['new_rows']})")
        for k in ("validation", "test", "v10_same_rows", "v11_same_rows"):
            if k in m:
                print(f"  {k:15} {json.dumps(m[k])}")
        print(f"  latest          {json.dumps(m['latest'])}")
