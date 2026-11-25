"""
Market States: a descriptive (not predictive) view of HMM regimes and drawdowns.

Regime labels are look-ahead free. The 3-state Gaussian HMM (daily return, 20-day volatility, India VIX change)
is refit every January on all data before that year, and each day's state comes from the FORWARD filter only
(p(state_t | data up to t)). A state switch needs the new state's probability to reach SWITCH_PROB (hysteresis),
which stops one-day flickers. States are ordered by their volatility (Calm, Normal, Stressed): they describe how
the market is moving, not which way. (Ordering by mean return mislabels violent rebound days as "bull".)

Outputs (data/analytics/):
    states_<sym>.json            current state, transition matrix with bootstrap bands, durations, survival,
                                 state-conditional behaviour, drawdown episodes and base rates
    states_<sym>_series.parquet  daily close, filtered probabilities, state label, drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.analytics import TRADING_DAYS, load_daily, write_frame, write_json
from quantos_platform.analytics.volatility import load_vix

STATE_SYMBOLS = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK"}
STATES = ["CALM", "NORMAL", "STRESSED"]
SWITCH_PROB = 0.6
MIN_TRAIN_YEARS = 2
N_BOOT, BLOCK, SEED = 1000, 63, 7
SURV_MAX = 250
FWD = 21
DD_LEVELS = (0.05, 0.10, 0.20, 0.30)


# ── regimes ───────────────────────────────────────────────────────────────────
def features(symbol: str) -> pd.DataFrame:
    df = load_daily(symbol)
    vix = load_vix()
    r = np.log(df["close"]).diff()
    f = pd.DataFrame({"tgt_r1": r, "tgt_vol20": r.rolling(20).std(),
                      "vix_chg1": np.log(vix).diff().reindex(df.index)}, index=df.index)
    return f.dropna()


def walk_forward_probs(f: pd.DataFrame) -> pd.DataFrame:
    """Refit each January on prior years, forward-filter through that year. No future data in any row."""
    from quantos_platform.models.regime_hmm import CausalRegimeHMM

    years = sorted(set(f.index.year))
    out = []
    for y in years[MIN_TRAIN_YEARS:]:
        train = f[f.index.year < y]
        upto = f[f.index.year <= y]
        hmm = CausalRegimeHMM().fit(train)
        if hmm.model is not None:
            hmm.order = np.argsort(hmm.model.means_[:, 1])     # by mean (standardised) 20-day volatility
        probs = hmm.transform(upto)
        out.append(probs[probs.index.year == y])
    p = pd.concat(out)
    p.columns = ["p_calm", "p_normal", "p_stressed"]
    return p


def label(probs: pd.DataFrame, switch: float = SWITCH_PROB) -> pd.Series:
    """Causal hysteresis: keep the current state until another state's filtered probability reaches `switch`."""
    arr = probs[["p_calm", "p_normal", "p_stressed"]].to_numpy()
    cur = int(arr[0].argmax())
    lab = np.empty(len(arr), dtype=int)
    for i, row in enumerate(arr):
        top = int(row.argmax())
        if top != cur and row[top] >= switch:
            cur = top
        lab[i] = cur
    return pd.Series(np.array(STATES)[lab], index=probs.index, name="state")


def episodes(states: pd.Series) -> pd.DataFrame:
    run = (states != states.shift()).cumsum()
    g = states.groupby(run)
    ep = pd.DataFrame({"state": g.first(), "start": g.apply(lambda s: s.index[0]),
                       "end": g.apply(lambda s: s.index[-1]), "days": g.size()}).reset_index(drop=True)
    ep["next"] = ep["state"].shift(-1)
    ep["ongoing"] = False
    ep.loc[ep.index[-1], "ongoing"] = True
    return ep


def transition_matrix(states: pd.Series) -> np.ndarray:
    idx = {s: i for i, s in enumerate(STATES)}
    a = states.map(idx).to_numpy()
    m = np.zeros((3, 3))
    np.add.at(m, (a[:-1], a[1:]), 1)
    with np.errstate(invalid="ignore"):
        return m / m.sum(1, keepdims=True)


def _counts(a: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3))
    np.add.at(m, (a[:-1], a[1:]), 1)
    return m


def bootstrap_transitions(states: pd.Series, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED) -> dict:
    """Moving-block bootstrap -> 90% bands for transition probabilities and implied durations. Transitions are
    counted inside each block only, so joining blocks does not invent switches."""
    rng = np.random.default_rng(seed)
    a = states.map({s: i for i, s in enumerate(STATES)}).to_numpy()
    n = len(a)
    blocks = np.array([_counts(a[i:i + block]) for i in range(n - block)])
    mats = []
    for _ in range(n_boot):
        m = blocks[rng.integers(0, len(blocks), size=n // block + 1)].sum(0)
        with np.errstate(invalid="ignore"):
            mats.append(m / m.sum(1, keepdims=True))
    mats = np.array(mats)
    lo, hi = np.nanpercentile(mats, 5, axis=0), np.nanpercentile(mats, 95, axis=0)
    dur = 1 / (1 - np.clip(mats[:, range(3), range(3)], 0, 0.9999))
    return {"lo": lo.tolist(), "hi": hi.tolist(),
            "dur_lo": np.nanpercentile(dur, 5, axis=0).tolist(), "dur_hi": np.nanpercentile(dur, 95, axis=0).tolist()}


def survival(ep: pd.DataFrame, state: str, kmax: int = SURV_MAX) -> list[float]:
    """Kaplan-Meier: share of episodes of `state` still running after k sessions (ongoing one censored)."""
    d = ep[ep["state"] == state]
    dur, done = d["days"].to_numpy(), (~d["ongoing"]).to_numpy()
    s, out = 1.0, []
    for k in range(1, kmax + 1):
        at_risk = (dur >= k).sum()
        ended = ((dur == k) & done).sum()
        if at_risk:
            s *= 1 - ended / at_risk
        out.append(float(s))
    return out


def conditional_stats(close: pd.Series, states: pd.Series) -> dict:
    r = np.log(close).diff()
    nxt = r.shift(-1).reindex(states.index)              # return of the session AFTER the labelled day
    fwd = (close.shift(-FWD) / close - 1).reindex(states.index)
    out = {}
    for s in STATES:
        m = states == s
        x, f = nxt[m].dropna(), fwd[m].dropna()
        out[s] = {"share_of_days": float(m.mean()), "days": int(m.sum()),
                  "next_day_mean": float(x.mean()), "next_day_up": float((x > 0).mean()),
                  "vol_annual": float(x.std() * np.sqrt(TRADING_DAYS)),
                  "worst_day": float(np.expm1(x.min())), "best_day": float(np.expm1(x.max())),
                  "big_down_share": float((x < np.log(0.98)).mean()),
                  "fwd21_mean": float(f.mean()), "fwd21_median": float(f.median()), "fwd21_up": float((f > 0).mean()),
                  "fwd21_p10": float(f.quantile(0.10)), "fwd21_p90": float(f.quantile(0.90)), "fwd21_n": int(len(f))}
    allx = r.shift(-1).reindex(states.index).dropna()
    allf = fwd.dropna()
    out["ALL"] = {"next_day_mean": float(allx.mean()), "next_day_up": float((allx > 0).mean()),
                  "vol_annual": float(allx.std() * np.sqrt(TRADING_DAYS)),
                  "big_down_share": float((allx < np.log(0.98)).mean()),
                  "fwd21_mean": float(allf.mean()), "fwd21_median": float(allf.median()),
                  "fwd21_up": float((allf > 0).mean()), "fwd21_p10": float(allf.quantile(0.10)),
                  "fwd21_p90": float(allf.quantile(0.90))}
    return out


# ── drawdowns ─────────────────────────────────────────────────────────────────
def drawdown_episodes(close: pd.Series, min_depth: float = 0.05) -> pd.DataFrame:
    """Peak -> trough -> recovery (close back at the old peak) for every fall deeper than min_depth."""
    peak = close.cummax()
    dd = close / peak - 1
    rows, i, n, idx = [], 0, len(close), close.index
    while i < n:
        if dd.iloc[i] < 0:
            j = i
            while j < n and dd.iloc[j] < 0:
                j += 1
            seg = dd.iloc[i:j]
            trough = seg.idxmin()
            depth = float(seg.min())
            if -depth >= min_depth:
                peak_day = idx[i - 1] if i > 0 else idx[0]
                rec = idx[j] if j < n else None
                rows.append({"peak": peak_day, "trough": trough, "recovered": rec, "depth": depth,
                             "peak_level": float(close.loc[peak_day]), "trough_level": float(close.loc[trough]),
                             "days_down": int(close.index.get_loc(trough) - close.index.get_loc(peak_day)),
                             "days_to_recover": (int(close.index.get_loc(rec) - close.index.get_loc(trough))
                                                 if rec is not None else None),
                             "ongoing": rec is None})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)


def drawdown_stage(dd: float) -> str:
    x = -dd
    return ("At or near the high" if x < 0.05 else "Pullback (5–10% off the high)" if x < 0.10 else
            "Correction (10–20% off the high)" if x < 0.20 else "Bear market (20%+ off the high)")


def drawdown_base_rates(ep: pd.DataFrame) -> list[dict]:
    """Of the falls that reached level a, how many went on to reach level b (completed episodes only)."""
    done = ep[~ep["ongoing"]]
    out = []
    for a, b in zip(DD_LEVELS[:-1], DD_LEVELS[1:]):
        reached = done[-done["depth"] >= a]
        out.append({"from": a, "to": b, "n": int(len(reached)),
                    "went_on": int((-reached["depth"] >= b).sum()),
                    "share": float((-reached["depth"] >= b).mean()) if len(reached) else None,
                    "median_recover_days": (float(reached["days_to_recover"].median()) if len(reached) else None)})
    return out


# ── build ─────────────────────────────────────────────────────────────────────
def build(symbol: str) -> dict:
    df = load_daily(symbol)
    close = df["close"]
    probs = walk_forward_probs(features(symbol))
    states = label(probs)
    ep = episodes(states)
    mat = transition_matrix(states)
    boot = bootstrap_transitions(states)
    cur = ep.iloc[-1]
    surv = {s: survival(ep, s) for s in STATES}
    streak = int(cur["days"])

    def still(k):
        sv = surv[cur["state"]]
        base = sv[streak - 1] if streak - 1 < len(sv) else None
        later = sv[streak - 1 + k] if streak - 1 + k < len(sv) else None
        return float(later / base) if base and later is not None else None

    full_dd = close / close.cummax() - 1
    dde = drawdown_episodes(close)
    ser = pd.DataFrame({"close": close}).join(probs, how="left").join(states, how="left")
    ser["drawdown"] = full_dd
    write_frame(f"states_{symbol.lower()}_series.parquet", ser)

    completed = ep[~ep["ongoing"]]
    summary = {
        "symbol": symbol, "name": STATE_SYMBOLS.get(symbol, symbol), "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "since": states.index[0].strftime("%Y-%m-%d"), "switch_prob": SWITCH_PROB,
        "current": {"state": cur["state"], "since": cur["start"].strftime("%Y-%m-%d"), "streak": streak,
                    "probs": {s: float(probs.iloc[-1][f"p_{s.lower()}"]) for s in STATES},
                    "still_after_5": still(5), "still_after_21": still(21)},
        "transition": {"states": STATES, "p": mat.tolist(), **boot,
                       "duration": [float(1 / (1 - mat[i, i])) if mat[i, i] < 1 else None for i in range(3)]},
        "episodes": {s: {"n": int((completed["state"] == s).sum()),
                         "median_days": float(completed.loc[completed["state"] == s, "days"].median())
                         if (completed["state"] == s).any() else None,
                         "longest": int(completed.loc[completed["state"] == s, "days"].max())
                         if (completed["state"] == s).any() else None,
                         "next": {t: int(((completed["state"] == s) & (completed["next"] == t)).sum()) for t in STATES}}
                     for s in STATES},
        "survival": {s: v[:120] for s, v in surv.items()},
        "conditional": conditional_stats(close, states),
        "recent_episodes": [{"state": r.state, "start": r.start.strftime("%Y-%m-%d"),
                             "end": r.end.strftime("%Y-%m-%d"), "days": int(r.days), "ongoing": bool(r.ongoing)}
                            for r in ep.tail(12).itertuples()],
        "drawdown": {"now": float(full_dd.iloc[-1]), "stage": drawdown_stage(float(full_dd.iloc[-1])),
                     "peak": float(close.cummax().iloc[-1]), "since_data": close.index[0].strftime("%Y-%m-%d"),
                     "episodes": [{k: (None if v is None or (not isinstance(v, str) and pd.isna(v)) else
                                       v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                                   for k, v in r.items()} for r in dde.to_dict("records")],
                     "base_rates": drawdown_base_rates(dde)},
    }
    write_json(f"states_{symbol.lower()}.json", summary)
    return summary


if __name__ == "__main__":
    import time
    for sym in STATE_SYMBOLS:
        t = time.time()
        s = build(sym)
        print(sym, s["current"], f"{time.time() - t:.0f}s")
        print(" P:", np.round(s["transition"]["p"], 3).tolist(), "dur", np.round(s["transition"]["duration"], 1))
        print(" cond:", {k: (round(v["next_day_mean"] * 1e4, 1), round(v["vol_annual"], 3)) for k, v in s["conditional"].items()})
        print(" dd:", s["drawdown"]["stage"], s["drawdown"]["base_rates"])
