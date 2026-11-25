"""
Systemic & Tail Risk: how fragile the market is and how bad bad days get. Measurements and history only.

Systemic (NIFTY sector indices):
    average pairwise correlation (63 sessions), absorption ratio (share of risk in the top 2 principal
    components, 252 sessions; Kritzman et al. 2011), turbulence (Mahalanobis distance of today's sector moves
    from the trailing year; Kritzman & Li 2010), cross-sector dispersion, current clustered correlation matrix,
    sector scoreboard
Tail (NIFTY 50, NIFTY BANK, SENSEX):
    1-day VaR / CVaR (historical, and GARCH-t filtered historical simulation), a walk-forward VaR backtest
    (exceedance rate + Kupiec test), Hill tail index, 21-session Monte Carlo from the GARCH-t model,
    and the worst historical drawdowns available for crash replays.

Outputs (data/analytics/): risk.json, risk_series.parquet, risk_returns.parquet (daily returns for replays).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from quantos_platform.analytics import TRADING_DAYS, load_daily, write_frame, write_json

SECTORS = {"NIFTY_AUTO": "Auto", "NIFTY_BANK": "Bank", "NIFTY_ENERGY": "Energy", "NIFTY_FIN_SERVICE": "Financials",
           "NIFTY_FMCG": "FMCG", "NIFTY_IT": "IT", "NIFTY_MEDIA": "Media", "NIFTY_METAL": "Metal",
           "NIFTY_PHARMA": "Pharma", "NIFTY_PSU_BANK": "PSU Bank", "NIFTY_REALTY": "Realty"}
INDICES = {"NIFTY_50": "NIFTY 50", "NIFTY_BANK": "NIFTY BANK", "SENSEX": "SENSEX",
           "NIFTY_MIDCAP_100": "NIFTY Midcap 100", "NIFTY_SMALLCAP_100": "NIFTY Smallcap 100"}
TAIL_SYMBOLS = ["NIFTY_50", "NIFTY_BANK", "SENSEX"]
CORR_WIN, AR_WIN, TURB_WIN, N_PC = 63, 252, 252, 2
MC_PATHS, MC_DAYS, SEED = 10_000, 21, 7


def returns_panel() -> pd.DataFrame:
    cols = {}
    for sym in {**SECTORS, **INDICES}:
        try:
            cols[sym] = np.log(load_daily(sym)["close"]).diff()
        except (OSError, KeyError):
            continue
    return pd.DataFrame(cols)


# ── systemic ──────────────────────────────────────────────────────────────────
def systemic(r: pd.DataFrame) -> pd.DataFrame:
    x = r[[c for c in SECTORS if c in r]].dropna()
    vals, idx = x.to_numpy(), x.index
    n, k = vals.shape
    iu = np.triu_indices(k, 1)
    avg_corr, absorb, turb = (np.full(n, np.nan) for _ in range(3))
    for t in range(n):
        if t >= CORR_WIN:
            c = np.corrcoef(vals[t - CORR_WIN + 1:t + 1].T)
            avg_corr[t] = c[iu].mean()
        if t >= AR_WIN:
            w = vals[t - AR_WIN + 1:t + 1]
            ev = np.linalg.eigvalsh(np.cov(w.T))[::-1]
            absorb[t] = ev[:N_PC].sum() / ev.sum()
        if t >= TURB_WIN:
            w = vals[t - TURB_WIN:t]                       # trailing year, excluding today
            d = vals[t] - w.mean(0)
            turb[t] = float(d @ np.linalg.pinv(np.cov(w.T)) @ d) / k
    ret21 = x.rolling(21).sum()
    out = pd.DataFrame({"avg_corr": avg_corr, "absorption": absorb, "turbulence": turb,
                        "dispersion": ret21.std(axis=1).to_numpy()}, index=idx)
    out["turb_p95_expanding"] = out["turbulence"].expanding(252).quantile(0.95).shift(1)
    return out


def clustered_corr(r: pd.DataFrame, window: int = CORR_WIN) -> dict:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    x = r[[c for c in SECTORS if c in r]].dropna().iloc[-window:]
    c = x.corr()
    dist = np.clip(1 - c.to_numpy(), 0, 2)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "average"))
    names = [SECTORS[c.columns[i]] for i in order]
    return {"names": names, "matrix": c.iloc[order, order].to_numpy().round(3).tolist()}


def turbulence_base_rates(sysdf: pd.DataFrame, nifty: pd.Series) -> dict:
    """After turbulent days (above the expanding 95th percentile, known at the time) vs all days."""
    r = np.log(nifty).diff()
    fwd_vol = r[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(TRADING_DAYS)
    fwd_ret = nifty.shift(-21) / nifty - 1
    s = sysdf.join(pd.DataFrame({"fwd_vol": fwd_vol, "fwd_ret": fwd_ret}), how="left").dropna(
        subset=["turbulence", "turb_p95_expanding", "fwd_vol", "fwd_ret"])
    hot = s["turbulence"] > s["turb_p95_expanding"]
    return {"n_turbulent": int(hot.sum()), "n_all": int(len(s)),
            "fwd_vol_turbulent": float(s.loc[hot, "fwd_vol"].mean()), "fwd_vol_all": float(s["fwd_vol"].mean()),
            "fwd_ret_turbulent": float(s.loc[hot, "fwd_ret"].mean()), "fwd_ret_all": float(s["fwd_ret"].mean()),
            "fwd_down10_turbulent": float((s.loc[hot, "fwd_ret"] < -0.10).mean()),
            "fwd_down10_all": float((s["fwd_ret"] < -0.10).mean())}


def scoreboard(r: pd.DataFrame) -> list[dict]:
    rows = []
    for sym, name in SECTORS.items():
        try:
            c = load_daily(sym)["close"]
        except (OSError, KeyError):
            continue
        hi, lo = c.iloc[-252:].max(), c.iloc[-252:].min()
        rows.append({"symbol": sym, "name": name, "w1": float(c.iloc[-1] / c.iloc[-6] - 1),
                     "m1": float(c.iloc[-1] / c.iloc[-22] - 1), "m3": float(c.iloc[-1] / c.iloc[-64] - 1),
                     "m6": float(c.iloc[-1] / c.iloc[-127] - 1), "y1": float(c.iloc[-1] / c.iloc[-253] - 1),
                     "range52": float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else None,
                     "off_high": float(c.iloc[-1] / hi - 1),          # vs the 52-week high
                     "vol21": float(np.log(c).diff().iloc[-21:].std() * np.sqrt(TRADING_DAYS))})
    return rows


# ── tail ──────────────────────────────────────────────────────────────────────
def hill_tail_index(losses: np.ndarray, tail_frac: float = 0.05) -> float:
    x = np.sort(losses[losses > 0])[::-1]
    k = max(int(len(losses) * tail_frac), 10)
    x = x[:k + 1]
    return float(1 / np.mean(np.log(x[:k] / x[k])))


def kupiec(n: int, x: int, p: float) -> float:
    """Kupiec proportion-of-failures p-value: is the exceedance count consistent with the VaR level?"""
    if n == 0:
        return float("nan")
    phat = x / n
    if phat in (0, 1):
        ll1 = 0.0
    else:
        ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    return float(1 - sps.chi2.cdf(-2 * (ll0 - ll1), 1))


def tail(symbol: str) -> dict:
    from arch import arch_model

    close = load_daily(symbol)["close"]
    r = np.log(close).diff().dropna()
    simple = np.expm1(r)
    rp = r * 100
    # walk-forward FHS VaR backtest: refit yearly on the prior 10 years, filter the year with fixed params
    years = sorted(set(rp.index.year))
    var_rows = []
    for y in years:
        train = rp[(rp.index.year < y) & (rp.index.year >= y - 10)]
        if len(train) < 3 * TRADING_DAYS:
            continue
        res = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        upto = rp[rp.index.year <= y].iloc[-(len(train) + (rp.index.year == y).sum()):]
        fixed = arch_model(upto, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fix(res.params)
        sig = fixed.conditional_volatility                     # sigma_t uses data up to t-1 only
        z = (train - res.params["mu"]) / res.conditional_volatility
        q95, q99 = np.quantile(z, 0.05), np.quantile(z, 0.01)
        yr = upto.index.year == y
        mu = res.params["mu"]
        var_rows.append(pd.DataFrame({"r": upto[yr] / 100, "var95": -(mu + sig[yr] * q95) / 100,
                                      "var99": -(mu + sig[yr] * q99) / 100}))
    bt = pd.concat(var_rows) if var_rows else pd.DataFrame(columns=["r", "var95", "var99"])
    n = len(bt)
    x95, x99 = int((bt["r"] < -bt["var95"]).sum()), int((bt["r"] < -bt["var99"]).sum())

    # today's FHS VaR from the last 10 years
    last = rp.iloc[-10 * TRADING_DAYS:]
    res = arch_model(last, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    z = ((last - res.params["mu"]) / res.conditional_volatility).to_numpy()
    sig_next = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]))
    sims = (res.params["mu"] + sig_next * z) / 100
    fhs = {lvl: {"var": float(-np.quantile(sims, 1 - lvl)), "cvar": float(-sims[sims <= np.quantile(sims, 1 - lvl)].mean())}
           for lvl in (0.95, 0.99)}
    hist = {}
    for lvl in (0.95, 0.99):
        for label, x in (("all", simple), ("2y", simple.iloc[-2 * TRADING_DAYS:])):
            q = np.quantile(x, 1 - lvl)
            hist[f"{label}_{int(lvl * 100)}"] = {"var": float(-q), "cvar": float(-x[x <= q].mean())}

    # 21-session Monte Carlo from the fitted GARCH-t (risk range, not a forecast)
    sim = res.forecast(horizon=MC_DAYS, method="simulation", simulations=MC_PATHS, reindex=False,
                       random_state=np.random.default_rng(SEED))
    paths = sim.simulations.values[-1] / 100                   # (paths, days) log returns
    tot = np.expm1(paths.sum(1))
    mc = {"p01": float(np.quantile(tot, 0.01)), "p05": float(np.quantile(tot, 0.05)),
          "p25": float(np.quantile(tot, 0.25)), "p50": float(np.quantile(tot, 0.50)),
          "p75": float(np.quantile(tot, 0.75)), "p95": float(np.quantile(tot, 0.95)),
          "down5": float((tot < -0.05).mean()), "down10": float((tot < -0.10).mean()),
          "hist": np.histogram(tot, bins=60)[0].tolist(), "edges": np.histogram(tot, bins=60)[1].tolist()}
    losses = -simple.to_numpy()
    return {"symbol": symbol, "name": INDICES.get(symbol, symbol), "close": float(close.iloc[-1]),
            "fhs": {str(k): v for k, v in fhs.items()}, "hist": hist, "sigma_next": sig_next / 100,
            "backtest": {"n": n, "start": bt.index[0].strftime("%Y-%m-%d") if n else None,
                         "x95": x95, "rate95": x95 / n if n else None, "p95": kupiec(n, x95, 0.05),
                         "x99": x99, "rate99": x99 / n if n else None, "p99": kupiec(n, x99, 0.01)},
            "hill_alpha": hill_tail_index(losses), "worst_days": [
                {"date": d.strftime("%Y-%m-%d"), "ret": float(v)} for d, v in simple.nsmallest(5).items()],
            "mc": mc, "garch_nu": float(res.params["nu"])}


def build() -> dict:
    from quantos_platform.analytics.states import drawdown_episodes

    r = returns_panel()
    sysdf = systemic(r)
    write_frame("risk_series.parquet", sysdf)
    write_frame("risk_returns.parquet", r.dropna(how="all"))
    nifty = load_daily("NIFTY_50")["close"]
    last = sysdf.dropna(subset=["avg_corr"]).iloc[-1]

    def pctile(col):
        s = sysdf[col].dropna()
        return float((s <= s.iloc[-1]).mean())

    crashes = drawdown_episodes(nifty, min_depth=0.15)
    out = {
        "as_of": sysdf.index[-1].strftime("%Y-%m-%d"), "sectors": SECTORS, "indices": INDICES,
        "now": {"avg_corr": float(last["avg_corr"]), "avg_corr_pct": pctile("avg_corr"),
                "absorption": float(sysdf["absorption"].dropna().iloc[-1]), "absorption_pct": pctile("absorption"),
                "absorption_shift": float(sysdf["absorption"].dropna().iloc[-1]
                                          - sysdf["absorption"].dropna().iloc[-252:].mean()),
                "turbulence": float(sysdf["turbulence"].dropna().iloc[-1]), "turbulence_pct": pctile("turbulence"),
                "turbulence_p95": float(sysdf["turbulence"].dropna().quantile(0.95)),
                "turbulent_days_21": int((sysdf["turbulence"].iloc[-21:] > sysdf["turb_p95_expanding"].iloc[-21:]).sum()),
                "dispersion": float(sysdf["dispersion"].dropna().iloc[-1]), "dispersion_pct": pctile("dispersion")},
        "turbulence_base": turbulence_base_rates(sysdf, nifty),
        "corr_matrix": clustered_corr(r),
        "scoreboard": scoreboard(r),
        "tail": {sym: tail(sym) for sym in TAIL_SYMBOLS},
        "crashes": [{"peak": c["peak"].strftime("%Y-%m-%d"), "trough": c["trough"].strftime("%Y-%m-%d"),
                     "depth": float(c["depth"]), "days_down": int(c["days_down"])}
                    for c in crashes.sort_values("depth").head(6).to_dict("records")],
    }
    write_json("risk.json", out)
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    o = build()
    print(f"{time.time() - t:.0f}s", o["now"])
    print(o["turbulence_base"])
    for s, v in o["tail"].items():
        print(s, v["fhs"], v["backtest"], round(v["hill_alpha"], 2), {k: round(v["mc"][k], 3) for k in ("p05", "p50", "p95")})
    print(o["crashes"])
