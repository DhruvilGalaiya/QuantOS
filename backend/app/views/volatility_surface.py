"""
QuantOS | Volatility Surface page (v2) of the main dashboard

    python -m streamlit run backend/app/regime_dashboard_v10.py   -> "Volatility Surface"

On first visit it refreshes the option snapshot if stale and rebuilds the surfaces
(vol_pipeline.py -> surface_engine.py). Everything shown here comes from
data/regime/volatility_surface/<symbol>_surface_v2.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui  # noqa: E402  (backend/app/ui.py)
from quantos_platform.vol import snapshot as vol_pipeline  # noqa: E402
from quantos_platform.vol import surface as surface_engine  # noqa: E402

SYMBOLS = surface_engine.SYMBOLS
# categorical order (validated palette), assigned to expiries in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = [[0.0, "#dbe9fa"], [0.5, "#2a78d6"], [1.0, "#0d2d57"]]
MUTED = "rgba(148,163,184,0.85)"
GRID = "rgba(148,163,184,0.18)"
PLOT_CFG = {"displaylogo": False}


# ============================================================
# REFRESH + LOAD
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_surface(symbol: str, file_mtime: float) -> dict | None:
    return surface_engine.load(symbol)  # file_mtime is part of the cache key


with st.spinner("Checking the option snapshot…"):
    refresh = ui.refresh_on_launch("vol") or {}
ui.job_health("vol")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Volatility")
    symbol = st.selectbox("Underlying", list(SYMBOLS), format_func=SYMBOLS.get)
    surface_view = st.radio("3D surface", ["Smoothed (SVI)", "SABR fit", "Market quotes only"])
    show_sabr = st.toggle("Show SABR on smile charts", value=True)
    if ui.refresh_button("vol", "Pull latest option data"):
        load_surface.clear()
    _ts = vol_pipeline.latest_snapshot_ts()
    snap = _ts.isoformat() if _ts is not None else None
    snap_text = _ts.strftime("%d %b %Y, %H:%M IST") if _ts is not None else "unknown"
    st.caption(f"Option data as of **{snap_text}**")
    st.caption("Source: Upstox option chain (scheduled `vol` job)")
    if refresh.get("errors"):
        with st.expander("Refresh warnings"):
            for k, v in refresh["errors"].items():
                st.caption(f"{k}: {v}")

_f = surface_engine.OUT_DIR / f"{symbol.lower()}_surface_v2.json"
data = load_surface(symbol, _f.stat().st_mtime if _f.exists() else 0.0)
if not data or not data.get("expiries"):
    st.error("No volatility surface for this underlying yet. It appears after the next `vol` job run.")
    st.stop()

exp = pd.DataFrame(data["expiries"]).sort_values("T").reset_index(drop=True)
pts = pd.DataFrame(data["points"])
colors = {e: SERIES[i % len(SERIES)] for i, e in enumerate(exp["expiry"])}
label = {r.expiry: f"{pd.Timestamp(r.expiry):%d %b} · {r.days:.1f}d" for r in exp.itertuples()}

# reference expiry for headline numbers: first one at least a week out (front weeklies are noisy)
ref = exp[exp["days"] >= 7].iloc[0] if (exp["days"] >= 7).any() else exp.iloc[-1]
rv = data.get("realized_vol_20d")


def pct(x, d=1, signed=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{d}f}" if signed else f"{x * 100:.{d}f}%"


# ============================================================
# HEADER
# ============================================================

st.title(f"{data['display_name']} volatility surface")
market_open = vol_pipeline.market_open_now()
st.caption(
    f"{len(exp)} expiries · {int(exp['n_points'].sum())} option quotes fitted · snapshot {snap_text}"
    + ("" if market_open else " · market closed: these are end-of-day quotes")
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Spot", f"{data['spot']:,.2f}",
          delta=f"forward {ref['forward']:,.0f} ({pd.Timestamp(ref['expiry']):%d %b})", delta_color="off",
          help="Forward = the price the market implies for expiry, from put-call parity (F = K + (C − P)/DF).")
c2.metric(f"ATM IV · {pd.Timestamp(ref['expiry']):%d %b}", pct(ref.get("atm_iv")),
          help="Implied volatility at the forward (log-moneyness 0), read off the fitted SVI smile.")
c3.metric("Realized vol (20d)", pct(rv),
          delta=f"IV − RV {pct((ref.get('atm_iv') or np.nan) - (rv or np.nan), 1, True)} pts", delta_color="off",
          help="Volatility the index actually showed over the last 20 sessions. IV above RV means options "
               "are priced richer than recent movement (the volatility risk premium).")
c4.metric("25Δ risk reversal", f"{pct(ref.get('rr25'), 2, True)} pts",
          help="IV of the 25-delta call minus IV of the 25-delta put. Negative = puts cost more than calls "
               "(downside protection in demand, normal for indices).")
c5.metric("25Δ butterfly", f"{pct(ref.get('bf25'), 2, True)} pts",
          help="Average of the 25-delta call and put IVs minus ATM IV: how curved (convex) the smile is, "
               "i.e. how much extra the market charges for big moves either way.")
n_bad = int((~exp["butterfly_ok"].astype(bool)).sum()) + sum(not c["ok"] for c in data.get("calendar", []))
c6.metric("Arbitrage checks", "All pass" if n_bad == 0 else f"{n_bad} warning(s)",
          help="Butterfly: the fitted smile implies a non-negative probability density. "
               "Calendar: total variance never falls for a later expiry.")


# ============================================================
# 3D SURFACE
# ============================================================

st.subheader("Volatility surface")
fit_pts = pts[pts["used_in_fit"].astype(bool)]
grid = data.get("grid") or {}
fig = go.Figure()
if surface_view != "Market quotes only" and grid:
    z = grid["svi_iv"] if surface_view == "Smoothed (SVI)" else grid.get("sabr_iv")
    if z:
        # Strike for every grid cell: K = (K/F) x forward at that maturity
        # (forward interpolated in log space between the fitted expiries).
        fwd_exp = exp.dropna(subset=["svi"]) if "svi" in exp else exp
        fwd_T = np.interp(np.array(grid["days"]), fwd_exp["days"], np.log(fwd_exp["forward"]))
        strikes = np.exp(fwd_T)[:, None] * np.array(grid["moneyness_fwd"])[None, :]   # (days, k)
        listed = np.sort(pts["strike"].unique())
        step = float(np.median(np.diff(listed))) if len(listed) > 1 else 50.0
        nearest = np.round(strikes / step) * step
        fig.add_trace(go.Surface(
            x=np.array(grid["moneyness_fwd"]) * 100, y=grid["days"], z=np.array(z, dtype=float) * 100,
            # plotly indexes Surface customdata as [x][y] (z is [y][x]), hence the transpose
            customdata=np.dstack([strikes, nearest,
                                  np.exp(fwd_T)[:, None].repeat(strikes.shape[1], 1)]).transpose(1, 0, 2),
            colorscale=SEQ_BLUE, opacity=0.92, showscale=True,
            colorbar=dict(title="IV %", thickness=12, len=0.6),
            hovertemplate="<b>Strike ≈ %{customdata[0]:,.0f}</b> (nearest listed %{customdata[1]:,.0f})"
                          "<br>K/F %{x:.1f}% · forward %{customdata[2]:,.0f}"
                          "<br>%{y:.1f} days to expiry<br>IV %{z:.2f}%<extra>"
                          + ("SVI" if surface_view == "Smoothed (SVI)" else "SABR") + "</extra>",
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
        ))
    kmin, kmax = min(grid["k"]) - 1e-9, max(grid["k"]) + 1e-9
    in_range = fit_pts[(fit_pts["k"] >= kmin) & (fit_pts["k"] <= kmax)]
    fig.add_trace(go.Scatter3d(
        x=in_range["moneyness_fwd"] * 100, y=in_range["days"], z=in_range["iv"] * 100, mode="markers",
        marker=dict(size=2.5, color="#eb6834"), name="Market (mid) IV",
        customdata=in_range[["strike"]].to_numpy(),
        hovertemplate="<b>Strike %{customdata[0]:,.0f}</b> (market quote)<br>K/F %{x:.1f}%<br>"
                      "%{y:.1f} days to expiry<br>market IV %{z:.2f}%<extra></extra>",
    ))
else:
    fig.add_trace(go.Scatter3d(
        x=fit_pts["moneyness_fwd"] * 100, y=fit_pts["days"], z=fit_pts["iv"] * 100, mode="markers",
        marker=dict(size=3, color=fit_pts["iv"] * 100, colorscale=SEQ_BLUE, showscale=True,
                    colorbar=dict(title="IV %", thickness=12, len=0.6)),
        customdata=fit_pts[["strike"]].to_numpy(),
        hovertemplate="<b>Strike %{customdata[0]:,.0f}</b><br>K/F %{x:.1f}%<br>"
                      "%{y:.1f} days to expiry<br>IV %{z:.2f}%<extra></extra>",
    ))
fig.update_layout(
    height=640, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    scene=dict(xaxis=dict(title="Moneyness K/F (%)", ticksuffix=""),
               yaxis=dict(title="Days to expiry"),
               zaxis=dict(title="Implied vol (%)", nticks=6),
               camera=dict(eye=dict(x=1.4, y=-1.6, z=0.75), center=dict(x=0, y=0, z=-0.15)),
               aspectmode="manual", aspectratio=dict(x=1.5, y=1.15, z=0.7)),
)
st.plotly_chart(fig, width="stretch", config={**PLOT_CFG, "scrollZoom": False})
st.caption(
    "x = strike as % of the forward (100 = at the money), y = days to expiry, z = implied volatility. "
    "The smooth sheet joins the per-expiry SVI fits by interpolating total variance (σ²·T) in time, "
    "over the strike range every expiry has quotes for; dots are market mid-price IVs."
)


# ============================================================
# SMILES
# ============================================================

st.subheader("Smile by expiry")
default = list(exp["expiry"][: min(3, len(exp))])
chosen = st.multiselect("Expiries", list(exp["expiry"]), default=default, format_func=label.get)
x_axis = st.radio("X axis", ["Moneyness K/F (%)", "Strike"], horizontal=True, label_visibility="collapsed")

smile = go.Figure()
for e in chosen:
    p = pts[pts["expiry"] == e].sort_values("k")
    used, unused = p[p["used_in_fit"].astype(bool)], p[~p["used_in_fit"].astype(bool)]
    xs = (lambda d: d["moneyness_fwd"] * 100) if x_axis.startswith("Moneyness") else (lambda d: d["strike"])
    col = colors[e]
    err_hi = ((used["iv_ask"] - used["iv"]) * 100).clip(lower=0).fillna(0)
    err_lo = ((used["iv"] - used["iv_bid"]) * 100).clip(lower=0).fillna(0)
    smile.add_trace(go.Scatter(
        x=xs(used), y=used["iv"] * 100, mode="markers", name=f"{label[e]} market", legendgroup=e,
        marker=dict(size=8, color=col, line=dict(width=1, color="rgba(255,255,255,0.9)")),
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, thickness=1, width=0,
                     color=col),
        customdata=np.c_[used["strike"], used["type"], used["iv_bid"] * 100, used["iv_ask"] * 100],
        hovertemplate="%{customdata[0]:,.0f} %{customdata[1]}<br>mid IV %{y:.2f}%<br>"
                      "bid %{customdata[2]:.2f}% · ask %{customdata[3]:.2f}%<extra>" + label[e] + "</extra>",
    ))
    if len(unused):
        smile.add_trace(go.Scatter(
            x=xs(unused), y=unused["iv"] * 100, mode="markers", name=f"{label[e]} not fitted",
            legendgroup=e, showlegend=False,
            marker=dict(size=7, symbol="circle-open", color=MUTED),
            customdata=np.c_[unused["strike"], unused["src"]],
            hovertemplate="%{customdata[0]:,.0f}<br>IV %{y:.2f}%<br>not fitted (%{customdata[1]} "
                          "quote or far wing)<extra></extra>",
        ))
    # model curves only across the fitted strike range (beyond it they're extrapolation)
    p = p[(p["k"] >= used["k"].min()) & (p["k"] <= used["k"].max())] if len(used) else p
    if "svi_iv" in p:
        smile.add_trace(go.Scatter(
            x=xs(p), y=p["svi_iv"] * 100, mode="lines", name=f"{label[e]} SVI", legendgroup=e,
            line=dict(color=col, width=2), hovertemplate="SVI %{y:.2f}%<extra></extra>"))
    if show_sabr and "sabr_iv" in p:
        smile.add_trace(go.Scatter(
            x=xs(p), y=p["sabr_iv"] * 100, mode="lines", name=f"{label[e]} SABR", legendgroup=e,
            line=dict(color=col, width=1.5, dash="dot"), hovertemplate="SABR %{y:.2f}%<extra></extra>"))
if x_axis.startswith("Moneyness"):
    smile.add_vline(x=100, line=dict(color=MUTED, width=1, dash="dash"))
smile.update_layout(
    height=460, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
    xaxis=dict(title=x_axis, gridcolor=GRID), yaxis=dict(title="Implied volatility (%)", gridcolor=GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(smile, width="stretch", config=PLOT_CFG)
st.caption(
    "Dots: IV from the mid price of the out-of-the-money option (puts left of the forward, calls right), "
    "with the bid–ask range as error bars. Solid line: SVI fit, dotted: SABR. "
    "Hollow grey dots are stale last-trade prices or far wings priced in paise: shown, not fitted."
)


# ============================================================
# TERM STRUCTURE + SKEW
# ============================================================

left, right = st.columns(2, gap="large")
with left:
    st.subheader("ATM term structure")
    ts = go.Figure()
    ts.add_trace(go.Scatter(
        x=exp["days"], y=exp["atm_iv"] * 100, mode="lines+markers", name="ATM IV",
        line=dict(color=SERIES[0], width=2), marker=dict(size=9),
        customdata=exp[["expiry"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>%{x:.1f} days<br>ATM IV %{y:.2f}%<extra></extra>"))
    if rv and np.isfinite(rv):
        ts.add_hline(y=rv * 100, line=dict(color=MUTED, width=1.5, dash="dash"),
                     annotation_text=f"20d realized {rv * 100:.1f}%", annotation_position="bottom right")
    # at least a 3-vol-point window so a flat curve doesn't look dramatic
    vals = list(exp["atm_iv"].dropna() * 100) + ([rv * 100] if rv and np.isfinite(rv) else [])
    mid, half = (max(vals) + min(vals)) / 2, max((max(vals) - min(vals)) / 2 + 0.4, 1.5)
    ts.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                     xaxis=dict(title="Days to expiry", gridcolor=GRID),
                     yaxis=dict(title="ATM implied vol (%)", gridcolor=GRID, range=[mid - half, mid + half]))
    st.plotly_chart(ts, width="stretch", config=PLOT_CFG)
    st.caption("Upward sloping = calm now, more uncertainty priced later (normal). "
               "Downward/inverted = stress or an event priced for the near term.")

with right:
    st.subheader("Skew term structure")
    sk = go.Figure()
    for name, col, c in (("25Δ risk reversal", "rr25", SERIES[1]), ("25Δ butterfly", "bf25", SERIES[2])):
        if col in exp:
            sk.add_trace(go.Scatter(
                x=exp["days"], y=exp[col] * 100, mode="lines+markers", name=name,
                line=dict(color=c, width=2), marker=dict(size=9),
                hovertemplate=name + " %{y:+.2f} pts<br>%{x:.1f} days<extra></extra>"))
    sk.add_hline(y=0, line=dict(color=MUTED, width=1))
    sk.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                     xaxis=dict(title="Days to expiry", gridcolor=GRID),
                     yaxis=dict(title="Vol points", gridcolor=GRID),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(sk, width="stretch", config=PLOT_CFG)
    st.caption("Risk reversal below zero = puts richer than calls (crash protection bid). "
               "Butterfly = smile curvature, the premium for large moves either way.")


# ============================================================
# CALIBRATION + ARBITRAGE
# ============================================================

st.subheader("Calibration by expiry")
table = pd.DataFrame({
    "Expiry": [label[e] for e in exp["expiry"]],
    "Forward": exp["forward"],
    "Carry (ann.)": exp["carry"] * 100,
    "Quotes fitted": exp["n_points"],
    "ATM IV": exp["atm_iv"] * 100,
    "25Δ RR": exp.get("rr25", pd.Series(np.nan, index=exp.index)) * 100,
    "25Δ BF": exp.get("bf25", pd.Series(np.nan, index=exp.index)) * 100,
    "SVI error": exp["svi_rmse"] * 100,
    "SABR error": exp["sabr_rmse"] * 100,
    "No butterfly arb": exp["butterfly_ok"].map({True: "✓", False: "✗"}),
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Forward": st.column_config.NumberColumn(format="%.1f"),
    "Carry (ann.)": st.column_config.NumberColumn(format="%.2f%%",
        help="Annualised forward premium over spot ≈ interest rate − dividend yield."),
    "ATM IV": st.column_config.NumberColumn(format="%.2f%%"),
    "25Δ RR": st.column_config.NumberColumn(format="%+.2f"),
    "25Δ BF": st.column_config.NumberColumn(format="%+.2f"),
    "SVI error": st.column_config.NumberColumn(format="%.2f", help="RMSE in vol points vs market mid IVs."),
    "SABR error": st.column_config.NumberColumn(format="%.2f", help="RMSE in vol points vs market mid IVs."),
})

with st.expander("Model parameters"):
    rows = []
    for r in exp.itertuples():
        svi, sabr = getattr(r, "svi", None) or {}, getattr(r, "sabr", None) or {}
        rows.append({"Expiry": label[r.expiry], "T (years)": r.T,
                     "SVI a": svi.get("a"), "SVI b": svi.get("b"), "SVI ρ": svi.get("rho"),
                     "SVI m": svi.get("m"), "SVI σ": svi.get("sigma"),
                     "SABR α": sabr.get("alpha"), "SABR β": sabr.get("beta"),
                     "SABR ρ": sabr.get("rho"), "SABR ν": sabr.get("nu")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in rows[0] if c != "Expiry"})
    st.caption("SVI: w(k) = a + b·(ρ·(k − m) + √((k − m)² + σ²)), w = σ_imp²·T, k = ln(K/F). "
               "SABR: Hagan lognormal approximation with β = 1.")

cal = data.get("calendar", [])
bad_fly = exp[~exp["butterfly_ok"].astype(bool)]
bad_cal = [c for c in cal if not c["ok"]]
if bad_fly.empty and not bad_cal:
    st.success(f"No-arbitrage checks passed: butterfly on all {len(exp)} expiries, "
               f"calendar on all {len(cal)} neighbouring pairs.", icon=":material/check_circle:")
else:
    for r in bad_fly.itertuples():
        st.warning(f"Butterfly arbitrage in the {label[r.expiry]} fit (min g = {r.butterfly_min_g:.4f}).")
    for c in bad_cal:
        st.warning(f"Calendar arbitrage: total variance falls from {c['near']} to {c['far']}.")
if data.get("skipped"):
    st.caption("Skipped expiries: " + "; ".join(
        f"{pd.Timestamp(s['expiry']):%d %b %Y} ({s['reason']})" for s in data["skipped"]))


# ============================================================
# LEARN
# ============================================================

with st.expander("How this surface is built (and why)"):
    st.markdown(
        """
**1. Exact time to expiry.** `T` = hours from the snapshot to 15:30 IST on expiry day ÷ (365 × 24).
For weekly options, counting "1 day" vs "21 hours" changes the IV a lot.

**2. The forward, not spot.** Put-call parity says `C − P = DF·(F − K)`, so `F = K + (C − P)/DF`.
We take the median over near-the-money strikes. The forward already includes interest and dividends
(the *carry* column ≈ rate − dividend yield), so calls and puts give the same IV around it.

**3. Our own implied vols.** Each option's bid, mid and ask price is inverted with **Black-76**
(the right model for European index options priced off a forward). The bid–ask range
becomes an IV range that tells us how precise each point is.

**4. Out-of-the-money options only.** Below the forward we use puts, above it calls.
In-the-money options are mostly intrinsic value, trade less and have wider spreads, so their IVs are noisy.

**5. Clean inputs.** We fit only two-sided quotes within ±4 standard deviations (4·σ·√T).
Stale last-trade prices and options priced in paise are shown, not fitted.

**6. SVI per expiry.** Gatheral's *raw SVI* models **total variance** `w = σ²·T` as a function of
log-moneyness `k = ln(K/F)` with 5 parameters: level (a), wing steepness (b), skew (ρ), shift (m),
ATM curvature (σ). Points with tighter bid–ask spreads get more weight. Bounds keep the wings
within Roger Lee's limit (`b·(1+|ρ|) ≤ 2`).

**7. SABR for comparison.** The Hagan formula with β = 1 has 3 parameters (α level, ρ skew,
ν vol-of-vol). It's simpler and has a dynamic meaning, but it often fits short-dated smiles less well than SVI.

**8. No-arbitrage checks.**
*Butterfly:* the smile must imply a non-negative probability for every price level (Durrleman's `g(k) ≥ 0`).
*Calendar:* total variance must not fall for a later expiry, or you could buy the later expiry and sell
the earlier one for free money.

**9. The smooth surface.** Between expiries, total variance is interpolated linearly in time at fixed
log-moneyness. This keeps the calendar condition intact, unlike interpolating vols directly.

**Reading it:** ATM IV vs realized vol tells you whether options are rich or cheap.
The risk reversal tells you how much more puts cost than calls. The butterfly tells you how much
the market pays for big moves. The term structure tells you *when* the market expects uncertainty.
"""
    )
