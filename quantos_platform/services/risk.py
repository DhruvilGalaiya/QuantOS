"""
Risk Monitor page service: how tightly sectors move together, how unusual today's moves are (turbulence), the
sector scoreboard, single-day loss levels and the 21-session Monte Carlo range for each index, and the daily
returns of every past crash so the page can replay them on the reader's own amount and holdings.
Reads analytics/risk.json, risk_series.parquet and risk_returns.parquet (built by the analytics job).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantos_platform.analytics import read_frame, read_json
from quantos_platform.services.common import level_word, ordinal

REPLAY_TAIL_DAYS = 90     # the replay runs from the peak to 90 days after the bottom


def _r(x, d=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def build() -> dict:
    R = read_json("risk.json")
    if not R:
        return {"as_of": None}
    ser = read_frame("risk_series.parquet")
    rets = read_frame("risk_returns.parquet")
    sv = ser.dropna(subset=["avg_corr"])
    sv = pd.concat([sv.iloc[:-756].iloc[::3], sv.iloc[-756:]])
    names = {**R["sectors"], **R["indices"]}
    crashes = []
    for c in R["crashes"]:
        w = rets.loc[(rets.index > pd.Timestamp(c["peak"])) & (rets.index <= pd.Timestamp(c["trough"]) + pd.Timedelta(days=REPLAY_TAIL_DAYS))]
        cols = [k for k in w.columns if w[k].notna().mean() > 0.9]
        crashes.append({**c, "dates": [d.strftime("%Y-%m-%d") for d in w.index],
                        "returns": {k: [_r(v, 4) if pd.notna(v) else 0.0 for v in w[k]] for k in cols}})
    N = R["now"]
    return {
        "as_of": R["as_of"], "names": names, "sectors": R["sectors"], "indices": R["indices"],
        "now": {**N, "avg_corr_level": level_word(N["avg_corr_pct"]), "avg_corr_ordinal": ordinal(N["avg_corr_pct"]),
                "turbulence_ordinal": ordinal(N["turbulence_pct"]), "dispersion_ordinal": ordinal(N["dispersion_pct"]),
                "absorption_level": level_word(N["absorption_pct"])},
        "turbulence_base": R["turbulence_base"],
        "corr_matrix": R["corr_matrix"],
        "scoreboard": sorted(R["scoreboard"], key=lambda r: r["m1"], reverse=True),
        "tail": R["tail"],
        "series": [{"time": d.strftime("%Y-%m-%d"), "avg_corr": _r(r.avg_corr, 4), "absorption": _r(r.absorption, 4),
                    "turbulence": _r(r.turbulence, 3), "p95": _r(r.turb_p95_expanding, 3)}
                   for d, r in sv[["avg_corr", "absorption", "turbulence", "turb_p95_expanding"]].iterrows()],
        "crashes": crashes,
    }
