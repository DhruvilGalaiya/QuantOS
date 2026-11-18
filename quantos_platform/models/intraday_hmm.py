"""
QuantOS V12 | Intraday Gaussian HMM on 15-minute bars

Learns hidden intraday states from bar-by-bar NIFTY and India VIX moves,
then for each session filters (forward algorithm only) from 09:30 to 15:15:

    ihmm_last_*  : state probabilities at 15:15
    ihmm_frac_*  : share of the session spent in each state (mean posterior)

States are ordered by mean NIFTY bar return: down / calm / up.
Fit only on training sessions; apply to later sessions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from quantos_platform.models.regime_hmm import _filtered
from quantos_platform.models.snapshot_1515 import _bars

warnings.filterwarnings("ignore")

NAMES = ["down", "calm", "up"]


def bar_matrix(target: str = "NIFTY_50") -> pd.DataFrame:
    """Per-bar standardisable features through 15:15, indexed by (date, hm)."""
    px = _bars(target)[["date", "hm", "close"]]
    vx = _bars("INDIA_VIX")[["date", "hm", "close"]].rename(columns={"close": "vix"})
    m = px.merge(vx, on=["date", "hm"], how="inner").sort_values(["date", "hm"])
    m["r"] = np.log(m["close"]).groupby(m["date"]).diff() * 100
    m["vr"] = np.log(m["vix"]).groupby(m["date"]).diff() * 100
    m = m.dropna(subset=["r", "vr"])
    m["absr"] = m["r"].abs()
    return m[["date", "hm", "r", "vr", "absr"]].reset_index(drop=True)


class IntradayHMM:
    cols = ["r", "vr", "absr"]

    def fit(self, bars: pd.DataFrame, seeds=(3, 17, 29)) -> "IntradayHMM":
        X = bars[self.cols].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        lengths = bars.groupby("date", sort=False).size().to_numpy()
        best, best_ll = None, -np.inf
        for seed in seeds:
            m = GaussianHMM(3, covariance_type="full", n_iter=100, random_state=seed, min_covar=1e-3)
            try:
                m.fit(Z, lengths)
                ll = m.score(Z, lengths)
            except Exception:
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        self.model = best
        self.order = np.argsort(best.means_[:, 0]) if best is not None else np.arange(3)
        return self

    def transform(self, bars: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for date, day in bars.groupby("date", sort=True):
            Z = (day[self.cols].to_numpy(float) - self.mu) / self.sd
            post = _filtered(self.model, Z)[:, self.order]
            rows[date] = np.concatenate([post[-1], post.mean(0)])
        cols = [f"ihmm_last_{n}" for n in NAMES] + [f"ihmm_frac_{n}" for n in NAMES]
        return pd.DataFrame.from_dict(rows, orient="index", columns=cols)
