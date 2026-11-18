"""
QuantOS V11 | Causal Gaussian-HMM regime layer

Fits a 3-state Gaussian HMM on training rows only and returns FILTERED
posteriors  p(s_t | x_1..x_t)  for any later rows (forward algorithm only;
hmmlearn's predict_proba is a smoothed posterior and would leak the future).

States are ordered by their mean daily return: 0 = BEAR, 1 = SIDE, 2 = BULL.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

warnings.filterwarnings("ignore")

HMM_INPUTS = ["tgt_r1", "tgt_vol20", "vix_chg1"]
SEEDS = (11, 23, 37, 51, 73)


def _fit(X: np.ndarray) -> GaussianHMM | None:
    best, best_ll = None, -np.inf
    for seed in SEEDS:
        m = GaussianHMM(
            n_components=3, covariance_type="full", n_iter=200,
            random_state=seed, min_covar=1e-4,
        )
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll and np.all(np.diag(m.transmat_) < 0.9999):
            best, best_ll = m, ll
    return best


def _filtered(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    log_b = model._compute_log_likelihood(X)
    log_A = np.log(model.transmat_ + 1e-300)
    log_alpha = np.log(model.startprob_ + 1e-300) + log_b[0]
    out = np.empty_like(log_b)
    out[0] = log_alpha - logsumexp(log_alpha)
    for t in range(1, len(X)):
        log_alpha = logsumexp(out[t - 1][:, None] + log_A, axis=0) + log_b[t]
        out[t] = log_alpha - logsumexp(log_alpha)
    return np.exp(out)


class CausalRegimeHMM:
    def __init__(self):
        self.model = None
        self.mean = self.std = None
        self.order = None

    def fit(self, frame: pd.DataFrame) -> "CausalRegimeHMM":
        X = frame[HMM_INPUTS].to_numpy(float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-12
        self.model = _fit((X - self.mean) / self.std)
        if self.model is not None:
            # sort by mean of the (standardised) return input
            self.order = np.argsort(self.model.means_[:, 0])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = ["hmm_p_bear", "hmm_p_side", "hmm_p_bull"]
        if self.model is None:
            return pd.DataFrame(np.full((len(frame), 3), 1 / 3), index=frame.index, columns=cols)
        X = (frame[HMM_INPUTS].to_numpy(float) - self.mean) / self.std
        post = _filtered(self.model, X)[:, self.order]
        return pd.DataFrame(post, index=frame.index, columns=cols)
