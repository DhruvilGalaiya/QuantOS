"""Models must not use future information."""

import numpy as np
import pandas as pd

from quantos_platform.models.regime_hmm import HMM_INPUTS, CausalRegimeHMM
from quantos_platform.strategies import expiry_playbook as pb


def _regime_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = np.concatenate([rng.normal(0.001, 0.006, n // 2), rng.normal(-0.001, 0.015, n - n // 2)])
    return pd.DataFrame({"tgt_r1": r, "tgt_vol20": pd.Series(r).rolling(20, min_periods=1).std().to_numpy(),
                         "vix_chg1": rng.normal(0, 0.05, n)}, index=idx)[HMM_INPUTS]


def test_hmm_filter_is_causal():
    """Filtered regime probabilities up to t must not change when later data arrives."""
    data = _regime_frame(900)
    model = CausalRegimeHMM().fit(data.iloc[:600])
    short = model.transform(data.iloc[:700])
    long = model.transform(data)
    pd.testing.assert_frame_equal(short, long.iloc[:700], atol=1e-10)


def test_playbook_training_ignores_the_future():
    """The 15:15 model for day T is identical whatever happens on/after T."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    frame = pd.DataFrame(rng.normal(size=(400, len(pb.FEATURES))), index=idx, columns=pb.FEATURES)
    frame["y_gap"] = rng.normal(size=400)
    cutoff = idx[300]
    m1, d1 = pb._fit(frame, frame, cutoff)
    future = frame.copy()
    future.loc[future.index >= idx[299], "y_gap"] = 99.0      # outcome of T-1 resolves at T's open
    future.loc[future.index >= cutoff, pb.FEATURES] = -50.0
    m2, d2 = pb._fit(future, future, cutoff)
    row = frame.loc[[cutoff]]
    assert d1 == d2
    assert m1.predict_proba(row[pb.FEATURES])[0, 1] == m2.predict_proba(row[pb.FEATURES])[0, 1]
