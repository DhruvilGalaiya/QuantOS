"""
V10 regime model (3-state Gaussian HMM) - bridge to the research implementation in src/.

Runs the original chain unchanged: build_live_features -> run_live_hmm ->
build_hmm_prediction_history, with its paths pinned to settings.DATA_DIR
(the original scripts use cwd-relative paths).
"""

from __future__ import annotations

import contextlib
import io

from quantos_platform.core import settings
from quantos_platform.legacy import legacy_module

PREDICTION_FILE = settings.PREDICTION_DIR / "latest_predictions.json"


def is_stale() -> bool:
    """True when any daily file is newer than the latest V10 prediction artifact."""
    if not PREDICTION_FILE.exists():
        return True
    mtime = PREDICTION_FILE.stat().st_mtime
    return any(p.stat().st_mtime > mtime for p in settings.DAILY_DIR.glob("*.parquet"))


def run() -> dict[str, str]:
    """Rebuild V10 features, live HMM states and prediction history. Returns step errors."""
    features = legacy_module("build_live_features")
    live_hmm = legacy_module("run_live_hmm")
    history = legacy_module("build_hmm_prediction_history")
    features.INPUT_DIR = settings.DAILY_DIR
    features.OUTPUT_DIR = settings.FEATURE_DIR
    history.OUTPUT_DIR = settings.PREDICTION_DIR
    history.LATEST_FILE = PREDICTION_FILE
    live_hmm.FEATURE_DIR = settings.FEATURE_DIR
    live_hmm.OUTPUT_DIR = settings.PREDICTION_DIR
    settings.FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for step, fn in (("features", features.main), ("hmm", live_hmm.main), ("history", history.main)):
            try:
                fn()
            except Exception as exc:
                errors[step] = f"{type(exc).__name__}: {exc}"
                if step != "history":   # later steps depend on this one
                    break
    return errors
