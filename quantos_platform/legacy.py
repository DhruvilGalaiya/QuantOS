"""
Bridge to research code in src/ (e.g. the V10 HMM chain).

This is the ONLY place that puts src/ on sys.path; production modules call
`legacy_module("build_live_features")` instead of importing src/ directly.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from quantos_platform.core import settings


def legacy_module(name: str) -> ModuleType:
    src = str(settings.LEGACY_SRC)
    if src not in sys.path:
        sys.path.append(src)   # appended: package modules always win over same-named legacy files
    return importlib.import_module(name)
