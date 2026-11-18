"""Moved to `quantos_platform.data.us_universe` (production package). This file keeps old imports and commands working."""

import importlib
import runpy  # noqa: F401
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __name__ == "__main__":
    runpy.run_module("quantos_platform.data.us_universe", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("quantos_platform.data.us_universe")
