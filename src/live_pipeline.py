"""Moved to `quantos_platform.data.daily` (production package). This file keeps old imports and commands working."""

import importlib
import runpy  # noqa: F401
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __name__ == "__main__":
    from quantos_platform.jobs import main
    raise SystemExit(main(['india'] + sys.argv[1:]))
else:
    sys.modules[__name__] = importlib.import_module("quantos_platform.data.daily")
