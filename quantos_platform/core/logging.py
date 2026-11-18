"""Logging: stdout + rotating file in data/system/logs/quantos.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from quantos_platform.core import settings

_CONFIGURED = False


def get_logger(name: str = "quantos") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        settings.ensure_dirs()
        root = logging.getLogger("quantos")
        root.setLevel(settings.LOG_LEVEL)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = RotatingFileHandler(settings.LOG_DIR / "quantos.log", maxBytes=2_000_000, backupCount=5)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("quantos") else f"quantos.{name}")
