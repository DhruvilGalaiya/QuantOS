"""Shared fixtures: an isolated data/system directory so tests never touch real status/locks."""

import pytest

from quantos_platform.core import settings


@pytest.fixture
def isolated_system(tmp_path, monkeypatch):
    sysdir = tmp_path / "system"
    monkeypatch.setattr(settings, "SYSTEM_DIR", sysdir)
    monkeypatch.setattr(settings, "STATUS_DB", sysdir / "status.sqlite")
    monkeypatch.setattr(settings, "LOG_DIR", sysdir / "logs")
    monkeypatch.setattr(settings, "CACHE_DIR", sysdir / "cache")
    monkeypatch.setattr(settings, "LOCK_DIR", sysdir / "locks")
    settings.ensure_dirs()
    return sysdir


def has_local_data() -> bool:
    return (settings.DAILY_DIR / "nifty_50.parquet").exists() and \
        (settings.SURFACE_DIR / "nifty_50_surface_v2.json").exists()
