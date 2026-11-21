"""
QuantOS | Job runner: the only code that fetches data or runs models.

    python -m quantos_platform.jobs india        # daily candles -> V10 HMM -> V11 forecast
    python -m quantos_platform.jobs vol          # option-chain snapshot -> IV surfaces
    python -m quantos_platform.jobs playbook     # 15-min bars + 15:15 signal -> paper log
    python -m quantos_platform.jobs us_equities  # US stock closes for the US Stocks page
    python -m quantos_platform.jobs analytics    # volatility & edge, event studies, Today brief (reads stored data)
    python -m quantos_platform.jobs all          # everything above
    python -m quantos_platform.jobs status       # last runs + data freshness
    add --force to re-run even when nothing is stale

Every job: takes a lock (no overlapping runs), records start/finish/result in the
status DB, logs to data/system/logs/quantos.log, and never raises to the caller
(the dashboard keeps serving the last good data).

Suggested schedule (IST, weekdays) for deployment:
    08:45  india            (V11 final forecast once the US session has closed)
    every 15 min 09:15-15:30  vol
    15:12  playbook          (signal at the 15:15 snapshot)
    15:50  india, vol, playbook   (end-of-day)
    07:00  us_equities       (after the US close)
    16:00  analytics         (after the end-of-day india/vol/playbook runs)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Callable

import pandas as pd

from quantos_platform.core import calendar, settings, status
from quantos_platform.core.logging import get_logger

log = get_logger("quantos.jobs")
LOCK_STALE_S = 30 * 60
MIN_GAP_S = 10 * 60   # ensure_fresh() won't re-run a job more often than this


class JobLocked(RuntimeError):
    pass


@contextmanager
def _lock(job: str):
    settings.ensure_dirs()
    path = settings.LOCK_DIR / f"{job}.lock"
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE_S:
        path.unlink(missing_ok=True)          # left behind by a crashed run
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise JobLocked(f"{job} is already running") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


# ============================================================
# JOBS
# ============================================================

def job_india(force: bool = False) -> dict:
    """Daily index candles -> V10 regime HMM -> V11 next-session forecast."""
    from quantos_platform.data import daily
    from quantos_platform.models import v10, v11

    report = daily.PipelineReport(started=calendar.now_ist())
    changed = daily.update_all_daily(report)
    out = {"updated": report.updated, "data_through": report.data_through, "errors": dict(report.errors)}
    if report.data_through:
        status.set_freshness("india_daily", report.data_through)
    if force or changed or v10.is_stale():
        errs = v10.run()
        out["v10"] = "ok" if not errs else errs
        out["errors"].update({f"v10:{k}": v for k, v in errs.items()})
    if force or changed or v11.needs_update():
        res = v11.run(rebuild=False, refresh_us=True)
        latest = {t: m["latest"] for t, m in res.items()}
        out["v11"] = latest
        first = next(iter(latest.values()))
        status.set_freshness("v11", f"{first['target_trading_date']} ({first['status']})")
    # a single failed instrument is a warning; a total fetch failure is an error
    if report.errors and len(report.errors) >= len(daily.INSTRUMENTS):
        raise RuntimeError(f"all daily fetches failed: {next(iter(report.errors.values()))}")
    return out


def job_vol(force: bool = False) -> dict:
    """Option-chain snapshot (when stale) -> volatility surfaces (SVI + SABR)."""
    from quantos_platform.vol import snapshot

    r = snapshot.run(force=force)
    if r.snapshot_ts:
        status.set_freshness("options_snapshot", r.snapshot_ts)
    if "snapshot" in r.errors and not r.snapshot_ts:
        raise RuntimeError(r.errors["snapshot"])
    return {"snapshot_ts": r.snapshot_ts, "new_snapshot": r.snapshot_taken,
            "surfaces_built": r.surfaces_built, "errors": r.errors}


def job_playbook(force: bool = False) -> dict:
    """15-min bars + Asia closes -> tonight's signal per index -> paper log scoring."""
    from quantos_platform.strategies import expiry_playbook as pb

    errors = pb.update_inputs(log=log.info)
    out = {"inputs_errors": errors, "tonight": {}, "signals": {}}
    for sym in pb.SYMBOLS:
        try:
            sig = pb.tonight(sym)
            save_tonight(sym, sig)
            out["tonight"][sym] = {k: sig.get(k) for k in ("phase", "active", "tdte", "side", "strike")}
        except Exception as exc:
            out.setdefault("errors", {})[f"tonight:{sym}"] = f"{type(exc).__name__}: {exc}"
        try:
            out["signals"][sym] = int(len(pb.track(sym, log=log.info)))
        except Exception as exc:
            out.setdefault("errors", {})[f"track:{sym}"] = f"{type(exc).__name__}: {exc}"
    status.set_freshness("playbook", calendar.now_ist().isoformat())
    if len(out.get("errors", {})) >= 2 * len(pb.SYMBOLS):
        raise RuntimeError(f"playbook failed: {out['errors']}")
    return out


def job_us_equities(force: bool = False) -> dict:
    """Top up daily closes for the US Stocks universe."""
    from quantos_platform.data.us_equities import load_us_asset, update_us_asset
    from quantos_platform.data.us_universe import load_universe

    tickers = load_universe()["ticker"].tolist() if len(load_universe()) else []
    failed = {}
    for t in tickers:
        try:
            update_us_asset(t)
        except Exception as exc:
            failed[t] = f"{type(exc).__name__}: {exc}"
    if tickers:
        last = max(pd.to_datetime(load_us_asset(t)["timestamp"]).max() for t in tickers[:5])
        status.set_freshness("us_equities", str(pd.Timestamp(last).date()))
    if tickers and len(failed) == len(tickers):
        raise RuntimeError(f"all US updates failed: {next(iter(failed.values()))}")
    return {"tickers": len(tickers), "failed": failed}


def job_analytics(force: bool = False) -> dict:
    """Descriptive analytics from stored data: Volatility & Edge, Event Lab, Today brief, plus the coming week's
    scheduled news events (free public calendars; failures there never fail the job)."""
    from quantos_platform.analytics import INDICES, brief, events, news_calendar, volatility

    out: dict = {"volatility": [], "events": [], "errors": {}}
    for sym in INDICES:
        for name, mod in (("volatility", volatility), ("events", events)):
            try:
                mod.build(sym)
                out[name].append(sym)
            except Exception as exc:
                out["errors"][f"{name}:{sym}"] = f"{type(exc).__name__}: {exc}"
    try:
        out["news"] = news_calendar.build()["sources"]
    except Exception as exc:
        out["errors"]["news"] = f"{type(exc).__name__}: {exc}"
    try:
        b = brief.build()
        as_of = max(v["market"]["as_of"] for v in b["symbols"].values())
        status.set_freshness("analytics", as_of)
        out["as_of"] = as_of
    except Exception as exc:
        out["errors"]["brief"] = f"{type(exc).__name__}: {exc}"
    if not out["volatility"] and not out["events"]:
        raise RuntimeError(f"analytics failed: {out['errors']}")
    return out


JOBS: dict[str, Callable[[bool], dict]] = {
    "india": job_india,
    "vol": job_vol,
    "playbook": job_playbook,
    "us_equities": job_us_equities,
    "analytics": job_analytics,   # last: `all` runs it after the data jobs
}


# ============================================================
# RUNNER
# ============================================================

def run(job: str, force: bool = False) -> dict:
    """Run one job with lock + status + logging. Never raises; returns {"ok": bool, ...}."""
    if job not in JOBS:
        raise KeyError(f"unknown job {job!r}; choose from {sorted(JOBS)}")
    try:
        with _lock(job):
            run_id = status.start_run(job)
            t0 = time.time()
            try:
                result = JOBS[job](force)
                ok, msg = True, f"ok in {time.time() - t0:.1f}s"
            except Exception as exc:
                result, ok = {"traceback": traceback.format_exc(limit=5)}, False
                msg = f"{type(exc).__name__}: {exc}"
            status.finish_run(run_id, ok, msg, result)
            (log.info if ok else log.error)(f"job {job}: {msg}")
            return {"job": job, "ok": ok, "message": msg, **result}
    except JobLocked as exc:
        log.info(str(exc))
        return {"job": job, "ok": True, "message": str(exc), "skipped": True}


def run_all(force: bool = False) -> list[dict]:
    return [run(j, force) for j in JOBS]


def is_stale(job: str) -> bool:
    """Cheap staleness checks used by pages in local mode to refresh on launch."""
    last = status.last_run(job)
    if last and last.get("started"):
        if time.time() - pd.Timestamp(last["started"]).timestamp() < MIN_GAP_S:
            return False
    if job == "india":
        from quantos_platform.models import v11

        fresh = status.freshness("india_daily")
        if not fresh or pd.Timestamp(fresh["as_of"]) < calendar.expected_latest_session():
            return True
        return v11.needs_update()
    if job == "vol":
        from quantos_platform.vol import snapshot

        return snapshot.snapshot_is_stale(snapshot.latest_snapshot_ts()) or \
            snapshot.surfaces_are_stale(snapshot.latest_snapshot_ts())
    if job == "playbook":
        fresh = status.freshness("playbook")
        if not fresh:
            return True
        age = calendar.now_ist() - pd.Timestamp(fresh["as_of"])
        return age > pd.Timedelta(minutes=10) if calendar.market_open_now() else \
            pd.Timestamp(fresh["as_of"]) < calendar.last_close()
    if job == "analytics":
        ok = status.last_run(job, ok_only=True)
        if not ok:
            return True
        built = pd.Timestamp(ok["started"]).timestamp()
        inputs = [settings.DAILY_DIR / "nifty_50.parquet", settings.PREDICTION_DIR / "latest_predictions.json",
                  settings.SURFACE_DIR / "nifty_50_surface_v2.json"]
        return any(p.exists() and p.stat().st_mtime > built for p in inputs)
    if job == "us_equities":
        ok = status.last_run(job, ok_only=True)
        return not ok or time.time() - pd.Timestamp(ok["started"]).timestamp() > 12 * 3600
    return False


def ensure_fresh(job: str) -> dict | None:
    """Run `job` if its data is stale (local mode only; public mode relies on the scheduler)."""
    if settings.PUBLIC:
        return None
    try:
        return run(job) if is_stale(job) else None
    except Exception as exc:   # staleness check itself failed: don't block the page
        log.warning(f"ensure_fresh({job}) failed: {exc}")
        return None


# ============================================================
# TONIGHT'S PLAYBOOK SIGNAL (artifact read by the page)
# ============================================================

def _tonight_path(symbol: str):
    return settings.PLAYBOOK_DIR / f"tonight_{symbol.lower()}.json"


def save_tonight(symbol: str, sig: dict) -> None:
    settings.PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _tonight_path(symbol).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sig, default=str, indent=1))
    tmp.replace(_tonight_path(symbol))


def load_tonight(symbol: str) -> dict:
    p = _tonight_path(symbol)
    return json.loads(p.read_text()) if p.exists() else {}


# ============================================================
# CLI
# ============================================================

def _print_status() -> None:
    pd.set_option("display.width", 200)
    for ds in ("india_daily", "v11", "options_snapshot", "playbook", "us_equities", "analytics"):
        f = status.freshness(ds)
        print(f"{ds:<18} as of {f['as_of'] if f else '-':<34} updated {f['updated'][:19] if f else '-'}")
    print()
    print(status.recent_runs(15).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")]
    if not names or names[0] == "status":
        _print_status()
        return 0
    results = run_all(force) if names[0] == "all" else [run(n, force) for n in names]
    for r in results:
        print(f"{r['job']:<12} {'OK ' if r['ok'] else 'ERR'} {r['message']}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
