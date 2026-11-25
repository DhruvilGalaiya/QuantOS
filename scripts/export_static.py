"""
Export every public API response to static JSON for the public site (no servers needed):

    python scripts/export_static.py [--out web/public/data/v1]

The files mirror the gateway routes: /api/v1/today -> <out>/today.json. Each payload is validated with the
same Pydantic models the API uses, so the static site and the live API share one contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantos_platform.analytics import INDICES  # noqa: E402
from quantos_platform.api import API_VERSION, schemas  # noqa: E402
from quantos_platform.core import calendar  # noqa: E402
from quantos_platform.services import common, events, regimes, risk, states, surface, today, us_stocks, volatility  # noqa: E402


def routes() -> dict:
    """route path (file name without .json) -> (builder, response model)."""
    out = {
        "today": (today.build, schemas.Today),
        "us": (us_stocks.universe, schemas.UsUniverse),
        "meta": (lambda: {"version": API_VERSION, "mode": "static", "market_open": False,
                          "now": calendar.now_ist().isoformat(), "freshness": common.freshness()}, schemas.Meta),
    }
    for sym in regimes.INSTRUMENTS:
        out[f"regimes/{sym}"] = (lambda s=sym: regimes.build(s), schemas.Regimes)
    for sym in INDICES:
        out[f"volatility/{sym}"] = (lambda s=sym: volatility.build(s), schemas.Volatility)
    out["risk"] = (risk.build, schemas.RiskMonitor)
    for sym in states.SYMBOLS:
        out[f"states/{sym}"] = (lambda s=sym: states.build(s), schemas.States)
    for sym in INDICES:
        out[f"events/{sym}"] = (lambda s=sym: events.build(s), schemas.Events)
    for sym in surface.SYMBOLS:
        out[f"surface/{sym}"] = (lambda s=sym: surface.build(s), schemas.Surface)
    for t in us_stocks.universe()["stocks"]:
        out[f"us/{t['ticker']}"] = (lambda tk=t["ticker"]: us_stocks.company(tk), schemas.UsCompany)
    return out


def export(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (build, model) in routes().items():
        payload = model.model_validate(build()).model_dump(mode="json", by_alias=True)
        path = out / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        tmp.replace(path)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "data" / f"v{API_VERSION}")
    args = ap.parse_args()
    files = export(args.out)
    for p in files:
        if p.parent == args.out:
            print(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}  {p.stat().st_size / 1024:.1f} KB")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
