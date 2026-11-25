"""ETags from artifact modification times: a response changes only when a file it reads changes."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from fastapi import Request, Response
from fastapi.responses import JSONResponse

# changes whenever the service (re)starts, so a code change that alters a payload is never served as "not modified"
_PROCESS_SALT = f"{os.getpid()}:{time.time_ns()}"


def etag_for(paths: Iterable[Path]) -> str:
    h = hashlib.sha1(_PROCESS_SALT.encode())
    for p in sorted(paths):
        try:
            st = p.stat()
            h.update(f"{p.name}:{st.st_mtime_ns}:{st.st_size};".encode())
        except OSError:
            h.update(f"{p.name}:missing;".encode())
    return f'W/"{h.hexdigest()[:20]}"'


def cached_json(request: Request, paths: Iterable[Path], build: Callable[[], object], model=None) -> Response:
    """304 when the client's If-None-Match matches the artifacts' ETag, else the built (validated) payload."""
    tag = etag_for(paths)
    headers = {"ETag": tag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers=headers)
    payload = build()
    if model is not None:
        payload = model.model_validate(payload).model_dump(mode="json", by_alias=True)
    return JSONResponse(payload, headers=headers)
