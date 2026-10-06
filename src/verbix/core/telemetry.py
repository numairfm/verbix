"""Minimal latency tracing: JSONL marks gated on VERBIX_TRACE=1. Stdlib only, never raises."""

from __future__ import annotations

import json
import os
import time
from typing import Any


def record_latency(mark: str, **extra: Any) -> None:
    if os.environ.get("VERBIX_TRACE") != "1":
        return
    path = os.environ.get("VERBIX_TRACE_FILE", "verbix-trace.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"mark": mark, "t": time.time(), **extra}) + "\n")
    except Exception:
        pass
