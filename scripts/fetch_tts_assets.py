#!/usr/bin/env python3
"""Fetch the pinned TTS runtime assets into a layout dir (CI entry point).

Usage: fetch_tts_assets.py <out-dir>
Requires TTS_BUNDLE_URL env (mirror holding the pinned files + bin/pocket-tts).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verbix.voice.tts_assets import ensure_tts_runtime

if __name__ == "__main__":
    base_url = os.environ.get("TTS_BUNDLE_URL")
    if not base_url:
        raise SystemExit("TTS_BUNDLE_URL env is required")
    layout = ensure_tts_runtime(Path(sys.argv[1]), base_url=base_url)
    print(f"runtime ready: {layout['bin']}")
