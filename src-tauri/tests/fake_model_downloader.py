#!/usr/bin/env python3
"""Fake model downloader for the shell setup tests.

Writes a small placeholder model under the requested root, or fails with a
message on standard error, so the setup tests never touch the network.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.sleep:
        time.sleep(args.sleep)

    if args.fail:
        print("Checksum mismatch for asr/fake.gguf", file=sys.stderr)
        return 1

    target = Path(args.root) / "stt" / "models" / "nemotron-gguf"
    target.mkdir(parents=True, exist_ok=True)
    (target / "fake.gguf").write_bytes(b"fake model")
    print("Fetching asr/fake.gguf ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
