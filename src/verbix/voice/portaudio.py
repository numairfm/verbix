"""Resolve the PortAudio runtime, including the AppImage-bundled copy.

The AppImage ships ``libportaudio.so.2`` under ``<AppDir>/usr/lib`` and the
AppRun hook exports ``VERBIX_PORTAUDIO_LIB`` with its absolute path. A host
without the ``portaudio`` package has no ``ldconfig`` entry, no ``libportaudio.so``
linker name, and may have no C compiler, so ``ctypes.util.find_library`` cannot
locate the bundled copy on its own.

``sounddevice`` copies ``ctypes.util.find_library`` by value at import time, so
this patch must run *before* ``sounddevice`` is imported. Call
:func:`ensure_loadable` immediately before importing ``sounddevice``.
"""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path

ENV_VAR = "VERBIX_PORTAUDIO_LIB"

_patched = False


def bundled_library() -> Path | None:
    """Return the bundled PortAudio path when the shell exported one."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def ensure_loadable() -> Path | None:
    """Point ``ctypes.util.find_library("portaudio")`` at the bundled library.

    Returns the bundled path when one is present, otherwise ``None``. The patch
    is installed once and only affects the ``"portaudio"`` lookup, so callers on
    an installed host keep their normal library resolution.
    """
    global _patched

    path = bundled_library()
    if path is None:
        return None
    if _patched:
        return path

    original = ctypes.util.find_library

    def find_library(name: str) -> str | None:
        if name == "portaudio":
            return str(path)
        return original(name)

    ctypes.util.find_library = find_library
    _patched = True
    return path
