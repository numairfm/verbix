from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import mimetypes
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import resolve_contained_path

_BLOCKED_EXTENSIONS = frozenset({
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".csh",
    ".ksh",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".js",
    ".jse",
    ".wsf",
    ".exe",
    ".msi",
    ".com",
    ".scr",
    ".pif",
    ".desktop",
    ".appimage",
    ".run",
    ".bin",
    ".command",
    ".tool",
})

_MACHO_MAGICS = (
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
)


def _is_executable_payload(target: Path) -> bool:
    suffix = target.suffix.lower()
    if suffix in _BLOCKED_EXTENSIONS or suffix == ".app":
        return True
    if not target.is_file():
        return False
    try:
        with target.open("rb") as f:
            head = f.read(4)
    except OSError:
        return False
    if head.startswith((b"#!", b"\x7fELF", b"MZ")):
        return True
    return head in _MACHO_MAGICS


@contextlib.contextmanager
def _suppress_stdio() -> Iterator[None]:
    """Redirects C-level stdout/stderr (fds 1 and 2) to /dev/null while spawning desktop children."""
    try:
        devnull_fd = os.open(os.devnull, os.O_RDWR)
    except OSError:
        yield
        return
    saved_out = -1
    saved_err = -1
    try:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    except OSError:
        pass
    try:
        yield
    finally:
        for saved, target_fd in ((saved_out, 1), (saved_err, 2)):
            if saved >= 0:
                try:
                    os.dup2(saved, target_fd)
                    os.close(saved)
                except OSError:
                    pass
        try:
            os.close(devnull_fd)
        except OSError:
            pass


def extract_raw_paths(args: dict[str, Any]) -> list[str]:
    """Extracts one or more path strings from 'path' and/or 'paths' arguments."""
    out: list[str] = []
    single = args.get("path")
    if isinstance(single, str) and single.strip():
        out.append(single.strip())
    multi = args.get("paths")
    if isinstance(multi, list):
        for item in multi:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                if s not in out:
                    out.append(s)
    elif isinstance(multi, str) and multi.strip():
        s = multi.strip()
        if s not in out:
            out.append(s)
    return out


def validate_openable_paths(raw_paths: list[str], workspace: Path) -> list[Path]:
    """Validates workspace containment, existence, and non-executable safety for paths."""
    resolved: list[Path] = []
    for path_str in raw_paths:
        target = resolve_contained_path(path_str, workspace)
        if not target.exists():
            raise ValueError(f"'{path_str}' does not exist in the workspace")
        if _is_executable_payload(target):
            raise ValueError(
                f"refusing to open executable or script file '{path_str}'"
            )
        resolved.append(target)
    return resolved


def _linux_gio_open(target: Path) -> str | None:
    """Opens target using libgio-2.0 g_app_info_launch_default_for_uri in C."""
    lib_name = ctypes.util.find_library("gio-2.0")
    if not lib_name:
        return None
    gio = ctypes.CDLL(lib_name)

    gio.g_app_info_get_default_for_type.argtypes = [ctypes.c_char_p, ctypes.c_int]
    gio.g_app_info_get_default_for_type.restype = ctypes.c_void_p
    gio.g_app_info_get_name.argtypes = [ctypes.c_void_p]
    gio.g_app_info_get_name.restype = ctypes.c_char_p
    gio.g_app_info_launch_default_for_uri.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    gio.g_app_info_launch_default_for_uri.restype = ctypes.c_int

    mime = "inode/directory" if target.is_dir() else (mimetypes.guess_type(str(target))[0] or "")
    handler_name = ""
    if mime:
        app_ptr = gio.g_app_info_get_default_for_type(mime.encode("utf-8"), 0)
        if app_ptr:
            raw_name = gio.g_app_info_get_name(app_ptr)
            if raw_name:
                handler_name = raw_name.decode("utf-8", errors="replace")

    uri = target.resolve().as_uri().encode("utf-8")
    with _suppress_stdio():
        ok = gio.g_app_info_launch_default_for_uri(uri, None, None)
    if not ok:
        return None
    via = f" with {handler_name}" if handler_name else ""
    return f"Opened '{target.name}'{via}."


def _dispatch_open(target: Path) -> str:
    if sys.platform == "win32" and hasattr(os, "startfile"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return f"Opened '{target.name}'."
    if sys.platform.startswith("linux"):
        res = _linux_gio_open(target)
        if res is not None:
            return res
        subprocess.Popen(
            ["xdg-open", str(target)],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return f"Opened '{target.name}'."
    if sys.platform == "darwin":
        subprocess.Popen(
            ["/usr/bin/open", str(target)],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return f"Opened '{target.name}'."
    raise RuntimeError(f"Unsupported platform '{sys.platform}'")


class OpenTool:
    """Opens one or more files or folders in the default or a specified desktop application."""

    name: str = "open"
    side_effecting: bool = True
    description: str = (
        "Opens one or more files or folders within the workspace in their default desktop application "
        "or in a specific installed desktop application (via 'app')."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a file or directory to open.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of relative file or directory paths to open.",
                },
                "app": {
                    "type": "string",
                    "description": "Optional installed application name to open the file(s) with (e.g. 'firefox', 'vlc').",
                },
            },
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_paths = extract_raw_paths(args)
        if not raw_paths:
            return ToolResult("error: 'path' or 'paths' is required", is_error=True)
        try:
            targets = validate_openable_paths(raw_paths, context.workspace)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)

        app_name = (args.get("app") or "").strip() if isinstance(args.get("app"), str) else ""
        if app_name:
            from verbix.tools.app import (
                _launch_entry,
                _match_app,
                discover_installed_apps,
            )

            catalog = await asyncio.to_thread(discover_installed_apps)
            entry = _match_app(app_name, catalog)
            if entry is None:
                return ToolResult(
                    f"error: no installed desktop application matching '{app_name}' found in the OS catalog",
                    is_error=True,
                )
            try:
                await asyncio.to_thread(_launch_entry, entry, targets)
                names_str = ", ".join(f"'{t.name}'" for t in targets)
                return ToolResult(f"Opened {names_str} with {entry.name}.")
            except OSError as e:
                return ToolResult(
                    f"error: could not open with '{entry.name}': {e}",
                    is_error=True,
                )

        try:
            results = []
            for target in targets:
                obs = await asyncio.to_thread(_dispatch_open, target)
                results.append(obs)
            return ToolResult(" ".join(results))
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                f"error: could not open {', '.join(raw_paths)}: {e}",
                is_error=True,
            )
