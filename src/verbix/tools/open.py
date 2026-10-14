from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import mimetypes
import os
import subprocess
import sys
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
    if not target.is_file():
        return False
    suffix = target.suffix.lower()
    if suffix in _BLOCKED_EXTENSIONS:
        return True
    try:
        with target.open("rb") as f:
            head = f.read(4)
    except OSError:
        return False
    if head.startswith((b"#!", b"\x7fELF", b"MZ")):
        return True
    return head in _MACHO_MAGICS


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
    """Opens a file or folder in the user's default desktop application."""

    name: str = "open"
    side_effecting: bool = True
    description: str = (
        "Opens a file or folder within the workspace in its default desktop application "
        "(like double-clicking it in the file manager)."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file or directory to open.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = (args.get("path") or "").strip()
        if not path_str:
            return ToolResult("error: 'path' is required", is_error=True)
        try:
            target = resolve_contained_path(path_str, context.workspace)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)

        if not target.exists():
            return ToolResult(
                f"error: '{path_str}' does not exist in the workspace",
                is_error=True,
            )
        if _is_executable_payload(target):
            return ToolResult(
                f"error: refusing to open executable or script file '{path_str}'",
                is_error=True,
            )
        try:
            obs = await asyncio.to_thread(_dispatch_open, target)
            return ToolResult(obs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                f"error: could not open '{path_str}': {e}",
                is_error=True,
            )
