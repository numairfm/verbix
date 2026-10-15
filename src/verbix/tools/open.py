from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]+:")


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


def extract_raw_paths(args: dict[str, Any]) -> list[str]:
    """Extracts one or more path or URL strings from 'path', 'paths', or 'url' arguments."""
    out: list[str] = []
    for key in ("path", "url"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            if s not in out:
                out.append(s)
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


def validate_openable_paths(raw_paths: list[str], workspace: Path) -> list[Path | str]:
    """Validates safe http(s) URLs or workspace-contained non-executable paths."""
    resolved: list[Path | str] = []
    for raw in raw_paths:
        if "://" in raw or _URI_SCHEME_RE.match(raw):
            parsed = urlparse(raw)
            if (
                parsed.scheme.lower() in ("http", "https")
                and parsed.netloc
                and not any(ch.isspace() for ch in raw)
            ):
                resolved.append(raw)
                continue
            raise ValueError(f"refusing to open unsupported or unsafe URI '{raw}'")
        target = resolve_contained_path(raw, workspace)
        if not target.exists():
            raise ValueError(f"'{raw}' does not exist in the workspace")
        if _is_executable_payload(target):
            raise ValueError(f"refusing to open executable or script file '{raw}'")
        resolved.append(target)
    return resolved


def _target_label(target: Path | str) -> str:
    return target.name if isinstance(target, Path) else str(target)


def _dispatch_open(target: Path | str) -> str:
    label = _target_label(target)
    if sys.platform == "win32" and hasattr(os, "startfile"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return f"Opened '{label}'."
    if sys.platform.startswith("linux"):
        gio_bin = shutil.which("gio")
        cmd = (
            [gio_bin, "open", str(target)]
            if gio_bin
            else ["xdg-open", str(target)]
        )
        subprocess.Popen(
            cmd,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return f"Opened '{label}'."
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
        return f"Opened '{label}'."
    raise RuntimeError(f"Unsupported platform '{sys.platform}'")


class OpenTool:
    """Opens one or more files, folders, or web URLs in the default or a specified desktop application."""

    name: str = "open"
    side_effecting: bool = True
    description: str = (
        "Opens one or more workspace files, folders, or http(s) URLs in their default desktop application "
        "or in a specific installed desktop application (via 'app')."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative workspace path or http(s) URL to open.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of relative workspace paths or http(s) URLs to open.",
                },
                "app": {
                    "type": "string",
                    "description": "Optional installed application name to open with (e.g. 'firefox', 'vlc').",
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

        app_name = (
            (args.get("app") or "").strip() if isinstance(args.get("app"), str) else ""
        )
        if app_name:
            from verbix.tools.app import (
                _match_app,
                discover_installed_apps,
                launch_app_entry,
            )

            catalog = await asyncio.to_thread(discover_installed_apps)
            entry = _match_app(app_name, catalog)
            if entry is None:
                return ToolResult(
                    f"error: no installed desktop application matching '{app_name}' found in the OS catalog",
                    is_error=True,
                )
            try:
                obs = await launch_app_entry(entry, targets)
                return ToolResult(obs)
            except OSError as e:
                return ToolResult(
                    f"error: could not open with '{entry.name}': {e}",
                    is_error=True,
                )

        try:
            results = [
                await asyncio.to_thread(_dispatch_open, target) for target in targets
            ]
            return ToolResult(" ".join(results))
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                f"error: could not open {', '.join(raw_paths)}: {e}",
                is_error=True,
            )
