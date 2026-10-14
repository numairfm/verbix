from __future__ import annotations

import asyncio
import configparser
import os
import plistlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from verbix.tools.base import ToolContext, ToolResult

_FIELD_CODE_RE = re.compile(r"^%[fFuUdDnNickvm]$")


@dataclass(frozen=True)
class AppEntry:
    id: str
    name: str
    exe_name: str
    argv: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    generic_name: str = ""


def _linux_desktop_dirs() -> list[Path]:
    data_home = Path(
        os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    )
    raw_dirs = os.environ.get(
        "XDG_DATA_DIRS",
        "/usr/local/share:/usr/share:/var/lib/flatpak/exports/share:/var/lib/snapd/desktop",
    )
    dirs = [data_home / "applications"]
    for part in raw_dirs.split(":"):
        if part.strip():
            dirs.append(Path(part.strip()) / "applications")
    return dirs


def _clean_desktop_exec(exec_line: str) -> tuple[str, ...]:
    try:
        tokens = shlex.split(exec_line)
    except ValueError:
        return ()
    cleaned: list[str] = []
    # Skip leading 'env VAR=VAL ...' wrapper if present so argv[0] is the binary
    i = 0
    if tokens and Path(tokens[0]).name == "env":
        i = 1
        while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
            i += 1
    for tok in tokens[i:]:
        if _FIELD_CODE_RE.match(tok):
            continue
        cleaned.append(tok.replace("%%", "%"))
    return tuple(cleaned)


def _discover_linux_apps() -> list[AppEntry]:
    seen_ids: set[str] = set()
    entries: list[AppEntry] = []
    for d in _linux_desktop_dirs():
        if not d.is_dir():
            continue
        for desktop_file in sorted(d.glob("**/*.desktop")):
            if not desktop_file.is_file():
                continue
            app_id = desktop_file.stem.lower()
            if app_id in seen_ids:
                continue
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                cp.read_string(
                    desktop_file.read_text(encoding="utf-8", errors="replace")
                )
            except (configparser.Error, OSError):
                continue
            if not cp.has_section("Desktop Entry"):
                continue
            sec = cp["Desktop Entry"]
            if sec.get("Type", "Application").strip() != "Application":
                continue
            if sec.get("NoDisplay", "false").strip().lower() == "true":
                continue
            if sec.get("Hidden", "false").strip().lower() == "true":
                continue
            if sec.get("Terminal", "false").strip().lower() == "true":
                continue
            name = sec.get("Name", "").strip()
            generic_name = sec.get("GenericName", "").strip()
            exec_line = sec.get("Exec", "").strip()
            if not name or not exec_line:
                continue
            argv = _clean_desktop_exec(exec_line)
            if not argv:
                continue
            exe_name = Path(argv[0]).name.lower()
            kw_raw = ";".join(
                filter(
                    None,
                    [
                        generic_name,
                        sec.get("Keywords", ""),
                        app_id,
                        exe_name,
                    ],
                )
            )
            keywords = tuple(
                k.strip().lower() for k in re.split(r"[;,]", kw_raw) if k.strip()
            )
            seen_ids.add(app_id)
            entries.append(
                AppEntry(
                    id=app_id,
                    name=name,
                    exe_name=exe_name,
                    argv=argv,
                    keywords=keywords,
                    generic_name=generic_name,
                )
            )
    return sorted(entries, key=lambda e: e.name.lower())


def _macos_app_dirs() -> list[Path]:
    return [
        Path("/Applications"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    ]


def _discover_macos_apps() -> list[AppEntry]:
    seen: set[str] = set()
    entries: list[AppEntry] = []
    for d in _macos_app_dirs():
        if not d.is_dir():
            continue
        for bundle in sorted(d.glob("*.app")):
            plist_path = bundle / "Contents" / "Info.plist"
            if not plist_path.is_file():
                continue
            try:
                with plist_path.open("rb") as f:
                    info = plistlib.load(f)
            except (plistlib.InvalidFileException, OSError):
                continue
            name = (
                info.get("CFBundleDisplayName")
                or info.get("CFBundleName")
                or bundle.stem
            )
            exe_rel = info.get("CFBundleExecutable") or bundle.stem
            exe_path = bundle / "Contents" / "MacOS" / str(exe_rel)
            app_id = str(info.get("CFBundleIdentifier") or bundle.stem).lower()
            if app_id in seen:
                continue
            seen.add(app_id)
            entries.append(
                AppEntry(
                    id=app_id,
                    name=str(name),
                    exe_name=str(exe_rel).lower(),
                    argv=(str(exe_path),),
                    keywords=(bundle.stem.lower(), app_id),
                )
            )
    return sorted(entries, key=lambda e: e.name.lower())


def _discover_windows_apps() -> list[AppEntry]:
    entries: list[AppEntry] = []
    seen: set[str] = set()
    start_dirs = [
        Path(os.environ.get("ProgramData", r"C:\ProgramData"))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs",
        Path(os.environ.get("APPDATA", str(Path.home())))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs",
    ]
    for d in start_dirs:
        if not d.is_dir():
            continue
        for lnk in sorted(d.glob("**/*.lnk")):
            key = lnk.stem.lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                AppEntry(
                    id=key,
                    name=lnk.stem,
                    exe_name=key,
                    argv=(str(lnk),),
                    keywords=(key,),
                )
            )
    return sorted(entries, key=lambda e: e.name.lower())


def discover_installed_apps() -> list[AppEntry]:
    if sys.platform.startswith("linux"):
        return _discover_linux_apps()
    if sys.platform == "darwin":
        return _discover_macos_apps()
    if sys.platform == "win32":
        return _discover_windows_apps()
    return []


def _match_app(query: str, catalog: list[AppEntry]) -> AppEntry | None:
    q = query.strip().lower()
    if not q or any(ch in q for ch in ("/", "\\", ";", "|", "&", "$", "`")):
        return None
    # 1. Exact match on id, name, or exe_name
    for entry in catalog:
        if q in (entry.id.lower(), entry.name.lower(), entry.exe_name.lower()):
            return entry
    # 2. Substring match on name or id
    for entry in catalog:
        if q in entry.name.lower() or q in entry.id.lower():
            return entry
    # 3. Keyword match
    for entry in catalog:
        if any(q in kw for kw in entry.keywords):
            return entry
    return None


def _running_user_procs() -> list[Any]:
    my_pid = os.getpid()
    my_uid = os.getuid() if hasattr(os, "getuid") else None
    out: list[Any] = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "uids"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if pid == my_pid:
                continue
            uids = info.get("uids")
            if my_uid is not None and uids is not None and getattr(uids, "real", my_uid) != my_uid:
                continue
            out.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _matching_procs(entry: AppEntry, procs: list[Any] | None = None) -> list[Any]:
    base_exe = entry.exe_name.lower().removesuffix("-bin")
    target_names = {
        entry.exe_name.lower(),
        base_exe,
        f"{base_exe}-bin",
        entry.id.lower(),
        entry.name.lower(),
    }
    matched: list[Any] = []
    for proc in (procs if procs is not None else _running_user_procs()):
        pname = (proc.info.get("name") or "").lower()
        pexe = Path(proc.info.get("exe") or "").name.lower()
        cmdline = proc.info.get("cmdline") or []
        cmd0 = Path(cmdline[0]).name.lower() if cmdline else ""
        if (
            pname in target_names
            or pname.removesuffix("-bin") in target_names
            or pexe in target_names
            or cmd0 in target_names
        ):
            matched.append(proc)
    return matched


def _launch_entry(entry: AppEntry) -> int | None:
    if sys.platform == "win32" and hasattr(os, "startfile"):
        os.startfile(entry.argv[0])  # type: ignore[attr-defined]
        return None
    proc = subprocess.Popen(
        list(entry.argv),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return getattr(proc, "pid", None)


class AppTool:
    """Discovers, lists, launches, and closes desktop applications using OS app catalogs and psutil."""

    name: str = "app"
    side_effecting: bool = True
    description: str = (
        "Lists, launches, or closes installed desktop applications using the OS application catalog "
        "(XDG .desktop entries on Linux, .app bundles on macOS, Start Menu on Windows) and psutil."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "launch", "close"],
                    "description": "Application action: 'list', 'launch', or 'close'.",
                },
                "name": {
                    "type": "string",
                    "description": "Application name (required for 'launch' and 'close'; optional filter for 'list').",
                },
            },
            "required": ["action"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        action = (args.get("action") or "").strip().lower()
        name = (args.get("name") or "").strip()
        if action not in ("list", "launch", "close"):
            return ToolResult(
                f"error: invalid action '{action}'; expected 'list', 'launch', or 'close'",
                is_error=True,
            )

        catalog = await asyncio.to_thread(discover_installed_apps)

        if action == "list":
            procs = await asyncio.to_thread(_running_user_procs)
            running_apps = [
                e.name for e in catalog if _matching_procs(e, procs)
            ]
            installed = [
                f"{e.name} ({e.generic_name})"
                if e.generic_name and e.generic_name.lower() != e.name.lower()
                else e.name
                for e in catalog
                if not name
                or name.lower() in e.name.lower()
                or name.lower() in e.id.lower()
                or name.lower() in e.generic_name.lower()
                or any(name.lower() in kw for kw in e.keywords)
            ]
            lines = []
            if running_apps:
                lines.append(f"Running desktop apps: {', '.join(sorted(set(running_apps)))}")
            if installed:
                lines.append(f"Installed desktop apps: {', '.join(installed)}")
            return ToolResult("\n".join(lines) if lines else "No matching desktop applications found.")

        if not name:
            return ToolResult(
                f"error: 'name' is required when action='{action}'",
                is_error=True,
            )

        entry = _match_app(name, catalog)
        if entry is None:
            return ToolResult(
                f"error: no installed desktop application matching '{name}' found in the OS catalog",
                is_error=True,
            )

        if action == "launch":
            try:
                spawned_pid = await asyncio.to_thread(_launch_entry, entry)
                running = await asyncio.to_thread(_matching_procs, entry)
                for _ in range(5):
                    if running:
                        break
                    await asyncio.sleep(0.05)
                    running = await asyncio.to_thread(_matching_procs, entry)
                pid = running[0].pid if running else spawned_pid
                pid_part = f", PID {pid}" if pid is not None else ""
                return ToolResult(
                    f"Launched '{entry.name}' (verified running{pid_part})."
                )
            except OSError as e:
                return ToolResult(f"error: failed to launch '{entry.name}': {e}", is_error=True)

        # action == "close"
        matched = await asyncio.to_thread(_matching_procs, entry)
        closed = 0
        for proc in matched:
            try:
                proc.terminate()
                closed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if closed == 0:
            return ToolResult(f"No running '{entry.name}' process found to close.")
        still_alive = False
        wait_fn = getattr(psutil, "wait_procs", None)
        if callable(wait_fn):
            try:
                _, alive = await asyncio.to_thread(wait_fn, matched, 2.0)
                still_alive = bool(alive)
            except (AttributeError, TypeError):
                pass
        state_str = "termination signaled" if still_alive else "verified stopped"
        return ToolResult(
            f"Closed '{entry.name}' ({closed} process(es) terminated; {state_str})."
        )
