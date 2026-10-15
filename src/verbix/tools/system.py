from __future__ import annotations

import asyncio
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import psutil

from verbix.tools.base import ToolContext, ToolResult

_SECTIONS = (
    "overview",
    "battery",
    "cpu",
    "memory",
    "disk",
    "network",
    "processes",
)


def _fmt_gib(n: float) -> str:
    return f"{n / (1024**3):.1f} GiB"


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _battery_line() -> str:
    try:
        batt = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        batt = None
    if batt is None:
        return "Battery: none (desktop / AC power)"
    pct = round(float(batt.percent))
    state = "charging" if batt.power_plugged else "discharging"
    secs = getattr(batt, "secsleft", None)
    if (
        isinstance(secs, (int, float))
        and secs > 0
        and secs
        not in (
            getattr(psutil, "POWER_TIME_UNLIMITED", -2),
            getattr(psutil, "POWER_TIME_UNKNOWN", -1),
        )
    ):
        return f"Battery: {pct}% ({state}, {_fmt_duration(secs)} remaining)"
    return f"Battery: {pct}% ({state})"


def _cpu_line() -> str:
    pct = psutil.cpu_percent(interval=0.05)
    logical = psutil.cpu_count(logical=True) or 1
    phys = psutil.cpu_count(logical=False) or logical
    load_part = ""
    if hasattr(os, "getloadavg"):
        try:
            l1, l5, l15 = os.getloadavg()
            load_part = f" (load avg: {l1:.2f}, {l5:.2f}, {l15:.2f})"
        except OSError:
            pass
    return f"CPU: {pct:.1f}% across {phys} physical / {logical} logical cores{load_part}"


def _memory_line() -> str:
    vm = psutil.virtual_memory()
    return (
        f"Memory: {_fmt_gib(vm.used)} / {_fmt_gib(vm.total)} "
        f"({vm.percent:.1f}% used, {_fmt_gib(vm.available)} available)"
    )


def _disk_line(workspace: Path) -> str:
    target = workspace if workspace.exists() else Path.home()
    du = psutil.disk_usage(str(target))
    return (
        f"Disk ({target}): {_fmt_gib(du.used)} / {_fmt_gib(du.total)} "
        f"({du.percent:.1f}% used, {_fmt_gib(du.free)} free)"
    )


def _network_line() -> str:
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
    except Exception:  # noqa: BLE001
        return "Network: status unavailable"
    active: list[str] = []
    for iface, st in sorted(stats.items()):
        if not getattr(st, "isup", False):
            continue
        low = iface.lower()
        if low in ("lo", "lo0") or "loopback" in low:
            continue
        ips = [
            a.address
            for a in addrs.get(iface, [])
            if getattr(a, "family", None) == socket.AF_INET
            and not str(a.address).startswith("127.")
        ]
        if ips:
            active.append(f"{iface} ({', '.join(ips)})")
        else:
            active.append(iface)
    if not active:
        return "Network: no active non-loopback interfaces"
    return f"Network: {', '.join(active)}"


def _processes_lines() -> str:
    rows: list[tuple[int, int, str]] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            mem = info.get("memory_info")
            rss = int(getattr(mem, "rss", 0) or 0)
            name = str(info.get("name") or "unknown")
            pid = int(info.get("pid") or 0)
            rows.append((rss, pid, name))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda r: r[0], reverse=True)
    top = [
        f"- {name} (PID {pid}, {rss / (1024**2):.0f} MiB RAM)"
        for rss, pid, name in rows[:5]
    ]
    return "Top processes by memory:\n" + ("\n".join(top) if top else "(none)")


class SystemTool:
    """Inspects live host system telemetry (CPU, memory, battery, disk, network, top processes)."""

    name: str = "system"
    side_effecting: bool = False
    description: str = (
        "Inspects live host system status (overview, battery, cpu, memory, disk, network, processes) "
        "across Linux, macOS, and Windows."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": list(_SECTIONS),
                    "description": "System section to inspect (default: 'overview').",
                    "default": "overview",
                }
            },
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        section = str(args.get("section") or "overview").strip().lower()
        if section not in _SECTIONS:
            return ToolResult(
                f"error: invalid section '{section}'; valid sections: {', '.join(_SECTIONS)}",
                is_error=True,
            )
        return await asyncio.to_thread(self._collect, section, context.workspace)

    def _collect(self, section: str, workspace: Path) -> ToolResult:
        try:
            if section == "battery":
                return ToolResult(_battery_line())
            if section == "cpu":
                return ToolResult(_cpu_line())
            if section == "memory":
                return ToolResult(_memory_line())
            if section == "disk":
                return ToolResult(_disk_line(workspace))
            if section == "network":
                return ToolResult(_network_line())
            if section == "processes":
                return ToolResult(_processes_lines())

            uptime = _fmt_duration(time.time() - psutil.boot_time())
            os_line = f"OS: {platform.system()} {platform.release()} ({platform.machine()}), uptime {uptime}"
            lines = [
                os_line,
                _cpu_line(),
                _memory_line(),
                _disk_line(workspace),
                _battery_line(),
                _network_line(),
            ]
            return ToolResult("\n".join(lines))
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: system check failed: {e}", is_error=True)
