from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ACTIVE_PGIDS: set[int] = set()
_BWRAP_PROBE_CACHE: tuple[bool, str] | None = None
_ENV_ALLOWLIST = ("PATH", "LANG", "TERM")


@dataclass(frozen=True)
class SandboxConfig:
    workspace: Path
    allow_network: bool = False
    timeout_seconds: float = 60.0
    extra_binds: tuple[tuple[Path, Path, bool], ...] = ()


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SandboxStatus:
    available: bool
    backend: str | None
    reason: str
    capabilities: str


def _probe_bwrap() -> tuple[bool, str]:
    global _BWRAP_PROBE_CACHE
    if _BWRAP_PROBE_CACHE is not None:
        return _BWRAP_PROBE_CACHE
    if sys.platform != "linux":
        return False, f"no sandbox backend for {sys.platform}"
    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        _BWRAP_PROBE_CACHE = (False, "bwrap executable not found on PATH")
        return _BWRAP_PROBE_CACHE
    try:
        res = subprocess.run(
            [bwrap_path, "--ro-bind", "/", "/", "--", "true"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if res.returncode == 0:
            _BWRAP_PROBE_CACHE = (True, "ok")
        else:
            detail = (res.stderr or res.stdout or f"exit {res.returncode}").strip()
            _BWRAP_PROBE_CACHE = (False, f"bwrap namespace check failed: {detail}")
    except (OSError, subprocess.SubprocessError) as e:
        _BWRAP_PROBE_CACHE = (False, f"bwrap probe error: {e}")
    return _BWRAP_PROBE_CACHE


def probe_sandbox(allow_network: bool = False) -> SandboxStatus:
    """Probes whether the mandatory OS sandbox backend is available and reports its capabilities."""
    if sys.platform != "linux":
        return SandboxStatus(
            available=False,
            backend=None,
            reason=f"no sandbox backend for {sys.platform}",
            capabilities="",
        )
    ok, reason = _probe_bwrap()
    if not ok:
        return SandboxStatus(
            available=False,
            backend=None,
            reason=reason,
            capabilities="",
        )
    net_desc = "network enabled" if allow_network else "network isolated (--unshare-net)"
    caps = (
        "Executes a bash command in the workspace inside a mandatory Linux bubblewrap (bwrap) sandbox "
        f"(workspace and /tmp read-write, system paths read-only, isolated PID namespace, masked .env/auth.json, {net_desc})."
    )
    return SandboxStatus(
        available=True,
        backend="bwrap",
        reason="ok",
        capabilities=caps,
    )


def build_sandbox_env(workspace: Path) -> dict[str, str]:
    """Builds the child environment strictly from an allowlist (PATH, LANG, TERM + scratch dirs)."""
    scratch_home = workspace / ".verbix" / "home"
    scratch_home.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(scratch_home),
        "TMPDIR": "/tmp",
        "HISTFILE": "/dev/null",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }
    for key in _ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def kill_active_processes() -> None:
    """Sends SIGKILL to all tracked active sandbox process groups."""
    for pgid in list(_ACTIVE_PGIDS):
        try:
            if os.name == "posix":
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _ACTIVE_PGIDS.clear()


async def execute_sandboxed(command: str, config: SandboxConfig) -> SandboxResult:
    """Executes a command inside a default-deny bubblewrap sandbox.

    Fails closed if the sandbox backend is unavailable.
    """
    status = probe_sandbox(allow_network=config.allow_network)
    if not status.available:
        raise RuntimeError(
            f"Sandbox unavailable ({status.reason}); sandboxed execution is required"
        )
    workspace_path = config.workspace.resolve()
    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        raise RuntimeError(
            "bwrap executable not found; sandboxed execution is required"
        )

    cmd_args = [
        bwrap_path,
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
        "--ro-bind-try",
        "/etc/alternatives",
        "/etc/alternatives",
        "--ro-bind-try",
        "/etc/ssl",
        "/etc/ssl",
        "--ro-bind-try",
        "/etc/pki",
        "/etc/pki",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(workspace_path),
        str(workspace_path),
    ]

    # Mask secret files inside workspace with a read-only empty file so bash cannot read or overwrite them
    secret_candidates = [
        workspace_path / ".env",
        workspace_path / ".verbix" / "auth.json",
    ]
    verbix_dir = workspace_path / ".verbix"
    if verbix_dir.is_dir():
        secret_candidates.extend(verbix_dir.glob("auth.*"))
    existing_secrets = list(dict.fromkeys(p for p in secret_candidates if p.is_file()))
    if existing_secrets:
        mask_file = workspace_path / ".verbix" / ".mask"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        mask_file.write_bytes(b"")
        cmd_args.extend(["--ro-bind", str(mask_file), str(mask_file)])
        for secret_path in existing_secrets:
            cmd_args.extend(["--ro-bind", str(mask_file), str(secret_path)])

    for src_path, dst_path, is_rw in config.extra_binds:
        src_res = src_path.resolve()
        if src_res.exists():
            flag = "--bind" if is_rw else "--ro-bind"
            cmd_args.extend([flag, str(src_res), str(dst_path)])

    cmd_args.extend(
        [
            "--chdir",
            str(workspace_path),
            "--unshare-pid",
            "--die-with-parent",
        ]
    )
    if not config.allow_network:
        cmd_args.append("--unshare-net")

    cmd_args.extend(["--", "bash", "-c", command])
    child_env = build_sandbox_env(workspace_path)

    proc = await asyncio.create_subprocess_exec(
        *cmd_args,
        cwd=str(workspace_path),
        env=child_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid) if os.name == "posix" else proc.pid
    _ACTIVE_PGIDS.add(pgid)

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=config.timeout_seconds
        )
        return SandboxResult(
            exit_code=proc.returncode if proc.returncode is not None else 1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )
    except (TimeoutError, asyncio.CancelledError):
        try:
            if os.name == "posix":
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    finally:
        _ACTIVE_PGIDS.discard(pgid)
