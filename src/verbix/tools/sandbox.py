from __future__ import annotations

import asyncio
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path


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


async def execute_sandboxed(command: str, config: SandboxConfig) -> SandboxResult:
    """Executes a command inside a default-deny bubblewrap sandbox.

    Fails closed if bwrap is not available.
    """
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

    for src_path, dst_path, is_rw in config.extra_binds:
        src_res = src_path.resolve()
        if src_res.exists():
            flag = "--bind" if is_rw else "--ro-bind"
            cmd_args.extend([flag, str(src_res), str(dst_path)])

    cmd_args.extend(
        [
            "--chdir",
            str(workspace_path),
            "--die-with-parent",
        ]
    )
    if not config.allow_network:
        cmd_args.append("--unshare-net")

    cmd_args.extend(["--", "bash", "-c", command])

    proc = await asyncio.create_subprocess_exec(
        *cmd_args,
        cwd=str(workspace_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=os.setsid,
    )

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
        # Kill the entire process group
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
