import asyncio
import os
import platform
import signal
import subprocess
from collections import deque
from typing import Any

from pydantic import BaseModel

from verbix.minisweagent.utils.serialize import recursive_merge

_HEAD_CAP = 500_000
_TAIL_CAP = 500_000
_KILL_GRACE_SECONDS = 2.0
_POST_EXIT_DRAIN_SECONDS = 0.2


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 600


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)
        self._active_pgids: set[int] = set()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = _run(command, cwd, os.environ | self.config.env, timeout or self.config.timeout)
            return {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:  # noqa: BLE001
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            return {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }

    async def aexecute(self, action: dict, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        """Async execute: new process group, SIGTERM/SIGKILL escalation, capped output, no API keys."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        timeout = timeout or self.config.timeout
        # ponytail: any *_API_KEY counts as an "active provider key" (LiteLLM auto-detects per provider)
        env = {k: v for k, v in os.environ.items() if k != "VERBIX_API_KEY" and not k.endswith("_API_KEY")}
        env |= self.config.env
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        pgid = os.getpgid(proc.pid) if os.name == "posix" else proc.pid
        self._active_pgids.add(pgid)
        out = _CappedOutput()
        drain_task = asyncio.create_task(_drain_stdout(proc, out))
        try:
            try:
                await asyncio.wait_for(proc.wait(), timeout)
            except TimeoutError:
                drain_task.cancel()
                await _kill_group(proc, pgid)
                await _drain_stdout(proc, out)
                raise subprocess.TimeoutExpired(command, timeout, output=out.text())
            try:
                # ponytail: shell exited; a grandchild may hold stdout open, so bound the tail drain and leave it — intentional daemons survive
                await asyncio.wait_for(asyncio.shield(drain_task), _POST_EXIT_DRAIN_SECONDS)
            except TimeoutError:
                drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                    pass
            return {"output": out.text(), "returncode": proc.returncode, "exception_info": ""}
        except asyncio.CancelledError:
            drain_task.cancel()
            await _kill_group(proc, pgid)
            raise
        except Exception as e:  # noqa: BLE001
            raw_output = getattr(e, "output", None) or ""
            return {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        finally:
            self._active_pgids.discard(pgid)

    def kill_all(self) -> None:
        """SIGKILL every tracked in-flight process group (shutdown path; fire-and-forget)."""
        for pgid in list(self._active_pgids):
            try:
                os.killpg(pgid, signal.SIGKILL) if os.name == "posix" else None
            except (ProcessLookupError, PermissionError):
                pass

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), dict(os.environ), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class _CappedOutput:
    """First 500KB + last 500KB of a byte stream, decoded on demand."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_len = 0
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if len(self.head) < _HEAD_CAP:
            self.head += chunk[: _HEAD_CAP - len(self.head)]
        self.tail.append(chunk)
        self.tail_len += len(chunk)
        while self.tail_len > _TAIL_CAP + 65536:
            self.tail_len -= len(self.tail.popleft())

    def text(self) -> str:
        if self.total <= _HEAD_CAP:
            return bytes(self.head).decode("utf-8", errors="replace")
        tail = b"".join(self.tail)
        if self.total <= _HEAD_CAP + _TAIL_CAP:
            return (bytes(self.head) + tail[len(self.head) - (self.total - self.tail_len) :]).decode(
                "utf-8", errors="replace"
            )
        skipped = self.total - _HEAD_CAP - _TAIL_CAP
        return (
            self.head.decode("utf-8", errors="replace")
            + f"\n...[truncated {skipped} bytes]...\n"
            + tail[-_TAIL_CAP:].decode("utf-8", errors="replace")
        )


async def _drain_stdout(proc: asyncio.subprocess.Process, out: _CappedOutput) -> None:
    """Read piped stdout to EOF so the child never blocks on a full pipe."""
    assert proc.stdout is not None
    while True:
        chunk = await proc.stdout.read(65536)
        if not chunk:
            break
        out.feed(chunk)


async def _kill_group(proc: asyncio.subprocess.Process, pgid: int) -> None:
    """SIGTERM the whole process group, escalating to SIGKILL after a grace period."""
    if os.name != "posix":
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), _KILL_GRACE_SECONDS)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), _KILL_GRACE_SECONDS)
    except TimeoutError:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await proc.wait()


def _run(command: str, cwd: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Like subprocess.run, but kills the whole process group on timeout so no children are orphaned."""
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
