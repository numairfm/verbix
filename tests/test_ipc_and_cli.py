import asyncio
import json
from unittest.mock import MagicMock

import pytest

from verbix.cli import build_parser
from verbix.core.events import TextDelta
from verbix.ipc.server import VerbixServer
from verbix.provider.models import Finish as ProviderFinish
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.provider.models import ToolCallDelta as ProviderToolCallDelta
from verbix.runtime.runtime import Runtime, WakeOutputFanOut


def _make_runtime(tmp_path, **kwargs):
    return Runtime(workspace=tmp_path, provider=MagicMock(), **kwargs)


def test_runtime_ipc_methods_exist(tmp_path):
    rt = _make_runtime(tmp_path)
    assert hasattr(rt, "get_task_status_summary")
    assert hasattr(rt, "cancel_task")


def test_get_task_status_summary_empty(tmp_path):
    rt = _make_runtime(tmp_path)
    assert "no active" in rt.get_task_status_summary().lower()
    assert "task-9" in rt.get_task_status_summary("task-9")


def test_get_task_status_summary_lists_ledger(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.ledger.register_task("task-1", "do things")
    assert "task-1" in rt.get_task_status_summary()
    single = rt.get_task_status_summary("task-1")
    assert "task-1" in single
    assert "do things" in single


@pytest.mark.asyncio
async def test_cancel_task_unknown_returns_false(tmp_path):
    rt = _make_runtime(tmp_path)
    assert await rt.cancel_task("task-nope") is False


@pytest.mark.asyncio
async def test_cancel_task_cancels_running_task(tmp_path):
    rt = _make_runtime(tmp_path)
    task = MagicMock()
    task.id = "task-1"
    task.description = "slow job"
    task.status = "running"

    async def _cancel():
        task.status = "cancelled"

    task.cancel = _cancel
    rt.tasks["task-1"] = task
    assert await rt.cancel_task("task-1") is True


@pytest.mark.asyncio
async def test_wake_output_fan_out_reaches_voice_and_broadcast(tmp_path):
    rt = _make_runtime(tmp_path)
    q = rt.subscribe()
    voice_seen = []

    class Voice:
        async def emit(self, event):
            voice_seen.append(event)

    fan = WakeOutputFanOut(Voice(), rt._broadcast_wake_event)
    await fan.emit(TextDelta(text="hello"))
    assert len(voice_seen) == 1
    evt = await asyncio.wait_for(q.get(), timeout=1.0)
    assert evt["type"] == "wake_event"
    assert evt["payload"]["text"] == "hello"
    rt.unsubscribe(q)


@pytest.mark.asyncio
async def test_spawn_broadcasts_ledger_transitions(tmp_path):
    provider = MagicMock()
    stream_calls = 0

    async def fake_stream(messages, tools=None, **kwargs):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield ProviderToolCallDelta(
                index=0,
                call_id="c1",
                name="read",
                arguments_delta=json.dumps({"path": "."}),
            )
            yield ProviderFinish(reason="tool_calls")
        else:
            yield ProviderTextDelta(text="done fast")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    q = rt.subscribe()
    try:
        task_id = await rt.spawn("quick job")
        evt = await asyncio.wait_for(q.get(), timeout=2.0)
        assert evt["type"] == "task_registered"
        assert evt["payload"]["task_id"] == task_id

        await rt.tasks[task_id].wait()
        found = False
        for _ in range(10):
            try:
                evt = await asyncio.wait_for(q.get(), timeout=2.0)
            except TimeoutError:
                break
            if evt["type"] == "task_completed" and evt["payload"]["task_id"] == task_id:
                found = True
                break
        assert found
    finally:
        await rt.aclose()
        rt.unsubscribe(q)


def test_cli_dead_flags_removed():
    parser = build_parser()
    args = parser.parse_args([])
    assert not hasattr(args, "isolate_worktrees")
    assert not hasattr(args, "allow_network")


@pytest.mark.asyncio
async def test_server_stops_in_flight_prompt_task(tmp_path):
    runtime = MagicMock()
    runtime.wake_output = None
    runtime.subscribe.return_value = asyncio.Queue()

    submit_started = asyncio.Event()
    submit_cancelled = asyncio.Event()

    async def slow_submit(prompt, output):
        submit_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            submit_cancelled.set()
            raise

    runtime.submit = slow_submit

    sock_path = tmp_path / "test.sock"
    server = VerbixServer(runtime=runtime, socket_path=sock_path)
    await server.start()

    _reader, writer = await asyncio.open_unix_connection(str(sock_path))
    writer.write(b'{"action": "prompt", "prompt": "hello", "id": "1"}\n')
    await writer.drain()

    await asyncio.wait_for(submit_started.wait(), timeout=2.0)
    assert not submit_cancelled.is_set()

    await server.stop()
    assert submit_cancelled.is_set()

    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_server_sends_terminal_reply_on_prompt_cancel(tmp_path):
    runtime = MagicMock()
    runtime.wake_output = None
    runtime.subscribe.return_value = asyncio.Queue()

    submit_started = asyncio.Event()
    submit_cancelled = asyncio.Event()

    async def slow_submit(prompt, output):
        submit_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            submit_cancelled.set()
            raise

    runtime.submit = slow_submit

    sock_path = tmp_path / "cancel.sock"
    server = VerbixServer(runtime=runtime, socket_path=sock_path)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    writer.write(b'{"action": "prompt", "prompt": "hello", "id": "7"}\n')
    await writer.drain()

    await asyncio.wait_for(submit_started.wait(), timeout=2.0)
    await server.stop()
    assert submit_cancelled.is_set()

    replies = []
    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if not line:
                break
            replies.append(json.loads(line.decode("utf-8")))
    except TimeoutError:
        pass

    writer.close()
    await writer.wait_closed()
    assert any(
        r.get("id") == "7" and r.get("status") == "error" for r in replies
    ), replies


@pytest.mark.asyncio
async def test_server_concurrent_writes_are_serialized(tmp_path):
    runtime = MagicMock()
    runtime.wake_output = None
    runtime.subscribe.return_value = asyncio.Queue()

    sock_path = tmp_path / "test_concurrent.sock"
    server = VerbixServer(runtime=runtime, socket_path=sock_path)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    while not server._clients:
        await asyncio.sleep(0.01)

    # Send many events concurrently from multiple tasks
    async def emit_batch(prefix: str, count: int):
        for i in range(count):
            await server.runtime.wake_output.emit(TextDelta(text=f"{prefix}-{i}"))

    tasks = [
        asyncio.create_task(emit_batch(f"worker-{t}", 50))
        for t in range(5)
    ]
    await asyncio.gather(*tasks)

    # Read all 250 lines and verify each is valid JSON
    lines_read = 0
    for _ in range(250):
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert line, "Premature EOF"
        data = json.loads(line.decode("utf-8"))
        assert data["type"] == "text_delta"
        lines_read += 1

    assert lines_read == 250

    writer.close()
    await writer.wait_closed()
    await server.stop()


def test_drain_stdin_uses_msvcrt_on_windows(monkeypatch):
    import sys
    import types

    from verbix.cli import drain_stdin

    select_called = False

    def exploding_select(*a, **k):
        nonlocal select_called
        select_called = True
        raise AssertionError("select.select must not be called on Windows stdin")

    chars = ["a", "\n"]
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.kbhit = lambda: bool(chars)
    fake_msvcrt.getwch = lambda: chars.pop(0)

    monkeypatch.setattr("verbix.cli.os.name", "nt")
    monkeypatch.setattr("verbix.cli.select.select", exploding_select)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    drain_stdin()
    assert select_called is False
    assert chars == []


def test_voice_cli_sigint_exits_cleanly_without_threadpool_hang_or_traceback(tmp_path):
    import signal
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from unittest.mock import AsyncMock, MagicMock\n"
        "import verbix.cli as cli_mod\n"
        "fake_rt = MagicMock()\n"
        "fake_rt.workspace = " + repr(str(tmp_path)) + "\n"
        "fake_rt.aclose = AsyncMock()\n"
        "fake_pipe = MagicMock()\n"
        "fake_pipe._active_voice_task = None\n"
        "fake_tts = MagicMock()\n"
        "fake_tts.aclose = AsyncMock()\n"
        "fake_sink = MagicMock()\n"
        "cli_mod.build_runtime = lambda args: fake_rt\n"
        "cli_mod._build_voice_pipeline = lambda rt, args, printer=None: (fake_pipe, fake_tts, fake_sink)\n"
        "cli_mod.main(['--voice', '--workspace', " + repr(str(tmp_path)) + "])\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Read until prompt appears so we know it is blocked waiting on stdin
        buf = ""
        assert proc.stdout is not None
        while "you>" not in buf:
            ch = proc.stdout.read(1)
            if not ch:
                break
            buf += ch
        assert "you>" in buf
        import time

        time.sleep(0.1)

        # Send a single SIGINT while stdin pipe stays open (just like a terminal)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=1.5)
        stderr = proc.stderr.read() if proc.stderr else ""
        assert "Traceback" not in stderr
        assert "RuntimeWarning" not in stderr
        assert "_python_exit" not in stderr
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdin:
            proc.stdin.close()


def test_cli_voice_mode_defaults_to_ptt():
    from verbix.cli import build_parser

    args = build_parser().parse_args(["--voice"])
    assert args.voice_mode == "ptt"



