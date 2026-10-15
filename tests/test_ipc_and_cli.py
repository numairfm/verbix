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


@pytest.mark.asyncio
async def test_cli_voice_mode_continuous_fails_with_clear_error(tmp_path, capsys):
    from verbix.cli import async_main, build_parser

    args = build_parser().parse_args(
        ["--voice", "--voice-mode", "continuous", "--workspace", str(tmp_path)]
    )
    code = await async_main(args)
    err = capsys.readouterr().err
    assert code == 2
    assert "Continuous voice mode has been removed" in err


@pytest.mark.asyncio
async def test_server_with_voice_pipeline_fans_out_wake_output_and_broadcasts_floor_state(
    tmp_path,
):
    from verbix.runtime.floor import FloorState

    rt = _make_runtime(tmp_path)
    voice_events = []

    class FakeVoiceOutput:
        async def emit(self, event):
            voice_events.append(event)

        def delivered_text(self):
            return "spoken"

        def reset_delivered(self):
            pass

    rt.wake_output = FakeVoiceOutput()
    sock_path = tmp_path / "fanout.sock"
    server = VerbixServer(runtime=rt, socket_path=sock_path)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        while not server._clients:
            await asyncio.sleep(0.01)

        # 1. Emitting on wake_output reaches BOTH FakeVoiceOutput AND IPC client
        await rt.wake_output.emit(TextDelta(text="Timer is up."))
        assert len(voice_events) == 1
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        msg = json.loads(line.decode("utf-8"))
        assert msg["type"] == "text_delta"
        assert msg["payload"]["text"] == "Timer is up."

        # 2. Floor state transition broadcasts floor_state runtime_event to IPC client
        rt.floor.set_state(FloorState.LISTENING)
        line2 = await asyncio.wait_for(reader.readline(), timeout=2.0)
        msg2 = json.loads(line2.decode("utf-8"))
        assert msg2["type"] == "runtime_event"
        assert msg2["payload"]["type"] == "floor_state"
        assert msg2["payload"]["payload"]["state"] == "listening"

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_streams_foreground_voice_pipeline_turn_events_to_ipc_clients(
    tmp_path,
):
    from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kwargs):
        yield ProviderTextDelta(text="Hello from voice turn.")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    pipeline = VoicePipeline(
        runtime=rt,
        stt=MockSpeechToText(),
        player=AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.0)),
    )
    sock_path = tmp_path / "voice_turn.sock"
    server = VerbixServer(runtime=rt, pipeline=pipeline, socket_path=sock_path)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        while not server._clients:
            await asyncio.sleep(0.01)

        await pipeline.execute_voice_turn("Hi there")
        received_types = []
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=0.3)
            except TimeoutError:
                break
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            received_types.append(msg["type"])
            if msg["type"] == "turn_complete":
                break
        assert "text_delta" in received_types
        assert "turn_complete" in received_types

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()
        await rt.aclose()


@pytest.mark.asyncio
async def test_ipc_server_and_client_fallback_to_tcp_when_unix_sockets_raise_not_implemented(
    tmp_path, monkeypatch
):
    from verbix.ipc.client import VerbixClient

    async def raise_not_impl(*a, **k):
        raise NotImplementedError("AF_UNIX not supported on Windows event loop")

    monkeypatch.setattr(asyncio, "start_unix_server", raise_not_impl, raising=False)
    monkeypatch.setattr(asyncio, "open_unix_connection", raise_not_impl, raising=False)

    rt = _make_runtime(tmp_path)
    sock_path = tmp_path / "win_fallback.sock"
    server = VerbixServer(runtime=rt, socket_path=sock_path)
    await server.start()
    try:
        assert sock_path.is_file()
        client = VerbixClient(socket_path=sock_path)
        await client.connect()
        assert await client.ping() is True
        await client.close()
    finally:
        await server.stop()
    assert not sock_path.exists()


def test_importing_cli_does_not_eagerly_import_litellm():
    import subprocess
    import sys

    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, verbix.cli; assert 'litellm' not in sys.modules, 'litellm eagerly imported'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr




