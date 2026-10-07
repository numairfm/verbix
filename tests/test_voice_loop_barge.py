"""Voice-loop barge tests: input must reach toggle even mid-turn.

Fakes only at hardware/provider edges; loop, pipeline, runtime are real.
"""

import asyncio
import contextlib
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix import cli
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech


class GateSink(MockAudioSink):
    def __init__(self):
        super().__init__(playback_delay=0)
        self.gate = asyncio.Event()
        self.played = []

    async def play_stream(self, chunks):
        async for c in chunks:
            self.played.append(c)
            await self.gate.wait()
        return not getattr(self, "_stopped", False)


class FakeCapture:
    def __init__(self):
        self.is_recording = False
        self.sample_rate = 16000
        self.on_chunk = None
        self.started = False

    async def start_recording(self):
        self.is_recording = True
        self.started = True

    async def stop_recording(self):
        self.is_recording = False
        return b""


@pytest.fixture
def voice_harness(monkeypatch, tmp_path):
    from verbix.runtime.runtime import Runtime

    entered = asyncio.Event()
    release_turn = asyncio.Event()
    gate_sink = GateSink()
    capture = FakeCapture()
    cancels = []

    async def hanging_stream(messages, tools=None, **kwargs):
        entered.set()
        yield ProviderTextDelta(text="Sentence one here. Sentence two here. ")
        await release_turn.wait()

    provider = MagicMock()
    provider.stream = hanging_stream
    provider.warm = AsyncMock(return_value=None)

    runtime = Runtime(workspace=tmp_path, provider=provider)
    orig_cancel = runtime.cancel_turn

    async def spy_cancel(*a, **k):
        cancels.append(True)
        return await orig_cancel(*a, **k)

    runtime.cancel_turn = spy_cancel  # type: ignore[method-assign]
    monkeypatch.setattr(cli, "build_runtime", lambda args: runtime)
    monkeypatch.setattr("verbix.voice.stt.FasterWhisperSTT", lambda: MockSpeechToText())
    tts = MockTextToSpeech()
    tts.aclose = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    monkeypatch.setattr("verbix.voice.pocket_helper.PocketHelperEngine", lambda *a, **k: tts)
    monkeypatch.setattr("verbix.voice.callback_sink.select_sink", lambda *a, **k: gate_sink)

    def fake_capture_cls(*a, **k):
        return capture

    monkeypatch.setattr("verbix.voice.capture.PTTCapture", fake_capture_cls)
    rt_dir = tmp_path / "ttsrt"
    (rt_dir / "bin").mkdir(parents=True)
    monkeypatch.setattr(
        "verbix.voice.tts_assets.ensure_tts_runtime",
        lambda *a, **k: {"bin": rt_dir / "bin", "models": rt_dir / "models", "voices": rt_dir / "voices"},
    )
    return {
        "cli": cli, "runtime": runtime, "entered": entered,
        "release_turn": release_turn, "sink": gate_sink,
        "capture": capture, "cancels": cancels, "tmp": tmp_path,
    }


async def _wait_for(cond, timeout=5.0):
    import time

    start = time.monotonic()
    while not cond():
        assert time.monotonic() - start < timeout, "timed out"
        await asyncio.sleep(0.01)


class _Out:
    def __init__(self, capsys):
        self.capsys = capsys
        self.text = ""

    def poll(self):
        self.text += self.capsys.readouterr().out
        return self.text


def _run_loop(harness, pipe_w):
    parser = cli.build_parser()
    args = parser.parse_args(["--voice", "--workspace", str(harness["tmp"])])
    return asyncio.create_task(cli.async_main(args))


@contextlib.asynccontextmanager
async def _session(harness, pipe_w):
    """Runs the loop; guarantees EOF-unblock and no stragglers on exit."""
    main = _run_loop(harness, pipe_w)
    try:
        yield main
    finally:
        try:
            os.close(pipe_w)
        except OSError:
            pass
        if not main.done():
            main.cancel()
            try:
                await asyncio.wait_for(main, timeout=10)
            except (asyncio.CancelledError, TimeoutError):
                pass


@pytest.mark.asyncio
async def test_typed_enter_during_playback_barges_instead_of_eaten(voice_harness, monkeypatch, capsys):
    h = voice_harness
    r, w = os.pipe()
    monkeypatch.setattr("sys.stdin", os.fdopen(r, "r"))
    out = _Out(capsys)
    async with _session(h, w) as main:
        await _wait_for(lambda: "you> " in out.poll())
        await asyncio.sleep(0.3)  # let drain_stdin + readline settle so input is not eaten
        os.write(w, b"hi\n")
        await _wait_for(lambda: h["entered"].is_set())
        await _wait_for(lambda: len(h["sink"].played) > 0)
        # Turn is mid-playback with audio held; Enter must reach toggle (barge).
        await asyncio.sleep(0.3)
        os.write(w, b"\n")
        await _wait_for(lambda: len(h["cancels"]) > 0)
        await _wait_for(lambda: h["capture"].started)
        assert "[Playback interrupted]" in out.poll()
        await asyncio.sleep(0.3)
        os.write(w, b"exit\n")
        await asyncio.wait_for(main, timeout=10)
    assert h["runtime"].floor.is_idle


@pytest.mark.asyncio
async def test_exit_with_live_turn_cancels_it_cleanly(voice_harness, monkeypatch, capsys):
    h = voice_harness
    r, w = os.pipe()
    monkeypatch.setattr("sys.stdin", os.fdopen(r, "r"))
    out = _Out(capsys)
    async with _session(h, w) as main:
        await _wait_for(lambda: "you> " in out.poll())
        await asyncio.sleep(0.3)  # let drain_stdin + readline settle so input is not eaten
        os.write(w, b"hi\n")
        await _wait_for(lambda: h["entered"].is_set())
        await asyncio.sleep(0.3)
        os.write(w, b"exit\n")
        await asyncio.wait_for(main, timeout=10)
    live = [t for t in asyncio.all_tasks() if t is not main and not t.done()]
    voice_tasks = [t for t in live if "execute_voice_turn" in str(t.get_coro())]
    assert voice_tasks == []
