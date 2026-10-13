"""S4: mic-audio dump flag — every finalized turn arrives with its evidence."""

import asyncio
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.cli import build_parser
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def _pcm(n: int = 1600, seed: int = 1) -> bytes:
    return bytes((seed + i) % 256 for i in range(n * 2))


class FakeCapture:
    def __init__(self, sample_rate: int = 16000, audio: bytes = b""):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.on_chunk = None
        self._audio = audio

    async def start_recording(self):
        self.is_recording = True

    async def stop_recording(self):
        self.is_recording = False
        return self._audio


class FakeLiveSession:
    def __init__(self, text: str):
        self.text = text
        self.fed_chunks: list[bytes] = []
        self.on_endpoint = None
        self.sample_rate = 16000
        self.speech_duration = 0.5

    def feed(self, pcm: bytes):
        self.fed_chunks.append(pcm)

    def feed_barge_in(self, pcm: bytes) -> bool:
        return False

    def cancel(self):
        pass

    def reset(self):
        pass

    async def finalize(self):
        return self.text


class FakeSTT:
    def __init__(self, texts: list[str] | None = None):
        self.sessions: list[FakeLiveSession] = []
        self._texts = list(texts or ["hello world"])

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        text = self._texts.pop(0) if self._texts else ""
        sess = FakeLiveSession(text)
        sess.on_endpoint = on_endpoint
        self.sessions.append(sess)
        return sess

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
        return "ptt transcription here"


def _pipeline(runtime, stt, capture, dump_dir=None, **kw):
    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    kwargs = {"dump_mic_dir": dump_dir} if dump_dir is not None else {}
    return VoicePipeline(runtime=runtime, stt=stt, player=player, capture=capture, **kwargs)


def _runtime():
    rt = MagicMock()
    rt.floor = FloorArbiter()
    rt.is_idle = True
    rt.user_active = False
    rt.provider = None
    rt.submit = AsyncMock(return_value="ok")
    rt.cancel_turn = AsyncMock(return_value=None)
    return rt


def _read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        return w.readframes(w.getnframes())


def test_cli_flag_defaults_off_and_parses():
    args = build_parser().parse_args([])
    assert args.dump_mic_dir is None
    args2 = build_parser().parse_args(["--dump-mic-dir", "/tmp/micd"])
    assert Path(args2.dump_mic_dir) == Path("/tmp/micd")


@pytest.mark.asyncio
async def test_unset_writes_nothing_ptt_and_continuous(tmp_path):
    rt = _runtime()
    audio = _pcm()
    cap = FakeCapture(audio=audio)
    pipe = _pipeline(rt, FakeSTT(["turn one"]), cap)
    await pipe.start_recording()
    cap.on_chunk(_pcm())
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "turn one"
    assert list(tmp_path.iterdir()) == []

    rt2 = _runtime()
    pipe2 = _pipeline(rt2, FakeSTT(["hi"]), FakeCapture())
    await pipe2.start_continuous_session()
    pipe2.capture.on_chunk(_pcm())
    pipe2._live_stt.on_endpoint()
    await asyncio.sleep(0.05)
    await pipe2.stop_continuous_session()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_ptt_dumps_wav_named_for_transcript(tmp_path):
    rt = _runtime()
    audio = _pcm()
    cap = FakeCapture(audio=audio)
    pipe = _pipeline(rt, FakeSTT(["ptt hi"]), cap, dump_dir=tmp_path)
    await pipe.start_recording()
    cap.on_chunk(audio)
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "ptt hi"
    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    assert "ptt_hi" in wavs[0].name
    assert _read_wav(wavs[0]) == audio


@pytest.mark.asyncio
async def test_continuous_dumps_finalized_turn_bytes(tmp_path):
    rt = _runtime()
    pipe = _pipeline(rt, FakeSTT(["hello world"]), FakeCapture(), dump_dir=tmp_path)
    await pipe.start_continuous_session()
    chunk = _pcm()
    pipe.capture.on_chunk(chunk)
    pipe._live_stt.on_endpoint()
    await asyncio.sleep(0.05)
    await pipe.stop_continuous_session()
    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    assert "hello_world" in wavs[0].name
    assert _read_wav(wavs[0]) == chunk


@pytest.mark.asyncio
async def test_barge_in_dumps_lookback_seed(tmp_path):
    rt = _runtime()
    stt = FakeSTT(["old", "new"])
    pipe = _pipeline(rt, stt, FakeCapture(), dump_dir=tmp_path)
    pipe._is_continuous_active = True
    pipe._loop = asyncio.get_running_loop()
    await pipe._setup_turn_listening()
    old = stt.sessions[0]
    old.feed_barge_in = lambda chunk: (old.fed_chunks.append(chunk), b"C" in chunk)[1]
    rt.floor.set_state(FloorState.SPEAKING)
    pipe.capture.on_chunk(b"A" * 10000)
    pipe.capture.on_chunk(b"B" * 10000)
    pipe.capture.on_chunk(b"C" * 2000)
    await asyncio.sleep(0.02)
    assert len(stt.sessions) == 2
    lookbacks = sorted(tmp_path.glob("*_lookback.wav"))
    assert len(lookbacks) == 1
    content = _read_wav(lookbacks[0])
    assert b"C" in content
    assert b"A" not in content
    await pipe.stop_continuous_session()


@pytest.mark.asyncio
async def test_dump_write_failure_never_breaks_turn(tmp_path):
    rt = _runtime()
    bad = tmp_path / "file-not-dir"
    bad.write_bytes(b"x")
    pipe = _pipeline(rt, FakeSTT(["still works"]), FakeCapture(audio=_pcm()), dump_dir=bad / "sub")
    await pipe.start_recording()
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "still works"
