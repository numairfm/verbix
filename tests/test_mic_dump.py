"""S4: mic-audio dump flag — every finalized PTT turn arrives with its evidence."""

import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.cli import build_parser
from verbix.runtime.floor import FloorArbiter
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def _pcm(n: int = 1600, seed: int = 1) -> bytes:
    return bytes((seed + i) % 256 for i in range(n * 2))


class FakeCapture:
    def __init__(self, sample_rate: int = 16000, audio: bytes = b""):
        self.sample_rate = sample_rate
        self.is_recording = False
        self._audio = audio

    async def start_recording(self):
        self.is_recording = True

    async def stop_recording(self):
        self.is_recording = False
        return self._audio


class FakeSTT:
    def __init__(self, texts: list[str] | None = None):
        self._texts = list(texts or ["hello world"])

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
        return self._texts.pop(0) if self._texts else ""


def _pipeline(runtime, stt, capture, dump_dir=None):
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
async def test_unset_writes_nothing_ptt(tmp_path):
    rt = _runtime()
    audio = _pcm()
    cap = FakeCapture(audio=audio)
    pipe = _pipeline(rt, FakeSTT(["turn one"]), cap)
    await pipe.start_recording()
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "turn one"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_ptt_dumps_wav_named_for_transcript(tmp_path):
    rt = _runtime()
    audio = _pcm()
    cap = FakeCapture(audio=audio)
    pipe = _pipeline(rt, FakeSTT(["ptt hi"]), cap, dump_dir=tmp_path)
    await pipe.start_recording()
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "ptt hi"
    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    assert "ptt_hi" in wavs[0].name
    assert _read_wav(wavs[0]) == audio


@pytest.mark.asyncio
async def test_dump_write_failure_never_breaks_turn(tmp_path):
    rt = _runtime()
    bad = tmp_path / "file-not-dir"
    bad.write_bytes(b"x")
    pipe = _pipeline(rt, FakeSTT(["still works"]), FakeCapture(audio=_pcm()), dump_dir=bad / "sub")
    await pipe.start_recording()
    res = await pipe.stop_recording_and_process()
    assert res["transcription"] == "still works"
