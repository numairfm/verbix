from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verbix.core.models import WorkerResult
from verbix.provider.client import ProviderClient
from verbix.provider.models import Finish, TextDelta, ToolCallDelta
from verbix.runtime.tasks import TaskRuntime
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline, split_sentences
from verbix.voice.player import AudioPlayer


def test_split_sentences():
    text = "Hello world! This is Verbix. How are you today? All good\nLet's go."
    sentences = split_sentences(text)
    assert len(sentences) == 5
    assert sentences[0] == "Hello world!"
    assert sentences[1] == "This is Verbix."
    assert sentences[2] == "How are you today?"
    assert sentences[3] == "All good"
    assert sentences[4] == "Let's go."


@pytest.mark.asyncio
async def test_audio_player_foreground_speech(tmp_path: Path):
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink)

    ok = await player.speak("First sentence.")
    assert ok is True
    assert len(tts.synthesized_texts) == 1
    assert len(sink.played_chunks) == 2


@pytest.mark.asyncio
async def test_audio_player_barge_in_interruption(tmp_path: Path):
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.05)
    player = AudioPlayer(tts=tts, sink=sink)

    task = asyncio.create_task(player.speak("Sentence one. Sentence two."))
    await asyncio.sleep(0.02)
    player.stop()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not player.is_playing


@pytest.mark.asyncio
async def test_audio_player_wake_turn_announcement(tmp_path: Path):
    class MockWakeProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            yield TextDelta(text="Build finished successfully in 4 seconds.")
            yield Finish(reason="stop")

    provider = MockWakeProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink)
    _pipeline = VoicePipeline(runtime=runtime, stt=MockSpeechToText(), player=player)

    task_id = await runtime.spawn_task(title="Background Build", prompt="build")
    worker_res = WorkerResult(
        success=True,
        speech_summary="Build finished successfully in 4 seconds.",
        details="Binaries in dist/.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    # Give wake turn time to synthesize and speak
    await asyncio.sleep(0.15)
    assert any("Build finished" in t for t in tts.synthesized_texts)


@pytest.mark.asyncio
async def test_voice_pipeline_toggle_flow(tmp_path: Path):
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    stt = MockSpeechToText(predefined_transcriptions=["Where is the workspace?"])
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink)

    pipeline = VoicePipeline(runtime=runtime, stt=stt, player=player)

    # 1. Idle toggle -> starts recording
    res1 = await pipeline.toggle()
    assert res1["action"] == "recording"

    # Stop recording simulation
    await pipeline.capture.stop_recording()

    # 2. Test audible cue playback
    await player.play_cue()
    assert len(sink.played_chunks) > 0

    # 3. Test toggle during active playback -> interrupts playback and starts recording
    player._is_playing = True
    res2 = await pipeline.toggle()
    assert res2["action"] == "interrupted_and_recording"
    await pipeline.capture.stop_recording()


@pytest.mark.asyncio
async def test_toggle_state_machine_transitions(tmp_path: Path):
    """Verifies all state transitions of the toggle state machine:

    Idle -> Recording -> Processing -> Idle, and Playing -> Interrupted + Recording.
    """
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    stt = MockSpeechToText(predefined_transcriptions=["Run tests"])
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink)

    class MockCapture:
        def __init__(self) -> None:
            self.is_recording = False
            self.sample_rate = 16000
            self.on_chunk = None

        async def start_recording(self) -> None:
            self.is_recording = True

        async def stop_recording(self) -> bytes:
            self.is_recording = False
            return b"\x00" * 3200

    mock_cap = MockCapture()
    pipeline = VoicePipeline(runtime=runtime, stt=stt, player=player, capture=mock_cap)  # type: ignore[arg-type]

    # 1. Idle -> Toggle -> Recording
    assert not pipeline.capture.is_recording
    assert not player.is_playing
    res1 = await pipeline.toggle()
    assert res1["action"] == "recording"
    assert pipeline.capture.is_recording

    # 2. Recording -> Toggle -> Processing
    res2 = await pipeline.toggle()
    assert res2["action"] == "processing"
    assert res2["transcription"] == "Run tests"
    assert not pipeline.capture.is_recording

    # 3. Playing -> Toggle -> Interrupted and Recording
    player._is_playing = True
    res3 = await pipeline.toggle()
    assert res3["action"] == "interrupted_and_recording"
    assert pipeline.capture.is_recording
    assert not player.is_playing


@pytest.mark.asyncio
async def test_runtime_spoken_acknowledgement_on_spawn(tmp_path: Path):
    """Verifies that spawning a background task immediately triggers spoken acknowledgement via turn complete."""

    class MockSpawnProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            yield ToolCallDelta(
                index=0,
                call_id="c1",
                name="task_spawn",
                arguments_delta='{"title": "compile binaries", "prompt": "cargo build"}',
            )
            yield Finish(reason="tool_calls")

    provider = MockSpawnProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    tts = MockTextToSpeech()
    sink = MockAudioSink()
    player = AudioPlayer(tts=tts, sink=sink)
    pipeline = VoicePipeline(runtime=runtime, stt=MockSpeechToText(), player=player)

    await pipeline.execute_voice_turn("Compile the code")
    assert any("Starting task: compile binaries." in t for t in tts.synthesized_texts)


@pytest.mark.asyncio
async def test_live_stt_session_lifecycle():
    """Verifies LiveSTTSession feed, background progress, and tail finalization."""
    from verbix.voice.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en")
    sess = stt.create_live_session()

    # 1. Empty finish
    sess.start()
    assert await sess.finish() == ""

    # 2. Simulate feeding silence chunks and finishing
    sess.start()
    silence_chunk = b"\x00" * 3200  # 100ms silence
    for _ in range(5):
        sess.feed(silence_chunk)
    text = await sess.finish()
    assert text == ""


@pytest.mark.asyncio
async def test_live_stt_never_drops_non_silent_tail():
    """Regression test: verifies that LiveSTT never drops a short non-silent tail (e.g. final word)."""
    import numpy as np

    from verbix.voice.stt import FasterWhisperSTT

    test_wav = Path("scratch/test_tts.wav")
    if not test_wav.exists():
        pytest.skip("scratch/test_tts.wav not found")

    raw = test_wav.read_bytes()
    data_idx = raw.find(b"data") + 8
    f32_24k = np.frombuffer(raw[data_idx:], dtype=np.float32)
    n_16k = int(len(f32_24k) * 16000 / 24000)
    f32_16k = np.interp(
        np.linspace(0, len(f32_24k) - 1, n_16k),
        np.arange(len(f32_24k)),
        f32_24k,
    ).astype(np.float32)
    i16_pcm = (f32_16k * 32767).astype(np.int16).tobytes()

    stt = FasterWhisperSTT(model_size="tiny.en")
    sess = stt.create_live_session()
    sess.start()
    sess.feed(i16_pcm)

    # Simulate that background worker only transcribed up to 1.8s
    split_byte_idx = int(16000 * 2 * 1.8)
    sess.latest_text = "The build finished successfully in 12"
    sess.last_transcribed_len = split_byte_idx

    final_text = await sess.finish()
    assert "seconds" in final_text.lower()
