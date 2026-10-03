from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verbix.core.models import WorkerResult
from verbix.provider.client import ProviderClient
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

    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    played = await player.speak_foreground_sentences(sentences)

    assert played == "First sentence. Second sentence. Third sentence."
    assert len(tts.synthesized_texts) == 3
    assert len(sink.played_chunks) == 6  # 2 chunks per sentence


@pytest.mark.asyncio
async def test_audio_player_barge_in_interruption(tmp_path: Path):
    tts = MockTextToSpeech()
    # 0.05s per chunk. 2 chunks per sentence = 0.10s per sentence
    sink = MockAudioSink(playback_delay=0.05)
    player = AudioPlayer(tts=tts, sink=sink)

    sentences = ["Sentence one.", "Sentence two.", "Sentence three."]

    async def speak_task():
        return await player.speak_foreground_sentences(sentences)

    task = asyncio.create_task(speak_task())

    # Wait for first sentence (0.10s) to finish, interrupt during sentence two (0.13s)
    await asyncio.sleep(0.13)
    played_so_far = player.interrupt()
    result = await task

    # Interruption stopped subsequent sentences
    assert "Sentence three." not in result
    assert "Sentence one." in played_so_far


@pytest.mark.asyncio
async def test_audio_player_announcement_queue(tmp_path: Path):
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)

    # Register task and finish it
    from verbix.core.models import TaskState

    task_id = "task-build-1"
    runtime.tasks[task_id] = TaskState(
        id=task_id, title="Background Build", status="running"
    )

    worker_res = WorkerResult(
        success=True,
        speech_summary="Build finished successfully in 4 seconds.",
        details="Binaries in dist/.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    task = runtime.get_task(task_id)
    assert task is not None
    assert task.announced is False

    # Drain announcement queue
    drained = await player.drain_announcements()
    assert drained == 1
    assert task.announced is True
    assert "Build finished successfully in 4 seconds." in tts.synthesized_texts


@pytest.mark.asyncio
async def test_voice_pipeline_toggle_flow(tmp_path: Path):
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    stt = MockSpeechToText(predefined_transcriptions=["Where is the workspace?"])
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)

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
    await player.speak_foreground_sentences(["First sentence is playing."])
    # Set playing state manually for test
    player._is_playing = True
    res2 = await pipeline.toggle()
    assert res2["action"] == "interrupted_and_recording"
    await pipeline.capture.stop_recording()

    # 4. Test confirmation and denial gestures
    async def _confirm_background():
        await asyncio.sleep(0.01)
        pipeline.confirm()

    task_c = asyncio.create_task(_confirm_background())
    confirmed = await pipeline.request_confirmation("Delete file?", timeout_seconds=1.0)
    assert confirmed is True
    await task_c

    # Test denial on timeout
    denied = await pipeline.request_confirmation("Drop database?", timeout_seconds=0.05)
    assert denied is False


@pytest.mark.asyncio
async def test_faster_whisper_stt_conformance():
    from verbix.voice.base import SpeechToText
    from verbix.voice.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en")
    assert isinstance(stt, SpeechToText)

    # Empty audio returns empty string without loading model
    assert await stt.transcribe(b"") == ""
    assert await stt.transcribe(b"\x00" * 100) == ""


@pytest.mark.asyncio
async def test_faster_whisper_stt_transcribe_speech():
    import numpy as np

    from verbix.voice.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en")

    # Read scratch/test_tts.wav (IEEE float32 24kHz mono) and resample to 16kHz int16 PCM
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
    )
    i16_pcm = (f32_16k * 32767).astype(np.int16).tobytes()

    text = await stt.transcribe(i16_pcm, sample_rate=16000)
    assert len(text) > 0
    assert any(w in text.lower() for w in ["build", "finished", "success", "seconds"])


@pytest.mark.asyncio
async def test_pocket_tts_engine_conformance():
    from verbix.voice.base import TextToSpeech
    from verbix.voice.pocket_tts import PocketTTSEngine

    tts = PocketTTSEngine()
    assert isinstance(tts, TextToSpeech)

    # Empty text yields nothing
    chunks = [c async for c in tts.synthesize("")]
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_pocket_tts_engine_synthesis():
    from verbix.voice.pocket_tts import PocketTTSEngine

    tts = PocketTTSEngine()
    try:
        chunks = [c async for c in tts.synthesize("Unit test playback.")]
        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        # 24kHz float32 is 4 bytes per sample, non-empty
        assert total_bytes > 0
        assert total_bytes % 4 == 0
    finally:
        await tts.aclose()


def test_subprocess_audio_sink_config():
    from verbix.voice.sinks import SubprocessAudioSink

    sink_f32 = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
    assert sink_f32.sample_rate == 24000
    assert sink_f32.audio_format == "f32"

    sink_s16 = SubprocessAudioSink(sample_rate=16000, audio_format="s16")
    assert sink_s16.sample_rate == 16000
    assert sink_s16.audio_format == "s16"
