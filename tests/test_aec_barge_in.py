from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from verbix.core.events import TextDelta
from verbix.runtime.floor import FloorArbiter
from verbix.voice.aec import EchoCanceller
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def test_echo_canceller_attenuates_speaker_echo():
    """Synthetic reference signal is attenuated from mic stream by WebRTC AEC."""
    aec = EchoCanceller(sample_rate=16000)
    if not aec.is_available:
        pytest.skip("pywebrtc_audio unavailable in current environment")

    # 1 second synthetic 880Hz speaker tone @ 24kHz
    t_24k = np.linspace(0, 1, 24000, endpoint=False)
    speaker_audio_24k = (np.sin(2 * np.pi * 880 * t_24k) * 15000).astype(np.int16).tobytes()

    # Feed speaker reference
    aec.feed_reference(speaker_audio_24k, source_sample_rate=24000)

    # Mic hears speaker echo (880Hz @ 16kHz) + user voice (440Hz @ 16kHz)
    t_16k = np.linspace(0, 1, 16000, endpoint=False)
    spk_16k = (np.sin(2 * np.pi * 880 * t_16k) * 15000).astype(np.int16)
    user_16k = (np.sin(2 * np.pi * 440 * t_16k) * 8000).astype(np.int16)
    mic_input = (spk_16k + user_16k).tobytes()

    cleaned_bytes = aec.process_mic(mic_input)
    cleaned_samples = np.frombuffer(cleaned_bytes, dtype=np.int16)

    # Initial energy of raw mic
    raw_energy = np.mean(np.abs(spk_16k + user_16k))
    # Adapted energy of cleaned mic (second half of audio after AEC filter converges)
    cleaned_energy = np.mean(np.abs(cleaned_samples[8000:]))

    # Cleaned energy should be significantly lower than combined energy because 880Hz was cancelled
    assert cleaned_energy < raw_energy


class FakeCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.on_chunk = None

    async def start_recording(self):
        self.is_recording = True

    async def stop_recording(self):
        self.is_recording = False
        return b""


@pytest.mark.asyncio
async def test_continuous_barge_in_interrupts_active_playback():
    """User speech detected during assistant playback immediately interrupts speech."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = False
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink(playback_delay=0.1)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)

    fake_session = MagicMock()
    fake_session.is_speech_detected = False
    fake_session.feed = MagicMock()
    fake_session.feed_barge_in = MagicMock(side_effect=lambda chunk: fake_session.is_speech_detected)

    stt = MagicMock()
    stt.open_session = MagicMock(return_value=fake_session)

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=FakeCapture(),
    )
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()

    # Start speech playback
    await pipeline.turn_output.emit(TextDelta(text="Verbix is currently talking to the user. "))
    await asyncio.sleep(0.02)
    assert pipeline.turn_output.is_playing is True

    # 1. Normal background chunk while assistant speaks: no user speech detected
    chunk_silence = b"\x00" * 320
    pipeline.capture.on_chunk(chunk_silence)
    assert pipeline.turn_output.is_playing is True
    runtime.cancel_turn.assert_not_called()

    # 2. User speaks over Verbix: VAD marks speech detected
    fake_session.is_speech_detected = True
    chunk_speech = b"\x10\x20" * 160
    pipeline.capture.on_chunk(chunk_speech)

    # Playback must be interrupted immediately
    assert pipeline.turn_output.is_playing is False
    assert sink.stopped is True
    await asyncio.sleep(0.01)
    runtime.cancel_turn.assert_called_once()


def test_echo_canceller_accepts_float32_reference_audio():
    """SherpaTTS produces 24kHz float32 mono PCM. AEC must accept float32 and convert to valid int16."""
    aec = EchoCanceller(sample_rate=16000)
    if not aec.is_available:
        pytest.skip("pywebrtc_audio unavailable in current environment")

    # 1 second synthetic 880Hz speaker tone @ 24kHz float32 in [-0.5, 0.5]
    t_24k = np.linspace(0, 1, 24000, endpoint=False)
    speaker_f32 = (np.sin(2 * np.pi * 880 * t_24k) * 0.5).astype(np.float32).tobytes()

    aec.feed_reference(speaker_f32, source_sample_rate=24000)

    # Reference buffer should contain resampled 16kHz int16 bytes
    assert len(aec._ref_buffer) == 16000 * 2
    ref_samples = np.frombuffer(bytes(aec._ref_buffer), dtype=np.int16)
    # Scaled int16 amplitude should be around ~16384 (0.5 * 32767), not noise around 32767
    assert np.max(np.abs(ref_samples)) <= 17000
    assert np.max(np.abs(ref_samples)) >= 15000


@pytest.mark.asyncio
async def test_continuous_loop_survives_barge_in_cancellation():
    """When a turn is cancelled by barge-in, the continuous loop task must survive and keep running."""
    from verbix.runtime.runtime import TurnHandle

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False

    turn_cancelled = False

    async def mock_submit(text, output):
        nonlocal turn_cancelled
        runtime.floor.set_state("SPEAKING")
        current_task = asyncio.current_task()
        runtime._turn_handle = TurnHandle(
            task=current_task,
            kind="foreground",
            delivered_text_provider=lambda: "",
        )
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            turn_cancelled = True
            raise
        return "finished"

    async def mock_cancel_turn(heard=None):
        if hasattr(runtime, "_turn_handle") and runtime._turn_handle.task:
            runtime._turn_handle.task.cancel()

    runtime.submit = mock_submit
    runtime.cancel_turn = mock_cancel_turn

    sink = MockAudioSink(playback_delay=0.1)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)

    fake_session = MagicMock()
    fake_session.is_speech_detected = False
    fake_session.feed = MagicMock()
    fake_session.feed_barge_in = MagicMock(return_value=False)
    fake_session.finalize = AsyncMock(return_value="tell me a story")

    stt = MagicMock()
    stt.open_session = MagicMock(return_value=fake_session)

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=FakeCapture(),
    )

    await pipeline.start_continuous_session()
    assert pipeline._continuous_task is not None
    assert not pipeline._continuous_task.done()

    # 1. Trigger speech endpoint for Turn 1
    pipeline._speech_endpoint_event.set()
    await asyncio.sleep(0.05)

    # Turn is now running in mock_submit
    assert runtime.floor.is_speaking is True
    assert not pipeline._continuous_task.done()

    # 2. Barge in while turn is running
    fake_session.is_speech_detected = True
    fake_session.feed_barge_in = MagicMock(return_value=True)
    pipeline.capture.on_chunk(b"\x10\x20" * 160)

    await asyncio.sleep(0.05)

    # Turn task was cancelled
    assert turn_cancelled is True
    # CRITICAL INVARIANT: The continuous loop task MUST NOT be dead
    assert not pipeline._continuous_task.done(), "Continuous task was killed by cancel_turn!"
    assert pipeline.is_continuous_active is True

    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_barge_in_cancels_turn_before_first_audio_playback():
    """User speech while LLM is generating (floor is SPEAKING but is_playing is False) must cancel turn."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.floor.set_state("SPEAKING")
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink(playback_delay=0.1)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    fake_session = MagicMock()
    fake_session.feed_barge_in = MagicMock(return_value=True)
    stt = MagicMock()
    stt.open_session = MagicMock(return_value=fake_session)

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=FakeCapture(),
    )
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()
    # Turn begins: LLM generating tokens (floor acquired as SPEAKING)
    runtime.floor.set_state("SPEAKING")

    # SpeechOutput has NOT started playing yet
    assert pipeline.turn_output.is_playing is False
    assert runtime.floor.is_speaking is True

    # User speaks "no no stop nevermind"
    pipeline.capture.on_chunk(b"\x10\x20" * 160)

    await asyncio.sleep(0.02)
    runtime.cancel_turn.assert_called_once()

