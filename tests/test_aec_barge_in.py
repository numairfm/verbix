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
