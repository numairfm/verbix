import asyncio
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from verbix.voice.capture import PTTCapture


def test_capture_init_defaults():
    cap = PTTCapture(sample_rate=16000, channels=1)
    assert cap.sample_rate == 16000
    assert cap.channels == 1
    assert cap.frame_bytes == 640
    assert not cap.is_recording


@pytest.mark.asyncio
async def test_capture_lifecycle_with_sounddevice():
    """Verify PTTCapture uses sounddevice.RawInputStream and delivers conditioned audio chunks."""
    chunks_received = []

    def on_chunk(data: bytes):
        chunks_received.append(data)

    cap = PTTCapture(sample_rate=16000, channels=1, on_chunk=on_chunk)

    fake_stream = MagicMock()
    fake_stream.start = MagicMock()
    fake_stream.stop = MagicMock()
    fake_stream.close = MagicMock()

    callback_holder = []

    def mock_stream_factory(*args, **kwargs):
        callback_holder.append(kwargs.get("callback"))
        return fake_stream

    with patch("sounddevice.RawInputStream", side_effect=mock_stream_factory):
        await cap.start_recording()
        assert cap.is_recording
        assert len(callback_holder) == 1
        stream_cb = callback_holder[0]

        # Simulate 2 frames (each 640 bytes) arriving from sounddevice callback
        # 16000 Hz mono s16le = 20ms per 640 bytes (320 samples)
        sample_frame = (np.ones(320, dtype=np.int16) * 1000).tobytes()
        stream_cb(sample_frame, 320, None, None)
        stream_cb(sample_frame, 320, None, None)

        # Allow async event loop to process queue
        await asyncio.sleep(0.05)

        recorded = await cap.stop_recording()
        assert not cap.is_recording
        assert fake_stream.stop.called
        assert fake_stream.close.called
        assert len(chunks_received) == 2
        assert len(recorded) == 1280


@pytest.mark.asyncio
async def test_audio_callback_never_raises_after_loop_teardown():
    """Ctrl+C teardown: a late PortAudio callback must not escape (no traceback noise)."""
    cap = PTTCapture(sample_rate=16000, channels=1, on_chunk=lambda _b: None)

    fake_stream = MagicMock()
    callback_holder = []

    def mock_stream_factory(*args, **kwargs):
        callback_holder.append(kwargs.get("callback"))
        return fake_stream

    with patch("sounddevice.RawInputStream", side_effect=mock_stream_factory):
        await cap.start_recording()
        assert cap.is_recording

    class _TornDownLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, *args, **kwargs):
            raise RuntimeError("Event loop is closed")

    cap._loop = _TornDownLoop()
    callback_holder[0](b"\x00" * 640, 320, None, None)  # must not raise
    await cap.stop_recording()


@pytest.mark.asyncio
async def test_start_recording_warns_and_idles_when_stream_fails_to_open(caplog):
    """No input device / broken stream: clear warning, graceful idle, no raise."""
    import sounddevice

    cap = PTTCapture(sample_rate=16000, channels=1)
    with (
        patch(
            "sounddevice.RawInputStream",
            side_effect=sounddevice.PortAudioError("Error opening InputStream: no device"),
        ),
        caplog.at_level(logging.WARNING, logger="verbix.voice.capture"),
    ):
        await cap.start_recording()
    assert not cap.is_recording
    assert any("idle" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_start_recording_warns_when_no_input_device(caplog):
    """Machine with zero input devices: warning names the problem, capture still attempted."""
    cap = PTTCapture(sample_rate=16000, channels=1)
    fake_stream = MagicMock()
    with (
        patch("sounddevice.query_devices", return_value=[]),
        patch("sounddevice.RawInputStream", return_value=fake_stream),
    ):
        with caplog.at_level(logging.WARNING, logger="verbix.voice.capture"):
            await cap.start_recording()
        assert cap.is_recording
        await cap.stop_recording()
    assert any(
        "no audio input device" in r.getMessage().lower() for r in caplog.records
    )
