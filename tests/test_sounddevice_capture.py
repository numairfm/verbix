import asyncio
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
