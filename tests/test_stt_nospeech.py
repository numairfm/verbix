"""No-speech guards at the single decode choke point (signal-based, no word lists)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from verbix.voice.transcribe_stt import TranscribeSession


class FakeSegmenter:
    def feed(self, pcm: bytes) -> None:
        pass

    def pop_segments(self) -> list[bytes]:
        return []

    def flush(self) -> list[bytes]:
        return []

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
async def test_vad_gate_skips_model_on_low_energy_noise():
    """When the segmenter yields no speech spans for noise, model recognizer is never called."""
    mock_stt = MagicMock()
    sess = TranscribeSession(stt=mock_stt, segmenter=FakeSegmenter(), sample_rate=16000)
    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.003, 0.003, 16000) * 32768).astype(np.int16).tobytes()
    sess.feed(noise)
    assert await sess.finalize() == ""
    mock_stt.transcribe.assert_not_called()
