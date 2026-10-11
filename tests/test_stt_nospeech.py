"""No-speech guards at the single decode choke point (signal-based, no word lists)."""


import numpy as np
import pytest

from verbix.voice.sherpa_stt import SherpaSTT


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
async def test_vad_gate_skips_model_on_low_energy_noise(tmp_path):
    """When the segmenter yields no speech spans for noise, model recognizer is never loaded."""
    stt = SherpaSTT(model_dir=tmp_path, segmenter=FakeSegmenter())
    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.003, 0.003, 16000) * 32768).astype(np.int16).tobytes()
    assert await stt.transcribe(noise) == ""
    assert not stt.is_loaded  # model never loaded/touched
