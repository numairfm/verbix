"""No-speech guards at the single decode choke point (signal-based, no word lists)."""

import numpy as np
import pytest

from verbix.voice.sherpa_stt import SherpaSTT


@pytest.mark.asyncio
async def test_vad_gate_skips_model_on_low_energy_noise(tmp_path):
    stt = SherpaSTT(model_dir=tmp_path)
    # Low-amplitude noise under RMS 200 threshold
    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.003, 0.003, 16000) * 32768).astype(np.int16).tobytes()
    assert await stt.transcribe(noise) == ""
    assert not stt.is_loaded  # model never loaded/touched
