"""No-speech guards at the single decode choke point (signal-based, no word lists)."""

from unittest.mock import MagicMock

import numpy as np

from verbix.voice.transcribe_stt import TranscribeSTT


def test_voiced_duration_gate_skips_model_on_low_energy_noise():
    """When PTT audio is low-energy noise below the voicing threshold, the neural ASR model is never called."""
    stt = object.__new__(TranscribeSTT)
    stt.sample_rate = 16000
    stt._model = MagicMock()

    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.003, 0.003, 16000) * 32768).astype(np.int16).tobytes()
    assert stt.transcribe(noise, sample_rate=16000) == ""
    stt._model.session.assert_not_called()
