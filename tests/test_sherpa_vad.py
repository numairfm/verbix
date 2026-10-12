from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from verbix.voice.sherpa_vad import WINDOW, SileroSegmenter


def test_flush_pads_and_accepts_pending_waveform():
    mock_det = MagicMock()
    mock_det.empty.return_value = True

    seg = SileroSegmenter(detector=mock_det)
    # Feed 200 samples (less than WINDOW=512)
    chunk = (np.ones(200, dtype=np.int16) * 1000).tobytes()
    seg.feed(chunk)

    # detector.accept_waveform should not be called yet because len < 512
    assert mock_det.accept_waveform.call_count == 0
    assert len(seg._pending) == 200

    # Call flush
    seg.flush()

    # detector.accept_waveform MUST have been called with padded WINDOW samples
    assert mock_det.accept_waveform.call_count == 1
    called_samples = mock_det.accept_waveform.call_args[0][0]
    assert len(called_samples) == WINDOW
    assert len(seg._pending) == 0
    mock_det.flush.assert_called_once()


def test_segmenter_bounds_overlong_speech_with_upstream_default():
    """Overlong speech must force-pop a bounded span: upstream default 20s, explicit."""
    from verbix.voice import sherpa_vad as vad_mod

    assert vad_mod.MAX_SPEECH_S == 20.0
    seg = SileroSegmenter(detector=MagicMock())
    assert seg.max_speech_s == 20.0


def test_endpoint_trailing_silence_pinned(tmp_path):
    """Trailing-silence endpoint stays 0.6s and is forwarded to the VAD config."""
    import sys
    import types

    from verbix.voice import sherpa_vad as vad_mod

    assert vad_mod.MIN_SILENCE_S == 0.6

    captured: dict = {}

    class _FakeSilero:
        def __init__(self) -> None:
            self.model: object = None
            self.threshold: object = None
            self.min_silence_duration: object = None
            self.min_speech_duration: object = None
            self.max_speech_duration: object = None

    class _FakeCfg:
        def __init__(self) -> None:
            self.silero_vad = _FakeSilero()
            self.sample_rate: object = None

    class _FakeDet:
        def __init__(self, cfg: object, buffer_size_in_seconds: object = None) -> None:
            captured["cfg"] = cfg

        def accept_waveform(self, samples): ...

        def flush(self): ...

        def empty(self): return True

        @property
        def front(self): raise AssertionError

        def pop(self): ...

        def reset(self): ...

    fake_mod = types.ModuleType("sherpa_onnx")
    fake_mod.VadModelConfig = _FakeCfg  # type: ignore[attr-defined]
    fake_mod.VoiceActivityDetector = _FakeDet  # type: ignore[attr-defined]
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"0")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_mod)
    try:
        seg = SileroSegmenter(model_path=model, detector=None)
        seg.ensure_loaded()
    finally:
        monkeypatch.undo()

    cfg = captured["cfg"]
    assert cfg.silero_vad.max_speech_duration == 20.0
    assert cfg.silero_vad.min_silence_duration == 0.6
