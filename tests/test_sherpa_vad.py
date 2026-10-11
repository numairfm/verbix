from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from verbix.voice.sherpa_vad import SileroSegmenter, WINDOW


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
