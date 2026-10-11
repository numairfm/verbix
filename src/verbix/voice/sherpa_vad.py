from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

PROVISION_CMD = "uv run scripts/fetch_voice_models.py"

WINDOW = 512  # 32 ms @16kHz: Silero's native window, also its minimum input
THRESHOLD = 0.5
MIN_SILENCE_S = 0.25
MIN_SPEECH_S = 0.15


class Detector(Protocol):
    """The sliver of sherpa_onnx.VoiceActivityDetector we use (fakes match it)."""

    def accept_waveform(self, samples: np.ndarray) -> None: ...
    def flush(self) -> None: ...
    def empty(self) -> bool: ...
    @property
    def front(self) -> Any: ...
    def pop(self) -> None: ...
    def reset(self) -> None: ...


def default_model_path() -> Path:
    from scripts.fetch_voice_models import default_root

    return default_root() / "models" / "silero-vad" / "silero_vad.onnx"


class SileroSegmenter:
    """Speech segmentation backed by sherpa-onnx's Silero VAD.

    Bytes in, speech spans out. Capture feeds arbitrary chunks; they are
    assembled into 512-sample windows here. Ready segments drain via
    pop_segments(); flush() ends the stream and drains the remainder.
    Anything that never becomes a segment (pops, breaths, noise, silence)
    is never decoded downstream — that is the whole design.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = THRESHOLD,
        min_silence_s: float = MIN_SILENCE_S,
        min_speech_s: float = MIN_SPEECH_S,
        sample_rate: int = 16000,
        detector: Detector | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.min_silence_s = min_silence_s
        self.min_speech_s = min_speech_s
        self.sample_rate = sample_rate
        self._detector = detector
        self._pending = np.zeros(0, dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._detector is not None

    def ensure_loaded(self) -> None:
        """Synchronous, idempotent load. First feed of a turn pays ~50 ms once."""
        if self.is_loaded:
            return
        path = self.model_path or default_model_path()
        if not path.is_file():
            raise RuntimeError(
                f"Silero VAD model missing at {path}. "
                f"Run '{PROVISION_CMD}' to download it."
            )
        try:
            import sherpa_onnx
        except (ImportError, OSError) as e:
            raise RuntimeError(f"sherpa-onnx failed to import: {e}.") from e

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(path)
        cfg.silero_vad.threshold = self.threshold
        cfg.silero_vad.min_silence_duration = self.min_silence_s
        cfg.silero_vad.min_speech_duration = self.min_speech_s
        cfg.sample_rate = self.sample_rate
        self._detector = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        logger.info("SileroSegmenter loaded from %s", path)

    def _require(self) -> Detector:
        self.ensure_loaded()
        assert self._detector is not None
        return self._detector

    def feed(self, pcm: bytes) -> None:
        """Accepts raw s16le mono bytes at the contract rate."""
        if not pcm:
            return
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._pending, audio]) if len(self._pending) else audio
        n_full = len(buf) // WINDOW
        det = self._require()
        for i in range(n_full):
            det.accept_waveform(buf[i * WINDOW : (i + 1) * WINDOW])
        self._pending = buf[n_full * WINDOW :]

    def pop_segments(self) -> list[bytes]:
        """Drains ready speech spans as s16le PCM. Short tails stay buffered."""
        return self._drain()

    def flush(self) -> list[bytes]:
        """Ends the stream and drains all speech spans.

        The detector scores the zero-padded remainder itself on flush, so the
        sub-window tail needs no special handling — it arrives as a segment
        if speech, or not at all.
        """
        det = self._require()
        if len(self._pending) > 0:
            padded = np.pad(self._pending, (0, WINDOW - len(self._pending)))
            det.accept_waveform(padded)
        det.flush()
        self._pending = np.zeros(0, dtype=np.float32)
        return self._drain()

    def reset(self) -> None:
        """Drops stream state for a new turn. Model stays loaded."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._require().reset()

    def _drain(self) -> list[bytes]:
        det = self._require()
        out: list[bytes] = []
        while not det.empty():
            seg = det.front
            det.pop()
            samples = np.asarray(seg.samples, dtype=np.float32)
            out.append((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        return out
