from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_VOICED_FRAME_S = 0.032
_MIN_VOICED_S = 0.15
_VOICED_RMS = 300.0


def _has_min_voiced_duration(
    audio_float: np.ndarray,
    sample_rate: int = 16000,
    min_voiced_s: float = _MIN_VOICED_S,
) -> bool:
    """True when at least min_voiced_s of audio sits above the voicing floor.

    Rejects transient clicks (Enter keypress, trackpad, mic pops) without calling
    the neural ASR model.
    """
    frame_len = int(sample_rate * _VOICED_FRAME_S)
    if audio_float.size < frame_len:
        return False
    n_full = len(audio_float) // frame_len
    voiced = 0
    for i in range(n_full):
        frame = audio_float[i * frame_len : (i + 1) * frame_len]
        rms = np.sqrt(float(np.mean(frame**2)) * 32768.0**2)
        if rms >= _VOICED_RMS:
            voiced += 1
    return (voiced * _VOICED_FRAME_S) >= min_voiced_s


def find_default_stt_model() -> Path:
    """Discovers the STT model path from env, local runtime, or default root.

    Decoupled and swappable: accepts any GGUF model path without hardcoding.
    """
    if env := os.environ.get("VERBIX_STT_MODEL"):
        return Path(env).resolve()

    from scripts.fetch_voice_models import default_root

    root = default_root() / "models"
    # 1. Look for specific nemotron-gguf folder
    nemotron_dir = root / "nemotron-gguf"
    if nemotron_dir.is_dir():
        for f in nemotron_dir.glob("*.gguf"):
            return f.resolve()

    # 2. Look for any .gguf in models directory
    if root.is_dir():
        for f in root.glob("**/*.gguf"):
            return f.resolve()

    return root / "nemotron-gguf" / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"


class TranscribeSession:
    """Incremental audio session that buffers PCM until endpointing,

    then transcribes the complete turn in a single clean pass.
    """

    def __init__(
        self,
        stt: TranscribeSTT,
        segmenter: Any | None = None,
        sample_rate: int = 16000,
        on_endpoint: Any | None = None,
    ) -> None:
        self._stt = stt
        self._segmenter = segmenter
        self.sample_rate = sample_rate
        self.on_endpoint = on_endpoint
        self._buf = bytearray()
        self._finalized = False
        self._duration: float = 0.0
        self.fed_chunks: list[bytes] = []

    @property
    def speech_duration(self) -> float:
        if self._duration > 0:
            return self._duration
        return (len(self._buf) // 2) / float(self.sample_rate) if self.sample_rate else 0.0

    def feed(self, chunk: bytes) -> None:
        if self._finalized or not chunk:
            return
        self.fed_chunks.append(chunk)
        self._buf.extend(chunk)
        if self._segmenter is not None and hasattr(self._segmenter, "feed"):
            try:
                self._segmenter.feed(chunk)
            except Exception as e:  # noqa: BLE001
                logger.debug("Segmenter feed error: %s", e)
            if hasattr(self._segmenter, "pop_segments"):
                spans = self._segmenter.pop_segments()
                if spans and self.on_endpoint:
                    try:
                        self.on_endpoint()
                    except Exception as e:  # noqa: BLE001
                        logger.debug("on_endpoint callback error: %s", e)

    def feed_barge_in(self, chunk: bytes) -> bool:
        if self._finalized or not chunk:
            return False
        if self._segmenter is not None and hasattr(self._segmenter, "feed"):
            try:
                self._segmenter.feed(chunk)
            except Exception as e:  # noqa: BLE001
                logger.debug("Segmenter feed error during barge-in: %s", e)
            return bool(getattr(self._segmenter, "is_speech_detected", False))
        return False

    def cancel(self) -> None:
        self._finalized = True
        self._buf.clear()
        if self._segmenter is not None and hasattr(self._segmenter, "reset"):
            self._segmenter.reset()

    def reset(self) -> None:
        self.cancel()
        self._finalized = False

    async def finalize(self) -> str:
        if self._finalized:
            return ""
        self._finalized = True
        if self._segmenter is not None and hasattr(self._segmenter, "flush"):
            try:
                self._segmenter.flush()
            except Exception as e:  # noqa: BLE001
                logger.debug("Segmenter flush error: %s", e)
        audio = bytes(self._buf)
        self._duration = (len(audio) // 2) / float(self.sample_rate) if self.sample_rate else 0.0
        self._buf.clear()
        if not audio or len(audio) < 2400:  # < 150ms
            return ""
        return await asyncio.to_thread(self._stt.transcribe, audio, self.sample_rate)


class TranscribeSTT:
    """High-performance GGUF Speech-to-Text inference using transcribe.cpp (GGML).

    Supports any GGUF speech model (Nemotron, Parakeet, Whisper).
    Transcribes complete utterances in a single clean pass without mid-sentence slicing.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        if transcribe_cpp is None:
            raise RuntimeError(
                "transcribe-cpp is not installed. Run 'uv add transcribe-cpp'."
            )

        self.model_path = Path(model_path).resolve() if model_path else find_default_stt_model()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"STT model not found at {self.model_path}. "
                f"Run 'uv run scripts/fetch_voice_models.py' or set VERBIX_STT_MODEL=<path.gguf>."
            )

        self.sample_rate = 16000
        self._model = transcribe_cpp.Model(str(self.model_path))
        logger.info("TranscribeSTT loaded GGUF model from %s", self.model_path)

    def open_session(
        self,
        sample_rate: int = 16000,
        on_endpoint: Any | None = None,
    ) -> TranscribeSession:
        """Opens an incremental session that buffers audio and finalizes in one clean pass."""
        from verbix.voice.sherpa_vad import SileroSegmenter

        seg = SileroSegmenter(sample_rate=sample_rate)
        seg.ensure_loaded()
        return TranscribeSession(
            stt=self,
            segmenter=seg,
            sample_rate=sample_rate,
            on_endpoint=on_endpoint,
        )

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribes raw PCM audio data (16-bit signed mono)."""
        if not audio_data or len(audio_data) < 320:
            return ""

        audio_data = audio_data[: len(audio_data) - (len(audio_data) % 2)]
        audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if audio_float.size < int(sample_rate * 0.15):  # under 150ms
            return ""

        if not _has_min_voiced_duration(audio_float, sample_rate=sample_rate):
            return ""

        try:
            with self._model.session() as session:
                result = session.run(audio_float)
                return result.text.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("transcribe.cpp error: %s", e)
            return ""
