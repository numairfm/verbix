from __future__ import annotations

import logging
import os
from pathlib import Path

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


def find_default_stt_model(workspace: Path | None = None) -> Path:
    """Discovers the STT model path from env, ~/.verbix/models, or default root."""
    if env := os.environ.get("VERBIX_STT_MODEL"):
        return Path(env).resolve()

    scoped_global = Path.home() / ".verbix" / "models"
    if scoped_global.is_dir():
        for f in sorted(scoped_global.glob("**/*.gguf")):
            if f.is_file():
                return f.resolve()

    from scripts.fetch_voice_models import default_root

    root = default_root() / "models"
    nemotron_dir = root / "nemotron-gguf"
    if nemotron_dir.is_dir():
        for f in nemotron_dir.glob("*.gguf"):
            return f.resolve()

    if root.is_dir():
        for f in root.glob("**/*.gguf"):
            return f.resolve()

    return root / "nemotron-gguf" / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"


class TranscribeSTT:
    """High-performance GGUF Speech-to-Text inference using transcribe.cpp (GGML).

    Supports any GGUF speech model (Nemotron, Parakeet, Whisper).
    Transcribes complete utterances in a single clean pass.
    """

    def __init__(
        self, model_path: Path | str | None = None, workspace: Path | None = None
    ) -> None:
        if transcribe_cpp is None:
            raise RuntimeError(
                "transcribe-cpp is not installed. Run 'uv add transcribe-cpp'."
            )

        self.model_path = (
            Path(model_path).resolve()
            if model_path
            else find_default_stt_model(workspace=workspace)
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"STT model not found at {self.model_path}. "
                f"Run 'uv run scripts/fetch_voice_models.py' or set VERBIX_STT_MODEL=<path.gguf>."
            )

        self.sample_rate = 16000
        self._model = transcribe_cpp.Model(str(self.model_path))
        logger.info("TranscribeSTT loaded GGUF model from %s", self.model_path)

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
