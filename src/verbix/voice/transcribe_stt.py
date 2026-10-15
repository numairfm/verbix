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


def default_voice_root() -> Path:
    """Resolves target voice root directory: VERBIX_VOICE_DIR -> ./voice-runtime -> ~/.verbix/voice."""
    if env := os.environ.get("VERBIX_VOICE_DIR"):
        return Path(env).resolve()
    local = Path.cwd() / "voice-runtime"
    if local.is_dir():
        return local.resolve()
    return (Path.home() / ".verbix" / "voice").resolve()


def default_stt_models_dir(voice_root: Path | None = None) -> Path:
    """Returns the canonical user-managed STT models directory (<voice_root>/stt/models)."""
    root = (voice_root or default_voice_root()).resolve()
    return root / "stt" / "models"


def migrate_legacy_stt_model(
    voice_root: Path | None = None, model_id: str = "nemotron-gguf"
) -> Path | None:
    """Safely migrates a legacy STT model from <root>/models/<id> to <root>/stt/models/<id>.

    Never overwrites an existing destination file and never touches sibling TTS files.
    """
    root = (voice_root or default_voice_root()).resolve()
    legacy_dir = root / "models" / model_id
    target_dir = root / "stt" / "models" / model_id
    try:
        if not legacy_dir.is_dir() or legacy_dir.is_symlink():
            return None
        files = [
            f
            for f in sorted(legacy_dir.iterdir())
            if f.is_file() and not f.is_symlink()
        ]
        if not files:
            return None
        for src in files:
            dest = target_dir / src.name
            if dest.exists():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            src.replace(dest)
        try:
            if not any(legacy_dir.iterdir()):
                legacy_dir.rmdir()
        except OSError:
            pass
        if target_dir.is_dir():
            for f in sorted(target_dir.glob("*.gguf")):
                if f.is_file():
                    return f.resolve()
    except OSError:
        return None
    return None


def find_default_stt_model(workspace: Path | None = None) -> Path:
    """Discovers the STT model path from env, ~/.verbix/voice/stt/models, or legacy fallbacks."""
    if env := os.environ.get("VERBIX_STT_MODEL"):
        return Path(env).resolve()

    scoped_global = Path.home() / ".verbix" / "models"
    if scoped_global.is_dir():
        for f in sorted(scoped_global.glob("**/*.gguf")):
            if f.is_file():
                return f.resolve()

    root = default_voice_root()
    user_voice_root = (Path.home() / ".verbix" / "voice").resolve()
    if root == user_voice_root:
        migrate_legacy_stt_model(root)

    stt_models = default_stt_models_dir(root)
    canonical_nemotron = stt_models / "nemotron-gguf"
    if canonical_nemotron.is_dir():
        for f in sorted(canonical_nemotron.glob("*.gguf")):
            if f.is_file():
                return f.resolve()

    if stt_models.is_dir():
        for f in sorted(stt_models.glob("**/*.gguf")):
            if f.is_file():
                return f.resolve()

    # Compatibility fallback for legacy or custom VERBIX_VOICE_DIR layouts
    legacy_models = root / "models"
    legacy_nemotron = legacy_models / "nemotron-gguf"
    if legacy_nemotron.is_dir():
        for f in sorted(legacy_nemotron.glob("*.gguf")):
            if f.is_file():
                return f.resolve()

    if legacy_models.is_dir():
        for f in sorted(legacy_models.glob("**/*.gguf")):
            if f.is_file():
                return f.resolve()

    return (
        stt_models
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    )


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
