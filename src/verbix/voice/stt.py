from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisperSTT:
    """Local Speech-to-Text engine using quantized faster-whisper on CPU.

    Transcribes raw 16-bit PCM audio in an asyncio worker thread for non-blocking execution.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int = 4,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.threads = threads
        self._model: WhisperModel | None = None
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading local Whisper model '%s' (device=%s, compute_type=%s, threads=%d)...",
                    self.model_size,
                    self.device,
                    self.compute_type,
                    self.threads,
                )
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.threads,
                )
            except ImportError as e:
                logger.error("faster-whisper is not installed: %s", e)
                raise
        return self._model

    async def warmup(self) -> None:
        """Pre-loads the model and runs a zero-sample inference to eliminate first-call latency."""
        import numpy as np

        def _do_warmup() -> None:
            model = self._load_model()
            dummy = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(dummy, beam_size=1)[0])

        await asyncio.to_thread(_do_warmup)

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribes raw PCM audio data (16-bit signed mono) and returns recognized text string."""
        if not audio_data or len(audio_data) < 320:  # < 10ms
            return ""

        import numpy as np

        # Convert raw 16-bit int PCM to normalized float32 in [-1.0, 1.0]
        audio_int = np.frombuffer(audio_data, dtype=np.int16)
        if audio_int.size == 0:
            return ""
        audio_float = audio_int.astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sample_rate != 16000 and len(audio_float) > 0:
            n_target = int(len(audio_float) * 16000 / sample_rate)
            audio_float = np.interp(
                np.linspace(0, len(audio_float) - 1, n_target),
                np.arange(len(audio_float)),
                audio_float,
            ).astype(np.float32)

        def _sync_transcribe() -> str:
            model = self._load_model()
            segments, _ = model.transcribe(
                audio_float,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
            )
            parts = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(parts).strip()

        try:
            return await asyncio.to_thread(_sync_transcribe)
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Local STT transcription error: %s", e)
            return ""
