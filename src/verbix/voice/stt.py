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

    def create_live_session(self, sample_rate: int = 16000) -> LiveSTTSession:
        """Creates a streaming live session that transcribes audio during active capture."""
        return LiveSTTSession(self, sample_rate=sample_rate)


class LiveSTTSession:
    """Manages active background transcription during audio recording.

    Continuously accumulates incoming PCM chunks from the microphone and runs
    progressive background inference so that speech is largely transcribed before key release.
    """

    def __init__(self, stt: FasterWhisperSTT, sample_rate: int = 16000) -> None:
        self.stt = stt
        self.sample_rate = sample_rate
        self.buf = bytearray()
        self.latest_text: str = ""
        self.last_transcribed_len: int = 0
        self._running: bool = False
        self._loop_task: asyncio.Task[None] | None = None
        self._in_flight_task: asyncio.Task[tuple[str, int]] | None = None

    def start(self) -> None:
        self.buf.clear()
        self.latest_text = ""
        self.last_transcribed_len = 0
        self._running = True
        self._loop_task = asyncio.create_task(self._transcribe_loop())

    def feed(self, pcm_chunk: bytes) -> None:
        if self._running:
            self.buf.extend(pcm_chunk)

    async def _transcribe_loop(self) -> None:
        while self._running:
            await asyncio.sleep(0.12)
            curr_len = len(self.buf)
            # If at least 0.4s new audio has accumulated and no worker is in flight
            if (
                curr_len - self.last_transcribed_len >= self.sample_rate * 2 * 0.4
                and self._in_flight_task is None
            ):
                snapshot = bytes(self.buf)
                snapshot_len = len(snapshot)

                async def _worker(
                    snap: bytes = snapshot, s_len: int = snapshot_len
                ) -> tuple[str, int]:
                    text = await self.stt.transcribe(snap, sample_rate=self.sample_rate)
                    return text, s_len

                self._in_flight_task = asyncio.create_task(_worker())
                try:
                    text, committed_len = await self._in_flight_task
                    self.latest_text = text
                    self.last_transcribed_len = committed_len
                except (asyncio.CancelledError, RuntimeError, OSError) as e:
                    logger.debug("Live STT worker interrupted: %s", e)
                finally:
                    self._in_flight_task = None

    async def finish(self, final_audio: bytes | None = None) -> str:
        self._running = False
        if final_audio:
            self.buf = bytearray(final_audio)

        # Await any in-flight background worker
        if self._in_flight_task is not None and not self._in_flight_task.done():
            try:
                text, committed_len = await self._in_flight_task
                self.latest_text = text
                self.last_transcribed_len = committed_len
            except (asyncio.CancelledError, RuntimeError, OSError) as e:
                logger.debug("Live STT finish worker error: %s", e)

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

        # If trailing audio since last transcription is small (<0.35s), use latest_text
        tail_bytes = len(self.buf) - self.last_transcribed_len
        if self.latest_text and tail_bytes < self.sample_rate * 2 * 0.35:
            return self.latest_text

        # Otherwise do final pass on accumulated buffer
        if len(self.buf) > 320:
            return await self.stt.transcribe(
                bytes(self.buf), sample_rate=self.sample_rate
            )
        return self.latest_text
