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
        threads: int = 6,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.threads = threads
        self._model: WhisperModel | None = None

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
            try:
                from faster_whisper.vad import get_speech_timestamps

                get_speech_timestamps(dummy)
            except (OSError, RuntimeError, ValueError):
                pass

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
        self._worker_task: asyncio.Task[None] | None = None

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
            await asyncio.sleep(0.04)
            curr_len = len(self.buf)
            # If idle and at least 0.3s new audio has accumulated since last snapshot
            if self._worker_task is None and (
                curr_len - self.last_transcribed_len >= self.sample_rate * 2 * 0.30
            ):
                snapshot = bytes(self.buf)
                snapshot_len = len(snapshot)

                async def _worker(
                    snap: bytes = snapshot, s_len: int = snapshot_len
                ) -> None:
                    try:
                        text = await self.stt.transcribe(
                            snap, sample_rate=self.sample_rate
                        )
                        self.latest_text = text
                        self.last_transcribed_len = s_len
                    except (asyncio.CancelledError, RuntimeError, OSError) as e:
                        logger.debug("Live STT worker error: %s", e)
                    finally:
                        self._worker_task = None

                self._worker_task = asyncio.create_task(_worker())

    async def finish(self, final_audio: bytes | None = None) -> str:
        self._running = False
        if final_audio:
            self.buf = bytearray(final_audio)

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

        # Await any in-flight background worker to incorporate its progress
        if self._worker_task is not None and not self._worker_task.done():
            try:
                await self._worker_task
            except (asyncio.CancelledError, RuntimeError, OSError) as e:
                logger.debug("Live STT finish worker await error: %s", e)

        tail_bytes = len(self.buf) - self.last_transcribed_len
        if tail_bytes <= 0:
            return self.latest_text

        # If background transcription has produced text, examine the trailing audio using neural VAD
        if self.latest_text:
            import numpy as np

            tail_pcm = bytes(self.buf[self.last_transcribed_len :])
            audio_int = np.frombuffer(tail_pcm, dtype=np.int16)
            if audio_int.size < 320:  # < 10ms
                return self.latest_text

            audio_float = (audio_int.astype(np.float32) / 32768.0).astype(np.float32)
            if self.sample_rate != 16000 and len(audio_float) > 0:
                n_target = int(len(audio_float) * 16000 / self.sample_rate)
                audio_float = np.interp(
                    np.linspace(0, len(audio_float) - 1, n_target),
                    np.arange(len(audio_float)),
                    audio_float,
                ).astype(np.float32)

            try:
                from faster_whisper.vad import get_speech_timestamps

                speech_segments = get_speech_timestamps(
                    audio_float, sampling_rate=16000
                )
                # If no speech is detected in the tail, return latest_text immediately
                if not speech_segments:
                    return self.latest_text
            except (OSError, RuntimeError, ValueError) as e:
                logger.debug(
                    "VAD tail check error, falling back to full transcribe: %s", e
                )

        # Speech detected in the untranscribed tail: transcribe only the tail
        # and append it. Re-running the whole utterance here was the finalize
        # latency regression; finalize cost must scale with the tail, not the
        # full recording.
        if not self.latest_text:
            # No background progress to build on: one full pass is the only
            # correct fallback.
            if len(self.buf) > 320:
                return await self.stt.transcribe(
                    bytes(self.buf), sample_rate=self.sample_rate
                )
            return ""

        tail_pcm = bytes(self.buf[self.last_transcribed_len :])
        if len(tail_pcm) <= 320:  # < 10 ms
            return self.latest_text
        tail_text = (
            await self.stt.transcribe(tail_pcm, sample_rate=self.sample_rate)
        ).strip()
        if tail_text:
            return f"{self.latest_text} {tail_text}".strip()
        return self.latest_text
