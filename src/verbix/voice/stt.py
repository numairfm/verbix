from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# S1 spec values: a pause >= 250ms commits the open segment; a segment is
# force-committed at 12s at the lowest-energy 32ms frame of the last 1s; each
# decode receives the last 200 chars of committed text as prompt context.
_PAUSE_COMMIT_S = 0.25
_FORCE_COMMIT_S = 12.0
_FORCE_SEARCH_S = 1.0
_FRAME_S = 0.032
_PROMPT_TAIL = 200
_WATCHDOG_INTERVAL_S = 0.05


def _default_vad(sample_rate: int) -> Callable[[bytes], bool]:
    """Bundled Silero VAD via faster-whisper (no model download).

    Silero needs >= 512-sample (32ms) windows but capture feeds 20ms
    chunks, so chunks accumulate until a full window is available; shorter
    tails trust the energy pre-gate. Pure digital silence short-circuits
    without touching VAD state.
    """

    import numpy as np

    _pending = [np.zeros(0, dtype=np.float32)]
    _WIN = 512  # 32ms @16kHz

    def _window_speech(window: Any) -> bool:
        try:
            from faster_whisper.vad import get_speech_timestamps

            return bool(get_speech_timestamps(window, sampling_rate=16000))
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.debug("VAD error, falling back to energy check: %s", e)
            return bool((window != 0).any())

    def _is_speech(pcm: bytes) -> bool:
        audio_int = np.frombuffer(pcm, dtype=np.int16)
        if audio_int.size == 0 or not np.any(audio_int):
            return False
        audio_float = (audio_int.astype(np.float32) / 32768.0).astype(np.float32)
        if sample_rate != 16000 and len(audio_float) > 0:
            n_target = int(len(audio_float) * 16000 / sample_rate)
            audio_float = np.interp(
                np.linspace(0, len(audio_float) - 1, n_target),
                np.arange(len(audio_float)),
                audio_float,
            ).astype(np.float32)
        buf = (
            np.concatenate([_pending[0], audio_float])
            if len(_pending[0])
            else audio_float
        )
        if len(buf) < _WIN:
            _pending[0] = buf
            return True
        n_full = len(buf) // _WIN
        speech = any(
            _window_speech(buf[i * _WIN : (i + 1) * _WIN]) for i in range(n_full)
        )
        _pending[0] = buf[n_full * _WIN :]
        return speech

    return _is_speech


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

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        initial_prompt: str | None = None,
    ) -> str:
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
            kwargs: dict[str, Any] = {}
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            segments, _ = model.transcribe(
                audio_float,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                **kwargs,
            )
            parts = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(parts).strip()

        try:
            return await asyncio.to_thread(_sync_transcribe)
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("Local STT transcription error: %s", e)
            return ""

    def open_session(
        self,
        sample_rate: int = 16000,
        on_partial: Callable[[str], Any] | None = None,
        on_endpoint: Callable[[str], Any] | None = None,
    ) -> STTSession:
        """Creates a segment-committed incremental session bound to this engine."""
        return SegmentCommittedSTT(stt=self, sample_rate=sample_rate).open_session(
            on_partial=on_partial, on_endpoint=on_endpoint
        )


class _Segment:
    """One committed audio span; its bytes are decoded exactly once."""

    __slots__ = ("pcm", "task", "text")

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.text: str | None = None
        self.task: asyncio.Task[None] | None = None


class STTSession:
    """Incremental transcription session with VAD-committed segments.

    `decode_fn` is `async def decode_fn(pcm: bytes, prompt: str = "") -> str`
    (prompt carries the last 200 chars of committed text as context).
    `feed()` buffers speech chunks and evaluates segment boundaries on every
    call against `clock`: a pause >= 250ms commits the open segment, and
    buffered audio >= 12s force-commits at the lowest-energy frame of the last
    1s. Committed segments decode eagerly when a running event loop is
    available, so `finalize()` only pays for the still-open tail. Silence
    chunks are never buffered, so every audio byte is decoded at most once.
    """

    def __init__(
        self,
        decode_fn: Callable[..., Awaitable[str]],
        vad_fn: Callable[[bytes], bool],
        clock: Callable[[], float],
        sample_rate: int = 16000,
        on_partial: Callable[[str], Any] | None = None,
        on_endpoint: Callable[[str], Any] | None = None,
        enable_watchdog: bool = False,
    ) -> None:
        self._decode_fn = decode_fn
        self._vad_fn = vad_fn
        self._clock = clock
        self.sample_rate = sample_rate
        self.on_partial = on_partial
        self.on_endpoint = on_endpoint
        self._buf = bytearray()
        self._segments: list[_Segment] = []
        self._last_speech_t: float | None = None
        self._finalized = False
        self._watchdog_enabled = enable_watchdog
        self._watchdog_task: asyncio.Task[None] | None = None
        self._ensure_watchdog()

    @property
    def committed_text(self) -> str:
        """Decoded text of committed segments, in order."""
        return " ".join(s.text for s in self._segments if s.text).strip()

    def feed(self, pcm: bytes) -> None:
        """Buffers one PCM chunk (16-bit signed mono); drops pure silence."""
        if self._finalized or not pcm:
            return
        now = self._clock()
        self._ensure_watchdog()
        if self._is_silence(pcm):
            self._maybe_pause_commit(now)
            return
        self._maybe_pause_commit(now)
        self._buf.extend(pcm)
        self._last_speech_t = now
        self._maybe_force_commit()

    async def finalize(self) -> str:
        """Decodes anything not yet decoded (committed backlog + open tail) and returns full text."""
        if self._finalized:
            return self.committed_text
        self._finalized = True
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        for seg in self._segments:
            if seg.text is None and seg.task is not None:
                try:
                    await seg.task
                except asyncio.CancelledError:
                    pass
            if seg.text is None:
                seg.text = await self._decode_once(seg.pcm)
                self._fire_callbacks(seg)
        if self._buf:
            tail = bytes(self._buf)
            self._buf.clear()
            text = await self._decode_once(tail)
            if text:
                seg = _Segment(tail)
                seg.text = text
                self._segments.append(seg)
                self._fire_callbacks(seg)
        return self.committed_text

    def cancel(self) -> None:
        """Abandons the session: stops the watchdog, cancels pending decodes, drops audio."""
        self._finalized = True
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        for seg in self._segments:
            if seg.task is not None and not seg.task.done():
                seg.task.cancel()
        self._buf.clear()

    def _is_silence(self, pcm: bytes) -> bool:
        if not any(pcm):  # pure digital silence carries no speech
            return True
        try:
            return not self._vad_fn(pcm)
        except Exception as e:  # noqa: BLE001
            logger.debug("VAD error, treating chunk as speech: %s", e)
            return False

    def _maybe_pause_commit(self, now: float) -> None:
        if (
            self._buf
            and self._last_speech_t is not None
            and now - self._last_speech_t >= _PAUSE_COMMIT_S
        ):
            self._push_segment(bytes(self._buf))
            self._buf.clear()

    def _maybe_force_commit(self) -> None:
        limit = int(self.sample_rate * _FORCE_COMMIT_S) * 2
        if len(self._buf) < limit:
            return
        cut = self._low_energy_cut(bytes(self._buf))
        if cut <= 0 or cut > len(self._buf):
            cut = len(self._buf)
        self._push_segment(bytes(self._buf[:cut]))
        del self._buf[:cut]

    def _low_energy_cut(self, buf: bytes) -> int:
        """Byte offset cutting buf at the end of the lowest-RMS 32ms frame in the last 1s."""
        import numpy as np

        frame_bytes = int(self.sample_rate * _FRAME_S) * 2
        window = buf[-int(self.sample_rate * _FORCE_SEARCH_S) * 2 :]
        n_frames = len(window) // frame_bytes
        if n_frames == 0:
            return len(buf)
        frames = (
            np.frombuffer(window[: n_frames * frame_bytes], dtype=np.int16)
            .reshape(n_frames, frame_bytes // 2)
            .astype(np.float32)
        )
        cut_in_window = (int(np.argmin((frames**2).mean(axis=1))) + 1) * frame_bytes
        return len(buf) - len(window) + cut_in_window

    def _push_segment(self, pcm: bytes) -> None:
        if not pcm:
            return
        seg = _Segment(pcm)
        self._segments.append(seg)
        seg.task = self._spawn_decode(seg)

    def _spawn_decode(self, seg: _Segment) -> asyncio.Task[None] | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None  # fed outside a loop (e.g. audio thread): finalize() decodes
        if self._finalized:
            return None
        return loop.create_task(self._decode_segment(seg))

    async def _decode_segment(self, seg: _Segment) -> None:
        try:
            seg.text = await self._decode_once(seg.pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("Segment decode error, will retry at finalize: %s", e)
            return
        self._fire_callbacks(seg)

    async def _decode_once(self, pcm: bytes) -> str:
        text = await self._decode_fn(pcm, prompt=self.committed_text[-_PROMPT_TAIL:])
        return text.strip() if text else ""

    def _fire_callbacks(self, seg: _Segment) -> None:
        try:
            if seg.text and self.on_endpoint is not None:
                self.on_endpoint(seg.text)
            if self.on_partial is not None:
                self.on_partial(self.committed_text)
        except Exception as e:  # noqa: BLE001
            logger.debug("STT callback error: %s", e)

    def _ensure_watchdog(self) -> None:
        if (
            not self._watchdog_enabled
            or self._watchdog_task is not None
            or self._finalized
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; retry on the next feed()
        self._watchdog_task = loop.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        try:
            while not self._finalized:
                await asyncio.sleep(_WATCHDOG_INTERVAL_S)
                if self._finalized:
                    break
                self._maybe_pause_commit(self._clock())
        except asyncio.CancelledError:
            pass


class SegmentCommittedSTT:
    """Streaming STT factory producing pause-committed `STTSession`s.

    Takes injectable `decode_fn`/`vad_fn`/`clock` so tests never load model
    weights. Production defaults decode via `FasterWhisperSTT.transcribe` (with
    committed-text prompt context) and VAD via bundled Silero timestamps.
    """

    def __init__(
        self,
        decode_fn: Callable[..., Awaitable[str]] | None = None,
        vad_fn: Callable[[bytes], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sample_rate: int = 16000,
        stt: FasterWhisperSTT | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        if decode_fn is None:
            stt_obj = stt if stt is not None else FasterWhisperSTT()

            async def _default_decode(pcm: bytes, prompt: str = "") -> str:
                return await stt_obj.transcribe(
                    pcm, sample_rate=sample_rate, initial_prompt=prompt or None
                )

            decode_fn = _default_decode
        self._decode_fn = decode_fn
        self._vad_fn = vad_fn if vad_fn is not None else _default_vad(sample_rate)
        self._clock = clock if clock is not None else time.monotonic
        self._realtime = clock is None

    def open_session(
        self,
        sample_rate: int | None = None,
        on_partial: Callable[[str], Any] | None = None,
        on_endpoint: Callable[[str], Any] | None = None,
    ) -> STTSession:
        """Opens a session; the background pause watchdog runs only on the real clock (production)."""
        return STTSession(
            self._decode_fn,
            self._vad_fn,
            self._clock,
            sample_rate=sample_rate or self.sample_rate,
            on_partial=on_partial,
            on_endpoint=on_endpoint,
            enable_watchdog=self._realtime,
        )

    async def transcribe(self, audio_data: bytes) -> str:
        """One-shot transcription through the injected decoder (no session)."""
        return await self._decode_fn(audio_data, prompt="")
