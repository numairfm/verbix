from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_STDERR_LOCK = threading.Lock()

PROVISION_CMD = "uv run scripts/fetch_voice_models.py"

REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokens.txt",
)



def _resolve_threads() -> int:
    n = os.cpu_count() or 1
    return min(4, max(1, n))


class _Segment:
    """One committed audio span; its bytes are decoded exactly once."""

    __slots__ = ("pcm", "task", "text")

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.text: str | None = None
        self.task: asyncio.Task[None] | None = None


class STTSession:
    """Incremental transcription session driven by Silero speech spans.

    Bytes in, span boundaries out: the segmenter (neural VAD) decides what is
    speech. Anything that never becomes a span — pops, breaths, noise,
    silence — is never decoded. The whole turn is judged once at finalize();
    per-segment re-gating would fragment real speech into sub-threshold
    pieces, so segments decode freely and only the turn total is gated.
    """

    def __init__(
        self,
        decode_fn: Callable[..., Awaitable[str]],
        segmenter: Any,
        sample_rate: int = 16000,
        on_endpoint: Callable[[], None] | None = None,
    ) -> None:
        self._decode_fn = decode_fn
        self._segmenter = segmenter
        self.sample_rate = sample_rate
        self.on_endpoint = on_endpoint
        self._segments: list[_Segment] = []
        self._finalized = False

    @property
    def committed_text(self) -> str:
        """Decoded text of committed segments, in order."""
        return " ".join(s.text for s in self._segments if s.text).strip()

    @property
    def speech_duration(self) -> float:
        """Total duration in seconds of VAD-classified speech spans in this session."""
        bytes_count = sum(len(s.pcm) for s in self._segments)
        return (bytes_count // 2) / float(self.sample_rate) if self.sample_rate else 0.0

    def feed(self, pcm: bytes) -> None:
        """Buffers one PCM chunk (16-bit signed mono) through the segmenter."""
        if self._finalized or not pcm:
            return
        self._segmenter.feed(pcm)
        spans = self._segmenter.pop_segments()
        for span in spans:
            self._push_segment(span)
        if spans and self.on_endpoint:
            try:
                self.on_endpoint()
            except Exception as e:  # noqa: BLE001
                logger.debug("on_endpoint callback error: %s", e)

    @property
    def is_speech_detected(self) -> bool:
        if hasattr(self._segmenter, "is_speech_detected"):
            return bool(self._segmenter.is_speech_detected)
        return False

    def feed_barge_in(self, pcm: bytes) -> bool:
        """Feeds PCM during playback. Returns True if user voice activity is detected."""
        if self._finalized or not pcm:
            return False
        self.feed(pcm)
        return self.is_speech_detected


    async def finalize(self) -> str:
        """Decodes every speech span and returns the joined text."""
        if self._finalized:
            return self.committed_text
        self._finalized = True
        for span in self._segmenter.flush():
            self._push_segment(span)
        for seg in self._segments:
            if seg.text is None and seg.task is not None:
                try:
                    await seg.task
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
            if seg.text is None:
                seg.text = await self._decode_once(seg.pcm)
        return self.committed_text

    def cancel(self) -> None:
        """Abandons the session: cancels pending decodes, drops audio."""
        self._finalized = True
        for seg in self._segments:
            if seg.task is not None and not seg.task.done():
                seg.task.cancel()

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
            return None  # fed outside a loop: finalize() decodes
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

    async def _decode_once(self, pcm: bytes) -> str:
        text = await self._decode_fn(pcm, prompt="")
        return text.strip() if text else ""


class SherpaSTT:
    """In-process Speech-to-Text engine using sherpa-onnx and Canary 180M Flash INT8."""

    def __init__(
        self,
        model_dir: Path | None = None,
        threads: int | None = None,
        src_lang: str = "en",
        tgt_lang: str = "en",
        segmenter: Any | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.threads = threads if threads is not None else _resolve_threads()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = 16000
        self._lock = asyncio.Lock()
        self._recognizer: Any = None
        self._loaded = False
        # Injected segmenter (tests). Production always builds a fresh one per
        # session: segmenter state must never leak across turns.
        self._segmenter = segmenter

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._recognizer is not None

    async def load(self) -> None:
        async with self._lock:
            await self._load_locked()

    async def _load_locked(self) -> None:
        if self.is_loaded:
            return

        from scripts.fetch_voice_models import default_root

        root = default_root()
        target_dir = self.model_dir or (root / "models" / "canary-int8")

        missing = [f for f in REQUIRED_FILES if not (target_dir / f).is_file()]
        if missing:
            raise RuntimeError(
                f"ASR model files missing under {target_dir}: {missing}. "
                f"Run '{PROVISION_CMD}' to download them."
            )

        try:
            import sherpa_onnx
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"sherpa-onnx is not installed or failed to import: {e}. "
                "Ensure sherpa-onnx is in your virtual environment."
            ) from e

        def _init() -> Any:
            return sherpa_onnx.OfflineRecognizer.from_nemo_canary(
                encoder=str(target_dir / "encoder.int8.onnx"),
                decoder=str(target_dir / "decoder.int8.onnx"),
                tokens=str(target_dir / "tokens.txt"),
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                num_threads=self.threads,
                provider="cpu",
            )

        self._recognizer = await asyncio.to_thread(_init)
        self._loaded = True
        logger.info("SherpaSTT loaded successfully from %s", target_dir)

    async def warmup(self) -> None:
        """Pre-loads the model and runs a zero-sample inference."""
        def _do_warmup() -> None:
            if not self.is_loaded:
                return
            dummy = np.zeros(16000, dtype=np.float32)
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, dummy)
            self._recognizer.decode_stream(stream)

        if not self.is_loaded:
            await self.load()
        await asyncio.to_thread(_do_warmup)

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribes raw PCM audio data (16-bit signed mono).

        Runs the same span pipeline as the live session: segment into speech
        spans, judge the whole turn once, decode spans. One entry point, one
        set of verdicts — no parallel gate logic to drift.
        """
        if not audio_data or len(audio_data) < 320:
            return ""

        audio_data = audio_data[: len(audio_data) - (len(audio_data) % 2)]
        audio_int = np.frombuffer(audio_data, dtype=np.int16)
        if audio_int.size == 0 or not np.any(audio_int):
            return ""

        if sample_rate != 16000:
            raise ValueError(f"SherpaSTT requires 16000 Hz audio, got {sample_rate} Hz.")

        async def _decode(pcm: bytes, prompt: str = "") -> str:
            audio_float = (
                np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            )
            return await self.transcribe_raw(audio_float)

        sess = STTSession(_decode, self._session_segmenter(), sample_rate=16000)
        sess.feed(audio_data)
        return await sess.finalize()

    async def transcribe_raw(self, audio_float: np.ndarray) -> str:
        """Model-only decode of 16 kHz float32 mono, no gates.

        For the live session: its chunks already passed the per-chunk VAD and
        the turn is judged once at finalize(), so re-gating per segment would
        fragment real speech into sub-threshold pieces. One-shot callers use
        transcribe(), which gates the full buffer as the full turn.
        """
        # The encoder's frontend needs a minimum frame count; anything shorter
        # raises ConvInteger {0,128} inside ORT. Skip silently — a sub-10 ms
        # span carries no word in any language.
        if audio_float.size < 160:
            return ""
        if not self.is_loaded:
            await self.load()

        def _sync_transcribe() -> str:
            import os

            with _STDERR_LOCK:
                devnull = os.open(os.devnull, os.O_WRONLY)
                saved = os.dup(2)
                try:
                    os.dup2(devnull, 2)
                    stream = self._recognizer.create_stream()
                    stream.accept_waveform(16000, audio_float)
                    self._recognizer.decode_stream(stream)
                    return stream.result.text.strip()
                finally:
                    os.dup2(saved, 2)
                    os.close(devnull)
                    os.close(saved)

        try:
            return await asyncio.to_thread(_sync_transcribe)
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("SherpaSTT transcription error: %s", e)
            return ""

    def _session_segmenter(self) -> Any:
        """Fresh-or-reset segmenter per session: stream state must never leak
        across turns."""
        if self._segmenter is not None:
            self._segmenter.reset()
            return self._segmenter
        from verbix.voice.sherpa_vad import SileroSegmenter

        seg = SileroSegmenter()
        seg.ensure_loaded()
        return seg

    def open_session(
        self,
        sample_rate: int = 16000,
        on_endpoint: Callable[[], None] | None = None,
    ) -> STTSession:
        """Creates a span-driven incremental session."""
        if sample_rate != 16000:
            raise ValueError(f"SherpaSTT requires 16000 Hz audio, got {sample_rate} Hz.")

        async def _decode(pcm: bytes, prompt: str = "") -> str:
            audio_float = (
                np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            )
            return await self.transcribe_raw(audio_float)

        return STTSession(
            _decode,
            self._session_segmenter(),
            sample_rate=16000,
            on_endpoint=on_endpoint,
        )
