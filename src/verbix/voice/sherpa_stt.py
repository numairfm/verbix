from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROVISION_CMD = "uv run scripts/fetch_voice_models.py"

REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokens.txt",
)

_PAUSE_COMMIT_S = 0.25
_FORCE_COMMIT_S = 12.0
_FORCE_SEARCH_S = 1.0
_FRAME_S = 0.032
_PROMPT_TAIL = 200
_WATCHDOG_INTERVAL_S = 0.05


def _resolve_threads() -> int:
    n = os.cpu_count() or 1
    return min(4, max(1, n))


# Exact decoder junk observed on noise/silence. The decoder exposes no
# confidence score (ys_log_probs is empty on Canary), so these are matched
# as data — never extended with per-case branches.
_JUNK_PHRASES = frozenset(
    {
        "i'm not sure.",
        "i am not sure.",
        "thank you.",
        "thank you",
        "thanks for watching.",
        "you",
    }
)

_PUNCT_ONLY = frozenset(".,…_ \t-?!")


def _looks_like_hallucination(text: str) -> bool:
    """True for decoder output that is punctuation, known junk, or single-char soup."""
    s = text.strip()
    if not s or all(ch in _PUNCT_ONLY for ch in s):
        return True
    if s.lower() in _JUNK_PHRASES:
        return True
    toks = s.split()
    return len(toks) > 1 and all(len(t) == 1 for t in toks)


def _default_vad(sample_rate: int = 16000, rms_threshold: float = 300.0) -> Callable[[bytes], bool]:
    """Energy-based VAD for speech chunks. Pure silence / low-energy noise returns False."""
    def _is_speech(pcm: bytes) -> bool:
        if not pcm or not any(pcm):
            return False
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if audio.size == 0 or not np.any(audio):
            return False
        rms = np.sqrt(np.mean(audio**2))
        return bool(rms >= rms_threshold)

    return _is_speech


class _Segment:
    """One committed audio span; its bytes are decoded exactly once."""

    __slots__ = ("pcm", "task", "text")

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.text: str | None = None
        self.task: asyncio.Task[None] | None = None


class STTSession:
    """Incremental transcription session with VAD-committed segments."""

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
        """Decodes anything not yet decoded and returns full text."""
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
        if not any(pcm):
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
            return None
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
            return
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
    """Streaming STT factory producing pause-committed `STTSession`s."""

    def __init__(
        self,
        decode_fn: Callable[..., Awaitable[str]] | None = None,
        vad_fn: Callable[[bytes], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sample_rate: int = 16000,
        stt: SherpaSTT | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        if decode_fn is None:
            stt_obj = stt if stt is not None else SherpaSTT()

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
        return STTSession(
            self._decode_fn,
            self._vad_fn,
            self._clock,
            sample_rate=sample_rate or self.sample_rate,
            on_partial=on_partial,
            on_endpoint=on_endpoint,
            enable_watchdog=self._realtime,
        )


class SherpaSTT:
    """In-process Speech-to-Text engine using sherpa-onnx and Canary 180M Flash INT8."""

    def __init__(
        self,
        model_dir: Path | None = None,
        threads: int | None = None,
        src_lang: str = "en",
        tgt_lang: str = "en",
    ) -> None:
        self.model_dir = model_dir
        self.threads = threads if threads is not None else _resolve_threads()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = 16000
        self._lock = asyncio.Lock()
        self._recognizer: Any = None
        self._loaded = False

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
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribes raw PCM audio data (16-bit signed mono)."""
        if not audio_data or len(audio_data) < 320:
            return ""

        audio_int = np.frombuffer(audio_data, dtype=np.int16)
        if audio_int.size == 0 or not np.any(audio_int):
            return ""

        audio_float = audio_int.astype(np.float32) / 32768.0

        # Resample to 16 kHz if needed
        if sample_rate != 16000 and len(audio_float) > 0:
            n_target = int(len(audio_float) * 16000 / sample_rate)
            audio_float = np.interp(
                np.linspace(0, len(audio_float) - 1, n_target),
                np.arange(len(audio_float)),
                audio_float,
            ).astype(np.float32)

        # Energy pre-gate: skip model if RMS is under 200 (silence or pure mic hiss)
        rms = np.sqrt(np.mean((audio_float * 32768.0) ** 2))
        if rms < 200.0:
            return ""

        if not self.is_loaded:
            await self.load()

        def _sync_transcribe() -> str:
            import os

            # sherpa-onnx's native code logs an endoftext-fallback notice to
            # stderr on every decode (upstream issue #3919). Silence the fd
            # during the blocking call so turns stay readable.
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved = os.dup(2)
            try:
                os.dup2(devnull, 2)
                stream = self._recognizer.create_stream()
                stream.accept_waveform(16000, audio_float)
                self._recognizer.decode_stream(stream)
                res = stream.result.text.strip()
            finally:
                os.dup2(saved, 2)
                os.close(devnull)
                os.close(saved)
            return "" if _looks_like_hallucination(res) else res

        try:
            return await asyncio.to_thread(_sync_transcribe)
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning("SherpaSTT transcription error: %s", e)
            return ""

    def open_session(
        self,
        sample_rate: int = 16000,
        on_partial: Callable[[str], Any] | None = None,
        on_endpoint: Callable[[str], Any] | None = None,
    ) -> STTSession:
        """Creates a segment-committed incremental session."""
        return SegmentCommittedSTT(stt=self, sample_rate=sample_rate).open_session(
            on_partial=on_partial, on_endpoint=on_endpoint
        )
