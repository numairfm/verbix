from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
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


class _Segment:
    """One committed audio span; its bytes are decoded exactly once."""

    __slots__ = ("pcm", "task", "text")

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.text: str | None = None
        self.task: asyncio.Task[None] | None = None


class STTSession:
    """Incremental transcription session driven by Silero speech spans."""

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

    def reset(self) -> None:
        """Resets session audio and segmenter state so no previous audio leaks."""
        self.cancel()
        self._segments.clear()
        self._finalized = False
        if hasattr(self._segmenter, "reset"):
            self._segmenter.reset()

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

    async def _decode_once(self, pcm: bytes) -> str:
        text = await self._decode_fn(pcm, prompt="")
        return text.strip() if text else ""


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
    """Discovers the STT model path from env, .verbix scopes, local runtime, or default root.

    Decoupled and swappable: accepts any GGUF model path without hardcoding.
    """
    if env := os.environ.get("VERBIX_STT_MODEL"):
        return Path(env).resolve()

    # Models layout reservation: models/stt|tts|llm (only stt is read here).
    if workspace is not None:
        scoped = Path(workspace) / ".verbix" / "models"
        if scoped.is_dir():
            for f in sorted(scoped.glob("**/*.gguf")):
                if f.is_file():
                    return f.resolve()
    scoped_global = Path.home() / ".verbix" / "models"
    if scoped_global.is_dir():
        for f in sorted(scoped_global.glob("**/*.gguf")):
            if f.is_file():
                return f.resolve()

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
        self._uses_spans = (
            segmenter is not None
            and hasattr(segmenter, "pop_segments")
            and hasattr(segmenter, "flush")
        )
        self._buf = bytearray()
        self._spans: list[bytes] = []
        self._finalized = False
        self._duration: float = 0.0
        self.fed_chunks: list[bytes] = []

    @property
    def is_speech_detected(self) -> bool:
        return bool(getattr(self._segmenter, "is_speech_detected", False))

    @property
    def speech_duration(self) -> float:
        if self._duration > 0:
            return self._duration
        if self._spans:
            return (sum(len(s) for s in self._spans) // 2) / float(self.sample_rate) if self.sample_rate else 0.0
        return (len(self._buf) // 2) / float(self.sample_rate) if self.sample_rate else 0.0

    def feed(self, chunk: bytes) -> None:
        if self._finalized or not chunk:
            return
        if not self._uses_spans:
            self.fed_chunks.append(chunk)
            self._buf.extend(chunk)
        if self._segmenter is not None and hasattr(self._segmenter, "feed"):
            try:
                self._segmenter.feed(chunk)
            except Exception as e:  # noqa: BLE001
                logger.debug("Segmenter feed error: %s", e)
            if hasattr(self._segmenter, "pop_segments"):
                spans = self._segmenter.pop_segments()
                if spans:
                    self._spans.extend(spans)
                    if self.on_endpoint:
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
            return self.is_speech_detected
        return False

    def cancel(self) -> None:
        self._finalized = True
        self._buf.clear()
        self._spans.clear()
        self.fed_chunks.clear()
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
                flushed = self._segmenter.flush()
                if flushed:
                    self._spans.extend(flushed)
            except Exception as e:  # noqa: BLE001
                logger.debug("Segmenter flush error: %s", e)
        if self._spans:
            audio = b"".join(self._spans)
        elif not self._uses_spans:
            audio = bytes(self._buf)
        else:
            audio = b""
        self._duration = (len(audio) // 2) / float(self.sample_rate) if self.sample_rate else 0.0
        self._buf.clear()
        self._spans.clear()
        self.fed_chunks.clear()
        if not audio or len(audio) < 2400:  # < 150ms
            return ""
        return await asyncio.to_thread(self._stt.transcribe, audio, self.sample_rate)


class TranscribeSTT:
    """High-performance GGUF Speech-to-Text inference using transcribe.cpp (GGML).

    Supports any GGUF speech model (Nemotron, Parakeet, Whisper).
    Transcribes complete utterances in a single clean pass without mid-sentence slicing.
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
