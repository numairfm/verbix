"""Callback-based audio sink with fade-out barge-in.

`CallbackAudioSink` feeds a PortAudio output stream (via `sounddevice`) from a
ring buffer instead of spawning a subprocess per utterance. `stop()` only sets
a flag — the audio callback applies a 10ms linear fade then silence, so
barge-in is click-free within one callback period (<=20ms).

`sounddevice` is imported lazily inside the constructor so the base install
never requires it (`pip install verbix[audio]` for the `callback` backend).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from array import array
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from verbix.voice.sinks import SubprocessAudioSink

logger = logging.getLogger(__name__)

FADE_SECONDS = 0.01


def apply_fade(samples: list[float], sample_rate: int = 24000) -> list[float]:
    """Linear fade-out over the last 10ms of `samples` (whole buffer if shorter)."""
    src = list(samples)
    n = len(src)
    if n == 0:
        return []
    fade_n = min(n, max(1, round(sample_rate * FADE_SECONDS)))
    out = list(src)
    start = n - fade_n
    denom = fade_n - 1 if fade_n > 1 else 1
    for i in range(fade_n):
        out[start + i] = src[start + i] * (1.0 - i / denom)
    return out


class CallbackAudioSink:
    """Plays f32/s16 PCM through a persistent `sounddevice` output stream."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        audio_format: str = "f32",
    ) -> None:
        assert channels == 1, "CallbackAudioSink is mono-only"
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            raise ImportError(
                "CallbackAudioSink needs the optional 'audio' extra: "
                "pip install verbix[audio]"
            ) from e
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format.lower()
        self._lock = threading.Lock()
        self._pending: deque[float] = deque()
        self._played_samples = 0
        self._stopping = False
        self._feeding = False
        self._fade_total = 0
        self._fade_left = 0
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    @property
    def played_samples(self) -> int:
        """Exact count of non-silent samples handed to the DAC (frames if mono)."""
        with self._lock:
            return self._played_samples

    def _fade_len(self) -> int:
        return max(1, round(self.sample_rate * FADE_SECONDS)) * self.channels

    def _enqueue(self, data: bytes) -> None:
        if not data:
            return
        if self.audio_format in ("f32", "float", "f32le"):
            n = len(data) // 4
            vals = [float(v) for v in array("f", data[: n * 4])]
        else:
            n = len(data) // 2
            vals = [v / 32768.0 for v in array("h", data[: n * 2])]
        with self._lock:
            self._pending.extend(vals)

    def _callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        need = frames * self.channels
        with self._lock:
            if self._stopping:
                k = min(need, self._fade_left, len(self._pending))
                taken = [self._pending.popleft() for _ in range(k)]
                if self._fade_total > 1:
                    pos = self._fade_total - self._fade_left
                    denom = self._fade_total - 1
                    taken = [v * (1.0 - (pos + j) / denom) for j, v in enumerate(taken)]
                else:
                    taken = [0.0] * k
                self._fade_left -= k
                if self._fade_left == 0:
                    self._pending.clear()
            else:
                k = min(need, len(self._pending))
                taken = [self._pending.popleft() for _ in range(k)]
            self._played_samples += k
        outdata.fill(0)
        if taken:
            col = outdata[:, 0] if getattr(outdata, "ndim", 1) == 2 else outdata
            col[:k] = taken

    def stop(self) -> None:
        """Flag fade-out: callback emits a 10ms fade then silence."""
        with self._lock:
            self._stopping = True
            # ponytail: snapshot at stop so a short tail ramps fully 1->0 instead of jumping
            self._fade_total = min(self._fade_len(), len(self._pending))
            self._fade_left = self._fade_total

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:  # noqa: BLE001 - PortAudio teardown must never raise
            logger.debug("Error closing audio stream: %s", e)

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> bool:
        with self._lock:
            self._stopping = False
            self._feeding = True
            self._fade_left = 0
        try:
            async for chunk in chunks:
                with self._lock:
                    if self._stopping:
                        return False
                self._enqueue(chunk)
            with self._lock:
                self._feeding = False
            while True:
                with self._lock:
                    if self._stopping:
                        return False
                    if not self._pending:
                        return True
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self.stop()
            raise

    async def play(self, audio_data: bytes) -> bool:
        async def _single():
            yield audio_data

        return await self.play_stream(_single())


def select_sink(kind: str = "auto", **kwargs: Any) -> Any:
    """Pick an AudioSink: `callback` iff a device opens, else subprocess.

    `auto` falls back to subprocess on any failure (missing/broken
    sounddevice, no output device); explicit `callback` raises a clean error.
    """
    kind = (kind or "auto").lower()
    if kind == "subprocess":
        return SubprocessAudioSink(**kwargs)
    if kind == "callback":
        try:
            return CallbackAudioSink(**kwargs)
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"callback audio sink unavailable: {e}") from e
    if kind == "auto":
        try:
            return CallbackAudioSink(**kwargs)
        except Exception as e:  # noqa: BLE001 - auto must fall back on any backend failure
            logger.debug("callback sink unavailable, using subprocess: %s", e)
            return SubprocessAudioSink(**kwargs)
    raise ValueError(
        f"unknown audio sink: {kind!r} (expected auto|callback|subprocess)"
    )
