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
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FADE_SECONDS = 0.01
_PORTAUDIO_HINT = (
    "Audio output unavailable (PortAudio missing or no output device). "
    "On Linux, install 'libportaudio2' (Debian/Ubuntu) or 'portaudio' (Fedora/Arch)."
)


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
            from verbix.voice.portaudio import ensure_loadable

            ensure_loadable()
            import sounddevice as sd
        except (ImportError, OSError) as e:
            raise RuntimeError(f"{_PORTAUDIO_HINT} ({e})") from e

        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format.lower()
        self._lock = threading.Lock()
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._played_samples = 0
        self._stopping = False
        self._feeding = False
        self._fade_total = 0
        self._fade_left = 0
        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                blocksize=sample_rate // 50,
                channels=channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            raise RuntimeError(f"{_PORTAUDIO_HINT} ({e})") from e

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
            if n == 0:
                return
            vals = np.frombuffer(data[: n * 4], dtype=np.float32).copy()
        else:
            n = len(data) // 2
            if n == 0:
                return
            vals = np.frombuffer(data[: n * 2], dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self._buf = np.concatenate((self._buf, vals)) if len(self._buf) else vals

    def _callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        need = frames * self.channels
        with self._lock:
            if self._stopping:
                k = min(need, self._fade_left, len(self._buf))
                taken = self._buf[:k].copy()
                self._buf = self._buf[k:]
                if self._fade_total > 1 and k > 0:
                    pos = self._fade_total - self._fade_left
                    denom = float(self._fade_total - 1)
                    ramp = 1.0 - (pos + np.arange(k, dtype=np.float32)) / denom
                    taken *= ramp
                elif k > 0:
                    taken.fill(0.0)
                self._fade_left -= k
                if self._fade_left == 0:
                    self._buf = np.zeros(0, dtype=np.float32)
            else:
                k = min(need, len(self._buf))
                taken = self._buf[:k].copy()
                self._buf = self._buf[k:]
            self._played_samples += k
        outdata.fill(0)
        if k > 0:
            col = outdata[:, 0] if getattr(outdata, "ndim", 1) == 2 else outdata
            col[:k] = taken

    def stop(self) -> None:
        """Flag fade-out: callback emits a 10ms fade then silence."""
        with self._lock:
            self._stopping = True
            # ponytail: snapshot at stop so a short tail ramps fully 1->0 instead of jumping
            self._fade_total = min(self._fade_len(), len(self._buf))
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
                    if len(self._buf) == 0:
                        return True
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self.stop()
            raise

    async def play(self, audio_data: bytes) -> bool:
        async def _single():
            yield audio_data

        return await self.play_stream(_single())
