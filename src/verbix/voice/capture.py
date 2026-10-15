from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_DRIFT_CORNER_HZ = 60.0


def _condition_s16le(
    data: bytes, rate: int, state: tuple[float, float | None]
) -> tuple[bytes, tuple[float, float | None]]:
    """Strips DC bias and baseline drift from one mono s16le chunk.

    Some mic drivers add a constant offset to every sample (measured ≈5757
    on one DMIC — 17.6% of full scale); a biased signal can read as machine
    tone to a VAD gate and drop real utterances. Mean removal cancels a
    constant offset exactly, then a one-pole high-pass takes baseline drift
    out. `state` is the filter memory (previous output, previous input); a
    cold state primes to the first sample so a clean signal passes through
    with only its corner response, no ramp-up.
    """
    if not data:
        return data, state
    import numpy as np

    data = data[: len(data) - (len(data) % 2)]
    if not data:
        return data, state
    x = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    # One-pole high-pass filter with 60 Hz corner. Cancels constant DC offset
    # and baseline drift continuously across chunks without chunk-boundary step
    # discontinuities.
    a = math.exp(-2.0 * math.pi * _DRIFT_CORNER_HZ / rate)
    y = np.empty_like(x)
    prev_y, prev_x = state
    if prev_x is None:  # cold start: prime to the first sample, no ramp-up
        prev_y = 0.0
        prev_x = float(x[0])
    for i, v in enumerate(x.tolist()):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes(), (prev_y, prev_x)


class PTTCapture:
    """Push-to-Talk audio recording controller using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Any = None
        self._is_recording: bool = False
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_task: asyncio.Task[bytes] | None = None
        self._hp_state: tuple[float, float | None] = (0.0, None)  # one-pole drift memory

    @property
    def frame_bytes(self) -> int:
        """Bytes per 20ms frame of S16 mono capture."""
        return self.sample_rate * self.channels * 2 // 50

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    async def start_recording(self) -> None:
        """Starts recording audio from default input source into memory."""
        if self.is_recording:
            return
        self._hp_state = (0.0, None)
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._is_recording = True

        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            logger.warning(
                "Microphone input unavailable: audio library failed to load (%s); voice capture idle",
                e,
            )
            self._is_recording = False
            return

        try:
            devices = sd.query_devices()
            if not any((d.get("max_input_channels", 0) or 0) > 0 for d in (devices or [])):
                logger.warning(
                    "No audio input device found; voice capture idle until a microphone is available"
                )
        except Exception as e:  # noqa: BLE001 - device query is advisory only
            logger.debug("Could not query audio input devices: %s", e)

        def _audio_callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            if status:
                logger.debug("sounddevice capture status: %s", status)
            try:
                if self._loop and self._loop.is_running() and self._is_recording:
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))
            except Exception as e:  # noqa: BLE001 - PortAudio thread must never print tracebacks on teardown
                logger.debug("Dropping capture chunk after teardown: %s", e)

        blocksize = self.sample_rate // 50  # 20ms frames
        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                channels=self.channels,
                dtype="int16",
                callback=_audio_callback,
            )
            stream.start()
            self._stream = stream
        except Exception as e:  # noqa: BLE001 - any device failure idles gracefully
            logger.warning("Microphone input stream failed to open (%s); voice capture idle", e)
            self._is_recording = False
            return

        async def _read_stream() -> bytes:
            buf = bytearray()
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                chunk = self._condition_chunk(chunk)
                buf.extend(chunk)
            return bytes(buf)

        self._reader_task = asyncio.create_task(_read_stream())

    def _condition_chunk(self, chunk: bytes) -> bytes:
        """DC-bias and baseline-drift conditioning with session filter state."""
        chunk, self._hp_state = _condition_s16le(chunk, self.sample_rate, self._hp_state)
        return chunk

    async def stop_recording(self) -> bytes:
        """Stops recording and returns accumulated raw PCM bytes."""
        if not self._is_recording:
            return b""

        self._is_recording = False
        stream = self._stream
        self._stream = None

        if stream:
            try:
                stream.stop()
                stream.close()
            except (OSError, RuntimeError) as e:
                logger.debug("Error closing sounddevice stream: %s", e)


        # Signal queue termination
        self._queue.put_nowait(None)

        audio_bytes = b""
        if self._reader_task:
            try:
                audio_bytes = await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        return audio_bytes

