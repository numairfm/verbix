from __future__ import annotations

import asyncio
import logging
import math
import shutil
from collections.abc import Callable

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

    x = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    x -= x.mean()
    # ponytail: one-pole high-pass, fixed 60 Hz corner, per-chunk python loop
    # (measured 0.25% of one core at the 16 kHz contract rate); a 2nd-order
    # corner or scipy.signal.lfilter only if a bar cares about low-frequency
    # fidelity
    a = math.exp(-2.0 * math.pi * _DRIFT_CORNER_HZ / rate)
    y = np.empty_like(x)
    prev_y, prev_x = state
    if prev_x is None:  # cold start: prime to the first sample, no ramp-up
        prev_y = prev_x = float(x[0])
    for i, v in enumerate(x.tolist()):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes(), (prev_y, prev_x)


class PTTCapture:
    """Push-to-Talk audio recording controller using native system capture (pw-record or arecord)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_chunk = on_chunk
        self._current_proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[bytes] | None = None
        self._hp_state: tuple[float, float | None] = (0.0, None)  # one-pole drift memory
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        if shutil.which("pw-record"):
            return "pw-record"
        if shutil.which("arecord"):
            return "arecord"
        return None

    @property
    def frame_bytes(self) -> int:
        """Bytes per 20ms frame of S16 mono capture."""
        return self.sample_rate * self.channels * 2 // 50

    @property
    def is_recording(self) -> bool:
        return self._current_proc is not None and self._current_proc.returncode is None

    async def start_recording(self) -> None:
        """Starts recording audio from default input source into memory."""
        if self.is_recording:
            return
        # One drift memory per session: cold at the start of a recording, so no
        # drift leaks in from the utterance before it.
        self._hp_state = (0.0, None)

        if not self._backend:
            logger.warning("No audio recording backend found (pw-record or arecord).")
            return

        cmd: list[str] = []
        if self._backend == "pw-record":
            cmd = [
                "pw-record",
                "--rate",
                str(self.sample_rate),
                "--channels",
                str(self.channels),
                "-",
            ]
        elif self._backend == "arecord":
            cmd = [
                "arecord",
                "-q",
                "-r",
                str(self.sample_rate),
                "-c",
                str(self.channels),
                "-f",
                "S16_LE",
                "-",
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._current_proc = proc

        async def _read_stream() -> bytes:
            buf = bytearray()
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(self.frame_bytes)
                if not chunk:
                    break
                chunk = self._condition_chunk(chunk)
                buf.extend(chunk)
                if self.on_chunk:
                    try:
                        self.on_chunk(chunk)
                    except (OSError, RuntimeError) as e:
                        logger.debug("on_chunk error: %s", e)
            return bytes(buf)

        self._reader_task = asyncio.create_task(_read_stream())

    def _condition_chunk(self, chunk: bytes) -> bytes:
        """The one seam every captured chunk crosses: DC-bias and baseline-drift
        conditioning with the session's single filter state, so `on_chunk` sees
        the same bytes `stop_recording` returns — the bytes the VAD gate scores."""
        chunk, self._hp_state = _condition_s16le(chunk, self.sample_rate, self._hp_state)
        return chunk

    async def stop_recording(self) -> bytes:
        """Stops recording and returns accumulated raw PCM bytes."""
        if not self._current_proc:
            return b""

        proc = self._current_proc
        self._current_proc = None

        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

        audio_bytes = b""
        if self._reader_task:
            try:
                audio_bytes = await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        try:
            await proc.wait()
        except asyncio.CancelledError:
            pass

        return audio_bytes
