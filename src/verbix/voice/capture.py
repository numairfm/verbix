from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable

logger = logging.getLogger(__name__)


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
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        if shutil.which("pw-record"):
            return "pw-record"
        if shutil.which("arecord"):
            return "arecord"
        return None

    @property
    def is_recording(self) -> bool:
        return self._current_proc is not None and self._current_proc.returncode is None

    async def start_recording(self) -> None:
        """Starts recording audio from default input source into memory."""
        if self.is_recording:
            return

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
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if self.on_chunk:
                    try:
                        self.on_chunk(chunk)
                    except (OSError, RuntimeError) as e:
                        logger.debug("on_chunk error: %s", e)
            return bytes(buf)

        self._reader_task = asyncio.create_task(_read_stream())

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
