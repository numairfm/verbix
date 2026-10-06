from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class SubprocessAudioSink:
    """Audio playback sink using system audio utilities (pw-play, aplay, or ffplay).

    Terminates the playback subprocess instantly on stop() for zero-latency barge-in.
    Supports both 16-bit integer PCM ("s16") and 32-bit float PCM ("f32").
    Streams directly into a single spawned process to eliminate playback gaps.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        audio_format: str = "f32",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format.lower()
        self._current_proc: asyncio.subprocess.Process | None = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        if shutil.which("pw-play"):
            return "pw-play"
        if shutil.which("aplay"):
            return "aplay"
        if shutil.which("ffplay"):
            return "ffplay"
        return None

    def _build_cmd(self) -> list[str]:
        if not self._backend:
            return []
        is_float = self.audio_format in ("f32", "float", "f32le")
        if self._backend == "pw-play":
            fmt = "f32" if is_float else "s16"
            return [
                "pw-play",
                "-a",
                "--rate",
                str(self.sample_rate),
                "--channels",
                str(self.channels),
                "--format",
                fmt,
                "-",
            ]
        elif self._backend == "aplay":
            fmt = "FLOAT_LE" if is_float else "S16_LE"
            return [
                "aplay",
                "-q",
                "-r",
                str(self.sample_rate),
                "-c",
                str(self.channels),
                "-f",
                fmt,
                "-",
            ]
        elif self._backend == "ffplay":
            fmt = "f32le" if is_float else "s16le"
            layout = "mono" if self.channels == 1 else "stereo"
            return [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-f",
                fmt,
                "-ar",
                str(self.sample_rate),
                "-ch_layout",
                layout,
                "-",
            ]
        return []

    def stop(self) -> None:
        """Immediately kills any active audio playback subprocess."""
        if self._current_proc and self._current_proc.returncode is None:
            try:
                self._current_proc.kill()
            except ProcessLookupError:
                pass
            self._current_proc = None

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> bool:
        """Streams raw PCM chunks into a single audio playback subprocess."""
        cmd = self._build_cmd()
        if not cmd:
            async for _ in chunks:
                pass
            return True

        self.stop()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._current_proc = proc
            assert proc.stdin is not None

            async for chunk in chunks:
                if proc.returncode is not None:
                    break
                proc.stdin.write(chunk)
                await proc.stdin.drain()

            try:
                proc.stdin.close()
                await proc.stdin.wait_closed()
            except (OSError, BrokenPipeError):
                pass

            await proc.wait()
            return proc.returncode == 0
        except asyncio.CancelledError:
            self.stop()
            raise
        except (OSError, RuntimeError) as e:
            logger.warning("Error during audio playback: %s", e)
            self.stop()
            return False
        finally:
            if proc is not None and self._current_proc is proc:
                self._current_proc = None

    async def play(self, audio_data: bytes) -> bool:
        """Plays raw PCM audio data through the system audio backend."""

        async def _single():
            yield audio_data

        return await self.play_stream(_single())
