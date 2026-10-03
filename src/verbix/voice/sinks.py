from __future__ import annotations

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)


class SubprocessAudioSink:
    """Audio playback sink using system audio utilities (pw-play, aplay, or ffplay).

    Terminates the playback subprocess instantly on stop() for zero-latency barge-in.
    Supports both 16-bit integer PCM ("s16") and 32-bit float PCM ("f32").
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

    def stop(self) -> None:
        """Immediately kills any active audio playback subprocess."""
        if self._current_proc and self._current_proc.returncode is None:
            try:
                self._current_proc.kill()
            except ProcessLookupError:
                pass
            self._current_proc = None

    async def play(self, audio_data: bytes) -> bool:
        """Plays raw PCM audio data through the system audio backend."""
        if not audio_data or not self._backend:
            return True

        self.stop()

        cmd: list[str] = []
        is_float = self.audio_format in ("f32", "float", "f32le")

        if self._backend == "pw-play":
            fmt = "f32" if is_float else "s16"
            cmd = [
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
            cmd = [
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
            cmd = [
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

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._current_proc = proc
            if proc.stdin:
                proc.stdin.write(audio_data)
                await proc.stdin.drain()
                proc.stdin.close()
            await proc.wait()
            return proc.returncode == 0
        except asyncio.CancelledError:
            self.stop()
            return False
        except (OSError, RuntimeError) as e:
            logger.warning("Error during audio playback: %s", e)
            self.stop()
            return False
        finally:
            if proc is not None and self._current_proc is proc:
                self._current_proc = None
