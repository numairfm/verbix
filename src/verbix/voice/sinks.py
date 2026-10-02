from __future__ import annotations

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)


class SubprocessAudioSink:
    """Audio playback sink using system audio utilities (pw-play, aplay, or ffplay).

    Terminates the playback subprocess instantly on stop() for zero-latency barge-in.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
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
        if self._backend == "pw-play":
            # pw-play supports raw input with parameters or wav
            cmd = ["pw-play", "--rate", str(self.sample_rate), "--channels", str(self.channels), "-"]
        elif self._backend == "aplay":
            cmd = ["aplay", "-q", "-r", str(self.sample_rate), "-c", str(self.channels), "-f", "S16_LE", "-"]
        elif self._backend == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-f", "s16le", "-ar", str(self.sample_rate), "-ac", str(self.channels), "-"]

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
