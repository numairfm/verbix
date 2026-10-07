from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import socket
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "alba.wav"


def _resolve_threads() -> int:
    n = os.cpu_count() or 1
    return min(4, max(1, n))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class PocketHelperEngine:
    """TTS via a resident PocketTTS.cpp helper process over localhost HTTP.

    Same surface as the previous engine: lazy ``load()``, ``synthesize()``
    yielding 24 kHz f32 mono PCM, ``aclose()``. No torch, no model weights
    in this process.
    """

    def __init__(
        self,
        helper_bin: Path | None = None,
        models_dir: Path | None = None,
        voices_dir: Path | None = None,
        voice: str = _DEFAULT_VOICE,
        threads: int | None = None,
    ) -> None:
        self.helper_bin = helper_bin
        self.models_dir = models_dir
        self.voices_dir = voices_dir
        self.voice = voice
        self.threads = threads if threads is not None else _resolve_threads()
        self.sample_rate = 24000
        self.channels = 1
        self.audio_format = "f32"
        self._lock = asyncio.Lock()
        self._proc: Any = None
        self._port = 0
        self._token = ""

    @property
    def is_loaded(self) -> bool:
        return self._proc is not None

    async def load(self) -> None:
        """Spawns the helper and waits until /health reports warm."""
        async with self._lock:
            await self._load_locked()

    async def _load_locked(self) -> None:
        if self.is_loaded:
            return
        from verbix.voice.tts_assets import _resolve_binary, default_root

        root = self.models_dir.parent if self.models_dir else default_root()
        binary = self.helper_bin or _resolve_binary(root)
        if binary is None:
            raise RuntimeError(
                f"TTS helper binary missing under {root / 'bin'}. Build one with "
                "scripts/build_tts_helper.sh or point VERBIX_TTS_DIR at a runtime dir."
            )
        models = self.models_dir or (root / "models")
        voices = self.voices_dir or (root / "voices")
        self._port = _free_port()
        self._token = secrets.token_hex(32)
        argv = [
            str(binary),
            "--server",
            "--port",
            str(self._port),
            "--precision",
            "int8",
            "--temperature",
            "0.3",
            "--lsd-steps",
            "1",
            "--first-chunk",
            "1",
            "--max-chunk",
            "1",
            "--threads",
            str(self.threads),
            "--models-dir",
            str(models),
            "--voices-dir",
            str(voices),
            "--tokenizer",
            str(models / "tokenizer.model"),
        ]
        self._proc = await asyncio.to_thread(
            subprocess.Popen, argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await self._await_healthy()
        logger.info("TTS helper ready on 127.0.0.1:%d", self._port)

    async def _await_healthy(self) -> None:
        for _ in range(200):
            try:
                await asyncio.to_thread(self._get, "/health")
                return
            except OSError:
                await asyncio.sleep(0.05)
        raise RuntimeError("TTS helper did not become healthy")

    def _get(self, path: str) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self._port}{path}")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                raise OSError(f"health status {resp.status}")

    def _post(self, text: str) -> bytes:
        body = json.dumps({"text": text, "voice": self.voice}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                raise _TTSStatusError(resp.status)
            return resp.read()

    async def synthesize(self, text: str):  # type: ignore[no-untyped-def]
        """Synthesizes text into 24 kHz f32 mono PCM chunks."""
        clean = text.strip()
        if not clean:
            return
        if not self.is_loaded:
            await self.load()
        try:
            pcm = await asyncio.to_thread(self._post, clean)
        except (OSError, _TTSStatusError) as first_err:
            if isinstance(first_err, _TTSStatusError):
                raise RuntimeError(f"tts chunk failed: HTTP {first_err.status}") from first_err  # noqa: TRY004
            await self._restart()
            try:
                pcm = await asyncio.to_thread(self._post, clean)
            except _TTSStatusError as e:
                raise RuntimeError(f"tts chunk failed: HTTP {e.status}") from e
            except OSError as e:
                raise RuntimeError(f"tts chunk failed: {e}") from e
        yield pcm

    async def _restart(self) -> None:
        """Terminate and respawn under one lock acquisition (no re-entry)."""
        async with self._lock:
            await self._terminate()
            await self._load_locked()

    async def aclose(self) -> None:
        """Terminates the helper, if running. Safe to call twice."""
        async with self._lock:
            await self._terminate()

    async def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        self._port = 0
        self._token = ""
        if proc is None:
            return
        await asyncio.to_thread(_stop_process, proc)


def _stop_process(proc: Any) -> None:
    try:
        proc.terminate()
    except OSError:
        pass


class _TTSStatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
