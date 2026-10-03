from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Finds an unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class PocketTTSEngine:
    """Local Text-to-Speech engine wrapping the resident PocketTTS C++ server helper.

    Outputs 24kHz float32 mono PCM audio bytes with low first-chunk latency.
    """

    def __init__(
        self,
        runtime_dir: Path | str | None = None,
        voice: str = "alba_1.10x.wav",
        threads: int = 4,
        temperature: float = 0.3,
    ) -> None:
        self.runtime_dir = self._resolve_runtime_dir(runtime_dir)
        self.voice = voice
        self.threads = threads
        self.temperature = temperature

        self.port: int | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None

        # Register exit hook to ensure subprocess termination
        atexit.register(self.stop)

    def _resolve_runtime_dir(self, explicit: Path | str | None) -> Path:
        if explicit:
            p = Path(explicit).resolve()
            if p.is_dir():
                return p

        env_dir = os.environ.get("VERBIX_TTS_DIR")
        if env_dir:
            p = Path(env_dir).resolve()
            if p.is_dir():
                return p

        candidates = [
            Path("tts/runtime"),
            Path("../verbix-old-go/core/tts/runtime"),
            Path("/home/numair/dev/projects/verbix-old-go/core/tts/runtime"),
        ]
        for cand in candidates:
            if cand.resolve().is_dir():
                return cand.resolve()

        return Path("tts/runtime").resolve()

    def _resolve_binary(self) -> Path:
        bin_env = os.environ.get("VERBIX_TTS_BIN")
        if bin_env and Path(bin_env).is_file():
            return Path(bin_env).resolve()

        candidates = [
            self.runtime_dir / "bin" / "verbix-tts-helper",
            self.runtime_dir / "bin" / "pocket-tts",
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"PocketTTS helper binary not found in {self.runtime_dir / 'bin'}"
        )

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Launches the resident pocket-tts HTTP server and waits for health check."""
        async with self._lock:
            if self.is_running:
                return

            bin_path = self._resolve_binary()
            lib_path = self.runtime_dir / "lib"
            models_dir = self.runtime_dir / "models"
            voices_dir = self.runtime_dir / "voices"
            tokenizer = models_dir / "tokenizer.model"

            self.port = _find_free_port()

            cmd = [
                str(bin_path),
                "--server",
                "--port",
                str(self.port),
                "--precision",
                "int8",
                "--mimi-precision",
                "fp32",
                "--temperature",
                str(self.temperature),
                "--lsd-steps",
                "1",
                "--first-chunk",
                "1",
                "--max-chunk",
                "1",
                "--threads",
                str(self.threads),
                "--models-dir",
                str(models_dir),
                "--voices-dir",
                str(voices_dir),
                "--tokenizer",
                str(tokenizer),
            ]

            env = dict(os.environ)
            ld_path = str(lib_path)
            if "LD_LIBRARY_PATH" in env:
                ld_path = f"{ld_path}:{env['LD_LIBRARY_PATH']}"
            env["LD_LIBRARY_PATH"] = ld_path
            env["DYLD_LIBRARY_PATH"] = str(lib_path)

            logger.info("Starting PocketTTS helper on port %d...", self.port)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._proc = proc

            # Poll health endpoint
            client = httpx.AsyncClient(timeout=1.0)
            self._http_client = httpx.AsyncClient(timeout=60.0)

            deadline = asyncio.get_running_loop().time() + 10.0
            healthy = False
            while asyncio.get_running_loop().time() < deadline:
                if proc.returncode is not None:
                    break
                try:
                    resp = await client.get(f"http://127.0.0.1:{self.port}/health")
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        healthy = True
                        break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(0.05)

            await client.aclose()

            if not healthy:
                self.stop()
                raise RuntimeError(
                    f"PocketTTS server failed to become healthy on port {self.port}"
                )

            logger.info("PocketTTS helper healthy on http://127.0.0.1:%d", self.port)

    def stop(self) -> None:
        """Terminates the helper subprocess."""
        if self._proc is not None:
            if self._proc.returncode is None:
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            self._proc = None

    async def aclose(self) -> None:
        """Async cleanup method."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self.stop()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesizes text into chunks of raw 24kHz float32 mono PCM audio bytes."""
        if not text.strip():
            return

        if not self.is_running:
            await self.start()

        assert self.port is not None
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)

        url = f"http://127.0.0.1:{self.port}/tts"
        payload = {"text": text, "voice": self.voice}

        try:
            async with self._http_client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
        except (httpx.HTTPError, OSError) as e:
            logger.warning("PocketTTS synthesis failed: %s", e)
            return
