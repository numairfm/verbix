from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pocket_tts import TTSModel  # pyright: ignore[reportMissingImports]
except ImportError:
    TTSModel = None  # type: ignore[assignment, misc]


class PocketTTSEngine:
    """Local Text-to-Speech engine using Kyutai's pocket-tts Python library.

    Outputs 24kHz float32 mono PCM audio bytes streamed in small chunks (~80ms each).
    Conforms to the TextToSpeech protocol.
    """

    def __init__(
        self,
        voice: str = "alba",
        temperature: float = 0.3,
        quantize: bool = False,
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.temperature = temperature
        self.quantize = quantize
        self.sample_rate = sample_rate
        self.channels = 1
        self.audio_format = "f32"

        self._model: Any = None
        self._voice_state: Any = None
        self._lock = asyncio.Lock()
        self._is_closed = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._voice_state is not None

    async def load(self) -> None:
        """Loads the TTSModel and pre-computes the target voice state in a thread."""
        if TTSModel is None:
            raise ImportError(
                "pocket-tts is not installed. Install it with `uv sync --extra tts` "
                "or `pip install pocket-tts`."
            )

        async with self._lock:
            if self.is_loaded:
                return

            def _init() -> tuple[Any, Any]:
                assert TTSModel is not None
                logger.info("Loading PocketTTS model (quantize=%s)...", self.quantize)
                model = TTSModel.load_model(
                    quantize=self.quantize, temp=self.temperature
                )
                voice_state = model.get_state_for_audio_prompt(self.voice)
                return model, voice_state

            self._model, self._voice_state = await asyncio.to_thread(_init)
            logger.info("PocketTTS loaded successfully with voice '%s'", self.voice)

    async def aclose(self) -> None:
        """Async cleanup method."""
        self._is_closed = True
        self._voice_state = None
        self._model = None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesizes text into chunks of raw 24kHz float32 mono PCM audio bytes."""
        clean_text = text.strip()
        if not clean_text or self._is_closed:
            return

        if not self.is_loaded:
            await self.load()

        assert self._model is not None
        assert self._voice_state is not None

        q: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()

        def _worker() -> None:
            try:
                for chunk in self._model.generate_audio_stream(
                    self._voice_state,
                    clean_text,
                    stop=stop_event,
                ):
                    if stop_event.is_set():
                        break
                    pcm_bytes = chunk.numpy().astype(np.float32).tobytes()
                    loop.call_soon_threadsafe(q.put_nowait, pcm_bytes)
                loop.call_soon_threadsafe(q.put_nowait, None)
            except (OSError, RuntimeError, ValueError) as e:
                loop.call_soon_threadsafe(q.put_nowait, e)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.warning("PocketTTS generation error: %s", item)
                    break
                yield item
        finally:
            stop_event.set()
