from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechToText(Protocol):
    """Protocol for transcribing speech audio bytes to text."""

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribes raw PCM audio data and returns recognized text string."""
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Protocol for synthesizing speech audio bytes from text."""

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesizes text into chunks of raw audio bytes (PCM/WAV)."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """Protocol for hardware or subprocess audio playback output."""

    async def play(self, audio_data: bytes) -> bool:
        """Plays a single chunk of raw audio data."""
        ...

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> bool:
        """Plays a stream of raw audio chunks to completion."""
        ...

    def stop(self) -> None:
        """Immediately halts any current audio playback."""
        ...


class MockSpeechToText:
    """Mock STT for testing and headless environments."""

    def __init__(self, predefined_transcriptions: list[str] | None = None) -> None:
        self.transcriptions = list(predefined_transcriptions or [])
        self.recorded_audios: list[bytes] = []

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        self.recorded_audios.append(audio_data)
        if self.transcriptions:
            return self.transcriptions.pop(0)
        return "mock transcription"


class MockTextToSpeech:
    """Mock TTS for testing and headless environments."""

    def __init__(self) -> None:
        self.synthesized_texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized_texts.append(text)
        # Yield simulated 100ms chunks of silence/tone
        chunk = b"\x00\x00" * 800  # 800 samples = 50ms at 16kHz 16-bit mono
        yield chunk
        yield chunk


class MockAudioSink:
    """Mock audio playback sink for testing."""

    def __init__(self, playback_delay: float = 0.05) -> None:
        self.playback_delay = playback_delay
        self.played_chunks: list[bytes] = []
        self._stopped = False

    async def play(self, audio_data: bytes) -> bool:
        async def _iter():
            yield audio_data

        return await self.play_stream(_iter())

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> bool:
        self._stopped = False
        try:
            async for chunk in chunks:
                if self._stopped:
                    return False
                self.played_chunks.append(chunk)
                elapsed = 0.0
                step = 0.01
                while elapsed < self.playback_delay:
                    if self._stopped:
                        return False
                    await asyncio.sleep(step)
                    elapsed += step
            return not self._stopped
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        self._stopped = True
