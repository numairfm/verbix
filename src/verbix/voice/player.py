from __future__ import annotations

import asyncio
import logging

from verbix.voice.base import AudioSink, TextToSpeech

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Manages serialized audio playback of TTS synthesis with instantaneous barge-in."""

    def __init__(
        self,
        tts: TextToSpeech,
        sink: AudioSink,
    ) -> None:
        self.tts = tts
        self.sink = sink
        self._playback_lock = asyncio.Lock()
        self._is_playing: bool = False
        self._active_play_task: asyncio.Task[bool] | None = None

    @property
    def is_playing(self) -> bool:
        return self._is_playing or (
            self._active_play_task is not None and not self._active_play_task.done()
        )

    def stop(self) -> None:
        """Instantly halts any active audio playback on the sink and cancels playback task."""
        self._is_playing = False
        self.sink.stop()
        if self._active_play_task and not self._active_play_task.done():
            self._active_play_task.cancel()

    async def speak(self, text: str) -> bool:
        """Synthesizes text and streams audio chunks through the sink with barge-in support."""
        if not text.strip():
            return True

        async with self._playback_lock:
            self._is_playing = True
            current_task = asyncio.current_task()
            self._active_play_task = current_task
            try:
                chunks = self.tts.synthesize(text)
                return await self.sink.play_stream(chunks)
            except asyncio.CancelledError:
                self.sink.stop()
                raise
            finally:
                if self._active_play_task is current_task:
                    self._active_play_task = None
                self._is_playing = False

    async def play_cue(self) -> None:
        """Plays an instant audible earcon/cue sound to acknowledge capture stop."""
        try:
            import numpy as np

            sample_rate = getattr(self.sink, "sample_rate", 24000)
            fmt = getattr(self.sink, "audio_format", "f32")
            duration = 0.05
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            envelope = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2
            wave_f32 = (np.sin(2 * np.pi * 880.0 * t) * envelope * 0.25).astype(
                np.float32
            )

            if fmt == "f32":
                cue_bytes = wave_f32.tobytes()
            else:
                cue_bytes = (wave_f32 * 32767).astype(np.int16).tobytes()

            await self.sink.play(cue_bytes)
        except (OSError, RuntimeError) as e:
            logger.debug("Failed to play cue sound: %s", e)
