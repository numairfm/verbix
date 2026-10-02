from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from verbix.voice.base import AudioSink, TextToSpeech

if TYPE_CHECKING:
    from verbix.core.models import WorkerResult
    from verbix.runtime.tasks import TaskRuntime

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Manages serialized audio playback of foreground dialogue and background task announcements.

    Enforces that speech items never collide or overlap, and provides instantaneous barge-in interruption.
    """

    def __init__(
        self,
        tts: TextToSpeech,
        sink: AudioSink,
        runtime: TaskRuntime | None = None,
    ) -> None:
        self.tts = tts
        self.sink = sink
        self.runtime = runtime

        self._active_play_task: asyncio.Task[None] | None = None
        self._is_interrupted: bool = False
        self._current_text: str = ""
        self._played_text: str = ""
        self._current_task_id: str | None = None

        # Lock to ensure only one item plays at a time
        self._playback_lock = asyncio.Lock()

    @property
    def is_playing(self) -> bool:
        return self._active_play_task is not None and not self._active_play_task.done()

    def interrupt(self) -> str:
        """Instantly halts any active audio playback on the sink and cancels the playback task.

        Returns the text that was successfully played before the interruption.
        """
        self._is_interrupted = True
        self.sink.stop()

        if self._active_play_task and not self._active_play_task.done():
            self._active_play_task.cancel()

        played = self._played_text.strip()
        return played

    async def speak_foreground_sentences(
        self,
        sentences: list[str],
    ) -> str:
        """Plays a sequence of foreground assistant sentences sequentially with barge-in support.

        Returns the accumulated text that was actually spoken before any interruption.
        """
        async with self._playback_lock:
            self._is_interrupted = False
            self._current_task_id = None
            self._played_text = ""

            for sentence in sentences:
                if self._is_interrupted:
                    break

                self._current_text = sentence
                completed = await self._synthesize_and_play(sentence)
                if completed:
                    self._played_text += sentence + " "
                else:
                    break

            return self._played_text.strip()

    async def play_announcement(
        self,
        task_id: str,
        result: WorkerResult,
    ) -> bool:
        """Plays a background task completion announcement sequentially.

        Sets task.announced = True only if playback completed without interruption.
        """
        async with self._playback_lock:
            self._is_interrupted = False
            self._current_task_id = task_id
            self._current_text = result.speech_summary
            self._played_text = ""

            completed = await self._synthesize_and_play(result.speech_summary)

            if completed:
                if self.runtime:
                    task = self.runtime.get_task(task_id)
                    if task:
                        task.announced = True
                return True
            return False

    async def drain_announcements(self) -> int:
        """Drains and sequentially plays all pending announcements in the runtime's announcement queue."""
        if not self.runtime:
            return 0

        played_count = 0
        while not self.runtime.announcement_q.empty():
            if self._is_interrupted:
                break
            task_id, result = self.runtime.announcement_q.get_nowait()
            success = await self.play_announcement(task_id, result)
            if success:
                played_count += 1
            else:
                break
        return played_count

    async def _synthesize_and_play(self, text: str) -> bool:
        """Synthesizes text through TTS and pipes audio chunks to the AudioSink."""
        if not text.strip():
            return True

        try:
            async for audio_chunk in self.tts.synthesize(text):
                if self._is_interrupted:
                    return False
                finished = await self.sink.play(audio_chunk)
                if not finished or self._is_interrupted:
                    return False
            return True
        except asyncio.CancelledError:
            return False
        except (OSError, RuntimeError) as e:
            logger.warning("Playback synthesis error: %s", e)
            return False
