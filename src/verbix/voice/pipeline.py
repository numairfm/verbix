from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from verbix.core.events import TextDelta, TurnComplete
from verbix.voice.capture import PTTCapture
from verbix.voice.player import AudioPlayer

if TYPE_CHECKING:
    from verbix.runtime.tasks import TaskRuntime
    from verbix.voice.base import SpeechToText

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Splits a stream of text into spoken sentences at punctuation or newline boundaries."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class VoicePipeline:
    """End-to-end voice interaction pipeline for Push-to-Talk and background audio orchestration."""

    def __init__(
        self,
        runtime: TaskRuntime,
        stt: SpeechToText,
        player: AudioPlayer,
        capture: PTTCapture | None = None,
    ) -> None:
        self.runtime = runtime
        self.stt = stt
        self.player = player
        self.capture = capture or PTTCapture()

        self._is_active_turn = False
        self._interrupted_playback: bool = False

    async def on_ptt_down(self) -> None:
        """Physical hotkey press down: instantly halt any audio playback and begin mic capture."""
        # 1. Instant barge-in: halt audio playback
        played_text = self.player.interrupt()
        if played_text and self.runtime.main_session:
            try:
                self.runtime.main_session.history.prune_foreground_barge_in(played_text)
            except ValueError:
                pass

        # 2. Begin recording mic
        await self.capture.start_recording()

    async def on_ptt_up(self) -> str | None:
        """Physical hotkey release: stop mic, transcribe speech, and execute conversational turn."""
        audio_bytes = await self.capture.stop_recording()
        if not audio_bytes or len(audio_bytes) < 1600:  # Less than 50ms of audio
            return None

        # Transcribe audio via STT
        transcribed_text = await self.stt.transcribe(audio_bytes)
        transcribed_text = transcribed_text.strip()
        if not transcribed_text:
            return None

        # Execute turn on receptionist agent
        await self.execute_voice_turn(transcribed_text)
        return transcribed_text

    async def execute_voice_turn(self, user_text: str) -> str:
        """Executes a complete receptionist turn from transcribed user text, streaming TTS audio."""
        self._is_active_turn = True
        accumulated_text = ""
        current_buffer = ""
        spoken_sentences: list[str] = []

        try:
            async for event in self.runtime.step_receptionist(user_input=user_text):
                if isinstance(event, TextDelta):
                    accumulated_text += event.text
                    current_buffer += event.text

                    # Check for sentence boundary to begin immediate TTS synthesis
                    if any(p in current_buffer for p in (". ", "! ", "? ", ".\n", "!\n", "?\n")):
                        sentences = split_sentences(current_buffer)
                        if len(sentences) > 1:
                            # Speak complete sentences, keep last partial sentence in buffer
                            complete = sentences[:-1]
                            current_buffer = sentences[-1]
                            spoken = await self.player.speak_foreground_sentences(complete)
                            spoken_sentences.append(spoken)

                elif isinstance(event, TurnComplete):
                    # Speak any remaining text in buffer
                    if current_buffer.strip():
                        remaining = split_sentences(current_buffer)
                        spoken = await self.player.speak_foreground_sentences(remaining)
                        spoken_sentences.append(spoken)

            # Drain any pending background task announcements that finished
            await self.player.drain_announcements()

        except asyncio.CancelledError:
            self.player.interrupt()
            raise
        finally:
            self._is_active_turn = False

        return accumulated_text
