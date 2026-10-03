from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

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
    """End-to-end toggle voice interaction pipeline with live STT during recording,

    instant audible cues, barge-in during playback, and gesture-less confirmation timeouts.
    """

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
        self._live_stt: Any = None
        self._pending_confirmation: asyncio.Future[bool] | None = None
        self._confirmation_prompt: str | None = None

    @property
    def is_confirmation_pending(self) -> bool:
        return (
            self._pending_confirmation is not None
            and not self._pending_confirmation.done()
        )

    @property
    def confirmation_prompt(self) -> str | None:
        return self._confirmation_prompt

    async def toggle(self) -> dict[str, Any]:
        """Toggles between listening and idle/processing.

        - If audio playback is active: halts playback instantly and begins recording.
        - If recording is active: ends capture, plays audible cue, transcribes, and executes turn.
        - If idle: starts mic capture with progressive live STT.
        """
        # 1. Playback barge-in
        if self.player.is_playing:
            played_text = self.player.interrupt()
            if played_text and self.runtime.main_session:
                try:
                    self.runtime.main_session.history.prune_foreground_barge_in(
                        played_text
                    )
                except ValueError:
                    pass
            await self.start_recording()
            return {"action": "interrupted_and_recording"}

        # 2. Recording active -> end capture and process
        if self.capture.is_recording:
            return await self.stop_recording_and_process()

        # 3. Idle -> begin recording
        await self.start_recording()
        return {"action": "recording"}

    async def start_recording(self) -> None:
        """Begins recording and attaches live STT worker."""
        if self.capture.is_recording:
            return

        create_live_fn = getattr(self.stt, "create_live_session", None)
        if callable(create_live_fn):
            self._live_stt = create_live_fn(sample_rate=self.capture.sample_rate)
            self._live_stt.start()
            self.capture.on_chunk = self._live_stt.feed

        await self.capture.start_recording()

    async def stop_recording_and_process(self) -> dict[str, Any]:
        """Stops recording, plays instant audible cue, finalizes STT, and starts receptionist turn."""
        audio_bytes = await self.capture.stop_recording()

        # Instant audible cue on capture stop
        asyncio.create_task(self.player.play_cue())

        transcribed_text = ""
        if self._live_stt is not None:
            transcribed_text = await self._live_stt.finish(audio_bytes)
            self._live_stt = None
        elif audio_bytes and len(audio_bytes) >= 1600:
            transcribed_text = await self.stt.transcribe(audio_bytes)

        transcribed_text = transcribed_text.strip()
        if not transcribed_text:
            return {"action": "idle", "transcription": ""}

        # Launch turn execution in background so toggle command returns immediately
        asyncio.create_task(self.execute_voice_turn(transcribed_text))
        return {"action": "processing", "transcription": transcribed_text}

    async def request_confirmation(
        self, prompt: str, timeout_seconds: float = 15.0
    ) -> bool:
        """Asks for confirmation (confirm/deny) without holding a key, denying automatically on timeout."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending_confirmation = fut
        self._confirmation_prompt = prompt

        try:
            confirmed = await asyncio.wait_for(fut, timeout=timeout_seconds)
            return confirmed
        except TimeoutError:
            logger.info(
                "Confirmation timed out after %.1fs; defaulting to deny.",
                timeout_seconds,
            )
            return False
        finally:
            self._pending_confirmation = None
            self._confirmation_prompt = None

    def confirm(self) -> bool:
        """Resolves a pending confirmation with True."""
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(True)
            return True
        return False

    def deny(self) -> bool:
        """Resolves a pending confirmation with False."""
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(False)
            return True
        return False

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
                    if any(
                        p in current_buffer
                        for p in (". ", "! ", "? ", ".\n", "!\n", "?\n")
                    ):
                        sentences = split_sentences(current_buffer)
                        if len(sentences) > 1:
                            complete = sentences[:-1]
                            current_buffer = sentences[-1]
                            spoken = await self.player.speak_foreground_sentences(
                                complete
                            )
                            spoken_sentences.append(spoken)

                elif isinstance(event, TurnComplete):
                    if current_buffer.strip():
                        remaining = split_sentences(current_buffer)
                        spoken = await self.player.speak_foreground_sentences(remaining)
                        spoken_sentences.append(spoken)

            await self.player.drain_announcements()

        except asyncio.CancelledError:
            self.player.interrupt()
            raise
        finally:
            self._is_active_turn = False

        return accumulated_text
