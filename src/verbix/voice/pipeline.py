from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from verbix.voice.capture import PTTCapture
from verbix.voice.output import SpeechOutput
from verbix.voice.player import AudioPlayer

if TYPE_CHECKING:
    from verbix.voice.base import SpeechToText

logger = logging.getLogger(__name__)


class VoicePipeline:
    """End-to-end toggle voice interaction pipeline with live STT during recording,

    instant audible cues, barge-in during playback, and single foreground turn ownership.
    """

    def __init__(
        self,
        runtime: Any,
        stt: SpeechToText,
        player: AudioPlayer,
        capture: PTTCapture | None = None,
        printer: Any = None,
    ) -> None:
        self.runtime = runtime
        self.stt = stt
        self.player = player
        self.capture = capture or PTTCapture()
        self.printer = printer

        self._speech_output = SpeechOutput(player.tts, player.sink, printer=self.printer)
        if self.runtime is not None:
            self.runtime.wake_output = self._speech_output

        self._live_stt: Any = None
        self._active_voice_task: asyncio.Task[str | None] | None = None

    async def toggle(self) -> dict[str, Any]:
        """Toggles between listening and idle/processing.

        - If audio playback is active or a turn is running: halts playback, cancels turn, begins recording.
        - If recording is active: ends capture, plays audible cue, transcribes, and executes turn.
        - If idle: starts mic capture with progressive live STT.
        """
        # 1. Recording active -> end capture and process
        if self.capture.is_recording:
            return await self.stop_recording_and_process()

        # 2. Playback / turn barge-in
        if self._speech_output.is_playing or self.player.is_playing or not self.runtime.is_idle:
            self._speech_output.interrupt()
            await self.runtime.cancel_turn()
            self.runtime.user_active = True
            await self.start_recording()
            return {"action": "interrupted_and_recording"}

        # 3. Idle -> begin recording
        self.runtime.user_active = True
        await self.start_recording()
        return {"action": "recording"}

    async def start_recording(self) -> None:
        """Begins recording and attaches incremental STT session."""
        if self.capture.is_recording:
            return

        open_session_fn = getattr(self.stt, "open_session", None)
        if callable(open_session_fn):
            self._live_stt = open_session_fn(sample_rate=self.capture.sample_rate)
            self.capture.on_chunk = self._live_stt.feed

        # Warm provider connection in background so TLS handshake is complete before speech ends
        if hasattr(self.runtime, "provider") and hasattr(self.runtime.provider, "warm"):
            asyncio.create_task(self.runtime.provider.warm())

        await self.capture.start_recording()

    async def stop_recording_and_process(self) -> dict[str, Any]:
        """Stops recording, plays instant audible cue, finalizes STT, and starts receptionist turn."""
        self.runtime.user_active = False
        audio_bytes = await self.capture.stop_recording()

        # Instant audible cue on capture stop
        asyncio.create_task(self._speech_output.play_cue())

        transcribed_text = ""
        if self._live_stt is not None:
            transcribed_text = await self._live_stt.finalize()
            self._live_stt = None
        elif audio_bytes and len(audio_bytes) >= 1600:
            transcribed_text = await self.stt.transcribe(audio_bytes)

        transcribed_text = transcribed_text.strip()
        if not transcribed_text:
            return {"action": "idle", "transcription": ""}

        # Launch turn execution in background so toggle command returns immediately
        self._active_voice_task = asyncio.create_task(
            self.execute_voice_turn(transcribed_text)
        )
        return {"action": "processing", "transcription": transcribed_text}

    async def execute_voice_turn(self, user_text: str) -> str | None:
        """Executes a complete receptionist turn from transcribed user text, streaming TTS audio."""
        if self.printer and hasattr(self.printer, "reset_delivered"):
            self.printer.reset_delivered()
        self._speech_output.reset()
        return await self.runtime.submit(user_text, self._speech_output)

    def delivered_text(self) -> str:
        return self._speech_output.delivered_text()
