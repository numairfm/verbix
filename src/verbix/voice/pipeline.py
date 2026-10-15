from __future__ import annotations

import asyncio
import inspect
import logging
import re
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.voice.capture import PTTCapture
from verbix.voice.output import SpeechOutput
from verbix.voice.player import AudioPlayer

if TYPE_CHECKING:
    from verbix.voice.base import SpeechToText

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Push-to-talk voice interaction pipeline with instant audible cues,

    barge-in during playback, and single foreground turn ownership.
    """

    def __init__(
        self,
        runtime: Any,
        stt: SpeechToText,
        player: AudioPlayer,
        capture: PTTCapture | None = None,
        printer: Any = None,
        dump_mic_dir: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.stt = stt
        self.player = player
        self.capture = capture or PTTCapture()
        self.printer = printer
        self._speech_output = SpeechOutput(
            player.tts,
            player.sink,
            printer=self.printer,
        )
        if self.runtime is not None:
            self.runtime.wake_output = self._speech_output

        self._active_voice_task: asyncio.Task[str | None] | None = None
        self._dump_mic_dir = Path(dump_mic_dir) if dump_mic_dir else None
        self._dump_turn = 0

    @staticmethod
    def _dump_slug(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (text or "").strip())[:40].strip("_")
        return slug or "nospeech"

    def _dump_wav(self, audio: bytes, *, text: str = "") -> None:
        """Writes mic PCM as 16kHz mono s16 WAV; never raises (evidence must not break turns)."""
        if self._dump_mic_dir is None or not audio:
            return
        try:
            self._dump_mic_dir.mkdir(parents=True, exist_ok=True)
            self._dump_turn += 1
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            label = self._dump_slug(text)
            path = self._dump_mic_dir / f"{ts}_{self._dump_turn:04d}_{label}.wav"
            audio = audio[: len(audio) - (len(audio) % 2)]
            rate = int(getattr(self.capture, "sample_rate", 16000) or 16000)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(audio)
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Mic dump failed: %s", e)

    def _stash_interruption(self, snapshot: Any) -> None:
        """Shelves unplayed text as a runtime side note (never in history)."""
        if self.runtime is None:
            return
        unplayed = (
            (getattr(snapshot, "unspoken_draft", "") or getattr(snapshot, "unplayed_text", "") or "").strip()
            if snapshot
            else ""
        )
        self.runtime.pending_continuation = unplayed or None

    async def toggle(self) -> dict[str, Any]:
        """Toggles between listening and idle/processing.

        - If recording is active (PTT): ends capture, plays audible cue, transcribes, and executes turn.
        - If playback is active or a turn is running: halts playback, cancels turn, begins recording.
        - If idle: starts recording.
        """
        # 1. Recording active -> end capture and process
        if self.capture.is_recording:
            return await self.stop_recording_and_process()

        # 2. Playback / turn barge-in
        if self._speech_output.is_playing or not self.runtime.is_idle:
            snapshot = self._speech_output.interrupt()
            self._stash_interruption(snapshot)
            heard = snapshot.confirmed_heard if snapshot is not None else ""
            await self.runtime.cancel_turn(heard=heard)
            self.runtime.user_active = True
            await self.start_recording()
            return {"action": "interrupted_and_recording"}

        # 3. Idle -> begin recording
        self.runtime.user_active = True
        await self.start_recording()
        return {"action": "recording"}

    def _log_user_transcript(self, text: str, duration: float | None = None) -> None:
        """Logs finalized recognized speech transcript via printer and debug logger."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if duration is not None:
            logger.info("Recognized user speech (%.1fs): %s", duration, cleaned)
        else:
            logger.info("Recognized user speech: %s", cleaned)
        if self.printer and hasattr(self.printer, "print_user"):
            try:
                if duration is not None:
                    self.printer.print_user(cleaned, duration=duration)
                else:
                    self.printer.print_user(cleaned)
            except TypeError:
                self.printer.print_user(cleaned)
            except Exception as e:  # noqa: BLE001
                logger.debug("Error logging user transcript to printer: %s", e)

    @property
    def turn_output(self) -> SpeechOutput:
        return self._speech_output

    async def start_recording(self) -> None:
        """Begins recording from the microphone."""
        if self.capture.is_recording:
            return

        # Hard-mute/interrupt speaker playback immediately so assistant cannot speak over recording
        if self._speech_output.is_playing:
            self._speech_output.interrupt()
        if self.runtime is not None and hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            self.runtime.floor.set_state(FloorState.LISTENING)

        # Warm provider connection in background so TLS handshake is complete before speech ends
        warm_fn = getattr(getattr(self.runtime, "provider", None), "warm", None)
        if asyncio.iscoroutinefunction(warm_fn):
            asyncio.create_task(warm_fn())

        await self.capture.start_recording()

    async def stop_recording_and_process(self) -> dict[str, Any]:
        """Stops recording, plays instant audible cue, transcribes audio, and starts receptionist turn."""
        self.runtime.user_active = False
        audio_bytes = await self.capture.stop_recording()

        # Instant audible cue on capture stop
        asyncio.create_task(self._speech_output.play_cue())

        transcribed_text = ""
        speech_dur: float | None = None
        if audio_bytes and len(audio_bytes) >= 1600:
            transcribe_fn = getattr(self.stt, "transcribe", None)
            if callable(transcribe_fn):
                if asyncio.iscoroutinefunction(transcribe_fn):
                    transcribed_text = await transcribe_fn(audio_bytes)
                else:
                    transcribed_text = await asyncio.to_thread(transcribe_fn, audio_bytes)
            speech_dur = len(audio_bytes) / 32000.0

        transcribed_text = transcribed_text.strip()
        self._dump_wav(audio_bytes, text=transcribed_text)
        if not transcribed_text:
            if self.runtime is not None and hasattr(self.runtime, "floor"):
                from verbix.runtime.floor import FloorState

                self.runtime.floor.set_state(FloorState.IDLE)
            secs = len(audio_bytes) / 32000 if audio_bytes else 0.0
            return {"action": "idle", "transcription": "", "detail": f"{secs:.1f}s captured"}

        self._log_user_transcript(transcribed_text, duration=speech_dur)

        # Bare continuation command handling
        norm = transcribed_text.lower().rstrip(".!? ")
        if norm in ("continue", "go on") and getattr(
            self.runtime, "pending_continuation", None
        ):
            stashed = self.runtime.pending_continuation
            self.runtime.pending_continuation = None
            transcribed_text = (
                f"[Stashed unplayed context: {stashed}]\n"
                f"User request: {transcribed_text}"
            )

        # Launch turn execution in background so toggle command returns immediately
        self._active_voice_task = asyncio.create_task(
            self.execute_voice_turn(transcribed_text)
        )
        return {"action": "processing", "transcription": transcribed_text}

    async def execute_voice_turn(self, user_text: str) -> str | None:
        """Executes a complete receptionist turn from transcribed user text, streaming TTS audio."""
        from verbix.runtime.floor import FloorState

        if self.printer and hasattr(self.printer, "reset_delivered"):
            self.printer.reset_delivered()
        interruption = getattr(self._speech_output, "last_interruption", None)
        self._speech_output.last_interruption = None
        self._speech_output.reset()
        if self.runtime is not None and hasattr(self.runtime, "floor"):
            self.runtime.floor.set_state(FloorState.SPEAKING)
        try:
            try:
                sig = inspect.signature(self.runtime.submit)
                if "interruption" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    return await self.runtime.submit(
                        user_text, self._speech_output, interruption=interruption
                    )
            except (ValueError, TypeError):
                pass
            return await self.runtime.submit(user_text, self._speech_output)
        finally:
            if (
                self.runtime is not None
                and hasattr(self.runtime, "floor")
                and self.runtime.floor.state == FloorState.SPEAKING
            ):
                self.runtime.floor.set_state(FloorState.IDLE)

    def delivered_text(self) -> str:
        return self._speech_output.delivered_text()
