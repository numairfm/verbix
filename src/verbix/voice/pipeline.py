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
        self._is_continuous_active: bool = False
        self._continuous_task: asyncio.Task[None] | None = None
        self._speech_endpoint_event: asyncio.Event = asyncio.Event()
        self._gate_stt_feed: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_continuous_active(self) -> bool:
        return self._is_continuous_active

    async def toggle(self) -> dict[str, Any]:
        """Toggles between listening and idle/processing.

        - If continuous session is active: closes continuous session.
        - If recording is active (PTT): ends capture, plays audible cue, transcribes, and executes turn.
        - If playback is active or a turn is running: halts playback, cancels turn, begins recording.
        - If idle: starts recording.
        """
        if self._is_continuous_active:
            return await self.stop_continuous_session()

        # 1. Recording active -> end capture and process
        if self.capture.is_recording:
            return await self.stop_recording_and_process()

        # 2. Playback / turn barge-in
        if self._speech_output.is_playing or not self.runtime.is_idle:
            heard = self._speech_output.delivered_text()
            self._speech_output.interrupt()
            await self.runtime.cancel_turn(heard=heard)
            self.runtime.user_active = True
            await self.start_recording()
            return {"action": "interrupted_and_recording"}

        # 3. Idle -> begin recording
        self.runtime.user_active = True
        await self.start_recording()
        return {"action": "recording"}


    async def start_continuous_session(self) -> dict[str, Any]:
        """Opens the hands-free conversational voice session."""
        if self._is_continuous_active:
            return {"action": "continuous_active"}

        if self._speech_output.is_playing or (self.runtime and not self.runtime.is_idle):
            heard = self._speech_output.delivered_text()
            self._speech_output.interrupt()
            if self.runtime:
                await self.runtime.cancel_turn(heard=heard)
        else:
            self._speech_output.interrupt()

        self._is_continuous_active = True
        self._loop = asyncio.get_running_loop()
        if self.runtime is not None:
            self.runtime.user_active = True
        if hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            self.runtime.floor.set_state(FloorState.LISTENING)

        await self._setup_turn_listening()
        self._continuous_task = asyncio.create_task(self._run_continuous_loop())
        return {"action": "continuous_started"}

    async def stop_continuous_session(self) -> dict[str, Any]:
        """Closes the hands-free conversational voice session and returns to idle."""
        if not self._is_continuous_active:
            return {"action": "idle"}

        self._is_continuous_active = False
        if self.runtime is not None:
            self.runtime.user_active = False

        if self._speech_output.is_playing or (self.runtime and not self.runtime.is_idle):
            heard = self._speech_output.delivered_text()
            self._speech_output.interrupt()
            if self.runtime:
                await self.runtime.cancel_turn(heard=heard)

        if self._continuous_task and not self._continuous_task.done():
            self._continuous_task.cancel()
            try:
                await self._continuous_task
            except asyncio.CancelledError:
                pass
            self._continuous_task = None

        if self.capture.is_recording:
            await self.capture.stop_recording()
        self.capture.on_chunk = None
        self._loop = None

        if self._live_stt is not None:
            if hasattr(self._live_stt, "cancel"):
                self._live_stt.cancel()
            self._live_stt = None

        if hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            self.runtime.floor.set_state(FloorState.IDLE)

        return {"action": "continuous_stopped"}

    @property
    def _is_speaking(self) -> bool:
        """Returns True if Verbix is actively speaking or playing audio."""
        floor_speaking = (
            getattr(getattr(self.runtime, "floor", None), "is_speaking", False)
            is True
        )
        output_playing = getattr(self._speech_output, "is_playing", False) is True
        player_playing = (
            hasattr(self, "player")
            and getattr(self.player, "is_playing", False) is True
        )
        return (
            self._gate_stt_feed
            or floor_speaking
            or output_playing
            or player_playing
        )

    async def _setup_turn_listening(self) -> None:
        """Sets up STT session and attaches capture callback for listening."""
        from verbix.runtime.floor import FloorState

        self._gate_stt_feed = False

        if hasattr(self.runtime, "floor"):
            self.runtime.floor.set_state(FloorState.LISTENING)

        self._speech_endpoint_event.clear()
        if self._live_stt is not None and hasattr(self._live_stt, "cancel"):
            self._live_stt.cancel()
        if hasattr(self.stt, "open_session"):
            sample_rate = getattr(self.capture, "sample_rate", 16000)
            self._live_stt = self.stt.open_session(
                sample_rate=sample_rate,
                on_endpoint=self._on_speech_endpoint,
            )

        def _feed_chunk(chunk: bytes) -> None:
            if self._is_speaking:
                return
            if not self._gate_stt_feed and self._live_stt is not None:
                try:
                    self._live_stt.feed(chunk)
                except (OSError, RuntimeError) as e:
                    logger.debug("Error feeding STT chunk: %s", e)

        self.capture.on_chunk = _feed_chunk

        if not self.capture.is_recording:
            await self.capture.start_recording()

    async def _run_continuous_loop(self) -> None:
        """Runs the hands-free conversational turn loop until stopped."""
        try:
            while self._is_continuous_active:
                # Wait for user speech endpoint
                await self._speech_endpoint_event.wait()
                if not self._is_continuous_active:
                    break

                # Gate STT feed immediately upon endpoint so self-speech is ignored
                self._gate_stt_feed = True

                # Play audible acknowledgement cue immediately while finalizing STT
                asyncio.create_task(self._speech_output.play_cue())

                transcribed_text = ""
                if self._live_stt is not None:
                    transcribed_text = await self._live_stt.finalize()
                    self._live_stt = None

                transcribed_text = transcribed_text.strip()
                if not transcribed_text:
                    if self._is_continuous_active:
                        await self._setup_turn_listening()
                    continue

                try:
                    await self.execute_voice_turn(transcribed_text)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning("Turn failed during continuous loop: %s", e)
                    if hasattr(self.runtime, "floor"):
                        from verbix.runtime.floor import FloorState

                        self.runtime.floor.set_state(FloorState.SPEAKING)
                    try:
                        await self.player.speak("Sorry, I ran into an error processing that.")
                    except asyncio.CancelledError:
                        raise
                    except Exception as speak_err:  # noqa: BLE001
                        logger.warning("Failed to speak error message: %s", speak_err)
                    while self.player.is_playing:
                        await asyncio.sleep(0.02)
                    await asyncio.sleep(0.15)

                if self._is_continuous_active:
                    await self._setup_turn_listening()

        except asyncio.CancelledError:
            pass

    def _on_speech_endpoint(self) -> None:
        """Signals speech endpoint safely across thread boundaries to the captured event loop."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._speech_endpoint_event.set)
        else:
            self._speech_endpoint_event.set()

    @property
    def turn_output(self) -> SpeechOutput:
        return self._speech_output

    async def start_recording(self) -> None:
        """Begins recording and attaches incremental STT session."""
        if self.capture.is_recording:
            return

        # Hard-mute/interrupt speaker playback immediately so assistant cannot speak over recording
        self._speech_output.interrupt()
        if self.runtime is not None and hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            self.runtime.floor.set_state(FloorState.LISTENING)

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
        self.capture.on_chunk = None

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
            if self.runtime is not None and hasattr(self.runtime, "floor"):
                from verbix.runtime.floor import FloorState

                self.runtime.floor.set_state(FloorState.IDLE)
            secs = len(audio_bytes) / 32000 if audio_bytes else 0.0
            return {"action": "idle", "transcription": "", "detail": f"{secs:.1f}s captured"}

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
        self._speech_output.reset()
        if self.runtime is not None and hasattr(self.runtime, "floor"):
            self.runtime.floor.set_state(FloorState.SPEAKING)
        try:
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
