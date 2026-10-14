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
        aec: Any = None,
        dump_mic_dir: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.stt = stt
        self.player = player
        self.capture = capture or PTTCapture()
        self.printer = printer
        if aec is not None:
            self.aec = aec
        else:
            from verbix.voice.aec import EchoCanceller

            self.aec = EchoCanceller(stream_delay_ms=20)
        self._speech_output = SpeechOutput(
            player.tts,
            player.sink,
            printer=self.printer,
            on_audio=self._feed_speaker_reference,
        )
        if self.runtime is not None:
            self.runtime.wake_output = self._speech_output

        self._live_stt: Any = None
        self._active_voice_task: asyncio.Task[str | None] | None = None
        self._is_continuous_active: bool = False
        self._continuous_task: asyncio.Task[None] | None = None
        self._speech_endpoint_event: asyncio.Event = asyncio.Event()
        self._gate_stt_feed: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mic_lookback: bytes = b""
        self._dump_mic_dir = Path(dump_mic_dir) if dump_mic_dir else None
        self._dump_turn = 0
        self._dump_buf = bytearray()

    def _feed_speaker_reference(self, audio: bytes) -> None:
        """Feeds outgoing speaker audio into AEC reference with explicit format parameters."""
        if hasattr(self, "aec") and callable(getattr(self.aec, "feed_reference", None)):
            sink = getattr(self.player, "sink", None)
            rate = int(getattr(sink, "sample_rate", 24000) or 24000)
            fmt = str(getattr(sink, "audio_format", "f32") or "f32")
            try:
                self.aec.feed_reference(audio, source_sample_rate=rate, audio_format=fmt)
            except TypeError:
                self.aec.feed_reference(audio)

    @staticmethod
    def _dump_slug(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (text or "").strip())[:40].strip("_")
        return slug or "nospeech"

    def _dump_wav(self, audio: bytes, *, kind: str, text: str = "") -> None:
        """Writes mic PCM as 16kHz mono s16 WAV; never raises (evidence must not break turns)."""
        if self._dump_mic_dir is None or not audio:
            return
        try:
            self._dump_mic_dir.mkdir(parents=True, exist_ok=True)
            self._dump_turn += 1
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            label = "lookback" if kind == "lookback" else self._dump_slug(text)
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
            snapshot = self._speech_output.interrupt()
            heard = snapshot.confirmed_heard if snapshot is not None else ""
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
            snapshot = self._speech_output.interrupt()
            heard = snapshot.confirmed_heard if snapshot is not None else ""
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
            snapshot = self._speech_output.interrupt()
            heard = snapshot.confirmed_heard if snapshot is not None else ""
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
            self._live_stt.cancel()
            self._live_stt = None

        if hasattr(self, "aec") and hasattr(self.aec, "reset"):
            self.aec.reset()

        if hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            self.runtime.floor.set_state(FloorState.IDLE)

        return {"action": "continuous_stopped"}

    def _stash_interruption(self, snapshot: Any) -> None:
        """Shelves unplayed text as a runtime side note (never in history)."""
        if self.runtime is None:
            return
        unplayed = (getattr(snapshot, "unplayed_text", "") or "").strip() if snapshot else ""
        self.runtime.pending_continuation = unplayed or None

    def _interrupt_active_speech(self) -> None:
        """Interrupts ongoing speech playback or active turn execution immediately upon user speech detection."""
        if not self._is_speaking and not self._speech_output.is_playing:
            return
        snapshot = self._speech_output.interrupt()
        heard = snapshot.confirmed_heard if snapshot is not None else ""
        self._stash_interruption(snapshot)
        if hasattr(self.runtime, "floor"):
            from verbix.runtime.floor import FloorState

            if hasattr(self.runtime.floor, "request_interrupt"):
                self.runtime.floor.request_interrupt()
            self.runtime.floor.set_state(FloorState.LISTENING)

        old = self._live_stt
        if old is not None:
            old.cancel()
        sample_rate = getattr(self.capture, "sample_rate", 16000)
        try:
            fresh = self.stt.open_session(
                sample_rate=sample_rate,
                on_endpoint=self._on_speech_endpoint,
            )
        except TypeError:
            fresh = self.stt.open_session(sample_rate=sample_rate)
        lookback = getattr(self, "_mic_lookback", b"") or b""
        if lookback:
            self._dump_buf = bytearray(lookback)
            self._dump_wav(lookback, kind="lookback")
            try:
                fresh.feed(lookback)
            except (OSError, RuntimeError) as e:
                logger.debug("Error seeding fresh STT session: %s", e)
        self._live_stt = fresh

        if (
            self.runtime is not None
            and self._loop is not None
            and not self._loop.is_closed()
        ):
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.runtime.cancel_turn(heard=heard))
            )

    @property
    def _is_speaking(self) -> bool:
        """Returns True if Verbix is actively speaking or playing audio."""
        floor_speaking = (
            getattr(getattr(self.runtime, "floor", None), "is_speaking", False)
            is True
        )
        output_playing = getattr(self._speech_output, "is_playing", False) is True
        return (
            self._gate_stt_feed
            or floor_speaking
            or output_playing
        )

    async def _setup_turn_listening(self) -> None:
        """Sets up STT session and attaches capture callback for listening."""
        from verbix.runtime.floor import FloorState

        self._gate_stt_feed = False
        self._dump_buf = bytearray()

        if hasattr(self.runtime, "floor"):
            self.runtime.floor.set_state(FloorState.LISTENING)

        self._speech_endpoint_event.clear()
        if self._live_stt is not None:
            self._live_stt.cancel()
        sample_rate = getattr(self.capture, "sample_rate", 16000)
        self._live_stt = self.stt.open_session(
            sample_rate=sample_rate,
            on_endpoint=self._on_speech_endpoint,
        )

        def _feed_chunk(chunk: bytes) -> None:
            if not chunk:
                return

            if not self._is_continuous_active:
                if self._is_speaking:
                    return
                if not self._gate_stt_feed and self._live_stt is not None:
                    try:
                        self._live_stt.feed(chunk)
                        self._dump_buf.extend(chunk)
                    except (OSError, RuntimeError) as e:
                        logger.debug("Error feeding STT chunk: %s", e)
                return

            # Barge-in: if Verbix is speaking, run AEC and test clean mic audio for user speech
            if self._is_speaking:
                clean_chunk = (
                    self.aec.process_mic(chunk)
                    if hasattr(self, "aec") and getattr(self.aec, "is_available", False)
                    else chunk
                )
                self._mic_lookback = (self._mic_lookback + bytes(clean_chunk))[-9600:]
                if self._live_stt is not None and not self._gate_stt_feed:
                    detected = False
                    try:
                        detected = self._live_stt.feed_barge_in(clean_chunk)
                    except (OSError, RuntimeError) as e:
                        logger.debug("Error feeding STT chunk during playback: %s", e)
                    if detected:
                        self._interrupt_active_speech()
                return

            self._mic_lookback = (self._mic_lookback + bytes(chunk))[-9600:]
            if not self._gate_stt_feed and self._live_stt is not None:
                try:
                    self._live_stt.feed(chunk)
                    self._dump_buf.extend(chunk)
                except (OSError, RuntimeError) as e:
                    logger.debug("Error feeding STT chunk: %s", e)

        self.capture.on_chunk = _feed_chunk

        if not self.capture.is_recording:
            await self.capture.start_recording()

    async def _run_continuous_loop(self) -> None:
        """Runs the hands-free conversational turn loop until stopped."""
        from verbix.runtime.floor import FloorState

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
                speech_dur: float | None = None
                if self._live_stt is not None:
                    transcribed_text = await self._live_stt.finalize()
                    speech_dur = self._live_stt.speech_duration
                    self._dump_wav(bytes(self._dump_buf), kind="turn", text=transcribed_text)
                    self._live_stt = None

                transcribed_text = transcribed_text.strip()
                if not transcribed_text:
                    if self._is_continuous_active:
                        self._gate_stt_feed = False
                        await self._setup_turn_listening()
                    continue

                self._log_user_transcript(transcribed_text, duration=speech_dur)

                # Bare continuation command handling
                norm = transcribed_text.lower().rstrip(".!? ")
                if norm in ("continue", "go on") and self.runtime.pending_continuation:
                    stashed = self.runtime.pending_continuation
                    self.runtime.pending_continuation = None
                    transcribed_text = (
                        f"[Stashed unplayed context: {stashed}]\n"
                        f"User request: {transcribed_text}"
                    )

                if self._is_continuous_active:
                    await self._setup_turn_listening()

                turn_task = asyncio.create_task(
                    self.execute_voice_turn(transcribed_text)
                )
                try:
                    await turn_task
                except asyncio.CancelledError:
                    # Turn was cancelled (e.g. by barge-in). The continuous loop must stay alive!
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.warning("Turn failed during continuous loop: %s", e)
                    if hasattr(self.runtime, "floor"):
                        self.runtime.floor.set_state(FloorState.SPEAKING)
                    try:
                        await self.player.speak("Sorry, I ran into an error processing that.")
                    except asyncio.CancelledError:
                        pass
                    except Exception as speak_err:  # noqa: BLE001
                        logger.warning("Failed to speak error message: %s", speak_err)
                    while self.player.is_playing:
                        await asyncio.sleep(0.02)
                    await asyncio.sleep(0.15)
                finally:
                    if self._is_continuous_active and self._live_stt is None:
                        await self._setup_turn_listening()

        except asyncio.CancelledError:
            pass

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
        if self._speech_output.is_playing:
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
        speech_dur: float | None = None
        if self._live_stt is not None:
            finalize_res = self._live_stt.finalize()
            if asyncio.iscoroutine(finalize_res):
                transcribed_text = await finalize_res
            else:
                transcribed_text = finalize_res
            speech_dur = self._live_stt.speech_duration
            self._live_stt = None
        elif audio_bytes and len(audio_bytes) >= 1600:
            transcribe_fn = getattr(self.stt, "transcribe", None)
            if callable(transcribe_fn):
                if asyncio.iscoroutinefunction(transcribe_fn):
                    transcribed_text = await transcribe_fn(audio_bytes)
                else:
                    transcribed_text = await asyncio.to_thread(transcribe_fn, audio_bytes)
            speech_dur = len(audio_bytes) / 32000.0

        transcribed_text = transcribed_text.strip()
        self._dump_wav(audio_bytes, kind="turn", text=transcribed_text)
        if not transcribed_text:
            if self.runtime is not None and hasattr(self.runtime, "floor"):
                from verbix.runtime.floor import FloorState

                self.runtime.floor.set_state(FloorState.IDLE)
            secs = len(audio_bytes) / 32000 if audio_bytes else 0.0
            return {"action": "idle", "transcription": "", "detail": f"{secs:.1f}s captured"}

        self._log_user_transcript(transcribed_text, duration=speech_dur)

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
            if hasattr(self, "aec") and callable(getattr(self.aec, "reset", None)):
                self.aec.reset()
            if (
                self.runtime is not None
                and hasattr(self.runtime, "floor")
                and self.runtime.floor.state == FloorState.SPEAKING
            ):
                if self._is_continuous_active:
                    self.runtime.floor.set_state(FloorState.LISTENING)
                else:
                    self.runtime.floor.set_state(FloorState.IDLE)

    def delivered_text(self) -> str:
        return self._speech_output.delivered_text()
