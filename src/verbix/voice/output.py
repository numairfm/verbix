"""Non-blocking speech output: one utterance streams through one sink stream.

`emit()` only enqueues epoch-tagged frames and returns; a background pump task
synthesizes each text frame (`normalize_for_tts` + per-chunk `synthesize`) and
yields audio into a single `play_stream` generator call. `interrupt()` bumps
the epoch, silences the sink, and drops stale frames synchronously.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from verbix.core.events import AgentEvent, TextDelta, TurnComplete
from verbix.core.telemetry import record_latency
from verbix.voice.segments import normalize_for_tts, split_chunks

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TERMINAL = (".", "!", "?")


def _sentences(text: str) -> list[str]:
    """Split text at sentence boundaries so sub-220 tails keep TTS pauses."""
    return [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]


class SpeechOutput:
    """TurnOutput that speaks agent text without blocking the turn loop."""

    def __init__(self, tts: Any, sink: Any, printer: Any = None) -> None:
        self.tts = tts
        self.sink = sink
        self.printer = printer
        self._buf = ""
        self._seen = ""
        self._delivered: list[str] = []
        self._epoch = 0
        self._frames: asyncio.Queue[tuple[int, str, Any]] = asyncio.Queue()
        self._synth_tasks: set[asyncio.Task] = set()
        self._pump: asyncio.Task[None] | None = None

    @property
    def is_playing(self) -> bool:
        return self._pump is not None and not self._pump.done()

    async def emit(self, event: AgentEvent) -> None:
        if self.printer and hasattr(self.printer, "emit"):
            res = self.printer.emit(event)
            if asyncio.iscoroutine(res):
                await res

        if isinstance(event, TextDelta):
            self._buf += event.text
            self._seen += event.text
            chunks = split_chunks(self._buf)
            if len(chunks) > 1:
                ready, self._buf = chunks[:-1], chunks[-1]
            elif self._buf.strip().endswith(_TERMINAL):
                ready, self._buf = chunks, ""
            else:
                return
            for c in ready:
                for s in _sentences(c) or [c]:
                    self._frames.put_nowait((self._epoch, "text", s))
            self._ensure_pump()
        elif isinstance(event, TurnComplete):
            buf, self._buf = self._buf, ""
            if buf.strip():
                for s in _sentences(buf) or [buf]:
                    self._frames.put_nowait((self._epoch, "text", s))
            rest = self._tail(event.final_text)
            if rest:
                for s in _sentences(rest) or [rest]:
                    self._frames.put_nowait((self._epoch, "text", s))
            self._frames.put_nowait((self._epoch, "end", ""))
            self._ensure_pump()

    def _tail(self, final_text: str) -> str:
        f = (final_text or "").strip()
        if not f or not self._seen:
            return f
        if f.startswith(self._seen.strip()):
            return f[len(self._seen.strip()):].strip()
        if self._seen.strip().startswith(f) or f in self._seen:
            return ""
        return f

    def interrupt(self) -> None:
        """Barge-in: silence audio now, discard everything not yet spoken."""
        playing = self.is_playing
        if playing:
            record_latency("interrupt_called")
        self._epoch += 1
        self.sink.stop()
        if playing:
            record_latency("audio_silent")
        if self._pump and not self._pump.done():
            self._pump.cancel()
        self._pump = None
        for task in list(self._synth_tasks):
            if not task.done():
                task.cancel()
        self._synth_tasks.clear()
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buf = ""
        self._seen = ""
        self._delivered.clear()

    def delivered_text(self) -> str:
        return " ".join(self._delivered).strip()

    def reset(self) -> None:
        self.interrupt()

    def reset_delivered(self) -> None:  # compat for WakeOutputFanOut/printer protocol
        self.reset()

    async def play_cue(self) -> None:
        """Earcon that never kills speech: direct play only when idle."""
        try:
            import numpy as np

            sample_rate = getattr(self.sink, "sample_rate", 24000)
            fmt = getattr(self.sink, "audio_format", "f32")
            n = int(sample_rate * 0.05)
            t = np.linspace(0, 0.05, n, endpoint=False)
            envelope = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
            wave = (np.sin(2 * np.pi * 880.0 * t) * envelope * 0.25).astype(np.float32)
            cue = wave.tobytes() if fmt == "f32" else (wave * 32767).astype(np.int16).tobytes()
        except (ImportError, OSError, RuntimeError, ValueError):
            logger.debug("Failed to build cue sound", exc_info=True)
            return
        if self.is_playing:
            self._frames.put_nowait((self._epoch, "cue", cue))
        else:
            await self.sink.play(cue)

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        current = asyncio.current_task()
        epoch = self._epoch
        try:
            while self._epoch == epoch and not self._frames.empty():
                frames_before = self._frames.qsize()
                await self.sink.play_stream(self._feed(epoch))
                if self._frames.qsize() >= frames_before and not self._frames.empty():
                    while not self._frames.empty():
                        try:
                            self._frames.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    break
        except asyncio.CancelledError:
            if self._epoch == epoch:
                self.sink.stop()
            raise
        finally:
            if self._pump is current:
                self._pump = None

    def _spawn_synth(
        self, payload: str, epoch: int
    ) -> asyncio.Task[list[bytes] | None]:
        """Synthesizes one text frame to buffered parts without playing them.

        Returns None when the epoch moved mid-synthesis (stale: discard).
        Playing still happens only in _feed, so prefetched audio is never
        heard unless the pump yields it — barge-in stays exact.
        """

        async def _run() -> list[bytes] | None:
            parts: list[bytes] = []
            async for audio in self.tts.synthesize(normalize_for_tts(payload)):
                if self._epoch != epoch:
                    return None
                parts.append(audio)
            return parts

        task = asyncio.create_task(_run())
        self._synth_tasks.add(task)
        task.add_done_callback(self._synth_tasks.discard)
        return task

    async def _feed(self, epoch: int):
        first_write = True
        first_pcm = True
        holdback: tuple[int, str, Any] | None = None
        carry: tuple[int, str, asyncio.Task] | None = None
        while True:
            if carry is None:
                if holdback is not None:
                    tag, kind, payload = holdback
                    holdback = None
                else:
                    tag, kind, payload = await self._frames.get()
                if tag != self._epoch or tag != epoch:
                    return
                if kind == "end":
                    if self._frames.empty() and holdback is None:
                        return
                    continue
                if kind == "cue":
                    if first_write:
                        record_latency("sink_first_write")
                        first_write = False
                    yield payload
                    continue
                carry = (tag, payload, self._spawn_synth(payload, epoch))
            tag, payload, task = carry
            carry = None
            # Prefetch exactly one upcoming text frame while this one runs.
            upcoming: tuple[int, str, asyncio.Task] | None = None
            try:
                tag2, kind2, payload2 = self._frames.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                if kind2 == "text" and tag2 == epoch and tag2 == self._epoch:
                    upcoming = (tag2, payload2, self._spawn_synth(payload2, epoch))
                else:
                    holdback = (tag2, kind2, payload2)
            try:
                parts = await task
            except (asyncio.CancelledError, Exception):
                if upcoming is not None and not upcoming[2].done():
                    upcoming[2].cancel()
                raise
            if parts is None:
                if upcoming is not None and not upcoming[2].done():
                    upcoming[2].cancel()
                return
            if tag != self._epoch or tag != epoch:
                if upcoming is not None and not upcoming[2].done():
                    upcoming[2].cancel()
                return
            if first_pcm:
                record_latency("tts_first_pcm")
                first_pcm = False
            if parts:
                self._delivered.append(payload)
                for audio in parts:
                    if first_write:
                        record_latency("sink_first_write")
                        first_write = False
                    yield audio
            carry = upcoming
