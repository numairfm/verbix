# Voice Latency (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut voice-to-voice latency to ≤900ms p50 by decoupling synthesis/playback from the LLM stream, decoding STT incrementally, and speaking the first chunk eagerly.

**Architecture:** Replaces `SpeechTurnOutput` (serial speak-per-sentence) with an epoch-tagged `SpeechOutput` that streams one utterance through a single sink stream; replaces the O(n²) live-STT loop with VAD-committed segments decoded once; moves pre-tool chatter release from turn-end to first-chunk boundary. No new required dependencies; `sounddevice` only as an optional `audio` extra.

**Tech Stack:** Python 3.13, asyncio, faster-whisper (bundled Silero VAD), PocketTTS, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-06-architecture-scaling-and-runtime-hardening.md` (§3.2 S1–S4, §3.3 T-Pipeline/sinks/barge-in, §5 budgets)

## Global Constraints
- **Zero new required dependencies:** everything runs on the existing tree; only addition allowed is the optional `audio` extra (`sounddevice`) that is never imported unless selected.
- **Strict PTT only (D2):** no always-open mic or background listening loops; mic active only during explicit toggle recording.
- **Invariants:** H1–H5, F1–F4, P1–P3, L1–L3 from the spec; all 121 existing tests green at every step.
- **No model downloads in tests:** STT tests inject a fake decoder/VAD; TTS/sink tests use `verbix.voice.base` mocks. `faster-whisper` model weights must never load in CI.
- **Backward Compatibility:** `transcribe()`, `TextToSpeech`/`AudioSink` protocols, CLI flags, and IPC contracts keep working.

## Review Focus
1. **Long first sentence with no clause boundary (>220 chars)** → S2 soft-max must force a split at the last clause boundary so first audio isn't held hostage. Pinned by Task 1's soft-max test.
2. **Interrupt mid-utterance, then speak again immediately** → stale PCM from the old epoch must never reach the sink. Pinned by Task 2's epoch test.
3. **No audio backend installed (`pw-play`/`aplay`/`ffplay` all missing)** → turn completes silently without crashing. Pinned by Task 2's no-backend test.
4. **12s+ continuous speech with no 250ms pause** → force-commit must fire so no segment grows unbounded. Pinned by Task 5's force-commit test.
5. **Tool call lands after the first chunk was already spoken** → spoken prefix stays in history, unspoken remainder is discarded, no dangling `tool_calls`. Pinned by Task 7's mid-speech tool test.

---

## File Structure

- `src/verbix/voice/segments.py` (new): pure segmenter + normalizer, no imports beyond stdlib. Used by `output.py` and `default.py`.
- `src/verbix/voice/output.py` (new): `SpeechOutput` — owns one utterance: segment buffer, synth worker task, sink pump via a single `play_stream` generator, epoch interrupt, delivered accounting.
- `src/verbix/voice/pipeline.py` (modify): wire `SpeechOutput` in place of `SpeechTurnOutput`; live-STT via `open_session`; cue only when idle; `delivered_text()` delegates to output.
- `src/verbix/voice/sinks.py` (modify): `SubprocessAudioSink` unchanged interface (`play`/`play_stream`/`stop`); `play_stream` already opens one process per call — one call per utterance now comes free from `SpeechOutput`.
- `src/verbix/voice/callback_sink.py` (new): `CallbackAudioSink` with ring buffer + 10ms fade on `stop()`; `audio.sink: auto | callback | subprocess` selection in `cli.py`.
- `src/verbix/voice/stt.py` (modify): add `STTSession` + `FasterWhisperSTT.open_session()`; replace `LiveSTTSession` usage in pipeline (delete `LiveSTTSession` + `create_live_session`).
- `src/verbix/voice/capture.py` (modify): 640-byte reads; pipeline clears `on_chunk` when session ends.
- `src/verbix/minisweagent/models/litellm_model.py` (modify): add `LitellmModel.warm()` (1-token completion, 60s rate limit).
- `src/verbix/minisweagent/agents/default.py` (modify): A4 first-chunk release — yield buffered `TextDelta`s at the first segment boundary instead of turn end; assistant content keeps the released prefix.
- `pyproject.toml` (modify): `[project.optional-dependencies] audio = ["sounddevice"]`.

---

### Task 1: Segmenter + Normalizer (pure, table-tested)

**Files:**
- Create: `src/verbix/voice/segments.py`
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Produces: `split_chunks(text: str) -> list[str]` (stable boundaries only — never splits inside a protected token)
- Produces: `normalize_for_tts(text: str) -> str` (TTS input only; never touches history/display)
- Consumes: nothing (stdlib only: `re`)

- [ ] **Step 1: Write failing table tests for S1–S4 + normalizer**

```python
# In tests/test_speech_segments.py
from verbix.voice.segments import split_chunks, normalize_for_tts

def test_eager_first_chunk_at_clause_boundary():
    assert split_chunks("Well, this is a much longer piece of spoken text here") == [
        "Well,", "this is a much longer piece of spoken text here"]

def test_no_split_inside_protected_tokens():
    assert split_chunks("Meet Dr. Smith at 3.14 for v1.2.3 release") == [
        "Meet Dr. Smith at 3.14 for v1.2.3 release"]
    assert split_chunks("see https://example.com/a.b for e.g. details here ok") == [
        "see https://example.com/a.b for e.g. details here ok"]

def test_soft_max_forces_split_at_last_clause():
    long_text = "First part here, " + "x" * 300
    chunks = split_chunks(long_text)
    assert len(chunks) == 2 and all(len(c) <= 221 for c in chunks)

def test_normalizer_strips_markdown_and_symbols():
    assert normalize_for_tts("Use *bold* and 50% & `code` → ok") == "Use bold and 50 percent and code to ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speech_segments.py -v`
Expected: FAIL (`No module named 'verbix.voice.segments'`).

- [ ] **Step 3: Implement `split_chunks` + `normalize_for_tts` in `src/verbix/voice/segments.py`**

Rules (spec §3.3): S1 eager first chunk — emit at first clause boundary (`, ; : —`) once ≥40 chars, or sentence end, whichever first. S2 later chunks break at sentence ends, soft max 220 chars with forced split at last clause boundary. S3 protected: abbreviations (`Mr. Mrs. Dr. St. vs. etc. e.g. i.e. U.S.`), decimals/versions, ellipses, whitespace-free tokens (URLs/paths), line-initial ordinals (`1.`). Normalizer: strip `* _ # \`` + leading bullets, URLs→"a link", long paths→final component, `%`→"percent", `&`→"and", `→`→"to". Numbers pass through.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_speech_segments.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat(voice): add clause-aware segmenter and TTS normalizer"
```

---

### Task 2: SpeechOutput Pipeline (non-blocking emit, epoch barge-in)

**Files:**
- Create: `src/verbix/voice/output.py`
- Modify: `src/verbix/voice/pipeline.py`
- Test: `tests/test_speech_output.py`

**Interfaces:**
- Consumes: `split_chunks`, `normalize_for_tts` from Task 1; `TextToSpeech`/`AudioSink` protocols + `MockTextToSpeech`/`MockAudioSink` from `verbix.voice.base`
- Produces: `SpeechOutput` in `src/verbix/voice/output.py` with `async emit(event)`, sync `interrupt()`, `delivered_text() -> str`, `reset()`

- [ ] **Step 1: Write failing tests for non-blocking emit, epoch discard, no double-speak**

```python
# In tests/test_speech_output.py
import asyncio
import pytest
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput
from verbix.core.events import TextDelta, TurnComplete

@pytest.mark.asyncio
async def test_emit_does_not_block_on_playback():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=5.0))
    await out.emit(TextDelta(text="Hello world. This is a test sentence here."))
    await asyncio.sleep(0.05)
    assert out.sink.played_chunks  # audio started without emit awaiting it

@pytest.mark.asyncio
async def test_interrupt_discards_stale_chunks():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.2))
    await out.emit(TextDelta(text="First sentence here spoken. Second sentence here spoken."))
    await asyncio.sleep(0.05)
    out.interrupt()
    n = len(out.sink.played_chunks)
    await out.emit(TextDelta(text="New sentence here spoken."))
    await asyncio.sleep(0.3)
    assert out.delivered_text().startswith("New")

@pytest.mark.asyncio
async def test_turn_complete_never_respeaks_delivered_text():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink())
    await out.emit(TextDelta(text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    await out.emit(TurnComplete(final_text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    assert out.tts.synthesized_texts.count("Hello world. This is spoken.") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speech_output.py -v`
Expected: FAIL (`No module named 'verbix.voice.output'`).

- [ ] **Step 3: Implement `SpeechOutput` in `src/verbix/voice/output.py`**

`emit()` only enqueues (L1 — never awaits playback/synth). One utterance = one `play_stream` generator call fed by a synth worker task with 1-chunk lookahead (`PocketTTSEngine.synthesize` per chunk, thread-safe stop via existing `stop_event`). Sync `interrupt()`: `epoch += 1`, cancel pump task, `sink.stop()`, clear queues — no awaits before audio silenced. `delivered_text()`: text of chunks handed to the sink (H2: counts once playback started). `TurnComplete`: speak only the unemitted tail, never `final_text` when deltas were delivered (fixes double-speak). Earcon: route through `SpeechOutput` as an epoch-tagged frame; `play_cue`-via-`sink.play` only when no utterance is active (never kills speech). `reset()` clears buffer + delivered (replaces `reset_delivered`).

- [ ] **Step 4: Wire into `VoicePipeline`, keep `SpeechTurnOutput` deleted**

Replace `self._speech_output` with `SpeechOutput(player.tts, player.sink)`; `delivered_text()` delegates; cue path uses output; `TurnHandle` provider already reads `delivered_text()` — unchanged.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_speech_output.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/voice/output.py src/verbix/voice/pipeline.py tests/test_speech_output.py
git commit -m "feat(voice): add non-blocking SpeechOutput with epoch barge-in"
```

---

### Task 3: CallbackAudioSink + Sink Selection (optional `audio` extra)

**Files:**
- Create: `src/verbix/voice/callback_sink.py`
- Modify: `src/verbix/cli.py`, `pyproject.toml`, `src/verbix/voice/__init__.py`
- Test: `tests/test_callback_sink.py`

**Interfaces:**
- Consumes: `AudioSink` protocol from `verbix.voice.base`
- Produces: `CallbackAudioSink` (same `play`/`play_stream`/`stop` surface); `audio.sink: auto | callback | subprocess` selection

- [ ] **Step 1: Write failing tests for fade math + selection fallback**

```python
# In tests/test_callback_sink.py
import math
from verbix.voice.callback_sink import apply_fade, select_sink

def test_fade_ramps_to_silence_in_10ms():
    faded = apply_fade([1.0] * 240, sample_rate=24000)
    assert faded[-1] == 0.0 and all(b <= a for a, b in zip(faded, [1.0] + faded))

def test_auto_falls_back_without_sounddevice(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert select_sink("auto").__class__.__name__ == "SubprocessAudioSink"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_callback_sink.py -v`
Expected: FAIL (`No module named 'verbix.voice.callback_sink'`).

- [ ] **Step 3: Implement `CallbackAudioSink` + selection**

Ring buffer feeding the PortAudio callback; `stop()` sets a flag — the callback applies a 10ms linear fade then silence (no clicks, silence within one ≤20ms callback period); exact played-sample cursor feeds `delivered_text`. `sounddevice` imported lazily inside the constructor (never at module import). `select_sink("auto")` uses callback iff `sounddevice` imports, else subprocess. `pyproject.toml`: `audio = ["sounddevice"]` optional extra. `cli.py`: `--audio-sink auto|callback|subprocess` (default `auto`).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_callback_sink.py -v`
Expected: PASS (no audio device needed — pure math + selection).

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/callback_sink.py src/verbix/cli.py pyproject.toml src/verbix/voice/__init__.py tests/test_callback_sink.py
git commit -m "feat(voice): add callback sink with fade-out barge-in"
```

---

### Task 4: Segment-Committed Incremental STT (S1, L2)

**Files:**
- Modify: `src/verbix/voice/stt.py`, `src/verbix/voice/pipeline.py`
- Test: `tests/test_incremental_stt.py`

**Interfaces:**
- Consumes: `get_speech_timestamps` from `faster_whisper.vad` (bundled, no model download)
- Produces: `StreamingSTT.open_session(sample_rate) -> STTSession` with `feed(pcm)`, `async finalize() -> str`, `cancel()`, optional `on_partial`/`on_endpoint`; `transcribe()` stays for one-shot use. `STTSession` takes an injectable monotonic `clock` (default `time.monotonic`) so tests control silence timing without sleeping.

- [ ] **Step 1: Write failing tests with injected fake decoder/VAD**

```python
# In tests/test_incremental_stt.py
import pytest
from verbix.voice.stt import SegmentCommittedSTT

def _speech_frames(n):  # fake VAD: nonzero energy
    return [b"\x01\x02" * 320] * n

@pytest.mark.asyncio
async def test_pause_commits_segment_exactly_once():
    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "hello"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: True,
                              clock=lambda: now[0])
    sess = stt.open_session()
    speech = _speech_frames(20)
    for chunk in speech:
        sess.feed(chunk)
    now[0] += 0.3  # 300ms of silence elapses on the fake clock
    sess.feed(b"\x00\x00" * 320)  # silence: VAD-false, triggers commit check
    await sess.finalize()
    assert len(calls) == 1  # committed audio never decoded twice
    assert calls[0] == b"".join(speech)

@pytest.mark.asyncio
async def test_force_commit_bounds_long_segments():
    seen = []
    async def fake_decode(pcm, prompt=""):
        seen.append(len(pcm))
        return "x"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: True,
                              clock=lambda: now[0])
    sess = stt.open_session()
    for chunk in _speech_frames(2000):  # ~20s of "speech"
        sess.feed(chunk)
        now[0] += 0.01
    await sess.finalize()
    assert len(seen) >= 2
    assert all(n <= 12 * 16000 * 2 + 16000 * 2 for n in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_incremental_stt.py -v`
Expected: FAIL (`cannot import name 'SegmentCommittedSTT'`).

- [ ] **Step 3: Implement `STTSession` + faster-whisper backend in `src/verbix/voice/stt.py`**

Spec S1 values: Silero VAD on 32ms frames as chunks arrive; internal pause ≥250ms commits `[seg_start, pause]` — decode once, append to `committed_text`, never decode again; `feed()` evaluates commit on every call against `clock`, plus a background watchdog covering trailing silence with no further feeds (production only); force-commit at 12s at the lowest-energy frame in the last 1s (numpy RMS); each decode passes last ~200 chars of `committed_text` as `initial_prompt`; `finalize()` decodes only the open segment (L2); constructor takes `decode_fn`/`vad_fn` (production defaults: `FasterWhisperSTT.transcribe` slice + `get_speech_timestamps` wrapper) so tests never load weights. Delete `LiveSTTSession`/`create_live_session`; pipeline uses `open_session` + `feed` as `on_chunk`, `finalize()` on PTT release.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_incremental_stt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/stt.py src/verbix/voice/pipeline.py tests/test_incremental_stt.py
git commit -m "feat(voice): add segment-committed incremental STT"
```

---

### Task 5: Capture Hygiene + Real Provider Warm-up (S3, S4)

**Files:**
- Modify: `src/verbix/voice/capture.py`, `src/verbix/voice/pipeline.py`, `src/verbix/minisweagent/models/litellm_model.py`
- Test: `tests/test_capture_warmup.py`

**Interfaces:**
- Produces: `LitellmModel.warm()` (idempotent, ≤1 call per 60s); 20ms capture frames
- Consumes: `record_latency` from `verbix.core.telemetry` (mark `provider_warm`)

- [ ] **Step 1: Write failing tests for frame size, on_chunk clearing, warm-up**

```python
# In tests/test_capture_warmup.py
import asyncio
from unittest.mock import AsyncMock, patch
from verbix.voice.capture import PTTCapture
from verbix.minisweagent.models.litellm_model import LitellmModel

def test_capture_reads_20ms_frames():
    assert PTTCapture(sample_rate=16000).frame_bytes == 640

@pytest.mark.asyncio
async def test_warm_is_rate_limited_to_60s():
    model = LitellmModel(model="fake-model")
    with patch("litellm.acompletion", new=AsyncMock(return_value={})) as ac:
        await model.warm()
        await model.warm()
        assert ac.await_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture_warmup.py -v`
Expected: FAIL (`No attribute 'frame_bytes'` / `No 'warm'`).

- [ ] **Step 3: Implement capture + warm-up**

`capture.py`: read 640-byte (20ms) frames instead of 4096; pipeline sets `capture.on_chunk = None` when the session ends (S4). `litellm_model.py`: `async def warm()` — one `litellm.acompletion(max_tokens=1)` (or provider models-list where supported), monotonic 60s rate limit, all exceptions swallowed (warm-up must never break recording), `record_latency("provider_warm")`. Pipeline keeps calling `provider.warm()` on record start (now real for `LitellmModel`).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_capture_warmup.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/capture.py src/verbix/voice/pipeline.py src/verbix/minisweagent/models/litellm_model.py tests/test_capture_warmup.py
git commit -m "feat(voice): 20ms capture frames and real provider warm-up"
```

---

### Task 6: First-Chunk Emission + Latency Marks (A4, L1 proof)

**Files:**
- Modify: `src/verbix/minisweagent/agents/default.py`, `src/verbix/voice/output.py`
- Test: `tests/test_first_chunk.py`

**Interfaces:**
- Consumes: `split_chunks` (Task 1); `SpeechOutput.delivered_text()` (Task 2); `record_latency` (Phase 1)
- Produces: first-chunk release in `step_stream`; marks `first_speakable_chunk`, `tts_first_pcm`, `sink_first_write`, `interrupt_called`, `audio_silent`

- [ ] **Step 1: Write failing tests for first-chunk release + spoken-prefix history**

```python
# In tests/test_first_chunk.py
import asyncio
import pytest
from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.core.events import TextDelta, ToolCallDelta

class ChunkThenToolModel:
    async def query_stream(self, messages, tools=None, **kw):
        yield TextDelta(text="Well, let me check that for you right now please")
        yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")

@pytest.mark.asyncio
async def test_first_chunk_released_before_tool_call():
    agent = DefaultAgent(model=ChunkThenToolModel(), tools=ToolRegistry(),
                         config=AgentConfig(system_template="sys"))
    seen = [e async for e in agent.step_stream("hi")]
    texts = [e.text for e in seen if isinstance(e, TextDelta)]
    assert any("Well," in t for t in texts)  # released at chunk boundary, not turn end
    asst = [m for m in agent.messages if m.get("role") == "assistant"][-1]
    assert asst.get("content") and "Well," in asst["content"]  # spoken prefix kept (H2)
    assert asst.get("tool_calls")  # pair intact (H1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_first_chunk.py -v`
Expected: FAIL (first chunk never yielded — full-turn buffering discards it).

- [ ] **Step 3: Implement A4 in `default.py` + marks in `output.py`**

Replace turn-end flush with first-speakable-chunk boundary (Task 1 `split_chunks` on the pending buffer): hold text only until the first boundary; if no tool call by then, yield buffered deltas and keep streaming; text suppressed by a tool call is still never stored, but already-released prefix IS stored as assistant content (H2). `delivered_text` stays the output-side truth via the `TurnHandle` provider. Marks: `record_latency("first_speakable_chunk")` at release, `"tts_first_pcm"` at first synth chunk, `"sink_first_write"` at first sink write, `"interrupt_called"`/`"audio_silent"` around `interrupt()` — all no-ops unless `VERBIX_TRACE=1`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_first_chunk.py tests/test_miniswe_agent_turn.py -v`
Expected: PASS (including the pre-existing suppression test).

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/agents/default.py src/verbix/voice/output.py tests/test_first_chunk.py
git commit -m "feat(agent): release first speakable chunk eagerly with latency marks"
```

---

### Task 7: Verbix-Owned Latency Budget Test (CI gate, fake timings)

**Files:**
- Create: `tests/test_voice_budgets.py`

**Interfaces:**
- Consumes: `SpeechOutput` (Task 2), `select_sink` (Task 3), mocks from `verbix.voice.base`
- Produces: CI assertions on Verbix-owned stages only (LLM TTFT excluded — provider-bound)

- [ ] **Step 1: Write failing budget tests with zero-delay fakes**

```python
# In tests/test_voice_budgets.py
import asyncio
import time
import pytest
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput
from verbix.core.events import TextDelta

@pytest.mark.asyncio
async def test_emit_returns_without_awaiting_playback():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=2.0))
    t0 = time.monotonic()
    await out.emit(TextDelta(text="Hello world. This is spoken aloud."))
    assert time.monotonic() - t0 < 0.5  # L1: emit never blocks on audio

@pytest.mark.asyncio
async def test_interrupt_silences_within_60ms():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=2.0))
    await out.emit(TextDelta(text="Hello world. This is spoken aloud."))
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    out.interrupt()
    assert time.monotonic() - t0 < 0.06  # subprocess-class bound (§5.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_budgets.py -v`
Expected: FAIL (whichever assertion the current serial pipeline violates — `emit` awaits `speak`).

- [ ] **Step 3: Fix whatever the budget test exposes (no new features)**

Only performance-shape fixes inside Task 2/3 code (e.g. an await left on the emit path). If everything passes, this step is a no-op — say so in the report.

- [ ] **Step 4: Run full suite to verify 0 regressions**

Run: `uv run pytest -q`
Expected: PASS (all 121 + new tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_voice_budgets.py
git commit -m "test(voice): gate Verbix-owned latency stages with fake timings"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-06-voice-latency-phase-2.md`. Please review the plan. Does it capture what you want?
