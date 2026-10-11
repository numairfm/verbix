# Conversational Interruption Handling & Turn Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide natural, contextual conversational interruption handling and turn continuity in Verbix continuous voice mode without keyword phrase matching.

**Architecture:** Mute audio deterministically upon user speech via Silero VAD/AEC; calculate the exact playback boundary to partition generated text into confirmed `heard_text` and `unspoken_draft`; recover deterministically on verified empty non-speech; and inject volatile interruption grounding on user turns so the foreground model contextually continues or adapts without repeating delivered speech.

**Tech Stack:** Python 3.13, asyncio, sherpa-onnx, sounddevice / PortAudio, LiteLLM.

**Spec:** `docs/superpowers/specs/2026-10-11-conversational-interruption-continuity-design.md`

## Global Constraints
- `assistant` role in `foreground.messages` carries ONLY prose actually voiced to the user (AGENTS.md).
- Seams: `voice/` owns the floor & playback cursor, `runtime/` owns grounding & history, `agent/` does generation.
- No hardcoded keyword phrase lists or regex-based intent classification.
- All 280 existing unit tests must remain passing; PTT mode must remain unaffected.
- Full test suite plus `ruff check` clean on all touched code.

## Review Focus
1. Interruption occurs before first sentence starts playing (`heard_text == ""`): history must not retain empty assistant message; `unspoken_draft` retains full text.
2. Interruption occurs during sentence 2: `heard_text` retains sentence 1; `unspoken_draft` starts at sentence 2.
3. User coughs/clears throat causing empty STT transcript: playback of `unspoken_draft` resumes deterministically after endpointing silence without an LLM call.
4. User speaks conversational interjection ("uh-huh", "mhm", "continue"): model receives grounding and continues `unspoken_draft` without repeating sentence 1.
5. User speaks redirection/question ("wait who was that?"): model answers question and adapts rather than blindly continuing.

---

### Task 1: Playback Position Accounting & Interruption Snapshotting in `SpeechOutput`

**Files:**
- Modify: `src/verbix/voice/output.py`
- Test: `tests/test_speech_output.py`

**Interfaces:**
- Produces: `SpeechOutput.interrupt() -> InterruptionSnapshot | None`
- Produces: `SpeechOutput.last_interruption: InterruptionSnapshot | None`
- Produces: `SpeechOutput.speak_text(text: str) -> None`

- [ ] **Step 1: Write failing tests for playback accounting and snapshotting**
In `tests/test_speech_output.py`: test that when speech with multiple sentences is interrupted, `SpeechOutput` records an `InterruptionSnapshot` with `heard_text` containing only completed sentences and `unspoken_draft` containing remaining sentences.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_speech_output.py -k test_interruption_snapshot -v`
Expected: FAIL

- [ ] **Step 3: Implement `InterruptionSnapshot` and playback tracking in `src/verbix/voice/output.py`**
Track cumulative sample counts per queued sentence against `sink.played_samples`. On `interrupt()`, construct and store `self.last_interruption = InterruptionSnapshot(heard_text=..., unspoken_draft=...)`. Provide `speak_text(text)` to queue new text without clearing the interruption history.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_speech_output.py -k test_interruption_snapshot -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/output.py tests/test_speech_output.py
git commit -m "feat(voice): add exact playback accounting and interruption snapshotting to SpeechOutput"
```

---

### Task 2: Deterministic Empty STT / Non-Speech Recovery in `VoicePipeline`

**Files:**
- Modify: `src/verbix/voice/pipeline.py`
- Test: `tests/test_conversational_voice_loop.py`

**Interfaces:**
- Consumes: `SpeechOutput.last_interruption`, `SpeechOutput.speak_text`
- Produces: Resumption of `unspoken_draft` when STT finalizes with empty transcription after user speech endpoint.

- [ ] **Step 1: Write failing test for empty STT recovery**
In `tests/test_conversational_voice_loop.py`: test that when an active turn is interrupted by VAD, and `finalize()` returns empty string `""` with no active speech, `VoicePipeline` resumes `unspoken_draft` without calling `runtime.submit`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_conversational_voice_loop.py -k test_empty_stt_resumes_unspoken_draft -v`
Expected: FAIL

- [ ] **Step 3: Implement safe resumption in `VoicePipeline._run_continuous_loop`**
In `_run_continuous_loop()`, if `transcribed_text` is empty, check if `self._speech_output.last_interruption` has `unspoken_draft`. If so, speak `unspoken_draft` via `_speech_output.speak_text()` and clear `last_interruption`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_conversational_voice_loop.py -k test_empty_stt_resumes_unspoken_draft -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/pipeline.py tests/test_conversational_voice_loop.py
git commit -m "feat(voice): resume unspoken draft deterministically on empty STT finalization"
```

---

### Task 3: Contextual Interruption Grounding in `Runtime` & Foreground Prompt

**Files:**
- Modify: `src/verbix/runtime/runtime.py`
- Modify: `src/verbix/runtime/foreground.py`
- Test: `tests/test_barge_in_heard.py`
- Test: `tests/test_conversational_voice_loop.py`

**Interfaces:**
- Consumes: `SpeechOutput.last_interruption`
- Produces: `Runtime.submit(user_text, output)` passes volatile interruption envelope into user prompt when an interruption snapshot exists.
- Produces: Foreground system prompt rules for conversational turn-taking and continuity.

- [ ] **Step 1: Write failing test for contextual interruption grounding**
In `tests/test_barge_in_heard.py`: test that when `cancel_turn(heard=heard)` occurs and `last_interruption` has `unspoken_draft`, the next `runtime.submit(user_text)` provides `[Floor context: You were interrupted while speaking...]` in the user message passed to the model, while history contains strictly `heard_text`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_barge_in_heard.py -k test_interruption_grounding -v`
Expected: FAIL

- [ ] **Step 3: Implement interruption grounding in `Runtime` and update `DEFAULT_VERBIX_PROMPT`**
In `Runtime._request_grounding()`: accept optional interruption snapshot and format context envelope. Clear snapshot once consumed.
In `DEFAULT_VERBIX_PROMPT`: add conversational turn-taking guidance.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_barge_in_heard.py -k test_interruption_grounding -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/verbix/runtime/runtime.py src/verbix/runtime/foreground.py tests/test_barge_in_heard.py
git commit -m "feat(runtime): inject contextual interruption grounding and update system prompt"
```

---

### Task 4: CLI Terminal Output Synchronization with Audio Delivery

**Files:**
- Modify: `src/verbix/cli.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: Clear visual indicator on interruption in `CliPrinter` or sentence-buffered flushing in voice mode.

- [ ] **Step 1: Write test for `CliPrinter` on turn interruption**
Test that `CliPrinter` handles interruption cleanly without merging subsequent text onto the aborted line.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_cli_smoke.py -k test_cli_printer_interruption -v`
Expected: FAIL

- [ ] **Step 3: Update `CliPrinter` in `src/verbix/cli.py`**
Add `on_interrupt()` or check delivered state so when playback is cut off, `CliPrinter` resets prefix and writes `[interrupted]\n`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_cli_smoke.py -k test_cli_printer_interruption -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/verbix/cli.py tests/test_cli_smoke.py
git commit -m "feat(cli): synchronize CliPrinter state on turn interruption"
```

---

### Task 5: End-to-End Test Suite & Verification

**Files:**
- Modify: `tests/test_conversational_voice_loop.py`

- [ ] **Step 1: Run full test suite and lint**
Run: `uv run pytest` and `uv run ruff check src/`
Expected: ALL PASS

- [ ] **Step 2: Final commit and verify git clean**
```bash
git status
```
