# Specification: Contextual Interruption Handling & Turn Continuity

- **Author**: Antigravity & Engineer
- **Status**: Draft (Approved in Brainstorming)
- **Target Subsystem**: `verbix.voice`, `verbix.runtime`, `verbix.cli`
- **Date**: 2026-10-11

---

## 1. Problem Statement

In conversational voice mode (`--voice --voice-mode continuous`), when the user speaks while Verbix is talking (barge-in):
1. **Destruction of Unspoken Work**: The LLM's streaming generation finishes in ~1–2 seconds, while TTS takes ~15–20 seconds to speak. When interrupted, all unplayed speech frames are purged, leaving Verbix with no memory of what it was saying or was about to say.
2. **Loss of Interruption Context**: The barge-in speech is transcribed and sent to the LLM as a totally disconnected prompt. Because the model has no knowledge that it was interrupted, it treats conversational interjections (e.g. "uh-huh", "wait", "continue") as brand-new standalone queries, leading to nonsensical responses like `"I am waiting."` or repeating the whole task from scratch.
3. **Visual Desync**: `CliPrinter` outputs tokens to `sys.stdout` as the LLM generates them, meaning the terminal displays text that was never voiced to the user.
4. **Playback Position Inaccuracy**: `SpeechOutput._delivered` records text as spoken when audio chunks are yielded to the sink generator, rather than when the DAC output actually plays them.

---

## 2. Goals & Non-Goals

### Goals
- **Precise Playback Accounting**: Accurately distinguish what the user heard (`heard_text`) from what was generated but not yet voiced (`unspoken_draft`).
- **Contextual Interruption Understanding**: Use the foreground LLM to interpret the relationship between the user's interjection, what was spoken, and the unspoken draft without brittle keyword lists or hardcoded phrase matching.
- **Natural Resumption & Adaptation**: Support seamless continuation on acknowledgments, relevant answers to mid-sentence questions, and flexible direction changes without repeating spoken content.
- **Terminal Synchronization**: Ensure the terminal transcript does not display unvoiced text as delivered speech.
- **Zero-Latency Non-Speech Recovery**: When a VAD barge-in is triggered but STT finalizes with empty speech after endpointing (e.g. cough, noise, breath), deterministically resume the unspoken draft without an unnecessary LLM round-trip.

### Non-Goals
- Building an offline audio classifier or separate intent-routing neural model.
- Preserving or splicing raw audio waveforms across turns (TTS synthesis is fast, ~50ms/sentence; continuation text is re-synthesized cleanly).
- Changing push-to-talk (`--voice-mode ptt`) interaction semantics.

---

## 3. Architecture & Data Flow

```
                      [User Speaks During Playback]
                                    │
                       (0 ms) Silero VAD / AEC
                                    │
                                    ▼
                     VoicePipeline._interrupt_active_speech()
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
   SpeechOutput.interrupt()                    Sink DAC Sample Accounting
  (Stops sink, captures exact                 (Determines exact sentence
   playback boundary & cursor)                 boundary reached by user)
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                  Snapshot: (heard_text, unspoken_draft)
                                    │
                  STT Endpoint Reached & Finalized
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
        [STT Text is Empty]                  [STT Text Has Words]
        (Acoustic noise/cough)              (User spoke an utterance)
                 │                                     │
                 ▼                                     ▼
     Deterministic Resumption:            Contextual Agent Turn:
     If safe (speech ended &              1. Assistant history repaired
     unspoken_draft exists),                 to contain ONLY heard_text.
     feed unspoken_draft directly         2. User turn dispatched with
     to SpeechOutput.                        volatile interruption grounding.
                                          3. LLM resolves intent & continues
                                             or adapts naturally.
```

---

## 4. Detailed Component Design

### 4.1 Playback Position & Cursor Tracking (`voice/output.py` & `voice/callback_sink.py`)

1. **DAC Position Accounting**:
   - `CallbackAudioSink` maintains `played_samples`, incremented strictly inside the PortAudio callback when non-silent samples are delivered to the hardware.
   - `SpeechOutput` tracks each sentence frame with its cumulative audio sample span `(start_sample, end_sample, text)`.
   - On `interrupt()`, `SpeechOutput` queries the sink for current `played_samples`:
     - Sentences with `end_sample <= played_samples` are marked **delivered** (`heard_text`).
     - The first sentence where `start_sample <= played_samples < end_sample` is the interruption point. For clean linguistic continuity, this sentence is treated as the start of `unspoken_draft` (or truncated at the last complete word boundary if length permits; sentence-level boundary prevents dangling phonemes).
     - Sentences with `start_sample > played_samples` are marked **undelivered** (`unspoken_draft`).
   - *Buffer Uncertainty Note*: In `SubprocessAudioSink` or systems without hardware callback feedback, `SpeechOutput` falls back to playback time elapsed $\times$ sample rate. Buffer latency jitter ($\le 50$ ms) is noted as the physical uncertainty limit.

2. **Retention of Unspoken Draft**:
   - Instead of discarding remaining frames on `interrupt()`, `SpeechOutput` stores:
     ```python
     self.last_interruption: InterruptionSnapshot | None
     ```
     containing `heard_text: str`, `unspoken_draft: str`, and `timestamp: float`.
   - `delivered_text()` returns `heard_text`.

### 4.2 Terminal Output Synchronization (`cli.py`)

- `CliPrinter` currently streams tokens immediately upon generation.
- To prevent displaying text the user never heard:
  - In voice mode, `CliPrinter` buffers generated text deltas per turn.
  - As sentences are handed to playback and confirmed voiced, `CliPrinter` flushes them to `sys.stdout`.
  - Alternatively, if text was already printed when interrupted, `CliPrinter` emits a visual interruption indicator (e.g. `\n[interrupted]\n`) and resets its line state so subsequent output does not appear merged.

### 4.3 Deterministic Recovery for Empty STT (`voice/pipeline.py`)

- When barge-in occurs, the continuous loop awaits the user speech endpoint (`_speech_endpoint_event`).
- `finalize()` decodes the captured audio.
- If `transcribed_text.strip() == ""`:
  - Check if Silero VAD still detects active speech:
    - If yes: user is still vocalizing; continue waiting.
    - If no: the vocalization has concluded and contained no decipherable speech (cough, sigh, background click).
  - If `last_interruption` contains non-empty `unspoken_draft`:
    - Resynthesize and speak `unspoken_draft` immediately via `SpeechOutput`.
    - No LLM call is made.

### 4.4 Contextual Grounding & Prompt Instructions (`runtime/` & `agent/`)

1. **Conversation History Invariant**:
   - In accordance with `AGENTS.md`, `self.foreground.messages` contains ONLY text Verbix actually spoke:
     ```python
     {"role": "assistant", "content": heard_text}
     ```
   - The unspoken draft is NEVER inserted into an assistant role message.

2. **Volatile Interruption Grounding**:
   - When an interruption snapshot is active and the user submits an utterance `user_text`:
     [`Runtime._step_inner`](file:///home/numair/dev/projects/verbix/src/verbix/runtime/runtime.py#L170) injects interruption grounding into the user message envelope alongside the clock grounding:
     ```text
     [Current local date and time: Monday, October 11, 2026, 10:45 PM.]
     [Floor context: You were interrupted by the user while speaking.]
     [What user actually heard: "{heard_text}"]
     [Your unspoken continuation was: "{unspoken_draft}"]
     [User interruption utterance: "{user_text}"]
     ```
   - The raw prompt text to the agent is constructed clearly so the model sees what the user heard, what was unsaid, and the user's interjection.

3. **System Prompt Rules for Turn Taking**:
   - In `DEFAULT_VERBIX_PROMPT`:
     ```text
     # CONVERSATIONAL TURN TAKING & INTERRUPTIONS
     - When interrupted while speaking, your prior spoken text is preserved and your unspoken draft is provided in floor context.
     - If the user's interjection is an acknowledgment, backchannel ("uh-huh", "yeah", "mhm", "continue", "go on"), smoothly continue the unfinished response from where it was cut off without repeating delivered text.
     - If the user asks a clarification question or requests a modification, address it directly using your context, then offer to continue the original task if relevant.
     - If the user explicitly cancels ("stop", "never mind"), acknowledge briefly and stop.
     - Never recite floor context tags, brackets, or system annotations aloud.
     ```

---

## 5. Acceptance Criteria

1. **Exact History Pinning**:
   - Given an assistant turn with 3 sentences, if interrupted during sentence 2, `runtime.foreground.messages[-1]["content"]` contains strictly sentence 1 (or sentence 1 + confirmed words of sentence 2).
   - Sentence 3 and unplayed parts are NOT in the message history.

2. **Contextual Resumption (No Keywords)**:
   - User says "Tell me a story." Verbix begins speaking.
   - User interjects with an acknowledgment ("uh-huh", "mhm", "yeah", "okay go on").
   - Verbix resumes the story from the unspoken draft without repeating the opening sentence and without saying `"I am waiting."`.

3. **Contextual Question Mid-Speech**:
   - User interjects with "Wait, who was the lighthouse keeper again?".
   - Verbix answers the question ("Elara was the keeper...") and continues or asks to continue.

4. **Task Redirection**:
   - User interjects with "Actually, change the setting to a spaceship instead."
   - Verbix pivots the story to a spaceship rather than completing the lighthouse story.

5. **Empty STT Resilience**:
   - A cough or loud breath trips VAD, pausing playback.
   - When acoustic silence resumes and STT returns `""`, Verbix resumes speaking the remaining draft without an error or LLM call.

6. **All Existing Tests & Lints Green**:
   - All 280 existing unit tests continue to pass.
   - PTT mode functionality is completely preserved.
   - `ruff check src/` is green.

---

## 6. Failure Modes & Mitigations

- **Model Ignores Continuity**: If a small model occasionally restarts from sentence 1 despite grounding, the system prompt guidance explicitly forbids repeating the delivered text snippet.
- **Race on Immediate Interruption**: If the user interrupts within the first 100ms before any sentence finishes playing, `heard_text` is empty `""`. `validate_and_repair_history` correctly pops the empty assistant turn, and the unspoken draft holds the entire response.
- **Microphone Echo / Self-Interruption**: AEC attenuation ensures speaker playback is not fed back as false user speech.
