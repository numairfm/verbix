# Implementation Plan: transcribe.cpp Migration & Clean Audio Loop Rebuild

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile Canary/sherpa-onnx STT slicing session with a clean, single-pass GGUF STT engine (`transcribe-cpp` + Nemotron 0.6B) and rebuild the continuous audio loop into a simple, rock-solid linear pipeline that cannot wipe audio or letter-loop on noise.

**Architecture:** 
1. **Audio In & VAD:** Microphones stream into a rolling 300ms look-back buffer. Silero VAD runs on incoming audio strictly as a speech gate and endpoint detector.
2. **Turn Buffering:** While speech is detected, incoming audio chunks accumulate into a single `bytearray` turn buffer. No 20ms audio slicing, no background tasks per segment.
3. **Turn Finalization:** Upon VAD endpoint (0.6s trailing silence) or PTT release, if the accumulated turn contains >=150ms of voiced energy (rejecting Enter key clicks and mic pops), the entire buffer is passed to `transcribe-cpp` in a single call.
4. **Clean Barge-In:** When user voice is detected during assistant playback, TTS stops synchronously, the floor state transitions to `LISTENING` immediately (preventing the 50x/sec wipe loop), and the new turn buffer begins accumulating immediately with the 300ms look-back onset.

**Tech Stack:** Python 3.13, `transcribe-cpp==0.4.0` (GGUF / GGML), `transcribe-cpp-native==0.4.0`, Nemotron 3.5 ASR Streaming 0.6B Q4_K_M (496 MB GGUF), `sherpa-onnx` (for Supertonic TTS only), `sounddevice`, `numpy`, `pytest`.

## Global Constraints

- **No git commits:** Do not execute `git commit` unless explicitly instructed.
- **PTT & Agent Invariants Unbroken:** PTT mode, agent tool calling, and conversation history repair invariants must remain 100% green.
- **No sidecar binaries:** Native libraries must be resolved via the official `transcribe-cpp-native` PyPI wheel across all platforms.
- **TDD:** Write failing tests first before implementing each component.
- **Quality Gate:** Full test suite (`pytest -q`) and linter (`ruff check src tests`) must be green at the end of each task.

## Review Focus

1. **Transient click rejection:** A single or double keypress click (e.g. Enter key) must return `""` immediately without calling the ASR model.
2. **Single barge-in trigger:** An interruption must transition the floor state immediately; subsequent chunks of the user's speech must NOT cancel or recreate the recording buffer.
3. **Onset preservation:** The 300ms look-back buffer must preserve the initial consonant/vowel of speech during an interruption without capturing old speaker echo.
4. **Clean linear buffer:** Audio is transcribed in a single contiguous pass; Canary's `_Segment` list and `_spawn_decode` background tasks are completely eliminated.
5. **TTS isolation:** `sherpa-onnx` is retained strictly for Supertonic TTS; it must not be touched or broken by STT changes.

---

### Task 1: Add `transcribe-cpp` Dependency and Update Model Fetcher

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/fetch_voice_models.py`
- Test: `tests/test_model_fetcher.py` (or execute script check)

**Interfaces:**
- Consumes: HuggingFace repository `handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf`
- Produces: `voice-runtime/models/nemotron-gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf`

- [ ] **Step 1: Add `transcribe-cpp` to `pyproject.toml`**

Add `"transcribe-cpp>=0.4.0"` to the dependencies list in `pyproject.toml`. Run `uv sync` to install `transcribe-cpp` and `transcribe-cpp-native`.

- [ ] **Step 2: Update `scripts/fetch_voice_models.py` for Nemotron GGUF**

In `scripts/fetch_voice_models.py`:
- Update the `"asr"` entry in `MANIFEST` to target `nemotron-gguf`:
  - `repo`: `"handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf"`
  - `file`: `"nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"`
  - Size: 520,093,696 bytes
  - SHA256: fetch or verify on first download.

- [ ] **Step 3: Verify model download and package import**

Run: `uv run python -c "import transcribe_cpp; print(transcribe_cpp.backends())"`
Expected: Imports cleanly, reports available backends (CPU, Vulkan).

---

### Task 2: Create Clean `TranscribeSTT` Engine & Click Filter

**Files:**
- Create: `src/verbix/voice/transcribe_stt.py`
- Delete: `src/verbix/voice/sherpa_stt.py` (once replaced)
- Modify: `src/verbix/voice/__init__.py`
- Test: `tests/test_transcribe_stt.py`

**Interfaces:**
- Produces: `TranscribeSTT(model_path: Path)` with method `transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> str`
- Replaces: `SherpaSTT`

- [ ] **Step 1: Write failing tests for `TranscribeSTT`**

In `tests/test_transcribe_stt.py`:
- Test that a clean 16kHz audio buffer transcribes accurately.
- Test that non-speech clicks (<150ms or lacking voiced energy) return `""` immediately without invoking the model session.
- Test that empty or sub-threshold audio returns `""`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcribe_stt.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `TranscribeSTT` in `src/verbix/voice/transcribe_stt.py`**

```python
from pathlib import Path
import numpy as np
import transcribe_cpp

def _has_min_voiced_duration(audio_float: np.ndarray, sample_rate: int = 16000, min_voiced_s: float = 0.15) -> bool:
    frame_len = int(sample_rate * 0.032) # 32ms frames
    if audio_float.size < frame_len:
        return False
    n_full = len(audio_float) // frame_len
    voiced = 0
    for i in range(n_full):
        frame = audio_float[i * frame_len : (i + 1) * frame_len]
        if np.sqrt(float(np.mean(frame**2)) * 32768.0**2) >= 300.0:
            voiced += 1
    return (voiced * 0.032) >= min_voiced_s

class TranscribeSTT:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or default_model_path()
        self._model = transcribe_cpp.Model(str(self.model_path))

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        if not pcm_bytes or len(pcm_bytes) < 320:
            return ""
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)]
        audio_float = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if not _has_min_voiced_duration(audio_float, sample_rate=sample_rate):
            return ""
        with self._model.session() as s:
            res = s.run(audio_float)
            return res.text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transcribe_stt.py -v`
Expected: PASS.

---

### Task 3: Rebuild the Linear Continuous Audio Loop in `VoicePipeline`

**Files:**
- Modify: `src/verbix/voice/pipeline.py:160-350`
- Test: `tests/test_conversational_voice_loop.py`

**Interfaces:**
- Consumes: `TranscribeSTT`, `SileroSegmenter`, `PTTCapture`, `SpeechOutput`
- Enforces linear state machine:
  - `IDLE` -> VAD speech start -> `ACCUMULATING` (collect chunks into buffer)
  - `ACCUMULATING` -> VAD endpoint (0.6s silence) -> `TRANSCRIBING` (one call to `stt.transcribe`) -> Execute Turn -> `IDLE`
  - Assistant speaking -> VAD detects speech -> `INTERRUPT` (stop TTS, floor=LISTENING immediately, seed buffer with lookback) -> `ACCUMULATING`.

- [ ] **Step 1: Write failing tests for single barge-in and linear accumulation**

In `tests/test_conversational_voice_loop.py`:
- Test that when an interrupt occurs, `_interrupt_active_speech` transitions floor to `LISTENING` immediately.
- Test that multiple consecutive chunks during user speech do NOT reset the audio buffer or recreate sessions.
- Test that upon speech endpoint, the full accumulated buffer is transcribed in one call.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conversational_voice_loop.py -k test_barge_in_single_trigger_and_accumulation -v`
Expected: FAIL.

- [ ] **Step 3: Rebuild audio feeding and continuous loop in `pipeline.py`**

- In `_interrupt_active_speech()`:
  - Call `self._speech_output.interrupt()`.
  - Set `self.runtime.floor.set_state(FloorState.LISTENING)` immediately.
  - Trigger `self.runtime.cancel_turn(heard=heard)`.
  - Reset active turn buffer to `self._mic_lookback`.
- In `_feed_chunk(chunk)`:
  - Update `self._mic_lookback`.
  - If Verbix is speaking (`_is_speaking`):
    - Run AEC: `clean = aec.process_mic(chunk)`.
    - If VAD detects speech on `clean`: trigger `_interrupt_active_speech()`.
  - If Verbix is not speaking (`LISTENING`):
    - Feed chunk to Silero VAD.
    - Append chunk to `self._current_turn_buffer`.
    - If VAD signals endpoint (0.6s silence): trigger `_speech_endpoint_event.set()`.
- In `_run_continuous_loop()`:
  - Await `_speech_endpoint_event`.
  - Grab accumulated `self._current_turn_buffer`.
  - Reset `self._current_turn_buffer = bytearray()`.
  - Transcribe buffer with `self.stt.transcribe(buffer)`.
  - If transcribed text is non-empty: execute turn.

- [ ] **Step 4: Run voice loop tests to verify they pass**

Run: `uv run pytest tests/test_conversational_voice_loop.py -v`
Expected: PASS.

---

### Task 4: Clean Up Dead STT Code and Slicing Artifacts

**Files:**
- Delete: `src/verbix/voice/sherpa_stt.py`
- Delete: `tests/test_incremental_stt.py` (legacy slicing tests)
- Modify: `src/verbix/cli.py` (point to `TranscribeSTT`)
- Modify: `src/verbix/voice/__init__.py`

- [ ] **Step 1: Remove `sherpa_stt.py` and obsolete slicing tests**

Remove `sherpa_stt.py` and `tests/test_incremental_stt.py`. Re-export `TranscribeSTT` from `src/verbix/voice/__init__.py`.

- [ ] **Step 2: Update CLI and test imports**

Ensure `cli.py` initializes `TranscribeSTT` using `voice-runtime/models/nemotron-gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf`.

- [ ] **Step 3: Run full test suite and linter**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: All tests pass, zero lint errors.

---

### Task 5: Verification & End-to-End Live Voice Test

- [ ] **Step 1: Run complete test suite**
Run: `uv run pytest -v`
Expected: ALL PASS.

- [ ] **Step 2: Run linter check**
Run: `uv run ruff check src tests`
Expected: ALL PASS.

- [ ] **Step 3: Live interactive smoke test**
Run: `uv run verbix --voice --voice-mode continuous`
Test:
1. "Hello" -> replies.
2. "Tell me a story" -> interrupts with "Wait" -> verifies clean single interruption, no repeated words, no letter hallucinations.
