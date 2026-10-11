# STT Hardening & Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 10 correctness, concurrency, and integrity issues identified during the STT review pass, following AGENTS.md (modular, simple, failing test first, green suite).

**Architecture:** 
1. Fix concurrency race in `SherpaSTT.transcribe_raw` by adding module-level thread lock around `os.dup2` stderr redirection.
2. Fix audio loss in `SileroSegmenter.flush` by padding and feeding remainder buffered samples before flush.
3. Fix session state leak on turn rejection by clearing segments list in `STTSession.finalize`.
4. Fix `asyncio.CancelledError` swallowing by distinguishing task cancellation from caller cancellation.
5. Fix unhandled `ValueError` on odd byte buffers in audio conditioning and transcription.
6. Clean up dead code, unused parameters, vestigial comments, and test coverage gaps.

**Tech Stack:** Python 3.13, asyncio, numpy, sherpa-onnx, pytest.

**Spec:** Review report from previous turn and `AGENTS.md`.

## Global Constraints

- One thing per file; minimal code; no speculative abstraction.
- TDD: write failing test first, run to confirm failure, write minimal code, verify green.
- Full suite plus `ruff check` green with `set -o pipefail`.
- Zero file mutations or test runs that leave uncommitted/untagged artifacts outside git clean.

## Review Focus

- Multi-span utterances concurrently calling `transcribe_raw` must never corrupt FD 2 (stderr).
- Spans with sub-512 remainder at stream end must be processed, never dropped.
- Turn-level gate reject must clear `committed_text` so subsequent queries return empty string.
- Odd byte length buffers must not raise `ValueError` from `numpy.frombuffer`.
- Tests must run hermetically without requiring external uncommitted model files.

---

### Task 1: Stderr Redirection Thread Lock (`SherpaSTT.transcribe_raw`)

**Files:**
- Modify: `src/verbix/voice/sherpa_stt.py:363-380`
- Test: `tests/test_sherpa_stt.py`

- [ ] **Step 1: Write failing test verifying thread-safe stderr redirection**
Add `test_concurrent_transcribe_raw_preserves_stderr` in `tests/test_sherpa_stt.py` simulating concurrent calls and asserting FD 2 remains valid and unchanged.

- [ ] **Step 2: Run test to verify it reproduces the concurrency risk/fails**
Run: `uv run pytest tests/test_sherpa_stt.py -k test_concurrent_transcribe_raw_preserves_stderr -v`

- [ ] **Step 3: Implement `_STDERR_LOCK` in `src/verbix/voice/sherpa_stt.py`**
Add `_STDERR_LOCK = threading.Lock()` and wrap `os.dup2` block in `with _STDERR_LOCK:`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_sherpa_stt.py -k test_concurrent_transcribe_raw_preserves_stderr -v`

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/sherpa_stt.py tests/test_sherpa_stt.py
git commit -m "fix(stt): synchronize stderr redirection across thread decodes"
```

---

### Task 2: Remainder Padding in `SileroSegmenter.flush()`

**Files:**
- Modify: `src/verbix/voice/sherpa_vad.py:125-128`
- Test: `tests/test_incremental_stt.py` (or new `tests/test_sherpa_vad.py`)

- [ ] **Step 1: Write failing test in `tests/test_sherpa_vad.py`**
Create `tests/test_sherpa_vad.py` asserting that sub-512 sample tails in `_pending` are zero-padded to `WINDOW` and passed to `detector.accept_waveform` before `detector.flush()`.

- [ ] **Step 2: Run test to verify it fails**
Run: `uv run pytest tests/test_sherpa_vad.py -v`

- [ ] **Step 3: Implement padding in `SileroSegmenter.flush()`**
In `src/verbix/voice/sherpa_vad.py`: if `len(self._pending) > 0`, pad to `WINDOW`, call `det.accept_waveform(...)`.

- [ ] **Step 4: Run test to verify it passes**
Run: `uv run pytest tests/test_sherpa_vad.py -v`

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/sherpa_vad.py tests/test_sherpa_vad.py
git commit -m "fix(vad): pad and accept pending audio tail before flush"
```

---

### Task 3: State Reset on Turn Gate Rejection & CancelledError Propagation

**Files:**
- Modify: `src/verbix/voice/sherpa_stt.py:116-126`
- Test: `tests/test_incremental_stt.py`

- [ ] **Step 1: Write failing tests for gate reject state clearing and cancellation**
Add tests in `tests/test_incremental_stt.py`:
- `test_gate_rejection_clears_committed_text`
- `test_finalize_cancellation_propagates`

- [ ] **Step 2: Run tests to verify failure**
Run: `uv run pytest tests/test_incremental_stt.py -k "test_gate_rejection_clears_committed_text or test_finalize_cancellation_propagates" -v`

- [ ] **Step 3: Implement fixes in `STTSession.finalize`**
1. Call `self._segments.clear()` on gate rejection.
2. In `await seg.task`, re-raise `CancelledError` if `asyncio.current_task().cancelling()`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `uv run pytest tests/test_incremental_stt.py -k "test_gate_rejection_clears_committed_text or test_finalize_cancellation_propagates" -v`

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/sherpa_stt.py tests/test_incremental_stt.py
git commit -m "fix(stt): clear segments on gate reject and propagate finalize cancellation"
```

---

### Task 4: Buffer Odd-Length Alignment Guard

**Files:**
- Modify: `src/verbix/voice/capture.py:31`, `src/verbix/voice/sherpa_stt.py:322`
- Test: `tests/test_capture_warmup.py`, `tests/test_sherpa_stt.py`

- [ ] **Step 1: Write failing tests with odd-length byte buffers**
Add `test_condition_handles_odd_bytes` and `test_transcribe_handles_odd_bytes`.

- [ ] **Step 2: Run tests to verify ValueError**
Run: `uv run pytest -k "odd_bytes" -v`

- [ ] **Step 3: Implement truncate to even byte length**
Ensure `data = data[:len(data) - len(data) % 2]` before `np.frombuffer`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `uv run pytest -k "odd_bytes" -v`

- [ ] **Step 5: Commit**
```bash
git add src/verbix/voice/capture.py src/verbix/voice/sherpa_stt.py tests/test_capture_warmup.py tests/test_sherpa_stt.py
git commit -m "fix(voice): guard against odd byte buffer lengths before int16 conversion"
```

---

### Task 5: Clean Dead Code, Unused Parameters, Vestigial Comments & Test Gaps

**Files:**
- Modify: `src/verbix/voice/sherpa_stt.py:31-33, 311`, `src/verbix/voice/sherpa_vad.py:69-71`
- Modify: `tests/test_stt_nospeech.py`, `tests/test_fetch_voice_models.py`

- [ ] **Step 1: Write tests for dict manifest in fetch_voice_models and hermetic test_stt_nospeech**
Update `tests/test_fetch_voice_models.py` to cover dict-shaped manifest entries. Make `tests/test_stt_nospeech.py` hermetic without requiring on-disk models.

- [ ] **Step 2: Remove dead code, vestigial comments, and unused `initial_prompt`**
Remove dead `SileroSegmenter.load()`, unused `initial_prompt` parameter, and stale comments.

- [ ] **Step 3: Run full suite and ruff check**
Run: `set -o pipefail && uv run pytest -v && uv run ruff check src tests scripts`

- [ ] **Step 4: Commit**
```bash
git add src/ tests/ scripts/
git commit -m "refactor(stt): prune dead code, remove unused parameters, and close test coverage gaps"
```
