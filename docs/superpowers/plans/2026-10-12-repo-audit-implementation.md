# Verbix Logic Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement confirmed logic fixes, regressions, justified simplifications, dead-code removal, and cross-platform execution alignment across Verbix.

**Architecture:** Enforce underlying invariants across layers: continuous 60Hz drift filter in audio capture without 20ms boundary clicks; sample-accurate playback accounting decoupled from clean resets; robust conversation history repair under turn cancellation; first-class unsandboxed execution across platforms with optional bubblewrap sandbox; and complete removal of obsolete parallel agent and provider stacks.

**Tech Stack:** Python 3.13, asyncio, NumPy, sherpa-onnx, LiteLLM, pytest.

**Spec:** `/home/numair/.gemini/antigravity-cli/brain/1fc61b8f-8bd3-4ee8-bd21-b5e70ecb578f/full_repo_logic_audit.md`

## Global Constraints

- **No git commits:** Do not run `git commit` unless explicitly instructed by the user.
- **Unsandboxed execution:** Normal execution in `cwd=workspace` is valid intentional behavior across platforms, retaining user permissions. Sandboxed execution via `bwrap` is optional when available and requested.
- **Minimal changes:** Fix at the root invariant; no speculative wrappers, parallel mechanisms, or keyword patches.
- **Commercial readiness:** Keep strict licensing, clean code, no bare pipes (`set -o pipefail`).

## Review Focus

1. **Chunk boundary clicks in STT:** Low noise and sub-100Hz audio must not have 50Hz step discontinuities injected across 20ms boundaries.
2. **Short transcript recognition:** Words like "Ask" or "Hi" must decode accurately into whole words rather than letter spelling or hallucinations.
3. **Interruption snapshot leakage:** Clean resets between completed turns must never set `last_interruption` or falsely mark turns as interrupted.
4. **History repair with tool calls:** Cancelled turns containing tool calls must still update assistant message content to accurately reflect delivered speech.
5. **Cross-platform bash execution:** Missing `bwrap` on macOS, Windows, or systems without bubblewrap must not prevent unsandboxed command execution.

---

### Task 1: Audio Signal Conditioning & STT Invariant Fixes

**Files:**
- Modify: `src/verbix/voice/capture.py:14-50`
- Modify: `tests/test_capture_warmup.py:27-42`
- Test: `tests/test_capture_warmup.py`, `tests/test_sounddevice_capture.py`, `tests/test_sherpa_stt.py`

**Interfaces:**
- Consumes: raw 16kHz s16le PCM byte chunks
- Produces: `_condition_s16le(data: bytes, rate: int, state: tuple[float, float | None]) -> tuple[bytes, tuple[float, float | None]]`

- [ ] **Step 1: Write the failing test for boundary continuity in audio capture**

Add a test in `tests/test_capture_warmup.py` verifying that chunked streaming of low-frequency audio (e.g. 75Hz) does not inject boundary discontinuities (>100 amplitude jumps) between consecutive 20ms frames:
```python
def test_conditioning_smooth_across_chunk_boundaries():
    t = np.arange(16000) / 16000
    wave75 = (3000 * np.sin(2 * np.pi * 75 * t)).astype(np.int16).tobytes()
    state = (0.0, None)
    chunks = []
    for i in range(0, len(wave75), 640):
        c, state = _condition_s16le(wave75[i:i+640], 16000, state)
        chunks.append(c)
    conditioned = np.frombuffer(b"".join(chunks), dtype=np.int16)
    jumps = [abs(int(conditioned[i]) - int(conditioned[i-1])) for i in range(320, 16000, 320)]
    assert max(jumps) < 100, f"Boundary jump {max(jumps)} too large!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture_warmup.py -k test_conditioning_smooth_across_chunk_boundaries -v`
Expected: FAIL with `AssertionError: Boundary jump 1190 too large!`

- [ ] **Step 3: Remove per-chunk mean subtraction in `capture.py:_condition_s16le`**

In `src/verbix/voice/capture.py`:
- Remove `x -= x.mean()` from `_condition_s16le`.
- Set cold start to `prev_y = 0.0, prev_x = float(x[0])` so constant DC is eliminated immediately from sample 0.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture_warmup.py -v`
Expected: PASS

---

### Task 2: Continuous Voice Loop & Playback Accounting Fixes

**Files:**
- Modify: `src/verbix/voice/output.py:206-238`
- Modify: `src/verbix/voice/output.py:276-297`
- Modify: `src/verbix/voice/sinks.py:20-38`
- Modify: `src/verbix/voice/sinks.py:87-148`
- Modify: `src/verbix/voice/pipeline.py:303-332`
- Test: `tests/test_speech_output.py`, `tests/test_conversational_voice_loop.py`, `tests/test_barge_in_heard.py`

**Interfaces:**
- Consumes: `SpeechOutput`, `SubprocessAudioSink`, `VoicePipeline`
- Produces: `SpeechOutput.reset()`, `SpeechOutput.interrupt() -> InterruptionSnapshot | None`, `SubprocessAudioSink.played_samples`

- [ ] **Step 1: Write failing tests for clean reset and playback completion accounting**

In `tests/test_speech_output.py`:
- Test that calling `output.reset()` clears state without leaving `output.last_interruption` populated.
- Test that normal playback completion clears delivered spans so subsequent turns start with 0 delivered samples.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speech_output.py -k test_clean_reset_leaves_no_interruption -v`
Expected: FAIL

- [ ] **Step 3: Implement clean `reset()`, normal completion accounting in `output.py`, and `played_samples` in `sinks.py`**

- In `src/verbix/voice/output.py`:
  - Separate `reset()` from `interrupt()`: `reset()` stops playback and clears state without creating `InterruptionSnapshot` or touching `last_interruption`.
  - In `_run()`: when all frames are consumed normally, reset `_delivered`, `_sent_spans`, and `_enqueued_samples`.
- In `src/verbix/voice/sinks.py`:
  - In `SubprocessAudioSink`: track `_samples_written` and `_stream_start_time`, expose `played_samples` property estimating elapsed playback samples capped at written samples.
- In `src/verbix/voice/pipeline.py`:
  - Do not eagerly call `_setup_turn_listening()` before `execute_voice_turn` finishes in `_run_continuous_loop`. Call `_setup_turn_listening()` after the turn completes (or is cancelled).

- [ ] **Step 4: Run voice tests to verify they pass**

Run: `uv run pytest tests/test_speech_output.py tests/test_barge_in_heard.py tests/test_conversational_voice_loop.py -v`
Expected: PASS

---

### Task 3: Conversation History Repair & Turn Cancellation Invariants

**Files:**
- Modify: `src/verbix/agent/utils/history.py:33-72`
- Test: `tests/test_history_invariants.py`

**Interfaces:**
- Consumes: `msgs: list[dict[str, Any]]`, `delivered_text: str = ""`
- Produces: `validate_and_repair_history(msgs: list[dict[str, Any]], delivered_text: str = "") -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing test for history repair with tool calls and delivered speech**

In `tests/test_history_invariants.py`:
```python
def test_repair_truncates_speech_even_with_tool_calls():
    msgs = [
        {"role": "user", "content": "check status"},
        {
            "role": "assistant",
            "content": "I will check that right now",
            "tool_calls": [{"id": "c1", "function": {"name": "status", "arguments": "{}"}}],
        },
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="I will check")
    asst = [m for m in repaired if m.get("role") == "assistant"][0]
    assert asst["content"] == "I will check"
    tool_resp = [m for m in repaired if m.get("role") == "tool"][0]
    assert tool_resp["tool_call_id"] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_history_invariants.py -k test_repair_truncates_speech_even_with_tool_calls -v`
Expected: FAIL (assistant content remains `"I will check that right now"`)

- [ ] **Step 3: Update `validate_and_repair_history` in `src/verbix/agent/utils/history.py`**

- Remove the premature `return _merge_assistant_text(msgs)` after synthesizing cancelled tool results.
- Update assistant message content to `delivered_text` for assistant messages in the turn, ensuring only actually spoken text is kept in history.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_history_invariants.py -v`
Expected: PASS

---

### Task 4: Cross-Platform Execution Policy & Tool Hardening

**Files:**
- Modify: `src/verbix/tools/bash.py:32-73`
- Modify: `src/verbix/tools/sandbox.py:26-119`
- Modify: `src/verbix/tools/edit.py:32-42`
- Modify: `src/verbix/tools/shared.py:14-43`
- Test: `tests/test_tools.py`, `tests/test_tool_output.py`

**Interfaces:**
- Consumes: `command: str`, `context: ToolContext`
- Produces: `BashTool.execute(...)`, `FileEditTool.execute(...)`, `_cap_text(...)`

- [ ] **Step 1: Write failing tests for unsandboxed bash execution and empty old_str edit**

In `tests/test_tools.py`:
- Test that `BashTool(sandboxed=False)` executes commands in `context.workspace` cleanly without requiring `bwrap`.
- Test that `FileEditTool` returns an error when `old_str` is empty (`""`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py -k "test_edit_empty_old_str or test_unsandboxed_bash_execution" -v`
Expected: FAIL

- [ ] **Step 3: Implement unsandboxed execution, edit guard, and optimized text capping**

- In `src/verbix/tools/sandbox.py`:
  - Add `execute_unsandboxed(command: str, workspace: Path, timeout_seconds: float = 60.0) -> SandboxResult`: executes `bash -c command` with `cwd=workspace`, process group tracking, and clean timeout escalation.
  - In `execute_sandboxed`: keep bubblewrap implementation; fail closed with clear error if `bwrap` is missing when sandboxed execution is requested.
- In `src/verbix/tools/bash.py`:
  - Add `sandboxed: bool = False` parameter to `BashTool.__init__`.
  - When `sandboxed=False`, call `execute_unsandboxed`. When `sandboxed=True`, call `execute_sandboxed`.
- In `src/verbix/tools/edit.py`:
  - Check `if not old_str: return ToolResult("error: old_str cannot be empty", is_error=True)`.
- In `src/verbix/tools/shared.py`:
  - Optimize `_cap_text` to slice UTF-8 bytes directly instead of encoding character-by-character in a loop.

- [ ] **Step 4: Run tool tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_tool_output.py -v`
Expected: PASS

---

### Task 5: Dead Code & Parallel Stack Pruning

**Files:**
- Delete: `src/verbix/core/agent.py`
- Delete: `src/verbix/core/history.py`
- Delete: `tests/test_agent.py`
- Delete: `tests/test_history.py`
- Delete: `src/verbix/provider/client.py`
- Delete: `src/verbix/provider/errors.py`
- Delete: `tests/test_provider.py`
- Modify: `src/verbix/core/__init__.py`
- Modify: `src/verbix/provider/__init__.py`
- Modify: `src/verbix/runtime/worker.py`
- Modify: `tests/test_runtime_e2e.py`
- Modify: `pyproject.toml` (remove `httpx2` if unused)

**Interfaces:**
- Consumes: repository codebase
- Produces: clean, dead-code-free codebase matching `AGENTS.md` ("the pollution around it goes")

- [ ] **Step 1: Remove legacy files**

Delete `src/verbix/core/agent.py`, `src/verbix/core/history.py`, `tests/test_agent.py`, `tests/test_history.py`, `src/verbix/provider/client.py`, `src/verbix/provider/errors.py`, `tests/test_provider.py`.

- [ ] **Step 2: Update imports in callers**

- In `src/verbix/core/__init__.py`: remove imports of `AgentSession`, `CycleDetector`, `ConversationHistory`.
- In `src/verbix/provider/__init__.py`: remove imports of `ProviderClient` and `errors`.
- In `src/verbix/runtime/worker.py`: remove `from verbix.core.agent import AgentSession`.
- In `tests/test_runtime_e2e.py`: replace `ProviderClient` reference with mock provider.
- In `pyproject.toml`: remove `httpx2>=2.13.0`.

- [ ] **Step 3: Run full test suite and verify no broken imports**

Run: `uv run pytest`
Expected: PASS (all tests pass)

---

### Task 6: Verification & Quality Gate

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Run linter check**

Run: `uv run ruff check src/ tests/`
Expected: ALL PASS

- [ ] **Step 3: Verify git status**

Run: `git status`
Expected: Clean, uncommitted working tree with all intended edits.
