# Explicit Finish Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile bash string interception (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) with an explicit, schema-validated `finish(summary="...")` tool for autonomous background workers.

**Architecture:** Add `FinishTool` to the worker's `ToolRegistry`. Update `DefaultAgent` and `WorkerTask` to recognize `finish(summary=...)` as the authoritative task termination signal that provides the canonical result to Verbix's announcer. Clean `LocalEnvironment` so it executes bash cleanly without scanning stdout for benchmark tokens.

**Tech Stack:** Python 3.12+, `asyncio`, `LiteLLM`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-06-natural-worker-completions.md`

## Global Constraints
- Zero reliance on `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` in `worker.yaml`, `LocalEnvironment`, or runtime.
- Background worker has access to both `bash` and `finish` tools.
- `finish` requires a non-empty `summary` string in its JSON schema.
- Native Python `asyncio` for all runtime coordination; zero blocking calls on the main loop.
- Full backwards-compatibility: existing voice companion, tools, and announcer remain unaffected.
- All tests must pass cleanly (`uv run pytest -v`).

## Review Focus
1. **Schema Validation & Information Density:** `finish` tool schema must require an information-dense `summary` parameter specifying exact metrics, exit codes, and verification evidence.
2. **Clean Execution:** `LocalEnvironment` must execute commands and return raw output without scanning for tokens or raising `Submitted`.
3. **Execution Before Completion:** Worker prompt must instruct discovering, executing, and verifying via `bash` before invoking `finish`.
4. **Fallback Handling:** If `finish` is called with empty text or no text is provided, `WorkerTask` must fall back to the stdout of the last bash command.
5. **Prompt Invariants:** Invariants in `tests/test_prompt_invariants.py` must enforce that no benchmark submission tokens exist in `worker.yaml`.

---

### Task 1: Define `FinishTool` and Clean Up `LocalEnvironment`

**Files:**
- Create/Modify: `src/verbix/runtime/tools.py`
- Modify: `src/verbix/minisweagent/environments/local.py`
- Test: `tests/test_runtime_tools.py`
- Test: `tests/test_miniswe_environment.py`

- [ ] **Step 1: Write failing tests**
  - In `tests/test_runtime_tools.py`: Test `FinishTool` schema contains `summary` required parameter and executes returning observation.
  - In `tests/test_miniswe_environment.py`: Test `LocalEnvironment.execute` does not intercept `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` or raise `Submitted`, but returns output normally.

- [ ] **Step 2: Run tests to verify RED**
  - Command: `uv run pytest tests/test_runtime_tools.py tests/test_miniswe_environment.py -v`

- [ ] **Step 3: Implement `FinishTool` and clean `LocalEnvironment`**
  - In `src/verbix/runtime/tools.py`: Implement `FinishTool` with name `"finish"`, description, and schema requiring `summary`.
  - In `src/verbix/minisweagent/environments/local.py`: Remove `_check_finished`, `_check_command_finished`, and benchmark marker parsing.

- [ ] **Step 4: Run tests to verify GREEN**
  - Command: `uv run pytest tests/test_runtime_tools.py tests/test_miniswe_environment.py -v`

- [ ] **Step 5: Commit changes**
  - `git add src/verbix/runtime/tools.py src/verbix/minisweagent/environments/local.py tests/test_runtime_tools.py tests/test_miniswe_environment.py`
  - `git commit -m "feat(tools): add FinishTool and remove bash token interception from LocalEnvironment"`

---

### Task 2: Support `finish` in `DefaultAgent`, `WorkerTask`, and `Runtime`

**Files:**
- Modify: `src/verbix/minisweagent/agents/default.py`
- Modify: `src/verbix/runtime/worker.py`
- Modify: `src/verbix/runtime/runtime.py`
- Test: `tests/test_miniswe_agent_turn.py`

- [ ] **Step 1: Write failing test in `tests/test_miniswe_agent_turn.py`**
  - Test worker agent executing `bash` then calling `finish(summary="All tests passed")`.
  - Verify that `step_stream()` yields `ToolCallFinished` and `TurnComplete(final_text="All tests passed")` and stops cleanly.

- [ ] **Step 2: Run test to verify RED**
  - Command: `uv run pytest tests/test_miniswe_agent_turn.py -k "finish" -v`

- [ ] **Step 3: Implement `finish` handling in `DefaultAgent`, `WorkerTask`, and `Runtime`**
  - In `src/verbix/minisweagent/agents/default.py`:
    - When executing tool named `"finish"`, extract `summary = args.get("summary", "")`, record tool message, yield `ToolCallFinished`, and yield `TurnComplete(final_text=summary)`. Stop turn loop immediately.
    - In synchronous `step()`: Remove `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` check.
  - In `src/verbix/runtime/worker.py`:
    - Set `self._result` from `TurnComplete.final_text`.
    - If `self._result` is empty, extract the last bash command output from session history before defaulting to generic text.
  - In `src/verbix/runtime/runtime.py`:
    - In `Runtime.spawn()`: Register `FinishTool` into the worker's `ToolRegistry` so the worker has `bash` and `finish`.

- [ ] **Step 4: Run test to verify GREEN**
  - Command: `uv run pytest tests/test_miniswe_agent_turn.py -k "finish" -v`

- [ ] **Step 5: Commit changes**
  - `git add src/verbix/minisweagent/agents/default.py src/verbix/runtime/worker.py src/verbix/runtime/runtime.py tests/test_miniswe_agent_turn.py`
  - `git commit -m "feat(runtime): wire FinishTool into worker loop and result capture"`

---

### Task 3: Update Worker Prompt Invariants and Clean Up Legacy References

**Files:**
- Modify: `src/verbix/minisweagent/config/worker.yaml`
- Modify: `src/verbix/runtime/runtime.py`
- Test: `tests/test_prompt_invariants.py`
- Test: `tests/test_runtime_e2e.py`

- [ ] **Step 1: Write failing tests in `tests/test_prompt_invariants.py`**
  - Assert that `worker.yaml` has `tools` containing both `bash` and `finish`.
  - Assert that `worker.yaml` does NOT contain `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  - Assert that `worker.yaml` instructs calling `finish` when verified.

- [ ] **Step 2: Run test to verify RED**
  - Command: `uv run pytest tests/test_prompt_invariants.py -v`

- [ ] **Step 3: Update `worker.yaml` and `DEFAULT_WORKER_PROMPT`**
  - In `src/verbix/minisweagent/config/worker.yaml`:
    - Add `finish` to `tools: [bash, finish]`.
    - In `instance_template`:
      1. Discover: Inspect files/environment using `bash`.
      2. Execute: Carry out actions directly using `bash`.
      3. Verify: Check command status and results using `bash`.
      4. Complete: Call `finish(summary="...")` with your final factual findings. Never call `finish` before executing the actual work.
    - Remove all benchmark token text and update `format_error_template`.
  - In `src/verbix/runtime/runtime.py`:
    - Update `DEFAULT_WORKER_PROMPT` to reference `finish` tool for task completion.

- [ ] **Step 4: Run full test suite and linters to verify GREEN**
  - Command: `uv run pytest -v`
  - Command: `uv run ruff check src/ tests/`
  - Command: `uv run pyright src/ tests/`

- [ ] **Step 5: Commit changes**
  - `git add src/verbix/minisweagent/config/worker.yaml src/verbix/runtime/runtime.py tests/test_prompt_invariants.py tests/test_runtime_e2e.py`
  - `git commit -m "feat(prompt): update worker prompt to use finish tool and enforce invariants"`
