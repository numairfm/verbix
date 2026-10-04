# Simple Foreground-Worker Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the minimal, robust Verbix foreground-worker runtime where Verbix (a foreground agent with strictly 3 tools: `spawn`, `status`, `cancel`) orchestrates independent, concurrent worker `AgentSession`s whose results are batched through a 300ms completion buffer.

**Architecture:** A lightweight `Runtime` orchestrator hosts Verbix (`AgentSession`) with a Minimum Effective Prompt and maintains a dictionary of running `WorkerTask`s. Background workers run independent `AgentSession`s with full tools (`bash`, `file_read`, `file_write`, `file_edit`) in `asyncio.Task`s. A 300ms debounce `CompletionBuffer` aggregates finished tasks and feeds them to Verbix to formulate natural spoken updates. `status` provides in-memory task inspection directly from `task.session.history.messages` (concise by default, full on demand).

**Tech Stack:** Python 3.14, `asyncio`, `pydantic`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-10-04-simple-foreground-worker-architecture.md`

## Global Constraints
- Identity is Verbix. Voice-first, plain spoken continuous prose; no markdown, no code syntax, no tool names or internal IDs spoken to the user.
- Native `asyncio` concurrency; zero custom locks, worktree managers, or disk log sprawl (Ponytail Ultra).
- Foreground agent strictly restricted to 3 tools: `spawn`, `status`, `cancel`.
- Dual-emission as default in `AgentSession`: text deltas and tool calls stream concurrently in the same turn.
- Immediate cancellation: `cancel()` cleanly halts `asyncio.Task` and cleans up bash subprocesses via `SIGKILL`.

## Review Focus
1. Rapid-fire worker completions within 300ms must be combined into a single wake notice rather than multiple separate turns.
2. `cancel` with no args must cancel the active task if exactly 1 is running, or list active tasks if 2+ are running.
3. `status` must return concise recent actions from worker history by default, and full history when `full=True`.
4. Worker failure must be captured gracefully in the buffer and communicated to Verbix without unhandled exceptions in background loops.
5. Inactive or unknown task queries in `status` and `cancel` must return clean error observations rather than crashing.

---

### Task 1: Foreground Tools (`spawn`, `status`, `cancel`)

**Files:**
- Create: `src/verbix/runtime/tools.py`
- Test: `tests/test_runtime_tools.py`

**Interfaces:**
- Produces:
  - `SpawnTool(runtime)` with execute `args: {"task": str}`
  - `StatusTool(runtime)` with execute `args: {"task_id": str | None, "full": bool}`
  - `CancelTool(runtime)` with execute `args: {"task_id": str | None}`

- [ ] **Step 1: Write failing tests for foreground tools**

```python
import pytest
from verbix.runtime.tools import SpawnTool, StatusTool, CancelTool
from verbix.tools.base import ToolContext

@pytest.mark.asyncio
async def test_foreground_tools_contract(dummy_runtime):
    spawn_tool = SpawnTool(dummy_runtime)
    status_tool = StatusTool(dummy_runtime)
    cancel_tool = CancelTool(dummy_runtime)

    # Test spawn
    res = await spawn_tool.execute({"task": "build binary"}, ToolContext(dummy_runtime.workspace))
    assert "task-1" in res.observation

    # Test status list and detail
    list_res = await status_tool.execute({}, ToolContext(dummy_runtime.workspace))
    assert "task-1" in list_res.observation

    # Test cancel single
    cancel_res = await cancel_tool.execute({}, ToolContext(dummy_runtime.workspace))
    assert "Cancelled" in cancel_res.observation or "halted" in cancel_res.observation.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_tools.py`
Expected: FAIL with ModuleNotFoundError or import error.

- [ ] **Step 3: Implement `SpawnTool`, `StatusTool`, and `CancelTool` in `src/verbix/runtime/tools.py`**

Implement tool definitions conforming to `verbix.tools.base.Tool` protocol, calling underlying methods on `runtime`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_tools.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/tools.py tests/test_runtime_tools.py
git commit -m "feat(runtime): add minimal foreground spawn, status, cancel tools"
```

---

### Task 2: WorkerTask and CompletionBuffer

**Files:**
- Create: `src/verbix/runtime/worker.py`
- Create: `src/verbix/runtime/buffer.py`
- Test: `tests/test_runtime_worker.py`

**Interfaces:**
- Produces:
  - `WorkerTask(id, description, session)`: methods `start()`, `cancel()`, `get_status(full: bool)`, properties `status`, `result`, `error`.
  - `CompletionBuffer(window_seconds=0.3, on_flush=callback)`: methods `push(task_id, result)`, `flush_now()`.

- [ ] **Step 1: Write failing tests for WorkerTask and CompletionBuffer**

```python
import asyncio
import pytest
from verbix.runtime.buffer import CompletionBuffer

@pytest.mark.asyncio
async def test_completion_buffer_batches_within_window():
    flushed = []
    buf = CompletionBuffer(window_seconds=0.1, on_flush=lambda batch: flushed.append(batch))
    buf.push("task-1", "Done 1")
    await asyncio.sleep(0.02)
    buf.push("task-2", "Done 2")
    await asyncio.sleep(0.15)
    assert len(flushed) == 1
    assert flushed[0] == [("task-1", "Done 1"), ("task-2", "Done 2")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_worker.py`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `WorkerTask` and `CompletionBuffer`**

- `WorkerTask`: Launches background `asyncio.create_task` executing worker `AgentSession.step()`. Captures final text or exception. Implements `get_status(full)` by extracting recent or all tool actions from `session.history.messages`.
- `CompletionBuffer`: Debounces incoming task completions over `0.3s` using an `asyncio.TimerHandle` or sleep task before invoking `on_flush`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_worker.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/worker.py src/verbix/runtime/buffer.py tests/test_runtime_worker.py
git commit -m "feat(runtime): add WorkerTask lifecycle and debounced CompletionBuffer"
```

---

### Task 3: Core Runtime Orchestration & Dual-Emission Support

**Files:**
- Create: `src/verbix/runtime/runtime.py`
- Modify: `src/verbix/runtime/__init__.py`
- Test: `tests/test_runtime_e2e.py`

**Interfaces:**
- Produces:
  - `Runtime(workspace, provider, ...)`:
    - `step(user_text)`: runs Verbix foreground turn.
    - `spawn(task_description)`: creates and starts `WorkerTask`.
    - `status(task_id, full)`: returns task status string.
    - `cancel(task_id)`: cancels `WorkerTask`.
    - `_on_buffer_flush(batch)`: injects completion notice into Verbix history and triggers a foreground turn to formulate spoken response.

- [ ] **Step 1: Write failing end-to-end runtime test**

```python
import pytest
from verbix.runtime.runtime import Runtime

@pytest.mark.asyncio
async def test_runtime_spawn_and_completion_loop(mock_provider, tmp_path):
    runtime = Runtime(workspace=tmp_path, provider=mock_provider)
    task_id = await runtime.spawn("calculate prime numbers")
    assert task_id == "task-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_e2e.py`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `Runtime` class in `src/verbix/runtime/runtime.py`**

- Wire `SpawnTool`, `StatusTool`, `CancelTool` to Verbix `AgentSession` with the Minimum Effective Prompt.
- Setup worker factory with `[FileReadTool(), FileWriteTool(), FileEditTool(), BashTool()]`.
- Connect `CompletionBuffer` to foreground completion notice injection.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_e2e.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/runtime.py src/verbix/runtime/__init__.py tests/test_runtime_e2e.py
git commit -m "feat(runtime): assemble Runtime with dual-emission and completion injection"
```

---

### Task 4: CLI, IPC, and Voice Integration

**Files:**
- Modify: `src/verbix/cli.py`
- Modify: `src/verbix/ipc/server.py`
- Modify: `src/verbix/voice/pipeline.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Adapts `build_runtime(args)` in `cli.py` to instantiate new `Runtime`.
- Connects `VerbixServer` and `VoicePipeline` to the new `Runtime` interface.

- [ ] **Step 1: Write smoke test for CLI runtime initialization**

```python
from verbix.cli import build_runtime, build_parser

def test_build_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    parser = build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path)])
    runtime = build_runtime(args)
    assert runtime is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py`
Expected: FAIL with `NotImplementedError("Runtime architecture is being rebuilt.")`

- [ ] **Step 3: Update `cli.py`, `ipc/server.py`, and `voice/pipeline.py`**

Replace stubs with new `Runtime` initialization and dispatch calls.

- [ ] **Step 4: Run test and full suite to verify all pass**

Run: `uv run pytest -v && uv run ruff check src/ tests/`
Expected: 100% pass across all tests, zero lint warnings.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/cli.py src/verbix/ipc/server.py src/verbix/voice/pipeline.py tests/test_cli_smoke.py
git commit -m "feat(cli): wire new runtime into CLI, IPC, and Voice pipeline"
```
