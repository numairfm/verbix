# Architecture Scaling & Runtime Hardening (Phase 0 & 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hardens the Verbix runtime loop, agent execution engine, task completion pipeline, and process isolation against races, orphaned state, context explosion, and resource leaks.

**Architecture:** Implements Phase 0 (instrumentation & regression harness) and Phase 1 (runtime hardening without new dependencies) from the approved specification. Replaces ad-hoc turn locks and buffer debounce with an explicit Floor Arbiter, a single-source-of-truth Task Ledger, a persistent Announcer Loop, an async cancellation-safe `step_stream`, and an async `LocalEnvironment` with process group lifecycle control.

**Tech Stack:** Python 3.11+, asyncio, LiteLLM, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-06-architecture-scaling-and-runtime-hardening.md`

## Global Constraints
- **Zero new dependencies:** Phase 0 & 1 must run entirely on the existing dependencies without pulling in new packages.
- **Invariants:** Strictly uphold H1-H5 (History), F1-F4 (Floor), P1-P3 (Isolation), and L1-L3 (Latency) defined in the spec.
- **Test Integrity:** All 87 existing tests must pass at every step; new tests must cover the failure states proven by the race probes.
- **Backward Compatibility:** CLI commands and IPC protocol contracts must remain functional.

## Review Focus
1. **Unpaired tool calls on cancellation:** A user interruption during tool call generation or execution must never leave dangling `tool_calls` without a matching `tool` result in history.
2. **Buffer loss on concurrent completions:** Multiple background workers completing within the batch window or mid-announcement must never drop or cancel pending completions.
3. **Orphaned child processes:** Cancelling a worker task or shutting down the runtime must reliably terminate the entire process group, not leave runaway background commands.
4. **Context window unbounded growth:** Foreground history must stay strictly within its token budget (16k default) without severing tool call-observation pairs.
5. **Deadlock on floor acquisition:** Floor transitions between `LISTENING`, `FOREGROUND`, and `ANNOUNCING` must never deadlock or leave the floor locked if a turn raises an unhandled exception.

---

### Task 1: Phase 0 — Instrumentation & Regression Test Harness

**Files:**
- Create: `src/verbix/core/telemetry.py`
- Create: `tests/test_runtime_races.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: `record_latency(mark: str, **extra)` in `src/verbix/core/telemetry.py`
- Produces: `assert_history_valid(messages: list[dict[str, Any]])` helper in `tests/conftest.py`

- [ ] **Step 1: Write the failing tests replicating the 3 empirical race conditions**

```python
# In tests/test_runtime_races.py
import asyncio
import os
import subprocess
import pytest
from verbix.runtime.buffer import CompletionBuffer
from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.core.events import ToolCallDelta

@pytest.mark.asyncio
async def test_regression_completion_buffer_drops_concurrent_finish():
    flushed = []
    completed = []

    async def on_flush(batch):
        flushed.append(batch)
        await asyncio.sleep(0.1)
        completed.append(batch)

    buf = CompletionBuffer(window_seconds=0.02, on_flush=on_flush)
    buf.push("task-1", "res-1")
    await asyncio.sleep(0.04)  # Flush task-1 begins
    buf.push("task-2", "res-2")  # Second push arrives while flush is in-flight
    await asyncio.sleep(0.2)

    # In current buggy buffer, task-1's flush was cancelled mid-flight, so completed only has task-2
    all_completed_ids = [t[0] for batch in completed for t in batch]
    assert "task-1" in all_completed_ids
    assert "task-2" in all_completed_ids

@pytest.mark.asyncio
async def test_regression_cancel_mid_tool_call_orphans_state():
    class SlowToolModel:
        async def query_stream(self, messages, tools=None, **kw):
            yield ToolCallDelta(index=0, call_id="c1", name="slow", arguments_delta="{}")

    reg = ToolRegistry()
    async def slow(args, **kw):
        await asyncio.sleep(5)
        return "done"
    reg.register("slow", {"type": "function", "function": {"name": "slow", "parameters": {}}}, slow)

    agent = DefaultAgent(model=SlowToolModel(), tools=reg, config=AgentConfig(system_template="sys"))
    async def run():
        async for _ in agent.step_stream("hi"):
            pass

    t = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # In current codebase, agent.messages ends with assistant message containing tool_calls without tool response
    last = agent.messages[-1]
    assert last.get("role") != "assistant" or not last.get("tool_calls"), "Dangling tool_calls message left in history!"

@pytest.mark.asyncio
async def test_regression_worker_cancel_kills_process_group():
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    marker = f"verbix_test_proc_{os.getpid()}_{int(asyncio.get_event_loop().time() * 1000)}"

    async def run():
        await asyncio.to_thread(env.execute, {"command": f"exec -a {marker} sleep 10"})

    t = asyncio.create_task(run())
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    await asyncio.sleep(0.1)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.strip()
    assert not alive, f"Process {alive} survived task cancellation!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_races.py -v`
Expected: FAIL (all 3 tests fail, confirming bugs).

- [ ] **Step 3: Implement `src/verbix/core/telemetry.py` and history validation fixture in `tests/conftest.py`**

Create `src/verbix/core/telemetry.py` to write JSONL latency marks when `VERBIX_TRACE=1`.
Add `assert_history_valid(messages)` to `tests/conftest.py` verifying invariants H1 (matching `tool_call_id`s, no dangling tool calls).

- [ ] **Step 4: Commit test harness**

```bash
git add src/verbix/core/telemetry.py tests/test_runtime_races.py tests/conftest.py
git commit -m "test: add Phase 0 regression harness for race conditions and telemetry"
```

---

### Task 2: History Validation, Repair, and Cancellation Safety (A1, A2, A3, H1, H2, H3)

**Files:**
- Create: `src/verbix/minisweagent/utils/history.py`
- Modify: `src/verbix/minisweagent/agents/default.py:229-446`
- Modify: `src/verbix/minisweagent/tools/registry.py:10-75`
- Modify: `src/verbix/runtime/runtime.py:483-510`
- Test: `tests/test_history_invariants.py`

**Interfaces:**
- Consumes: `DefaultAgent.messages`, `AgentEvent`
- Produces: `validate_and_repair_history(messages: list[dict[str, Any]], delivered_text: str = "") -> list[dict[str, Any]]`
- Produces: `TurnHandle` dataclass in `src/verbix/runtime/runtime.py`

- [ ] **Step 1: Write failing unit tests for history repair and cancellation safety**

```python
# In tests/test_history_invariants.py
from verbix.minisweagent.utils.history import validate_and_repair_history

def test_repair_unanswered_tool_calls():
    msgs = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]},
    ]
    repaired = validate_and_repair_history(msgs)
    assert len(repaired) == 3
    assert repaired[-1]["role"] == "tool"
    assert repaired[-1]["tool_call_id"] == "call_1"
    assert "cancelled" in repaired[-1]["content"]

def test_repair_truncates_to_delivered_speech():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "This is a very long response that was interrupted halfway."},
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="This is a very")
    assert repaired[-1]["content"] == "This is a very"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_history_invariants.py -v`
Expected: FAIL (`validate_and_repair_history` not implemented).

- [ ] **Step 3: Implement `validate_and_repair_history` in `src/verbix/minisweagent/utils/history.py`**

Port and adapt `ConversationHistory.close_aborted_turn` from `core/history.py` to work with standard message dictionaries. Guarantee:
1. Every assistant `tool_calls` entry is paired with a subsequent `tool` message. Missing responses receive synthetic cancelled responses.
2. If interrupted during pure speech, assistant content is truncated to `delivered_text` (or popped if empty).
3. Consecutive assistant text messages are merged for API compatibility.

- [ ] **Step 4: Update `ToolRegistry` and `DefaultAgent.step_stream` for shielded execution and cancellation cleanup**

In `ToolRegistry`, support `side_effecting: bool = False`. Mark `spawn` and `cancel` as `side_effecting=True`.
In `DefaultAgent.step_stream`:
1. Side-effecting tool executions are wrapped in `asyncio.shield` so started spawns/cancels complete and are recorded.
2. Wrap `step_stream` in `try...except asyncio.CancelledError`: invoke `validate_and_repair_history(self.messages, delivered_text=...)` before re-raising.
3. Replace the arbitrary 100-char pending buffer check with proper first-chunk emission, and ensure suppressed pre-tool chatter is NOT stored in `assistant.content`.

- [ ] **Step 5: Introduce `TurnHandle` in `src/verbix/runtime/runtime.py`**

Replace `_active_turn_task` with a dedicated `TurnHandle` holding the turn's `task`, `kind`, and `delivered_text` provider. Ensure `cancel_turn()` idempotently cancels the handle and invokes history repair.

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_history_invariants.py tests/test_runtime_races.py -k "test_regression_cancel_mid_tool_call_orphans_state" -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/verbix/minisweagent/utils/history.py src/verbix/minisweagent/agents/default.py src/verbix/minisweagent/tools/registry.py src/verbix/runtime/runtime.py tests/test_history_invariants.py
git commit -m "feat(agent): implement history repair on cancellation and shielded tool execution"
```

---

### Task 3: Observation Caps, Scoped Limits, and Grounding Separation (A5, A6, H4, L3)

**Files:**
- Modify: `src/verbix/minisweagent/agents/default.py`
- Modify: `src/verbix/minisweagent/models/litellm_model.py`
- Modify: `src/verbix/runtime/runtime.py`
- Test: `tests/test_agent_limits_and_caps.py`

**Interfaces:**
- Produces: `cap_observation(text: str, max_chars: int = 10000) -> str`
- Modifies: `DefaultAgent.step_stream` limit checking and usage calculation

- [ ] **Step 1: Write failing tests for observation cap and per-turn limits**

```python
# In tests/test_agent_limits_and_caps.py
import pytest
from verbix.minisweagent.agents.default import cap_observation, DefaultAgent, AgentConfig

def test_cap_observation_bounds_large_outputs():
    huge = "a" * 50000
    capped = cap_observation(huge, max_chars=10000)
    assert len(capped) <= 10200
    assert "elided" in capped or "..." in capped
    assert capped.startswith("aaaa")
    assert capped.endswith("aaaa")

def test_foreground_limits_reset_per_turn():
    # Test that DefaultAgent can be configured with turn-scoped limits
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_limits_and_caps.py -v`
Expected: FAIL (`cap_observation` not defined).

- [ ] **Step 3: Implement `cap_observation` and integrate into streaming tool execution**

In `src/verbix/minisweagent/agents/default.py`:
Implement `cap_observation(text: str, max_chars: int = 10000) -> str`.
In `_execute_tool`, apply `cap_observation(obs, max_chars=10000)` for worker tools and `cap_observation(obs, max_chars=1500)` for foreground tools.

- [ ] **Step 4: Update LiteLLM streaming cost tracking and scoped limits**

In `LitellmModel.query_stream`:
Enable `stream_options={"include_usage": True}` on `litellm.acompletion` call.
When usage arrives in stream chunk, compute completion cost using `litellm.cost_calculator.completion_cost` and update agent cost.
In `DefaultAgent`:
Support `turn_scoped_limits: bool = False`. When True (foreground agent), reset `turn_calls` and `turn_cost` at the start of each `step_stream` call.

- [ ] **Step 5: Separate static system prompt from volatile grounding (L3)**

In `Runtime`:
Keep the system prompt purely static (identity, role, permanent tools).
Move volatile runtime grounding (current local time, dynamic status) out of `system_prompt` updates. Prepend dynamic time/grounding to the user message at request time only, preserving provider prompt-cache prefixes.

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_agent_limits_and_caps.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/verbix/minisweagent/agents/default.py src/verbix/minisweagent/models/litellm_model.py src/verbix/runtime/runtime.py tests/test_agent_limits_and_caps.py
git commit -m "feat(runtime): add tool observation caps, scoped limit tracking, and cache-friendly grounding"
```

---

### Task 4: Tier 1 Context Window Budget & Rolling Compaction (A7 Tier 1, H5)

**Files:**
- Create: `src/verbix/minisweagent/utils/compaction.py`
- Modify: `src/verbix/minisweagent/agents/default.py`
- Test: `tests/test_compaction.py`

**Interfaces:**
- Produces: `compact_messages(messages: list[dict[str, Any]], max_tokens: int, is_worker: bool = False) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests for Tier 1 deterministic compaction**

```python
# In tests/test_compaction.py
from verbix.minisweagent.utils.compaction import compact_messages

def test_compact_messages_preserves_recent_turns():
    msgs = [
        {"role": "system", "content": "You are Verbix"},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Resp 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Resp 2"},
    ]
    compacted = compact_messages(msgs, max_tokens=1000)
    assert len(compacted) == 5

def test_compact_messages_elides_old_tool_observations_without_breaking_pairs():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
        {"role": "assistant", "content": "old task done"},
        {"role": "user", "content": "new task"},
        {"role": "assistant", "content": "new resp"},
    ]
    compacted = compact_messages(msgs, max_tokens=500)
    # The tool result for c1 must still exist, but content elided
    tool_msg = next(m for m in compacted if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert len(tool_msg["content"]) < 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_compaction.py -v`
Expected: FAIL (`compact_messages` not defined).

- [ ] **Step 3: Implement `compact_messages` in `src/verbix/minisweagent/utils/compaction.py`**

Implement Tier 1 deterministic compaction:
1. Calculate token count via `litellm.token_counter` (fallback: `chars // 4`).
2. If token count <= `max_tokens`, return messages unchanged.
3. Group messages into turn groups (foreground) or steps (workers). Pin system message and initial task.
4. Stage 1: Elide tool observations older than last 3 turns to `"[status output elided]"`. Truncate tool arguments > 500 chars.
5. Stage 2: In workers, truncate observations older than last 8 steps to first 200 chars.
6. Stage 3: Drop oldest turn groups/steps entirely (never drop system or split tool-call/result pairs).

- [ ] **Step 4: Integrate compaction into `DefaultAgent.step_stream`**

Before dispatching `self.model.query_stream(self.messages)`, apply `compact_messages(self.messages, max_tokens=self.config.max_context_tokens or 16000)`.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_compaction.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/minisweagent/utils/compaction.py src/verbix/minisweagent/agents/default.py tests/test_compaction.py
git commit -m "feat(agent): implement Tier 1 deterministic context compaction and pairing-safe pruning"
```

---

### Task 5: Task Ledger and Floor Arbiter (R1, R2, F1, F2, F3)

**Files:**
- Create: `src/verbix/runtime/ledger.py`
- Create: `src/verbix/runtime/floor.py`
- Modify: `src/verbix/runtime/runtime.py`
- Test: `tests/test_ledger_and_floor.py`

**Interfaces:**
- Produces: `TaskLedger`, `TaskEntry` in `src/verbix/runtime/ledger.py`
- Produces: `FloorArbiter`, `FloorState` in `src/verbix/runtime/floor.py`

- [ ] **Step 1: Write failing tests for TaskLedger and FloorArbiter**

```python
# In tests/test_ledger_and_floor.py
import pytest
from verbix.runtime.ledger import TaskLedger
from verbix.runtime.floor import FloorArbiter, FloorState

def test_ledger_state_transitions():
    ledger = TaskLedger()
    ledger.register_task("task-1", "calculate metrics")
    entry = ledger.get("task-1")
    assert entry.status == "running"
    ledger.mark_completed("task-1", "metric result = 42")
    assert entry.status == "completed"
    assert entry.delivery == "pending"

@pytest.mark.asyncio
async def test_floor_arbiter_transitions_and_interrupt():
    floor = FloorArbiter()
    assert floor.state == FloorState.IDLE
    async with floor.acquire(FloorState.FOREGROUND):
        assert floor.state == FloorState.FOREGROUND
    assert floor.state == FloorState.IDLE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ledger_and_floor.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `TaskLedger` in `src/verbix/runtime/ledger.py`**

Store task metadata (`id`, `objective`, `status`, `started_at`, `finished_at`, `result_summary`, `delivery`).
Support `get_compact_summary() -> str` formatting at most 6 recent tasks in <= 600 characters for request-time injection into the companion.

- [ ] **Step 4: Implement `FloorArbiter` in `src/verbix/runtime/floor.py`**

Implement states: `IDLE`, `LISTENING`, `FOREGROUND`, `ANNOUNCING`.
Enforce invariant F1 (single floor owner) and F3 (immediate synchronous interrupt flag).
Support async context manager `acquire(state)`.

- [ ] **Step 5: Wire FloorArbiter and TaskLedger into `Runtime`**

Replace `_turn_lock` and `user_active` in `Runtime` with `self.floor = FloorArbiter()` and `self.ledger = TaskLedger()`.
Ensure foreground `submit()` and `step()` acquire `FloorState.FOREGROUND`.

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_ledger_and_floor.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/verbix/runtime/ledger.py src/verbix/runtime/floor.py src/verbix/runtime/runtime.py tests/test_ledger_and_floor.py
git commit -m "feat(runtime): introduce single-truth TaskLedger and FloorArbiter"
```

---

### Task 6: Completion Queue & Dedicated Announcer Loop (R3, F4, E1-E4)

**Files:**
- Create: `src/verbix/runtime/announcer.py`
- Modify: `src/verbix/runtime/runtime.py`
- Remove: `src/verbix/runtime/buffer.py`
- Test: `tests/test_announcer_loop.py`

**Interfaces:**
- Produces: `AnnouncerLoop` in `src/verbix/runtime/announcer.py`

- [ ] **Step 1: Write failing tests for AnnouncerLoop handling concurrent and burst arrivals**

```python
# In tests/test_announcer_loop.py
import asyncio
import pytest
from verbix.runtime.announcer import AnnouncerLoop
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.ledger import TaskLedger

@pytest.mark.asyncio
async def test_announcer_loop_batches_without_dropping_items():
    floor = FloorArbiter()
    ledger = TaskLedger()
    announced = []

    async def speak_fn(batch):
        announced.append(batch)
        await asyncio.sleep(0.05)

    announcer = AnnouncerLoop(floor=floor, ledger=ledger, speak_fn=speak_fn, batch_window=0.03)
    announcer.start()

    ledger.register_task("t1", "obj 1")
    ledger.mark_completed("t1", "res 1")
    announcer.push("t1")

    await asyncio.sleep(0.04)  # Announcement in progress
    ledger.register_task("t2", "obj 2")
    ledger.mark_completed("t2", "res 2")
    announcer.push("t2")       # Second push during active announcement

    await asyncio.sleep(0.15)
    await announcer.stop()

    flattened = [item for batch in announced for item in batch]
    assert "t1" in flattened
    assert "t2" in flattened
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_announcer_loop.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `AnnouncerLoop` in `src/verbix/runtime/announcer.py`**

Replace `CompletionBuffer` with a non-cancelling persistent queue loop:
1. `push(task_id)` enqueues the task ID into an `asyncio.Queue`.
2. Worker loop: waits until items are available and `floor.state == IDLE`.
3. Collects pending arrivals up to `batch_window` (default 0.3s).
4. Acquires `floor.acquire(FloorState.ANNOUNCING)`.
5. Executes announcer speech synthesis.
6. Once speech is completed/delivered, marks tasks as `delivery="heard"` in `TaskLedger` and appends delivered text to foreground history.
7. Any subsequent task arriving during announcement stays safely queued for the next idle window.

- [ ] **Step 4: Wire `AnnouncerLoop` into `Runtime` and delete `CompletionBuffer`**

Update `Runtime` to start `AnnouncerLoop` on initialization and stop on teardown.
Remove `src/verbix/runtime/buffer.py`.

- [ ] **Step 5: Run tests including the regression test to verify pass**

Run: `uv run pytest tests/test_announcer_loop.py tests/test_runtime_races.py -k "test_regression_completion_buffer_drops_concurrent_finish" -v`
Expected: PASS (regression test passes!).

- [ ] **Step 6: Commit**

```bash
git add src/verbix/runtime/announcer.py src/verbix/runtime/runtime.py tests/test_announcer_loop.py
git rm src/verbix/runtime/buffer.py
git commit -m "feat(runtime): replace CompletionBuffer with non-cancelling AnnouncerLoop"
```

---

### Task 7: Async LocalEnvironment & Process Isolation (R5, P1, P2)

**Files:**
- Modify: `src/verbix/minisweagent/environments/local.py`
- Modify: `src/verbix/runtime/worker.py`
- Test: `tests/test_local_environment.py`

**Interfaces:**
- Modifies: `LocalEnvironment.execute` to handle process groups asynchronously with SIGTERM/SIGKILL escalation and output caps.

- [ ] **Step 1: Write failing tests for process group cleanup and output stream capping**

```python
# In tests/test_local_environment.py
import asyncio
import os
import subprocess
import pytest
from verbix.minisweagent.environments.local import LocalEnvironment

@pytest.mark.asyncio
async def test_local_env_kills_process_group_on_cancel():
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    marker = f"verbix_kill_test_{os.getpid()}"

    t = asyncio.create_task(env.aexecute({"command": f"exec -a {marker} sleep 10"}))
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    await asyncio.sleep(0.1)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.strip()
    assert not alive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_local_environment.py -v`
Expected: FAIL (`aexecute` not implemented or process survives).

- [ ] **Step 3: Implement async `aexecute` with process group tracking in `LocalEnvironment`**

In `src/verbix/minisweagent/environments/local.py`:
1. Implement `async def aexecute(self, action: dict, cwd: str = "", timeout: int | None = None) -> dict[str, Any]`.
2. Use `asyncio.create_subprocess_shell` with `start_new_session=True` on POSIX.
3. Track active process group IDs in a set.
4. On `asyncio.CancelledError` or timeout: send `os.killpg(pgid, signal.SIGTERM)`, wait up to 2 seconds, then escalate to `os.killpg(pgid, signal.SIGKILL)`.
5. Implement streaming output accumulation capped at 1MB (500KB head + 500KB tail) to prevent pipe deadlock and OOM.
6. Strip Verbix's provider API keys (`VERBIX_API_KEY`, active provider keys) from the worker subshell environment by default.

- [ ] **Step 4: Update `DefaultAgent._execute_tool` and `WorkerTask` to use `aexecute`**

Make tool execution of `bash` await `env.aexecute(...)` instead of `asyncio.to_thread(env.execute, ...)`.

- [ ] **Step 5: Run tests including the process leak regression test**

Run: `uv run pytest tests/test_local_environment.py tests/test_runtime_races.py -k "test_regression_worker_cancel_kills_process_group" -v`
Expected: PASS (all process leak regressions resolved!).

- [ ] **Step 6: Commit**

```bash
git add src/verbix/minisweagent/environments/local.py src/verbix/minisweagent/agents/default.py src/verbix/runtime/worker.py tests/test_local_environment.py
git commit -m "feat(env): implement async process group management and output stream capping in LocalEnvironment"
```

---

### Task 8: IPC, Tools, and CLI Cleanup (R6, Q4)

**Files:**
- Modify: `src/verbix/runtime/runtime.py`
- Modify: `src/verbix/runtime/tools.py`
- Modify: `src/verbix/ipc/server.py`
- Modify: `src/verbix/cli.py`
- Test: `tests/test_ipc_and_cli.py`

**Interfaces:**
- Produces: `Runtime.get_task_status_summary(task_id)` and `Runtime.cancel_task(task_id)` for IPC server
- Produces: `WakeOutputFanOut` delivering to voice sink and IPC broadcast

- [ ] **Step 1: Write failing tests for IPC status/cancel methods and WakeOutputFanOut**

```python
# In tests/test_ipc_and_cli.py
import pytest
from verbix.runtime.runtime import Runtime
from unittest.mock import MagicMock

def test_runtime_ipc_methods_exist(tmp_path):
    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    assert hasattr(rt, "get_task_status_summary")
    assert hasattr(rt, "cancel_task")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipc_and_cli.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement missing IPC methods in `Runtime`**

Add `get_task_status_summary(self, task_id: str | None = None) -> str` and `async def cancel_task(self, task_id: str) -> bool` to `Runtime`, delegating directly to `self.ledger` and `self.cancel()`.

- [ ] **Step 4: Implement `WakeOutputFanOut` and broadcast events**

In `src/verbix/runtime/runtime.py`:
Implement a fan-out helper so that when announcements occur, events are sent to both the voice output and all connected IPC client queues.
Connect `_broadcast_event` to ledger transitions so IPC subscribers receive live updates.

- [ ] **Step 5: Clean up dead CLI options and fix FinishTool observation formatting**

In `src/verbix/runtime/tools.py`:
Fix `FinishTool.execute` to return `ToolResult(f"Task completed: {summary}")` and ensure `DefaultAgent` handles observation string cleanly without repr leakage.
In `src/verbix/cli.py`:
Remove unused `--isolate-worktrees` and `--allow-network` argument definitions.

- [ ] **Step 6: Run full test suite to verify 0 regressions**

Run: `uv run pytest -v`
Expected: PASS (All tests pass, including all newly added tests).

- [ ] **Step 7: Commit**

```bash
git add src/verbix/runtime/runtime.py src/verbix/runtime/tools.py src/verbix/ipc/server.py src/verbix/cli.py tests/test_ipc_and_cli.py
git commit -m "fix(runtime): wire IPC methods, fan out wake output, fix finish observation, and clean CLI flags"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-06-architecture-scaling-and-runtime-hardening-phase-0-1.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** - A fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** - I implement every task myself in this session, the way this harness runs work, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end. Runs well with a mid-tier session model, since the plan carries the design.

For this plan I recommend **Subagent-driven**, because the tasks touch concurrent asyncio loops, process signals, and state transitions across agent and runtime layers where isolated verification per task prevents race regressions from compounding. Does the plan capture what you want, and which approach should we use?
