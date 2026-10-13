# Agent Mechanics Redo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cleanly separate Verbix's three agent roles into dedicated modules under `verbix/agent/` (`worker.py`, `voice.py`, `announcer.py`), restoring the autonomous worker to a pure, non-streaming `mini-swe-agent` loop with strict tool enforcement, while `verbix/runtime/` acts purely as coordinator.

**Architecture:** Split the tangled monolithic `DefaultAgent` into three single-purpose agent classes: `WorkerAgent` (non-streaming, strict tool enforcement), `VoiceAgent` (streaming voice receptionist with speech chunking), and `AnnouncerAgent` (tool-less spoken debriefer). The runtime coordinates background worker execution, enqueues results on `finish`, and speaks summaries through the announcer into shared voice memory.

**Tech Stack:** Python 3.13, LiteLLM, asyncio, pytest, ruff.

**Spec:** [`docs/superpowers/specs/2026-10-13-agent-mechanics-redo-design.md`](file:///home/numair/dev/projects/verbix/docs/superpowers/specs/2026-10-13-agent-mechanics-redo-design.md)

## Global Constraints

- Python 3.13+ compatible.
- All dynamic tools in `src/verbix/tools/` must remain untouched and configurable via YAML `tools:` lists.
- Worker agent must use non-streaming LLM completions (`stream=False`), eliminating SSE stream deadlocks.
- The assistant role is strictly reserved for spoken text; worker tool observations return terse status or raw evidence.
- No scaffolding for later; minimal code that satisfies requirements.
- Full test suite plus `ruff check` must pass cleanly before any completion claim.

## Review Focus

1. **Worker model generates raw code/HTML without a tool call**: Worker must not exit or treat this as speech; it must reject with a `FormatError` and prompt the model to use a tool.
2. **Worker executes large file write (multi-KB HTML/JS)**: Tool arguments must not be prematurely sliced into invalid JSON by context compaction.
3. **Worker finishes via `finish` tool**: The finish summary must be delivered to `runtime._pending_announcements`, spoken by the announcer when floor is idle, and appended to foreground history.
4. **User cancels a running worker task**: `WorkerTask.cancel()` must cleanly halt the background task and record cancellation without hanging.
5. **Tool catalog dynamic discovery**: Modifying `worker.yaml`'s `tools:` list immediately changes the available tools without touching agent code.

---

### Task 1: Autonomous Worker Agent (`WorkerAgent`)

**Files:**
- Create: `src/verbix/agent/worker.py`
- Test: `tests/test_worker_agent.py`

**Interfaces:**
- Consumes: `ToolRegistry` from `verbix.agent.tools.registry`, `LitellmModel` from `verbix.agent.models.litellm_model`.
- Produces: `WorkerAgent` class with `run(task: str) -> dict[str, Any]` and async `step() -> ToolResult`.

- [ ] **Step 1: Write the failing test for `WorkerAgent` single-step and format error**

```python
import pytest
from unittest.mock import MagicMock
from verbix.agent.worker import WorkerAgent
from verbix.agent.tools.registry import ToolRegistry
from verbix.tools.base import ToolResult

@pytest.mark.asyncio
async def test_worker_rejects_chatter_without_tool_call():
    mock_model = MagicMock()
    # First response is plain text (no tool calls), second response calls finish
    mock_model.query.side_effect = [
        {"role": "assistant", "content": "I will create index.html now", "tool_calls": []},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
        },
    ]
    tools = ToolRegistry()
    tools.register("finish", {"type": "function", "function": {"name": "finish"}}, lambda a: ToolResult("done", is_terminal=True))
    
    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("build something")
    
    assert result["status"] == "completed"
    assert result["summary"] == "done"
    # Verify format error was fed back after the first response
    assert any("MUST execute a tool" in m.get("content", "") for m in agent.messages if m.get("role") == "user")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker_agent.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'verbix.agent.worker'`

- [ ] **Step 3: Implement `WorkerAgent` in `src/verbix/agent/worker.py`**

Implement `WorkerAgent` channeling `mini-swe-agent`:
- Non-streaming model calls via `model.query(self.messages, tools=...)`.
- Strict tool-calling guard: if `not tool_calls`, add user message reminding model that every step must execute a tool, increment format errors.
- Dispatches tool calls through `self.tools.aexecute()`.
- Terminates cleanly when `res.is_terminal` (i.e. `finish` tool is called) or limits are exceeded.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_worker_agent.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/worker.py tests/test_worker_agent.py
git commit -m "feat: add WorkerAgent with mini-swe single loop and strict tool guard"
```

---

### Task 2: Voice and Announcer Agent Classes (`VoiceAgent` and `AnnouncerAgent`)

**Files:**
- Create: `src/verbix/agent/voice.py`
- Create: `src/verbix/agent/announcer.py`
- Test: `tests/test_voice_agent.py`

**Interfaces:**
- Consumes: `LitellmModel` / provider stream adapter, `split_chunks` from `verbix.voice.segments`.
- Produces: `VoiceAgent` (live streaming + voice segmentation for receptionist), `AnnouncerAgent` (tool-less voice presenter).

- [ ] **Step 1: Write the failing tests for `VoiceAgent` and `AnnouncerAgent`**

```python
import pytest
from unittest.mock import MagicMock
from verbix.agent.voice import VoiceAgent
from verbix.agent.announcer import AnnouncerAgent
from verbix.core.events import TextDelta, TurnComplete

@pytest.mark.asyncio
async def test_voice_agent_streams_text():
    mock_model = MagicMock()
    async def fake_stream(messages, tools=None, **kw):
        yield TextDelta(text="Hello there. How can I help you today?")
        yield TurnComplete(final_text="Hello there. How can I help you today?")
    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)
    events = [e async for e in agent.step_stream("hi")]
    assert any(isinstance(e, TextDelta) and "Hello there" in e.text for e in events)

@pytest.mark.asyncio
async def test_announcer_agent_has_no_tools():
    agent = AnnouncerAgent(model=MagicMock())
    assert agent.get_tool_definitions() is None or agent.get_tool_definitions() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_voice_agent.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'verbix.agent.voice'`

- [ ] **Step 3: Implement `VoiceAgent` in `src/verbix/agent/voice.py` and `AnnouncerAgent` in `src/verbix/agent/announcer.py`**

- `VoiceAgent`: Dedicated conversational streaming agent using `query_stream()`, speech segmentation via `split_chunks`, pre-tool suppression, and tool execution for `spawn`/`status`/`cancel`.
- `AnnouncerAgent`: Subclass or dedicated variant of `VoiceAgent` with `tool_choice="none"`, `tools=[]`, strictly producing spoken debriefs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_voice_agent.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/voice.py src/verbix/agent/announcer.py tests/test_voice_agent.py
git commit -m "feat: add VoiceAgent and AnnouncerAgent classes in verbix/agent/"
```

---

### Task 3: Runtime Wiring and Worker-to-Announcer Handoff

**Files:**
- Modify: `src/verbix/runtime/workers.py`
- Modify: `src/verbix/runtime/worker.py`
- Modify: `src/verbix/runtime/foreground.py`
- Modify: `src/verbix/runtime/announcer.py`
- Test: `tests/test_runtime_worker.py`

**Interfaces:**
- Consumes: `WorkerAgent`, `VoiceAgent`, `AnnouncerAgent` from `verbix.agent.*`.
- Produces: Integrated runtime where `spawn` runs `WorkerAgent`, completion enqueues to announcer, announcer speaks when floor is idle.

- [ ] **Step 1: Write the failing test for worker finish handoff to announcer queue**

```python
import pytest
from unittest.mock import MagicMock
from verbix.runtime.worker import WorkerTask

@pytest.mark.asyncio
async def test_worker_finish_completes_and_reports_summary():
    mock_agent = MagicMock()
    async def fake_run(task):
        return {"status": "completed", "summary": "Wrote flappy.html (2400 bytes, verified)."}
    mock_agent.run = fake_run
    mock_agent.messages = []
    
    task = WorkerTask(id="task-1", description="create flappy bird", session=mock_agent)
    task.start()
    await task.wait()
    
    assert task.status == "completed"
    assert "Wrote flappy.html" in (task.result or "")
```

- [ ] **Step 2: Run test to verify it fails or exposes previous loop mismatch**

Run: `pytest tests/test_runtime_worker.py::test_worker_finish_completes_and_reports_summary -v`  
Expected: FAIL (or mismatch with old `step_stream` wrapper)

- [ ] **Step 3: Update `src/verbix/runtime/workers.py` and `worker.py`**

- In `workers.py`: `spawn_worker` instantiates `WorkerAgent` using `worker.yaml`.
- In `worker.py`: `WorkerTask._run()` invokes `session.run(self.description)` directly without an outer looping re-prompt mechanism.
- On completion: appends `(task_id, result)` to `runtime._pending_announcements` and calls `drain_pending_announcements(runtime)`.
- Update `foreground.py` and `runtime/announcer.py` to instantiate `VoiceAgent` and `AnnouncerAgent`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime_worker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/workers.py src/verbix/runtime/worker.py src/verbix/runtime/foreground.py src/verbix/runtime/announcer.py tests/test_runtime_worker.py
git commit -m "refactor: wire runtime to use WorkerAgent, VoiceAgent, and AnnouncerAgent"
```

---

### Task 4: Context Compaction Fix & Full Regression

**Files:**
- Modify: `src/verbix/agent/utils/compaction.py`
- Test: `tests/test_compaction.py`

**Interfaces:**
- Consumes: message histories > 16k tokens.
- Produces: valid compacted JSON messages without argument truncation that corrupts JSON syntax.

- [ ] **Step 1: Write test verifying large tool arguments remain valid JSON during compaction**

```python
import json
from verbix.agent.utils.compaction import compact_messages

def test_compaction_never_breaks_json_arguments():
    huge_code = "<!DOCTYPE html>\\n" + "x" * 20000
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "create app"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "write", "arguments": json.dumps({"path": "app.html", "content": huge_code})}
            }]
        },
        {"role": "tool", "tool_call_id": "c1", "name": "write", "content": "Wrote 20016 characters"},
    ]
    compacted = compact_messages(messages, max_tokens=1000)
    for m in compacted:
        for tc in m.get("tool_calls", []):
            args_str = tc.get("function", {}).get("arguments", "")
            # Must always remain decodable JSON
            json.loads(args_str)
```

- [ ] **Step 2: Run test to verify if old code corrupts JSON**

Run: `pytest tests/test_compaction.py -k test_compaction_never_breaks_json_arguments -v`

- [ ] **Step 3: Fix `_truncate_args` in `src/verbix/agent/utils/compaction.py`**

Ensure `_truncate_args` parses the JSON, elides the inner string value cleanly (e.g. `{"path": "app.html", "content": "[truncated]"}`), and dumps valid JSON back, instead of blindly slicing raw characters with `args[:500]`.

- [ ] **Step 4: Run full test suite and linter**

Run:
```bash
uv run pytest
uv run ruff check
```
Expected: PASS for all tests and clean linter.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/utils/compaction.py tests/test_compaction.py
git commit -m "fix: preserve valid JSON in compaction and clean full test suite"
```
