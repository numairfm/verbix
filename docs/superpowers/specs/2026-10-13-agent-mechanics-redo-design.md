# Design Spec: Agent Mechanics Redo (Worker, Voice, Announcer Seam Separation)

**Date**: 2026-10-13  
**Status**: Draft for Review  
**Context**: Verbix desktop assistant agent architecture  

---

## 1. Executive Summary & Goals

### Problem
Verbix previously forced three fundamentally different roles—the foreground voice receptionist, the autonomous background worker, and the spoken announcer—into a single monolithic agent class (`DefaultAgent`). This caused:
1. **Voice buffer bleed**: Autonomous workers were subjected to TTS speech segmentation (`split_chunks`), buffering generated code and text in memory.
2. **Missing tool enforcement**: When a worker generated text/code without a tool call, the turn silently ended as "speech". The outer task runner then re-injected duplicate user prompts in a 30-round loop.
3. **Streaming deadlocks**: Background workers used unshielded SSE streaming (`litellm.acompletion(stream=True)`) with no per-chunk idle timeout, causing multi-minute hangs on large code generations.

### Goals
1. **Clean Seams**: Completely decouple the autonomous background worker from the voiced foreground/announcer agents.
2. **Mini-SWE Worker Parity**: Restore the worker agent to a pure `mini-swe-agent` execution loop (`run()` / `step()`) with strict tool enforcement and non-streaming model calls.
3. **Preserve Dynamic Tools**: Keep tool registration fully dynamic per `AGENTS.md`: tools live in `tools/<name>.py`, are listed in agent YAML `tools:`, and can be added or removed with zero agent code changes.
4. **Seamless Voice & Announcement Handoff**: When a worker calls `finish`, its summary automatically enqueues into the announcer, which presents the result to the user when the voice floor is idle and records the outcome into shared conversation memory.

---

## 2. Architecture & Seams

```
+-------------------------------------------------------------------+
|                        VERBIX RUNTIME                             |
|                                                                   |
|  +--------------------+         spawn(objective)                  |
|  |  Foreground Voice  | ------------------------------+           |
|  |  Agent             |                               |           |
|  |  (voice.yaml)      | <-------------+               v           |
|  |  [spawn, status,   |               |      +-----------------+  |
|  |   cancel]          |               |      |  Worker Task    |  |
|  +--------------------+               |      |  (worker.yaml)  |  |
|            |                          |      |  [read, write,  |  |
|     speaks |                          |      |   edit, bash,   |  |
|            v                          |      |   finish]       |  |
|  +--------------------+               |      +-----------------+  |
|  | Audio Playback /   |               |               |           |
|  | Floor Arbiter      |               |        finish | (summary) |
|  +--------------------+               |               v           |
|            ^                          |      +-----------------+  |
|     speaks |                          |      |  Announcer      |  |
|  +--------------------+               |      |  (announcer.y.) |  |
|  |  Announcer Voice   |               |      |  [tools: none]  |  |
|  |  Agent             | --------------+      +-----------------+  |
|  +--------------------+   appends result to                       |
|                           foreground memory                       |
+-------------------------------------------------------------------+
```

### The Three Agents

1. **Foreground Voice Agent (`verbix/runtime/foreground.py`)**
   - **Role**: Conversational desktop receptionist.
   - **Config**: `verbix/agent/config/voice.yaml`.
   - **Tools**: Dynamic list from config, strictly `spawn`, `status`, `cancel`.
   - **Wire**: Live streaming (`query_stream`) to feed early audio to TTS via `split_chunks`.
   - **Behavior**: Direct conversational dialog; delegates tasks to `spawn`.

2. **Autonomous Worker Agent (`verbix/agent/worker.py`)**
   - **Role**: Autonomous task executor.
   - **Config**: `verbix/agent/config/worker.yaml`.
   - **Tools**: Dynamic list from config, typically `read`, `write`, `edit`, `bash`, `finish`.
   - **Wire**: Standard, reliable non-streaming model queries (`litellm.completion`).
   - **Behavior**: Pure `mini-swe-agent` step loop. Every turn must call a tool. Stops only when `finish` is called or limits are reached.

3. **Announcer Voice Agent (`verbix/runtime/announcer.py`)**
   - **Role**: Spoken debriefer for completed worker tasks.
   - **Config**: `verbix/agent/config/announcer.yaml`.
   - **Tools**: None (`tools: []`, `tool_choice: "none"`).
   - **Wire**: Live streaming (`query_stream`) to TTS.
   - **Behavior**: Activated when worker finishes and floor is idle. Synthesizes concise spoken debrief, plays audio, and appends the final text into `runtime.foreground.messages`.

---

## 3. Dynamic Tool Architecture

As required by `AGENTS.md`:
* Each tool is an independent file in `verbix/tools/<name>.py` exposing standard `execute(args, context)` and schema.
* Tools are assigned dynamically to agents via their YAML `tools:` list.
* Adding a tool: write `verbix/tools/<name>.py`, register in `TOOL_CATALOG` in `verbix/tools/__init__.py`, and add the name to the agent YAML `tools:` list.
* Removing a tool: delete or comment the name in the agent YAML.
* Tools have zero knowledge of whether they are run in a worker or voice agent.

---

## 4. Component Mechanics & Control Flow

### Autonomous Worker Execution (`WorkerAgent`)

```
Initial Prompt: system_template + instance_template(task=objective)
  |
  v
Loop:
  1. Check step_limit, cost_limit, wall_time_limit.
  2. Query model with messages + tool definitions (non-streaming).
  3. Inspect model output:
     - If NO tool call present:
         Add FormatError message: "Error: No tool call found. Every step MUST execute a tool."
         Continue loop.
     - If tool calls present:
         For each tool call:
             Execute tool handler.
             Record tool result message in history.
             If tool is 'finish':
                 Set status = 'completed', extract summary.
                 Break loop.
  4. Repeat until finish or limits reached.
```

### Key Differences from Old Implementation:
- **No outer re-prompting loop**: `WorkerTask` no longer runs an outer `while True` loop that re-injects `self.description`. `WorkerAgent.run()` is self-contained.
- **No speech segmentation**: `split_chunks` is completely removed from worker execution.
- **No streaming stalls**: `WorkerAgent` queries synchronously via `litellm.completion` wrapped in `asyncio.to_thread` (or clean non-streaming async call), honoring standard timeout budgets.
- **Safe compaction**: Multi-turn history pruning never truncates JSON strings inside `tool_calls.function.arguments`.

### Worker-to-Announcer Handoff

1. Worker executes `finish(summary="...")`.
2. `WorkerTask` completes, sets `task.status = "completed"`, `task.result = summary`.
3. Completion callback enqueues `(task_id, summary)` into `runtime._pending_announcements`.
4. `drain_pending_announcements(runtime)` runs:
   - Waits for `runtime.floor.is_idle`.
   - Takes announcer snapshot of `runtime.foreground.messages` (text and context).
   - Prompts announcer with the worker's summary.
   - Announcer streams spoken response to user.
   - Appends the spoken prose to `runtime.foreground.messages` so voice continuity is preserved.

---

## 5. Error Handling & Edge Cases

1. **Unformatted / Chatty Worker Output**:
   If the worker LLM outputs text/code without a tool call, `WorkerAgent` immediately adds a user format observation reminding it to execute a tool, resetting consecutive errors upon a valid tool call.
2. **Worker Timeouts**:
   Each worker task is bounded by `wall_time_limit_seconds` (or environment timeout). If exceeded, it terminates cleanly with `TimeExceeded` and marks status `failed`.
3. **User Cancellation**:
   If the user says "cancel task-X" or "cancel all", `cancel` tool calls `task.cancel()`, which halts the background task and records cancellation in the ledger.
4. **Tool Failures (e.g. non-zero bash exit code)**:
   Tool failures return `is_error=True` with the error output in the observation, allowing the worker to read the failure and self-correct on the next step.

---

## 6. Verification & Proof Strategy

Following the project's strict TDD workflow:
1. **Failing Unit Tests**:
   - `test_worker_multi_turn_executes_to_finish`: Verify worker runs multi-turn tool calls (e.g. `bash` -> `write` -> `finish`) in a single self-contained run without hanging.
   - `test_worker_rejects_chatter_without_tool_call`: Verify that model output lacking tool calls raises format feedback instead of ending the turn.
   - `test_worker_finish_enqueues_announcement`: Verify that when worker finishes, summary is queued for the announcer and synced to foreground memory.
2. **Regression & Full Suite**:
   - Full test suite passes: `pytest`.
   - Linter clean: `ruff check`.
