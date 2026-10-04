# Specification: Simple Foreground-Worker Runtime Architecture

## 1. Overview & Principles
A clean, minimal, subagent-driven runtime for Verbix.
- **Identity**: The being is **Verbix** — a voice-operated assistant and calm, capable peer. Responses are spoken aloud in plain, continuous conversational prose without narrating internal tool names or task IDs.
- **YAGNI & Simplicity (Ponytail Ultra)**: Native Python `asyncio` primitives. Zero custom locks, worktree orchestrators, or disk log clutter.
- **Latency Decoupling**: Verbix responds to conversation in ~300–500ms; background workers execute long multi-turn tool loops independently without stalling voice interaction.
- **Dual-Emission as Default**: Any `AgentSession` naturally streams spoken words (`TextDelta`) while simultaneously formulating and executing tool calls in the same turn.

---

## 2. Architecture Diagram

```
                         ┌──────────────┐
                         │     USER     │
                         └──────┬───────┘
                                │
                         voice / text / CLI
                                │
                                ▼
                    ┌──────────────────────┐
                    │        VERBIX        │
                    │  (Foreground Agent)  │
                    │                      │
                    │  conversation        │
                    │  quick answers       │
                    │  orchestration       │
                    │                      │
                    │  Tools ONLY:         │
                    │  • spawn(task)       │
                    │  • status(task_id?,  │
                    │           full?)     │
                    │  • cancel(task_id?)  │
                    └──────────┬───────────┘
                               │
                         spawn(task)
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │  WORKER 1  │   │  WORKER 2  │   │  WORKER 3  │
       │            │   │            │   │            │
       │ AgentSession│  │ AgentSession│  │ AgentSession│
       │ own context│   │ own context│   │ own context│
       │ full tools │   │ full tools │   │ full tools │
       └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
             │                │                │
             └────────────────┼────────────────┘
                              │ worker finished (task_id, result_text)
                              ▼
                 ┌────────────────────────┐
                 │    COMPLETION BUFFER   │
                 │                        │
                 │ 300ms batching window  │
                 └────────────┬───────────┘
                              │
                    batch of 1..N results
                              │
                              ▼
                    ┌──────────────────────┐
                    │        VERBIX        │
                    │                      │
                    │ interpret results    │
                    │ decide what to say   │
                    │ formulate response   │
                    └──────────┬───────────┘
                               │
                               ▼
                           TTS / CLI
```

---

## 3. Core Components

### 3.1 WorkerTask
- Represents an asynchronous background task.
- **Fields**:
  - `id: str`: Sequential identifier (e.g. `task-1`, `task-2`).
  - `description: str`: User instruction for the worker.
  - `status: Literal["running", "completed", "failed", "cancelled"]`.
  - `result: str | None`: Final text output produced by the worker `AgentSession`.
  - `error: str | None`: Exception details if failed.
  - `session: AgentSession`: Worker's dedicated agent session with independent history and full tools (`FileReadTool`, `FileWriteTool`, `FileEditTool`, `BashTool`).
  - `asyncio_task: asyncio.Task[None] | None`: The running background task.
- **Cancellation**:
  - Calling `cancel()` invokes `asyncio_task.cancel()`.
  - `BashTool` terminates running subprocesses (`SIGKILL`) in its `finally` block; LLM stream aborts immediately. Status is set to `"cancelled"`.

### 3.2 CompletionBuffer
- Batches worker finish notifications over a sliding window (`300ms`).
- **Mechanism**:
  - When a worker finishes, it posts `(task_id, result)` to the buffer.
  - A 300ms debounce timer starts. If additional workers finish before the timer fires, they are appended to the pending batch.
  - Once the window elapses, the complete batch is flushed to the foreground wake handler.
- **Benefit**: Prevents back-to-back rapid audio speech overlaps when multiple background workers finish in close succession.

### 3.3 Foreground Tools (Strictly 3 Tools)
1. **`spawn(task: str) -> str`**:
   - Spawns a new background worker with `task` description; returns `task_id`.
2. **`status(task_id: str | None = None, full: bool = False) -> str`**:
   - If `task_id` is omitted: lists all active and recently completed tasks with their IDs and descriptions.
   - If `task_id` is provided:
     - `full=False` (default): returns summary and last 2–3 actions from `task.session.history.messages` (lightweight, voice-friendly).
     - `full=True`: returns complete transcript of tool calls and messages from that worker.
3. **`cancel(task_id: str | None = None) -> str`**:
   - If `task_id` is provided: halts that specific task immediately.
   - If `task_id="all"`: halts all running tasks.
   - If `task_id` is omitted:
     - If exactly 1 task is running: cancels that task.
     - If 0 tasks are running: returns informative message.
     - If 2+ tasks are running: returns the list of running tasks without guessing, prompting Verbix to specify the ID.

### 3.4 Minimum Effective Prompt (MEP)
The foreground system prompt defines Verbix's identity without rigid sentence caps or laundry lists:
```text
You are Verbix, a voice-operated assistant and calm, capable peer. You say "I," speak plainly, admit limits honestly, and avoid flattery.

- Your answers are spoken aloud. Speak in plain, continuous conversational prose. Never use markdown, bullet points, asterisks, backticks, or code syntax.
- Be concise. Give the direct answer without narrating internal steps or tool mechanics.
- Answer conversational and general knowledge questions directly.
- Delegate tasks requiring shell commands, file tools, or multi-step execution to spawn. When you call spawn, begin with one short spoken sentence stating what you are starting.
- Inspect active background work with status, and stop work with cancel.
```

### 3.5 Runtime Manager
- **Class `Runtime`**:
  - Holds `foreground_session: AgentSession`.
  - Manages active tasks: `dict[str, WorkerTask]`.
  - Holds `CompletionBuffer`.
  - Dispatches `step(user_text)`: Runs foreground turn with user input.
  - Dispatches `on_completions(batch)`: Injects batch into foreground conversation history as `[Background task completed: task-X -> result]` and executes a foreground turn to formulate a concise spoken summary for the user.

---

## 4. Verification Strategy
1. **Unit Tests**:
   - `test_foreground_tools`: Verify `spawn`, `status` (summary and full), and `cancel` (single, all, and multi-task prompt).
   - `test_worker_task_lifecycle`: Start, execution, completion, and in-memory history inspection.
   - `test_worker_task_cancellation`: Immediate task cancellation and subprocess cleanup.
   - `test_completion_buffer_batching`: Debounced batching within 300ms window.
   - `test_dual_emission`: Verify model emits text tokens and tool calls concurrently.
   - `test_runtime_e2e`: End-to-end user turn -> spawn worker -> worker finishes -> buffer flushes -> Verbix speaks.
2. **Quality & Regression Checks**:
   - `uv run ruff check src/ tests/` (zero warnings/errors).
   - `uv run pytest -v` (100% pass across all tests).
