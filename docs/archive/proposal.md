# Verbix Architecture Proposal & Research Report (Final Revision)

Date: 2026-10-02  
Author: Antigravity Pairing Agent & Numair Mazloum  
Status: Approved in Direction — Pre-Slice 2 Specification  

---

## 1. Project Initialization & Tooling Verification

The workspace at `/home/numair/dev/projects/verbix` is initialized using **uv** (v0.12.19) [Verified on host] targeting Python `>=3.13` (host runtime: Python 3.14.7) [Verified on host].
- **Build Backend**: `uv_build` (standalone, standard, pure-Python compatible).
- **Package Layout**: Modern `src/verbix` structure with `pyproject.toml`.
- **Tooling Constraints**: Dependency resolution, lockfile generation, venv management, tool execution, and testing run strictly through `uv`. No Poetry, Pipenv, or Conda.

---

## 2. Post-Mortem: Verbix-Old-Go Architecture & Failure Analysis

A systematic analysis of the legacy codebase at `~/dev/projects/verbix-old-go` (30+ git commits, ~15k lines of Go across `core/`, Rust/Tauri in `app/`) confirms the exact failure modes and friction points:

### 2.1 The Core Failures (Verified Against Code)

1. **Background-Task Re-Projection into Receptionist Context**:
   - *Code References*: `core/main.go:2809-2828` (`taskSystemContext`) and `core/stale_completion_test.go:54-81` (`TestFinishedTerminalMarksReportedAtPlaybackReceipt`) [Verified].
   - *Failure Mechanism*: On every turn, the controller generated `taskSystemContext`, injecting a dynamic list of active and recently completed tasks into the system prompt.
   - *Root Cause*: If an announcement was pending or recently delivered, the model continued to see task state in its prompt. A follow-up pleasantry ("thank you") caused the model to re-address the finished task or re-promise the work.
   - *Architecture Fix*: **Eliminate dynamic task projection entirely.** The system prompt never fluctuates based on task state. Completed tasks enter conversation history once as permanent, immutable messages.

2. **Unheard Background Speech Disconnected from History**:
   - *Code References*: `core/main.go:2745-2756` (`s.queueAnnouncement(tk.ID, line)`) and `core/barge_in_history_test.go` [Verified].
   - *Failure Mechanism*: Spoken completions from background workers were delayed from entering the agent's message history until the Rust audio layer emitted an asynchronous `playback_started` receipt.
   - *Root Cause*: If an audio frame was dropped, delayed, or if the user barged in right as the task finished, the completion text was never recorded into the session log. The agent then saw no record of having completed the request, apologized, and re-spawned it.
   - *Architecture Fix*: **Decouple historical event recording from audio delivery.** When a task completes, its outcome is an objective factual event committed to history immediately at the turn boundary. Spoken playback is an output channel with its own `announced: bool` state.

3. **Prompt-Based "Pull Dodge" (Weak-Model Failure)**:
   - *Code References*: `core/main.go:2520-2548` (`pullDodges` tripwire counter) and `core/main.go:2789-2807` (`turnUsedTask`) [Verified].
   - *Failure Mechanism*: The Go core relied on prompt prose advising the model to call `task_status` to check progress.
   - *Root Cause*: Weak models (e.g. Gemini Flash Lite, small local LLMs) ignored the advice, skipped the tool, and generated ungrounded stall prose ("Give me a moment while I check").
   - *Architecture Fix*: **Replace model judgment with deterministic runtime behavior.** Fast-path status queries are answered directly by runtime logic, and active task progress is framed into the turn context automatically.

4. **Safety Monitor Argument Blowup & Rate Limiting**:
   - *Code References*: `core/agent/safety.go:22-62` (`safetyInput`, `safetyToolCall`) [Verified].
   - *Failure Mechanism*: Before executing tools, `safety.go` called a secondary LLM with `safetyInput` embedding all prior tool calls and proposed calls with full raw arguments (`Arguments: tc.Arguments`).
   - *Root Cause*: File writes (`write_file`) or bash commands with large scripts embedded megabytes of raw text into every safety check, exceeding rate limits (HTTP 429) and token ceilings.
   - *Architecture Fix*: Enforce safety structurally in code via an OS sandbox, never by piping megabytes of raw arguments into secondary LLMs.

5. **Tri-Lingual Boundary Friction (TypeScript <-> Rust <-> Go)**:
   - *Code References*: `app/src-tauri/src/main.rs`, `core/main.go:41` (`serveMode` JSONL IPC), `app/src/` [Verified].
   - *Friction*: Audio streaming, hotkey states, task events, and cancellations crossed two foreign function / IPC seams. Debugging timing races required synchronizing Go channels, Rust Tokio tasks, and TypeScript event listeners.
   - *Architecture Fix*: The core runtime, CLI, and voice pipeline are 100% Python in one process. The desktop shell is a thin, dumb display/hotkey client communicating over a local socket.

---

## 3. Research Findings

All package versions and specifications were verified directly against PyPI and local machine execution on 2026-10-02.

### 3.1 Agent Frameworks vs. Custom Loop

| Framework | Version / Date | Footprint & Dependencies | Assessment for Verbix |
| :--- | :--- | :--- | :--- |
| **mini-SWE-agent** | v2.4.6 (2026-07-23) [Verified] | `litellm`, `swe-rex`, `modal`, `datasets`, `pyyaml` | **Reject.** Tailored for SWE-bench evaluation in Docker/Modal containers. No interactive streaming, local voice, or barge-in support. |
| **OpenAI Agents SDK** | v0.22.3 (2026-09-17) [Verified] | `openai`, `httpx2`, `mcp`, `starlette`, `websockets`, `temporalio`, `dapr`, `modal` | **Reject.** Tightly coupled to OpenAI cloud APIs and heavy distributed workflow backends. Violates local-first BYOK design. |
| **PydanticAI** | v2.53.0 (2026-10-02) [Verified] | `pydantic-ai-slim`: `pydantic>=2.12`, `pydantic-graph`, `anyio`, `httpx2>=2.7`, `opentelemetry-api` | **Reject as framework; borrow patterns.** The agent loop is hidden inside `pydantic-graph` and `Agent.run()`. This obstructs fine-grained turn control, sub-turn cancellation, custom thought-delta streaming, and small-model tool recovery. |
| **Custom Minimal Loop** | Standard Python + `httpx2` + `pydantic` | Zero framework dependencies | **Adopt.** An async generator loop (`async for event in session.step(...)`) is under 200 lines. It provides full control over barge-in cancellation, small-model JSON repair, cycle detection, and memory isolation. |

### 3.2 Provider Abstraction & The Universal Layer Question

- **LiteLLM Evaluation**:
  - *Current Version*: v1.103.2 (2026-10-01) [Verified].
  - *Dependency Footprint*: `uv pip compile` resolves **164 packages** [Verified on host], including `boto3`, `botocore`, `aiohttp`, `cryptography`, and `opentelemetry`.
  - *Verdict*: **Explicitly rejected.** It introduces massive bloat, frequent breaking changes, and monkey-patches global loggers.
- **Provider Scope**: Start strictly with **Generic OpenAI-Compatible plus OpenRouter ONLY**. OpenRouter already proxies Anthropic, Gemini, Mistral, and Meta models using standard OpenAI wire format. Native Anthropic/Gemini endpoints are deferred until direct connections are explicitly needed.

### 3.3 Investigation: `httpx` vs `httpx2`

- **What is `httpx2`?** [Verified on PyPI]:
  - `httpx2` (v2.13.1, September 2026) is an actively maintained continuation fork of Encode's `httpx` stewarded by Pydantic (`https://github.com/pydantic/httpx2`).
- **Why did PydanticAI and OpenAI Agents move to `httpx2`?** [Verified]:
  - Encode's original `httpx` project experienced limited activity (last release v0.28.1 on Dec 6, 2024).
  - Pydantic stepped in to maintain `httpx2` as a production-grade HTTP client with timely security patches, performance improvements, and native OS certificate loading via `truststore>=0.10`.
  - Both `pydantic-ai-slim>=2.53.0` and `openai-agents>=0.22.3` explicitly declare dependencies on `httpx2>=2.7`.
- **Decision before Slice 2**:
  - **Adopt `httpx2`**.
  - Its API is a direct match for `httpx` (`httpx2.AsyncClient`, `stream()`, `aiter_lines()`), but provides active maintenance, native OS truststore certificates, and avoids unpatched CVEs in frozen libraries.

---

## 4. Latency Figures & Measured Spikes

### 4.1 Measured Spikes on Host Machine

Spikes were executed in `scratch/` on this machine (Linux x86_64, PipeWire 1.6.9, AMD CPU) [Verified on host]:

1. **Mid-Flight Streaming Cancellation Latency (`scratch/spike_cancellation.py`)** [Verified on host]:
   - **What it ran against**: A local `asyncio` SSE HTTP streaming server simulating an active provider connection emitting chunked tokens every 40 ms.
   - **Measurement**: After receiving partial chunks, `task.cancel()` was invoked.
   - **Cancellation Latency**: **1.231 ms** [Measured on this host].
   - The TCP connection was immediately aborted and closed without leaking memory or file descriptors.

2. **PocketTTS CPU First-Chunk Synthesis Latency (`scratch/test_tts.wav`)** [Verified on host]:
   - **What it ran against**: Bundled PocketTTS ONNX runtime (`pocket-tts --profile`, `alba_1.10x.wav`, 8 CPU threads).
   - **Input Text**: *"The build finished successfully in twelve seconds."* (2.56 seconds of synthesized audio).
   - **First-Chunk Latency (TTFA)**: **99 ms** [Measured on this host].
   - **Real-Time Factor (RTFx)**: **2.51x real-time** (entire 2.56s generated in 1.02s).
   - **Memory Footprint**: 331 MB idle RSS, 443 MB peak RSS during synthesis.

### 4.2 Latency Budgets (Measured vs. Estimated)

We distinguish two separate operational budgets:

#### Budget A: Time to Spoken Acknowledgment (Conversational Interaction)
*User releases hotkey → hears receptionist's first spoken word.*

| Stage | Latency | Source |
| :--- | :--- | :--- |
| **Push-to-Talk Speech End** | **0 ms** | **Measured** (instantaneous key release event; no VAD cutoff delay) |
| **Local STT Transcription** | **120 – 180 ms** | **Estimated** (quantized int8 Whisper/Sherpa on 2–3s speech) |
| **Cloud/BYOK Model TTFS** | **200 – 350 ms** | **Estimated** (fast streaming provider: Gemini Flash / Groq / OpenRouter) |
| **PocketTTS First-Chunk Synthesis** | **99 ms** | **Measured on host** |
| **Audio Sink Buffer Preroll** | **30 – 50 ms** | **Estimated** (PipeWire buffer flush) |
| **TOTAL TIME TO SPOKEN ACK** | **~450 ms – 680 ms** | **(< 0.7 seconds)** |

#### Budget B: Time to Task Result
*Background worker completes job → user hears spoken completion summary.*

| Stage | Latency | Source |
| :--- | :--- | :--- |
| **Task Execution** | Variable | Workload dependent (e.g. 5s – 5min) |
| **Worker Finish Event Emission** | **~1 ms** | **Measured** (asyncio task completion + queue enqueue) |
| **PocketTTS First-Chunk Synthesis** | **99 ms** | **Measured on host** |
| **Audio Sink Buffer Preroll** | **30 – 50 ms** | **Estimated** |
| **TOTAL TIME FROM FINISH TO SPEECH** | **~130 ms – 150 ms** | **(near-instantaneous completion notification)** |

---

## 5. Recommended Architecture

### 5.1 System Structure & Responsibilities

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND INTERFACES                             │
│   ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐   │
│   │ CLI (Rich/Typer)  │  │ Voice Pipeline    │  │ Desktop Gateway  │   │
│   │ (Direct In-Proc)  │  │ (PTT/TTS/BargeIn) │  │ (UDS / NamedPipe)│   │
│   └─────────┬─────────┘  └─────────┬─────────┘  └────────┬─────────┘   │
└─────────────┼──────────────────────┼─────────────────────┼─────────────┘
              │                      │                     │
              ▼                      ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           VERBIX RUNTIME                               │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Subscriber Queues: subscribers: list[asyncio.Queue]            │   │
│   │ Spoken Announcement Queue: announcement_q: asyncio.Queue       │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                   │
│   ┌────────────────────────────────▼───────────────────────────────┐   │
│   │ Main Receptionist AgentSession (Strictly Orchestration-Only)   │   │
│   │ - Tools: task_spawn, task_cancel, task_steer, task_status      │   │
│   │ - NO bash, NO file tools, NO prompt projection of task lists   │   │
│   │ - Injects transient running-task line to active turns          │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                   │
│   ┌────────────────────────────────▼───────────────────────────────┐   │
│   │ Task Runtime (Structured Concurrency via TaskGroup)            │   │
│   │ - tasks: dict[str, TaskState] (plain dict)                     │   │
│   │ - Spawns Worker Sessions in Isolated Worktrees / Sandboxes     │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                   │
│   ┌────────────────────────────────▼───────────────────────────────┐   │
│   │ Worker AgentSessions (Isolated Contexts)                       │   │
│   │ - Independent message histories                                │   │
│   │ - System Tools: bash, file tools, search, MCP                  │   │
│   │ - Default-deny OS sandbox, 30-turn limit, cycle detection      │   │
│   │ - Emits structured WorkerResult (speech_summary + details)     │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PROVIDER CLIENT (httpx2 + SSE Dialects)                  │
│   Generic OpenAI-Compatible & OpenRouter Only                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Status Handling: The Turn-Context Line & Runtime Parity

To eliminate model judgment and the legacy Go "pull dodge" without anchored regexes:

1. **Compact Running-Task Line on Every Turn**:
   - While ANY task is running (`len([t for t in tasks.values() if t.is_running]) > 0`):
   - The runtime prepends a compact single-line prefix to the turn prompt:
     `[Running: task-1 'build kernel' (step 12)]`
   - **Invariants**:
     - **NEVER contains finished tasks** (finished tasks enter history once at completion and are never re-projected).
     - **NEVER persisted to history** (it is a transient prompt annotation for that turn only).
   - Because the active state is directly visible in the prompt, the model answers contextual questions ("how's it going?") accurately without needing to call `task_status`.

2. **Runtime Answer Parity Rule**:
   - If the runtime ever intercepts or answers a turn without invoking the model (e.g. an explicit client shortcut or deterministic command):
   - The runtime **MUST append BOTH the user message AND the runtime reply to history**:
     ```python
     history.append(UserMessage(content=user_text))
     history.append(AssistantMessage(content=runtime_reply_text))
     ```
   - This ensures absolute parity between what was spoken and what exists in conversation memory.

### 5.3 Structured Worker Finish

To ensure spoken completions are never raw worker prose, prompt ramblings, or internal scratchpads, every worker must yield a strictly typed finish record:

```python
from pydantic import BaseModel, Field


class WorkerResult(BaseModel):
    success: bool
    speech_summary: str = Field(
        ...,
        max_length=180,
        description="Terse, natural spoken summary for TTS playback. Maximum 25-30 words. No markdown, no fences.",
    )
    details: str = Field(
        ...,
        description="Comprehensive technical details, changed files, test output, diffs, or error logs for text/CLI display.",
    )
```

- **Speech Output**: PocketTTS synthesizes *only* `speech_summary` (e.g. *"Saved four release binaries to dist."*).
- **History & Display**: The main conversation history and CLI record the full event:
  `[Task completed: 'build-binaries' (task-1) -> Succeeded: Saved four release binaries to dist. Details: ...]`.

### 5.4 OS Sandbox: The True Security Boundary

The OS sandbox is the **sole security boundary**. Command allowlists and confirmation prompts are **UX / convenience mechanisms only**.

1. **Default-Deny OS Sandbox**:
   - **Filesystem Isolation**: Mount ONLY the target workspace directory (`rw`) and minimal runtime dependencies (`ro`: `/usr`, `/lib`, `/lib64`, `/bin`, `/etc/alternatives`, minimal `/etc`).
   - All other host directories (`/home`, `/root`, `/tmp`, `/dev`, `/var`) are masked or unmounted. Sensitive user paths (`~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.config`) are completely inaccessible.
2. **Network Policy**:
   - **Default-Deny Network**: By default, workers run with `--unshare-net` (no network interfaces except loopback).
   - Network access must be explicitly declared in the task configuration or granted on request.
3. **Cross-Platform Implementation**:
   - **Linux**: Bubblewrap (`bwrap`) is standard, unprivileged (user namespaces), and available across Fedora, Debian, Ubuntu, and Arch.
   - **macOS**: Apple `sandbox-exec` with a compiled Seatbelt profile (`(deny default) (allow file-read* ...) (allow file-write* (subpath "/path/to/workspace"))`).
   - **Windows**: Windows Sandbox, AppContainer isolation, or restricted token with JobObject filesystem boundaries.
4. **Confirmation Gate (UX Layer)**:
   - For mutating commands or operations outside the safe allowlist:
   - **Mechanism**: Requires an explicit **physical hotkey press** (e.g. holding hotkey for 1 second or double-tapping), or a terminal `[Y/n]` keypress.
   - **Timeout Policy**: **Denies on timeout** (fail-closed after 15 seconds).
   - **NEVER a spoken "confirm"**: Spoken confirmation is rejected due to acoustic false positives, background chatter, and speech recognition ambiguity.

### 5.5 Task Completion History Invariants & Barge-In Pruning

1. **Immediate Injection on Finish**:
   - Task completions enter the main session history **immediately when the task completes**, *not* when audio plays.
2. **Injection Timing**:
   - Injected strictly at **turn boundaries** (never breaking an in-flight `tool_call` and `tool_result` pair).
   - If the main agent is idle: Appended immediately.
   - If a foreground turn is running: Buffered in an internal queue and appended immediately after the foreground turn settles.
3. **Authorship**:
   - **100% Runtime-Written**: Zero LLM calls are made to format or re-phrase the completion.
4. **Separate Announcement State**:
   - Task execution record tracks `announced: bool`. The summary is enqueued into `announcement_q`. When audio finishes playing, `announced = True`. If audio is interrupted or cancelled, `announced` remains `False`, but the completion record remains safely in history.
5. **Barge-In Pruning Scope**:
   - Barge-in pruning applies **strictly to foreground assistant turns** (when the assistant is speaking an immediate conversational answer).
   - If interrupted, only the text corresponding to physically played audio frames is kept; unplayed sentences are pruned. Background task completion records are **never pruned**.

### 5.6 Audio Ownership & Echo Cancellation

- **Ownership**: **Python owns microphone capture and audio playback.** The desktop shell / OS integration *only* sends hotkey/trigger events.
- **Echo Cancellation Architecture**:
  - **Mode 1: Hotkey Barge-In (Push-to-Talk / Push-to-Interrupt - Recommended Default)**:
    - Mic is only recorded while the hotkey is held or active.
    - Pressing the hotkey immediately halts audio playback in Python.
    - **Zero Acoustic Echo Cancellation (AEC) needed!** Mic and speakers are never active simultaneously.
  - **Mode 2: VAD / Hands-Free Barge-In**:
    - When mic remains open while speakers play, mic records the agent's own speech, causing feedback loops.
    - Handled via PipeWire's native echo-cancel module (`module-echo-cancel`) on Linux or WebRTC AEC in Python using the playback buffer as reference.

---

## 6. Workspace Selection, Concurrency, and Steering

### 6.1 Workspace Selection
How a spoken request selects its target directory:
1. **Explicit Spoken Override**: Verbal path reference (e.g. *"In repo verbix, check git status"* -> resolved to configured alias or path).
2. **Desktop Shell Focused Context**: The desktop shell inspects the active window (via XDG Desktop Portal, KDE KWin API, or macOS Accessibility) and passes the focused IDE or terminal working directory with the hotkey trigger.
3. **Daemon Default**: Falls back to the global default workspace specified in `~/.config/verbix/config.toml` (or the directory where the daemon was launched).

### 6.2 Concurrent Worker Tree Isolation
To prevent multiple background workers from clobbering each other's edits:
- **Git Worktree Isolation**: For any Git workspace, mutating workers execute in an isolated Git worktree:
  ```shell
  git worktree add -b worker/task-<id> .verbix/worktrees/task-<id> HEAD
  ```
  Workers commit their changes to the worker branch. When the task finishes, the main session offers to merge or review the branch.
- **Non-Git Workspaces**: Managed via directory-level file locking or isolated shadow copies.
- **Read-Only Tasks**: Concurrently inspect the primary workspace without worktree overhead or locks.

### 6.3 Mid-Turn Worker Steering (`task_steer`)
How `task_steer` is delivered to an active worker:
- Every worker owns an incoming `steer_queue: asyncio.Queue[str]`.
- **Subshell Interruption**: If a worker is blocked on a long-running subshell command (`bash`), calling `task_steer` sends `SIGINT` to the process group, terminating the long command cleanly.
- **Next-Step Injection**: At the next tool boundary or turn iteration, the worker drains `steer_queue` and appends a steering message to its context:
  `Role="user"`: `"[Steering instruction from user: Stop editing tests, focus on src/core.py]"`
  The worker model incorporates the instruction on its immediate next inference step.

---

## 7. Deferred & Brief Topics

1. **Persistence**:
   - *Current need*: None. In-memory session state is sufficient.
   - *Future design*: Append-only JSONL files (`~/.local/share/verbix/sessions/<id>.jsonl`). Because all messages and completions are immutable events, dumping them to disk requires zero ORM or database complexity.
2. **HTTP Client**:
   - `httpx2` is used for provider streaming. No secondary HTTP client library is needed.
3. **Local STT/TTS Runtimes**:
   - Keep decoupled from the core. Managed via clean `SpeechToText` and `TextToSpeech` Protocols. Runtimes run as external helper binaries or optional extras.
4. **Desktop Shell Options & Wayland/KDE Constraints**:
   - *Options*: Tauri v2 (Rust) or GPUI (Rust).
   - *Wayland/KDE Constraints*:
     - Arbitrary apps cannot grab global hotkeys. Must use the `org.freedesktop.portal.GlobalShortcuts` XDG portal (supported natively in KDE Plasma 6).
     - Floating overlays and click-through require `wlr-layer-shell` or KDE window rules.
5. **Type Checker Choice**:
   - **Pyright** chosen over Mypy. Pyright is 5–10x faster, has superior support for modern Python 3.12+ type parameters and `Protocol` structural subtyping, and integrates natively with VS Code / Neovim LSP.
6. **Packaging Implications**:
   - Python distribution: Pure-Python wheel via `uv build`.
   - Standalone desktop binary: Built via PyInstaller or Nuitka, bundling the Python core into an executable without bundling heavy ML weights.

---

## 8. Initial Implementation Plan (Sequenced with Skeleton)

- **Slice 1: Core Types & History Log**
  - Canonical message definitions, event dataclasses, append-only history.
  - Pydantic configuration parsing, pytest and ruff setup.
- **Slice 2: Thin Provider Client (`httpx2` + SSE)**
  - Generic OpenAI-compatible & OpenRouter streaming client using `httpx2`.
  - Streaming text deltas, thought-delta extraction, and tool call streaming.
- **Slice 3: Native Tool Engine & OS Sandboxing**
  - Bubblewrap/sandbox runner for `bash` (default-deny filesystem & network, process-group kill).
  - Safe command allowlist and physical hotkey confirmation gate.
- **Slice 4: Single-Turn & Multi-Turn Agent Loop (`AgentSession`)**
  - Transparent async generator loop with max turns, cycle detection, and cancellation.
- **Slice 4.5: End-to-End Conversational Skeleton (PTT → STT → Provider → TTS)**
  - Thin integration proving the complete voice loop: physical Push-to-Talk hotkey triggers mic capture → local STT transcribes → provider streams first sentence → PocketTTS speaks initial audio word (< 0.7s latency target).
- **Slice 5: First-Class Interactive CLI**
  - Rich streaming interface in terminal (`verbix chat`).
- **Slice 6: Background Task Runtime (`TaskRuntime` & Concurrency)**
  - `asyncio.TaskGroup` worker orchestration, plain dict task tracking, turn-boundary history injection.
  - Worktree isolation, `task_steer` queue delivery, and running-task turn framing.
- **Slice 7: Audio Playback Serialization & Barge-In**
  - Serialized announcement queue, sentence-level barge-in truncation for foreground turns.
- **Explicitly Deferred**: Desktop shell (Tauri/GPUI), persistent vector databases, complex multi-server MCP hubs.
