# Specification: Mini-SWE-Agent Unified Agent Architecture

## 1. Overview & Motivation
This specification defines the transition of Verbix to a commercial-ready, unified, config-driven agent architecture based on **Mini-SWE-Agent**.

### Core Goals
1. **Commercial Provider Layer**: Use `litellm` as the universal model backend to natively support all major model providers (OpenAI, Anthropic, OpenRouter, AWS Bedrock, Google Vertex, Azure, and local Ollama) without bespoke API client code.
2. **Unified Agent Engine**: Replace bespoke, fragile `AgentSession` loops with a single, battle-tested Mini-SWE-Agent core powering **both** the voice receptionist (foreground) and autonomous coding agents (background workers).
3. **Config-Driven Tooling**: Agents are defined by declarative YAML configurations (`configs/voice.yaml`, `configs/swe_worker.yaml`). Adding new tools, adjusting prompts, or creating new subagents is achieved via configuration without editing Python agent loops.
4. **Preserved Voice & UX Excellence**: Maintain Verbix's core strengths:
   - Push-to-Talk (PTT) toggle loop with audible earcons and zero-latency audio barge-in.
   - Real-time token streaming to PocketTTS for ~200–300ms Time-To-First-Token (TTFT) speech output.
   - Asynchronous foreground companion and background worker coordination via `CompletionBuffer`.

---

## 2. Architecture & Data Flow

```
                              ┌──────────────┐
                              │     USER     │
                              └──────┬───────┘
                                     │
                              Voice / Terminal PTT
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     VOICE PIPELINE      │
                        │  (PTT Capture -> STT)   │
                        └────────────┬────────────┘
                                     │
                           user_text │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             VERBIX RUNTIME                               │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ FOREGROUND COMPANION                                             │   │
│   │ MiniSWEAgent (config: configs/voice.yaml)                        │   │
│   │ • Persona: Calm, conversational peer, spoken prose.              │   │
│   │ • Tools: [spawn, status, cancel]                                 │   │
│   │ • Streaming: async streaming via litellm.acompletion             │   │
│   └────────────────────────────────┬─────────────────────────────────┘   │
│                                    │ spawn(objective)                    │
│                                    ▼                                     │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ BACKGROUND WORKER TASK                                           │   │
│   │ MiniSWEAgent (config: configs/swe_worker.yaml)                   │   │
│   │ • Persona: Expert SWE-bench software engineer.                   │   │
│   │ • Tools: [bash] (process-group timeout subshell)                 │   │
│   │ • Protocol: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT           │   │
│   └────────────────────────────────┬─────────────────────────────────┘   │
│                                    │ task.result                         │
│                                    ▼                                     │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ COMPLETION BUFFER (Debounce window: 0.3s)                        │   │
│   └────────────────────────────────┬─────────────────────────────────┘   │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │ Direct Announcement
                                     ▼
                        ┌─────────────────────────┐
                        │     AUDIO & DISPLAY     │
                        │  • PocketTTSEngine      │
                        │  • AudioPlayer          │
                        │  • CliPrinter (stdout)  │
                        └─────────────────────────┘
```

---

## 3. Package Structure

The real, literal mini-swe-agent codebase is vendored into `src/verbix/minisweagent/`:

```
src/verbix/
├── cli.py                     # Entry points (text mode + --voice toggle)
├── voice/                     # STT, PocketTTS, AudioPlayer, Sinks, Pipeline
├── ipc/                       # Unix socket server/client daemon
├── runtime/                   # Runtime orchestrator, CompletionBuffer, WorkerTask
└── minisweagent/              # Vendored Mini-SWE-Agent package
    ├── __init__.py
    ├── agents/
    │   ├── default.py         # DefaultAgent core loop
    │   └── interactive.py     # InteractiveAgent
    ├── environments/
    │   ├── local.py           # LocalEnvironment (isolated bash execution)
    │   └── docker.py          # DockerEnvironment (container isolation)
    ├── models/
    │   ├── litellm_model.py   # LitellmModel with async streaming extensions
    │   └── openrouter_model.py# Lightweight direct OpenRouter model
    ├── tools/                 # Dynamic tool registry for extensible tooling
    │   ├── base.py            # Base Tool definition
    │   ├── registry.py        # Named tool lookup registry
    │   └── bash.py            # Mini-SWE bash tool
    ├── config/
    │   ├── voice.yaml         # Foreground receptionist config
    │   ├── swe_worker.yaml    # Autonomous SWE worker config (from mini.yaml)
    │   └── default.yaml
    └── utils/
        ├── serialize.py
        └── log.py
```

### Dependencies
Added to `pyproject.toml`:
- `litellm`: Multi-provider commercial LLM abstraction.
- `jinja2`: Dynamic template rendering for system and instance prompts.
- `pyyaml`: Configuration parsing.
- `tenacity`: Retry handling on rate limits and transient errors.
- `rich`: Formatting utilities.

---

## 4. Universal Tooling System

### 4.1 Declarative Configuration
Instead of hardcoding tools to `[BASH_TOOL]`, `AgentConfig` supports a dynamic list of tool identifiers:

```yaml
agent:
  system_template: "..."
  instance_template: "..."
  tools:
    - spawn
    - status
    - cancel
```

### 4.2 Dynamic Tool Registry
A lightweight registry in `minisweagent.tools.registry` binds named tools to their JSON Schema definitions and execution callables:

```python
class ToolRegistry:
    def register(self, name: str, definition: dict, handler: Callable):
        ...
    def get_definitions(self, tool_names: list[str]) -> list[dict]:
        ...
    async def execute(self, tool_name: str, arguments: dict, context: Any) -> Any:
        ...
```

- When the **Foreground Agent** initializes with `tools: ["spawn", "status", "cancel"]`, it receives schemas and executors bound to `Runtime.spawn`, `Runtime.status`, `Runtime.cancel`.
- When the **Worker Agent** initializes with `tools: ["bash"]`, it executes via `LocalEnvironment.execute()`.
- Future tools (`file_edit`, `ripgrep`, `code_search`, `git`) register with this system without modifying agent loops.

---

## 5. Streaming & Voice Latency Contract

### 5.1 Async Streaming Model Adapter
For voice interaction, waiting for a full response introduces unacceptable latency. `LitellmModel` will provide an async generator interface:

```python
async def query_stream(
    self, messages: list[dict], tools: list[dict] | None = None
) -> AsyncIterator[AgentEvent]:
    # Calls litellm.acompletion(..., stream=True)
    # Yields TextDelta for speech chunks
    # Yields ToolCallDelta for actions
```

### 5.2 Speech Pipeline Integration
1. Tokens emitted by `query_stream` are immediately buffered by sentence in `SpeechTurnOutput`.
2. As soon as a full sentence is formed, PocketTTS begins synthesis and audio playback via `pw-play`.
3. TTFT remains ~200–300ms, preserving native conversational feel.

---

## 6. Worker Execution & Termination Protocol

1. **Autonomous Workflow**: Background workers run Mini-SWE-Agent's proven SWE-bench workflow (Codebase analysis -> Reproducer script -> Edit -> Test -> Verify).
2. **Termination Protocol**:
   The worker finishes by issuing:
   ```bash
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
   <Spoken summary of actions taken and results>
   ```
   `LocalEnvironment._check_finished()` detects this signal and raises `Submitted`.
3. **Announcement**:
   The submission text is assigned to `task.result`, picked up by `CompletionBuffer`, and announced directly to the user over PocketTTS and `CliPrinter`.

---

## 7. Implementation Plan & Milestones

### Milestone 1: Core Vendoring & Dependencies
- Copy `minisweagent/` from `/home/numair/dev/projects/mini-swe-agent/src/minisweagent` into `src/verbix/minisweagent/`.
- Add dependencies (`litellm`, `jinja2`, `pyyaml`, `tenacity`, `rich`) to `pyproject.toml` and sync environment via `uv sync`.
- Verify vendored unit tests pass.

### Milestone 2: Extensible Tool Registry & Async Streaming
- Introduce `minisweagent.tools.registry` for dynamic tool resolution.
- Add `query_stream` with `litellm.acompletion(stream=True)` in `LitellmModel`.
- Support custom `tools:` lists in `AgentConfig`.

### Milestone 3: Configs & Runtime Adapter
- Create `configs/voice.yaml` for foreground voice receptionist.
- Create `configs/swe_worker.yaml` for autonomous SWE workers.
- Replace legacy `AgentSession` in `src/verbix/runtime/runtime.py` and `worker.py` with `MiniSWEAgent`.

### Milestone 4: Verification & Acceptance
- Run full pytest test suite.
- Test interactive terminal conversation: `uv run verbix "Hello"`.
- Test voice loop with background delegation: `uv run verbix --voice` -> "Check the current directory and list files".
