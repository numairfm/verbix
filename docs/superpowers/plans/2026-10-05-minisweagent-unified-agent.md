# Mini-SWE-Agent Unified Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the core Mini-SWE-Agent package into `src/verbix/minisweagent/`, add LiteLLM multi-provider support, and wire it cleanly as the unified agent engine powering both the voice receptionist and background coding workers.

**Architecture:** A streamlined vendoring of Mini-SWE-Agent with unnecessary benchmark harness files stripped. A simple 30-line `ToolRegistry` handles dynamic tools. `LitellmModel` provides async streaming (`acompletion(stream=True)`) for voice, while `LocalEnvironment` provides subshell bash execution for workers.

**Tech Stack:** Python 3.14, `litellm`, `jinja2`, `pyyaml`, `tenacity`, `pydantic`, `pytest`, `asyncio`.

**Spec:** [`docs/superpowers/specs/2026-10-05-minisweagent-unified-agent-architecture.md`](file:///home/numair/dev/projects/verbix/docs/superpowers/specs/2026-10-05-minisweagent-unified-agent-architecture.md)

## Global Constraints
- Native Python `asyncio` for all runtime coordination; zero blocking calls on the main loop.
- No unrequested abstractions: keep the vendored package lean; omit unused benchmark harnesses (Docker, Singularity, benchmarks).
- Time-to-first-token in voice mode must remain ~200–300ms using streaming LiteLLM.
- All existing tests (voice pipeline, PTT capture, CompletionBuffer) must pass.

## Review Focus
1. Provider credential errors: LitellmModel missing API keys raises clear, user-visible errors rather than silent hangs.
2. Voice latency regression: First sentence from `query_stream` must yield immediately on sentence boundary without waiting for turn completion.
3. Tool execution errors in bash: Non-zero exit codes from bash must be wrapped in XML observation tags and fed back to the model without crashing the agent.
4. Process termination: `LocalEnvironment` kills process groups on timeout so child processes never orphan.
5. Task result propagation: Worker completion with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` correctly assigns `task.result` and triggers `CompletionBuffer`.

---

### Task 1: Dependencies and Core Package Vendoring

**Files:**
- Modify: `pyproject.toml`
- Create: `src/verbix/minisweagent/__init__.py`
- Create: `src/verbix/minisweagent/exceptions.py`
- Create: `src/verbix/minisweagent/environments/local.py`
- Create: `src/verbix/minisweagent/utils/serialize.py`
- Test: `tests/test_miniswe_environment.py`

**Interfaces:**
- Produces: `LocalEnvironment(cwd: str, timeout: int)` with `.execute(action: dict) -> dict`

- [ ] **Step 1: Write the failing test for LocalEnvironment execution**

```python
import pytest
from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.minisweagent.exceptions import Submitted

def test_local_environment_echo(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    res = env.execute({"command": "echo 'hello world'"})
    assert res["returncode"] == 0
    assert "hello world" in res["output"]

def test_local_environment_submission(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    with pytest.raises(Submitted) as exc_info:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nDone!"})
    assert "Done!" in exc_info.value.messages[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_miniswe_environment.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'verbix.minisweagent')

- [ ] **Step 3: Add dependencies and vendor core environment files**

Add `litellm`, `jinja2`, `pyyaml`, `tenacity` to `pyproject.toml`.
Copy `exceptions.py`, `environments/local.py`, and `utils/serialize.py` from `/home/numair/dev/projects/mini-swe-agent/src/minisweagent/` into `src/verbix/minisweagent/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_miniswe_environment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/verbix/minisweagent/ tests/test_miniswe_environment.py
git commit -m "feat(miniswe): vendor core exceptions and local environment"
```

---

### Task 2: Extensible Tool Registry and Streaming LitellmModel

**Files:**
- Create: `src/verbix/minisweagent/tools/registry.py`
- Create: `src/verbix/minisweagent/models/litellm_model.py`
- Test: `tests/test_miniswe_tools_and_model.py`

**Interfaces:**
- Produces: `ToolRegistry` with `.register(name, definition, handler)` and `.get_definitions(names)`.
- Produces: `LitellmModel` with async generator `query_stream(messages, tools) -> AsyncIterator[AgentEvent]`.

- [ ] **Step 1: Write failing test for ToolRegistry and LitellmModel**

```python
import pytest
from verbix.minisweagent.tools.registry import ToolRegistry

def test_tool_registry():
    reg = ToolRegistry()
    reg.register("greet", {"type": "function", "function": {"name": "greet"}}, lambda args: "hello " + args["name"])
    defs = reg.get_definitions(["greet"])
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "greet"
    assert reg.execute("greet", {"name": "world"}) == "hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_miniswe_tools_and_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ToolRegistry and LitellmModel with async streaming**

Implement minimal `ToolRegistry` (~25 LOC).
Implement `LitellmModel` using `litellm.acompletion(stream=True)` yielding `TextDelta` and `ToolCallDelta`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_miniswe_tools_and_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/tools/ src/verbix/minisweagent/models/ tests/test_miniswe_tools_and_model.py
git commit -m "feat(miniswe): add tool registry and streaming litellm model"
```

---

### Task 3: Mini-SWE Agent Core and YAML Configurations

**Files:**
- Create: `src/verbix/minisweagent/agents/default.py`
- Create: `src/verbix/minisweagent/config/voice.yaml`
- Create: `src/verbix/minisweagent/config/swe_worker.yaml`
- Test: `tests/test_miniswe_agent_turn.py`

**Interfaces:**
- Produces: `DefaultAgent(model, env, tools, config)` with `.step_stream(user_input)` yielding events.

- [ ] **Step 1: Write failing test for DefaultAgent turn**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from verbix.minisweagent.agents.default import DefaultAgent
from verbix.core.events import TextDelta, TurnComplete

@pytest.mark.asyncio
async def test_agent_step_stream_yields_text():
    mock_model = MagicMock()
    async def mock_stream(messages, tools=None):
        yield TextDelta(text="Hello from MiniSWE!")
        yield TurnComplete(final_text="Hello from MiniSWE!")
    mock_model.query_stream = mock_stream

    agent = DefaultAgent(model=mock_model, system_prompt="You are Verbix.")
    events = [e async for e in agent.step_stream("hi")]
    assert any(isinstance(e, TextDelta) and "Hello from MiniSWE!" in e.text for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_miniswe_agent_turn.py -v`
Expected: FAIL

- [ ] **Step 3: Implement DefaultAgent with streaming step and YAML configs**

Port `DefaultAgent` from mini-swe-agent with native async event forwarding. Add `voice.yaml` and `swe_worker.yaml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_miniswe_agent_turn.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/agents/ src/verbix/minisweagent/config/ tests/test_miniswe_agent_turn.py
git commit -m "feat(miniswe): implement streaming DefaultAgent and configs"
```

---

### Task 4: Runtime Integration & Verification

**Files:**
- Modify: `src/verbix/runtime/runtime.py`
- Modify: `src/verbix/runtime/worker.py`
- Modify: `src/verbix/cli.py`
- Test: `tests/test_runtime_e2e.py`

**Interfaces:**
- Connects: `Runtime.foreground` and `WorkerTask` to use `DefaultAgent` instances.

- [ ] **Step 1: Write integration test verifying MiniSWEAgent runs in Runtime**

```python
@pytest.mark.asyncio
async def test_runtime_uses_miniswe_agent(tmp_path):
    from verbix.runtime.runtime import Runtime
    from verbix.provider.client import ProviderClient
    # Ensure runtime foreground is powered by DefaultAgent
    runtime = Runtime(workspace=tmp_path, provider=ProviderClient())
    assert hasattr(runtime.foreground, "config") or hasattr(runtime.foreground, "step_stream")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_e2e.py -k "test_runtime_uses_miniswe_agent" -v`
Expected: FAIL

- [ ] **Step 3: Wire Runtime and WorkerTask to use MiniSWEAgent**

Replace legacy `AgentSession` invocations in `Runtime.__init__` and `WorkerTask` with `DefaultAgent`.

- [ ] **Step 4: Run full test suite to verify all 40+ tests pass**

Run: `uv run pytest -v`
Run: `uv run ruff check src/ tests/`
Run: `uv run pyright src/ tests/`
Expected: All tests pass, 0 lints, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/ src/verbix/cli.py tests/
git commit -m "feat(runtime): wire MiniSWEAgent into Runtime and background workers"
```
