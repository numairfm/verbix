# General Worker Architecture and Prompt Invariants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Verbix background worker configuration and prompts into a domain-agnostic execution engine governed by structural prompt invariants with zero few-shot examples.

**Architecture:** Rename `swe_worker.yaml` to `worker.yaml` and replace benchmark-specific phrasing and examples with a general autonomous execution model (Discover $\rightarrow$ Execute $\rightarrow$ Verify $\rightarrow$ Submit). Enforce epistemic delegation boundaries in `SpawnTool.parameters_schema`, `voice.yaml`, and `runtime.py` so the companion describes user intent rather than guessing concrete shell commands.

**Tech Stack:** Python 3.12+, LiteLLM, Mini-SWE-Agent, PyYAML, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-general-worker-and-prompt-invariants.md`

## Global Constraints

- Native Python `asyncio` for all runtime coordination; zero blocking calls on the main loop.
- No few-shot command examples in prompts (e.g. `npm test`, `pytest`, `cargo test`, `nl -ba`, `sed`); enforce structural invariants instead.
- Backwards compatibility: `Runtime.spawn` loads `worker.yaml` by default, falling back gracefully to `swe_worker.yaml` if an external override exists.
- All existing tests (voice pipeline, PTT capture, CompletionBuffer, MiniSWEAgent turns) must pass.
- Clean code quality: 100% passing tests, zero Ruff lint/format errors, zero Pyright type errors.

## Review Focus

1. **Companion Command Guessing:** Spoken user requests lacking command details (e.g., "run tests", "check temperature") must produce high-level intent in `objective`, not hallucinated shell commands.
2. **Worker Overfitting to Examples:** Worker must not default to `npm test` or `sed` when instructed to execute general objectives.
3. **Persona Bleed into Tool Schema:** Roleplay or voice style prompts (e.g. caveman speak) must only affect conversational replies, not tool calling arguments or objective formulations.
4. **General Task Support:** Worker prompt must support non-code tasks (media, system inspection, app launching) without referencing "software engineering" or "SWE-bench".
5. **Config Fallback:** If `worker.yaml` is renamed or accessed via legacy `swe_worker` config specifiers, the system handles it gracefully without crashing.

---

### Task 1: SpawnTool Schema Contract and Tests

**Files:**
- Modify: `src/verbix/runtime/tools.py`
- Modify: `tests/test_runtime_tools.py`

**Interfaces:**
- Consumes: `SpawnTool` in `src/verbix/runtime/tools.py`
- Produces: Updated `SpawnTool.parameters_schema` with epistemic boundary description

- [ ] **Step 1: Write the failing test**

In `tests/test_runtime_tools.py`, add `test_spawn_tool_schema_enforces_intent_invariant`:
```python
def test_spawn_tool_schema_enforces_intent_invariant(runtime: Runtime) -> None:
    tool = SpawnTool(runtime)
    schema = tool.parameters_schema
    desc = schema["properties"]["objective"]["description"]
    assert "high-level goal" in desc.lower() or "intent" in desc.lower()
    assert "never guess" in desc.lower() or "do not guess" in desc.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_tools.py -k "test_spawn_tool_schema_enforces_intent_invariant" -v`
Expected: FAIL with assertion error.

- [ ] **Step 3: Update `SpawnTool.parameters_schema` in `src/verbix/runtime/tools.py`**

Update `objective` property description in `SpawnTool.parameters_schema`:
```python
"objective": {
    "type": "string",
    "description": (
        "The high-level goal and verification criteria for the background worker to accomplish. "
        "State what needs to be done. Never guess or insert specific shell commands, package managers, "
        "or file paths unless explicitly provided by the user."
    ),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_tools.py -k "test_spawn_tool_schema_enforces_intent_invariant" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/tools.py tests/test_runtime_tools.py
git commit -m "feat(runtime): enforce intent delegation invariant in SpawnTool schema"
```

---

### Task 2: Worker Configuration Renaming and Invariant Prompt Engineering

**Files:**
- Create: `src/verbix/minisweagent/config/worker.yaml`
- Delete: `src/verbix/minisweagent/config/swe_worker.yaml`
- Modify: `src/verbix/runtime/runtime.py`
- Create: `tests/test_prompt_invariants.py`

**Interfaces:**
- Consumes: `worker.yaml` via `Runtime.spawn`
- Produces: General-purpose worker prompt with 4-phase invariant (Discover, Execute, Verify, Submit) and zero command examples

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_invariants.py`:
```python
from pathlib import Path
import yaml
from verbix.runtime.runtime import DEFAULT_WORKER_PROMPT

def test_worker_config_exists_and_swe_worker_removed():
    config_dir = Path("src/verbix/minisweagent/config")
    worker_yaml = config_dir / "worker.yaml"
    assert worker_yaml.is_file(), "worker.yaml must exist"

def test_worker_prompt_contains_zero_examples_and_has_invariants():
    config_dir = Path("src/verbix/minisweagent/config")
    worker_data = yaml.safe_load((config_dir / "worker.yaml").read_text(encoding="utf-8"))
    prompt = worker_data["agent"]["instance_template"] + " " + worker_data["agent"]["system_template"]
    
    # Assert zero command examples
    forbidden_examples = ["npm test", "cargo test", "nl -ba", "sed -n", "pytest"]
    for ex in forbidden_examples:
        assert ex not in prompt, f"Forbidden example found in worker prompt: {ex}"
        
    # Assert general execution invariants
    assert "Discover" in prompt
    assert "Execute" in prompt
    assert "Verify" in prompt
    assert "Submit" in prompt
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_invariants.py -v`
Expected: FAIL (`worker.yaml` does not exist)

- [ ] **Step 3: Implement `worker.yaml` and update `runtime.py`**

1. Create `src/verbix/minisweagent/config/worker.yaml` without command examples, framing the worker as a general autonomous background worker executing tasks for Verbix.
2. Remove `src/verbix/minisweagent/config/swe_worker.yaml`.
3. In `src/verbix/runtime/runtime.py`:
   - Update `DEFAULT_WORKER_PROMPT`:
     ```python
     DEFAULT_WORKER_PROMPT = """You are an autonomous background worker executing tasks for Verbix.
     Discover the necessary environment and tools, execute actions directly, verify outcomes, and submit a clear factual summary of results upon completion."""
     ```
   - In `Runtime.spawn`, resolve `worker.yaml` first, with fallback to `swe_worker.yaml`:
     ```python
     worker_path = (
         Path(__file__).resolve().parent.parent
         / "minisweagent"
         / "config"
         / "worker.yaml"
     )
     if not worker_path.is_file():
         worker_path = (
             Path(__file__).resolve().parent.parent
             / "minisweagent"
             / "config"
             / "swe_worker.yaml"
         )
     ```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_invariants.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/config/ src/verbix/runtime/runtime.py tests/test_prompt_invariants.py
git commit -m "feat(worker): rename swe_worker to worker and enforce general execution invariants"
```

---

### Task 3: Foreground Companion and Announcer Prompt Invariants

**Files:**
- Modify: `src/verbix/minisweagent/config/voice.yaml`
- Modify: `src/verbix/runtime/runtime.py`
- Modify: `tests/test_prompt_invariants.py`

**Interfaces:**
- Consumes: `voice.yaml` and `DEFAULT_VERBIX_PROMPT`
- Produces: Epistemic boundary and persona orthogonality invariants in companion prompts

- [ ] **Step 1: Write the failing test**

In `tests/test_prompt_invariants.py`, add `test_companion_prompt_invariants`:
```python
def test_companion_prompt_invariants():
    config_dir = Path("src/verbix/minisweagent/config")
    voice_data = yaml.safe_load((config_dir / "voice.yaml").read_text(encoding="utf-8"))
    sys_prompt = voice_data["agent"]["system_template"]
    
    assert "never guess" in sys_prompt.lower() or "do not guess" in sys_prompt.lower()
    assert "objective" in sys_prompt.lower()
    assert "spawn" in sys_prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_invariants.py -k "test_companion_prompt_invariants" -v`
Expected: FAIL

- [ ] **Step 3: Update `voice.yaml` and `DEFAULT_VERBIX_PROMPT` in `runtime.py`**

Add epistemic boundary and persona orthogonality invariants to `voice.yaml` and `DEFAULT_VERBIX_PROMPT`:
```yaml
    # CAPABILITY ROUTING & DELEGATION
    - Answer quick conversational and general knowledge questions directly.
    - Delegate multi-step tasks, background operations, desktop control, system commands, or file interactions to spawn.
    - You have no direct access to the filesystem, shell, or execution environment.
    - When calling spawn, formulate the objective as the user's intended goal and expected outcome.
    - Never guess, invent, or substitute concrete shell commands, scripts, package managers, or file paths into the objective unless the user explicitly specified them.
    - Persona, voice, and style rules apply strictly to spoken text spoken to the user; tool arguments and objectives must always remain exact, factual, and unstylized.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompt_invariants.py -k "test_companion_prompt_invariants" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/config/voice.yaml src/verbix/runtime/runtime.py tests/test_prompt_invariants.py
git commit -m "feat(companion): add epistemic boundary and persona orthogonality invariants"
```

---

### Task 4: Full Suite Verification & Backwards-Compatibility Check

**Files:**
- Modify: `tests/test_miniswe_agent_turn.py`
- Test: All tests across `tests/`

**Interfaces:**
- Consumes: `get_config_from_spec("worker")` and `get_config_from_spec("swe_worker")`
- Produces: 100% green test suite across legacy and new configurations

- [ ] **Step 1: Update `tests/test_miniswe_agent_turn.py`**

In `tests/test_miniswe_agent_turn.py`, update tests that imported `swe_worker` config to test `worker` config, and verify `swe_worker` falls back to `worker` in `get_config_from_spec` if desired, or test `worker` directly.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS all tests (80+ passing, 0 failures).

- [ ] **Step 3: Run linters and type checkers**

Run: `uv run ruff check src/ tests/`
Expected: PASS (no violations)
Run: `uv run pyright src/ tests/`
Expected: PASS (0 errors, 0 warnings)

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(worker): update turn tests for worker config and verify clean suite"
```
