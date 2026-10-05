# Specification: General Worker Architecture and Prompt Invariants

## 1. Overview & Motivation

Verbix is a voice-operated desktop assistant and agent platform. It is designed to be a calm, capable peer that assists the user with general desktop operations: researching information, launching applications, controlling media, inspecting system state, running shell scripts, and developing software.

### The Problem
During recent interactive voice testing, a failure mode emerged:
1. When asked to "run tests in my project here", the conversational companion spawned `spawn({'objective': 'npm test'})` in a Python project.
2. The companion lacked an **epistemic boundary invariant**: having no filesystem tools, it guessed a concrete shell command rather than describing the user's intent.
3. The background worker configuration was named `swe_worker.yaml` and contained hardcoded command examples (`npm test`, `pytest`, `cargo test`, `nl -ba`, `sed`). Models overfit to few-shot command examples in prompts, biasing execution toward those commands regardless of the environment.
4. The system over-indexed on software engineering benchmarks (`swe_` prefix and code-only framing), contradicting Verbix's core identity as a general-purpose desktop companion capable of general tasks (media control, app launching, system administration, research, and coding).

### Core Goals
1. **General-Purpose Worker Identity**: Rename `swe_worker.yaml` to `worker.yaml` and eliminate SWE-bench benchmark artifacts.
2. **Structural Prompt Invariants**: Eliminate few-shot command examples. Define the general autonomous execution model: **Discover $\rightarrow$ Execute $\rightarrow$ Verify $\rightarrow$ Submit**.
3. **Strict Epistemic Delegation**: Mandate that the companion delegates declarative goals and acceptance criteria, never guessing or substituting shell commands, tools, or file paths not specified by the user.
4. **Persona Orthogonality**: Ensure that custom system prompts (such as voice personas or style constraints) govern only conversational prose spoken to the user, keeping tool calling and task arguments uncorrupted.

---

## 2. Core Prompt Invariants

### Invariant 1: Epistemic Boundary Invariant (Companion cannot assume environment state)
The conversational companion has access only to task control tools (`spawn`, `status`, `cancel`). It has zero direct access to the filesystem, shell, or installed applications.
- **Rule**: The companion must never guess, assume, or invent specific commands, tools, frameworks, package managers, or file paths into the `objective`.
- **Rule**: The companion must formulate the `objective` as the user's intended outcome and constraints, delegating discovery and execution to the worker.

### Invariant 2: Objective Contract Invariant (`objective` is an Intent Specification)
In `SpawnTool.parameters_schema`:
- **Rule**: The `objective` parameter is a declarative description of the goal, acceptance criteria, and user constraints. It is not a single shell command or script.
- **Rule**: The parameter schema description must explicitly instruct the model to describe what to accomplish and verify, prohibiting command guessing.

### Invariant 3: General Autonomous Execution Invariant (Discover $\rightarrow$ Execute $\rightarrow$ Verify $\rightarrow$ Submit)
In `worker.yaml`:
- **Rule**: Zero few-shot command examples.
- **Rule**: The worker prompt defines the general execution loop applicable to any domain (media, desktop apps, research, scripts, code):
  1. **Discover**: Inspect relevant environment state, system configuration, files, or tools before acting.
  2. **Execute**: Carry out actions directly via available tools to accomplish the objective.
  3. **Verify**: Check outcomes (outputs, exit codes, process state) against the objective.
  4. **Submit**: Conclude with a factual summary of actions and results.

### Invariant 4: Persona Orthogonality Invariant (Style vs. Tool Semantics)
- **Rule**: Persona, tone, and spoken formatting rules govern only conversational prose spoken to the user.
- **Rule**: Tool calls, parameter schemas, and task objectives must remain precise, factual, and unstylized regardless of active personas.

---

## 3. Detailed Component Changes

### 3.1 `src/verbix/runtime/tools.py` (`SpawnTool`)
Update `parameters_schema` for `objective`:
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

### 3.2 `src/verbix/minisweagent/config/worker.yaml` (Renamed from `swe_worker.yaml`)
- Delete lines 36–51 containing few-shot examples (`npm test`, `pytest`, `cargo test`, `nl -ba`, `sed`, `anything`).
- Remove code-only framing. Frame the worker as an autonomous worker executing general tasks for Verbix.
- Enforce the 4-phase execution loop: Discover, Execute, Verify, Submit.

### 3.3 `src/verbix/minisweagent/config/voice.yaml` & `DEFAULT_VERBIX_PROMPT`
- Update `# CAPABILITY ROUTING` and `# TOOL OPERATIONS & BACKGROUND WORK` sections to explicitly establish:
  - Direct answers for quick conversational questions.
  - Delegation to `spawn` for multi-step tasks, background operations, system commands, or file interactions.
  - Invariant that the companion describes the user's intent and never guesses commands or paths.
  - Invariant that persona/spoken styling applies only to speech, never tool arguments.

### 3.4 `src/verbix/runtime/runtime.py`
- In `Runtime.spawn`:
  - Check for `worker.yaml` first in `src/verbix/minisweagent/config/`.
  - Fallback to `swe_worker.yaml` if present for backwards compatibility.
- Update `DEFAULT_WORKER_PROMPT` to mirror general execution invariants.
- Update `DEFAULT_VERBIX_PROMPT` to mirror `voice.yaml`.

---

## 4. Testing & Verification

1. **Schema & Contract Tests** (`tests/test_runtime_tools.py`):
   - Assert `SpawnTool.parameters_schema` documents the intent specification invariant and prohibits command guessing.
2. **Prompt Invariant Tests** (`tests/test_prompt_invariants.py`):
   - Verify `worker.yaml` exists and `swe_worker.yaml` is retired or alias-redirected.
   - Verify zero few-shot command examples (`npm test`, `cargo test`, `nl -ba`, `sed`) in `worker.yaml`, `voice.yaml`, `runtime.py`.
   - Verify presence of Discover, Execute, Verify, Submit invariants in `worker.yaml` and `DEFAULT_WORKER_PROMPT`.
   - Verify `Runtime.spawn` loads `worker.yaml` by default.
3. **Regression Tests**:
   - Full test suite passes: `pytest tests/test_runtime_tools.py tests/test_miniswe_agent_turn.py tests/test_runtime_e2e.py`.
