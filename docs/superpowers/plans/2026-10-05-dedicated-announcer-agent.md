# Dedicated Announcer Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent autonomous tool execution loops and history pollution during background task completion announcements by introducing a dedicated tool-less `announcer` agent.

**Architecture:** A lightweight `announcer.yaml` configuration defines a pure-synthesis agent (`tools: []`). On buffer flush, `Runtime` delegates synthesis to `self.announcer` under `_turn_lock`, then records only the clean spoken result into `foreground.messages` as an assistant turn.

**Tech Stack:** Python 3.13+, AsyncIO, PyYAML, MiniSWE-Agent (`DefaultAgent`, `AgentConfig`).

**Spec:** Scalpel feature slice in conversation `bc1c5c67-4bd3-46a0-ac3a-50abbffb44f0`.

## Global Constraints
- `announcer.yaml` MUST declare `tools: []`.
- `announcer` MUST NOT have access to bash, spawn, status, or cancel tools.
- `_on_buffer_flush` MUST commit only `{"role": "assistant", "content": spoken_text}` to `foreground.messages`.
- Preserved behavior: 300ms buffer debounce window, turn lock synchronization, and silent failure logging.

## Review Focus
1. Synthesis failure: Announcer fails after retries -> logged silently, no raw output blurted.
2. Tool execution attempt: Announcer prompt receives text suggesting tools -> cannot call tools because none exist in its schema.
3. Clean companion history: Companion messages contain only real user turns and assistant responses, never internal bracketed wake prompts.

---

### Task 1: Dedicated Announcer YAML Configuration

**Files:**
- Create: `src/verbix/minisweagent/config/announcer.yaml`

**Interfaces:**
- Produces: YAML config loaded by `Runtime` with `tools: []` and spoken prose prompt.

- [ ] **Step 1: Create `src/verbix/minisweagent/config/announcer.yaml`**
```yaml
agent:
  system_template: |
    You are Verbix's voice announcer. Your only job is to announce background task completion results directly to the user in spoken, conversational prose.
    - Write for listening: lead with the answer and speak in plain, continuous conversational prose. Never use markdown syntax, asterisks, bullet points (* or -), backticks, code blocks, tables, headings, or numbered lists.
    - Be concise. Give the shortest response that completely satisfies the update; when the task produced detailed results, explain them clearly without technical syntax.
    - Never call or mention tools. Speak answers and messages directly to the user.
  instance_template: "{{task}}"
  tools: []
  step_limit: 1
  cost_limit: 3.0
model:
  model_kwargs:
    drop_params: true
    num_retries: 3
```

- [ ] **Step 2: Commit**
```bash
git add src/verbix/minisweagent/config/announcer.yaml
git commit -m "feat(config): add tool-less announcer agent configuration"
```

---

### Task 2: Wire Announcer into Runtime Buffer Flush & Test TDD

**Files:**
- Modify: `src/verbix/runtime/runtime.py`
- Modify: `tests/test_runtime_e2e.py`

**Interfaces:**
- Consumes: `announcer.yaml` via `AgentConfig` and `DefaultAgent`.
- Modifies: `Runtime._on_buffer_flush` to run `self.announcer.step_stream`.

- [ ] **Step 1: Write failing test in `tests/test_runtime_e2e.py`**
Test that `runtime.announcer` exists, has `tools == []`, that buffer flush uses announcer, and that `foreground.messages` contains ONLY the assistant announcement without any bracketed `[Background task completion update...]` user prompt.

- [ ] **Step 2: Run test to verify RED**
```bash
uv run pytest tests/test_runtime_e2e.py -k "test_runtime_announcer_clean_history" -v
```

- [ ] **Step 3: Implement `self.announcer` and `_on_buffer_flush` in `src/verbix/runtime/runtime.py`**
Initialize `self.announcer` in `Runtime.__init__` from `announcer.yaml`. In `_on_buffer_flush`:
```python
async for event in self.announcer.step_stream(wake_prompt):
    ...
if final_text:
    self.foreground.messages.append({"role": "assistant", "content": final_text})
```

- [ ] **Step 4: Run test to verify GREEN**
```bash
uv run pytest tests/test_runtime_e2e.py -k "test_runtime_announcer_clean_history" -v
```

- [ ] **Step 5: Run full test suite & lint verification**
```bash
uv run pytest && uv run ruff check src/ tests/ && uv run pyright src/ tests/
```

- [ ] **Step 6: Commit**
```bash
git add src/verbix/runtime/runtime.py tests/test_runtime_e2e.py
git commit -m "feat(runtime): delegate completion synthesis to dedicated tool-less announcer"
```
