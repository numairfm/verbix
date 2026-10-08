# Tool + Runtime Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One tool per file, yaml list as the only switch, runtime split by agent, engine free of tool names.

**Architecture:** Move code without changing behavior first (tools split, runtime split), then delete the three special cases (foreground caps, finish name-check, bash triple-default) replacing each with a per-tool field or a loud error. Each task is independently testable and committable.

**Tech Stack:** Python 3.14, existing `Tool` protocol (`src/verbix/tools/base.py:33-45`), `ToolRegistry` (`src/verbix/minisweagent/tools/registry.py:9-83`), pytest, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-tool-runtime-reorg-design.md`

## Global Constraints

- One tool bug touches exactly `src/verbix/tools/<name>.py` plus its test — never `default.py` or the runtime coordinator.
- Yaml `tools:` list is the only switch: add works, remove stops, unknown name raises naming the bad key.
- No tool name string (`spawn`, `status`, `cancel`, `finish`, `bash`, `read`, `write`, `edit`) remains in `src/verbix/minisweagent/agents/default.py` outside comments.
- `bwrap` missing still fails closed (`src/verbix/tools/sandbox.py:32-36`).
- Full suite + `ruff check` green before any completion claim; bare-pipe `| tail` forbidden, use `set -o pipefail`.

## Review Focus

- Yaml typo (`tools: [raed]`) raises at build naming `raed` and listing catalog keys, instead of silently offering no tools.
- Worker given `spawn` in yaml actually spawns (dynamic path works both directions, not just the current hardcoded sets).
- `finish` ends the turn through generic `is_terminal` with no `name ==` check anywhere in the engine.
- Announcer with `tools: []` sends `tool_choice="none"` and zero tool definitions; empty means none, not bash-by-default.
- 2.5 MB photo through the new `tools/read.py` still returns a multimodal payload identical to today.

---

### Task 1: Split read/write/edit plus shared helper

**Files:**
- Create: `src/verbix/tools/shared.py`, `src/verbix/tools/read.py`, `src/verbix/tools/write.py`, `src/verbix/tools/edit.py`
- Modify: none (old `src/verbix/tools/filesystem.py` deleted in Task 3 after all importers move)
- Test: `tests/test_tools_split.py`

**Interfaces:**
- Consumes: `ToolContext`, `ToolResult` from `src/verbix/tools/base.py:11-30`
- Produces: `resolve_contained_path(path_str: str, workspace: Path) -> Path` and `record_version(target: Path, raw: bytes, context: ToolContext) -> None` + `check_fresh(target: Path, context: ToolContext) -> str | None` in `shared.py`; `FileReadTool`, `FileWriteTool`, `FileEditTool` classes with identical `name`, `description`, `parameters_schema`, `execute(args, context)` signatures as today (`filesystem.py:33-265`)

- [ ] **Step 1: Write the failing test**

```python
def test_split_tools_importable_from_new_paths():
    from verbix.tools.read import FileReadTool
    from verbix.tools.write import FileWriteTool
    from verbix.tools.edit import FileEditTool
    from verbix.tools.shared import resolve_contained_path
    assert FileReadTool().name == "read"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py::test_split_tools_importable_from_new_paths -q 2>&1 | tail -1`
Expected: FAIL with "No module named 'verbix.tools.read'"

- [ ] **Step 3: Implement by moving code verbatim: `resolve_contained_path` (`filesystem.py:20-30`) plus version-hash lines (`filesystem.py:116,164-184,226-240`) into `shared.py`; each tool class into its own file with zero behavior change**

- [ ] **Step 4: Run new test plus existing filesystem tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py tests/test_miniswe_tools_and_model.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/shared.py src/verbix/tools/read.py src/verbix/tools/write.py src/verbix/tools/edit.py tests/test_tools_split.py
git commit -m "feat: split file tools into one file per tool plus shared helper"
```

### Task 2: Bash becomes a normal registered tool

**Files:**
- Modify: `src/verbix/tools/bash.py` (add `observation_cap = 10000`), `src/verbix/minisweagent/agents/default.py:198-204,224-252,656-685`, `src/verbix/minisweagent/models/litellm_model.py:120`
- Test: `tests/test_tools_split.py` (append)

**Interfaces:**
- Consumes: `SandboxConfig`, `execute_sandboxed` from `src/verbix/tools/sandbox.py:12-26`
- Produces: `BashTool.execute(args: dict, context: ToolContext) -> ToolResult` unchanged; engine exposes no bash default

- [ ] **Step 1: Write the failing test**

```python
def test_engine_offers_no_tools_when_registry_empty():
    from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
    from verbix.minisweagent.tools.registry import ToolRegistry
    agent = DefaultAgent(model=None, env=object(), tools=ToolRegistry(), config=AgentConfig(tools=[]))
    assert agent.get_tool_definitions() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py::test_engine_offers_no_tools_when_registry_empty -q 2>&1 | tail -1`
Expected: FAIL (today returns `[BASH_TOOL]` via the `elif not definitions and self.env is not None` fallback at `default.py:203-204`)

- [ ] **Step 3: Implement: delete `BASH_TOOL` fallback (`default.py:198-204`), delete the `BASH_TOOL` import (`default.py:35`, dead after the fallback goes or ruff F401 fires), delete the `BASH_TOOL` import (`litellm_model.py:29`, dead after the `:120` change below), delete env-execute branch (`default.py:224-252`), delete the sync-path env-first routing plus `"bash"` default (`default.py:656-685`, including `action.get("tool_name", "bash")` at `:664` — sync path routes through the registry like the async path), change `litellm_model.py:120` to `kwargs_dict.pop("tools", None)`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/agents/default.py src/verbix/minisweagent/models/litellm_model.py src/verbix/tools/bash.py tests/test_tools_split.py
git commit -m "fix: bash is a registered tool only, no engine defaults"
```

### Task 3: Catalog plus runtime tools move

**Files:**
- Create: `src/verbix/tools/spawn.py`, `src/verbix/tools/status.py`, `src/verbix/tools/cancel.py`, `src/verbix/tools/finish.py`
- Modify: `src/verbix/tools/__init__.py` (add `TOOL_CATALOG` + `build_registry(names: list[str], runtime: Any = None, workspace: Path | None = None) -> ToolRegistry`, workspace defaulting to cwd so two-arg calls stay valid), `src/verbix/minisweagent/tools/registry.py:32-36` (raise on missing). Old files stay in place this task; deletions plus importer migration happen in Task 5.
- Test: `tests/test_tools_split.py` (append)

**Interfaces:**
- Consumes: tool classes from Task 1-2 plus moved `SpawnTool(runtime)`, `StatusTool(runtime)`, `CancelTool(runtime)`, `FinishTool()` (signatures identical to `runtime/tools.py:12-146`)
- Produces: `TOOL_CATALOG: dict[str, Callable[[Any], Any]]`, `build_registry(names, runtime=None, workspace=None) -> ToolRegistry` raising `KeyError` naming bad key + valid keys; each registry owns one `ToolContext(workspace)` shared by its handlers so the stale guard accumulates per agent session (fixes the proven `missing 'context'` failure: bare `tool.execute` registered today, `aexecute(name, args)` called with no context at `default.py:216`)

- [ ] **Step 1: Write the failing test**

```python
def test_unknown_tool_name_raises_loudly():
    import pytest
    from verbix.tools import build_registry
    with pytest.raises(KeyError, match="raed"):
        build_registry(["raed"], runtime=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py::test_unknown_tool_name_raises_loudly -q 2>&1 | tail -1`
Expected: FAIL with "build_registry not defined"

- [ ] **Step 3: Implement: move the four classes verbatim (runtime-taking constructors kept), write catalog mapping all 8 names, `build_registry` creates one `ToolContext(workspace)` and each handler closes over it (`tool.execute(args, ctx)`), `build_registry` raises on unknown, `get_definitions` raises on missing instead of `if name in self._tools` skip**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/ tests/test_tools_split.py src/verbix/minisweagent/tools/registry.py
git commit -m "feat: tool catalog with loud unknown-name errors"
```

### Task 4: Engine depollution (caps + finish)

**Files:**
- Modify: `src/verbix/minisweagent/agents/default.py:88-97,208-209,456-479`, `src/verbix/tools/spawn.py`, `src/verbix/tools/status.py`, `src/verbix/tools/cancel.py` (add `observation_cap = 1500`), `src/verbix/tools/finish.py` (return `is_terminal=True`), `src/verbix/runtime/runtime.py:440-446` (preserve `ToolResult`)
- Test: `tests/test_tools_split.py` (append)

**Interfaces:**
- Consumes: `ToolResult(observation: str, is_error: bool, is_terminal: bool)` from `base.py:11-14`
- Produces: no `FOREGROUND_TOOL_NAMES`, no `_observation_cap`, no `name == "finish"` in `default.py`; turn ends when `res.is_terminal is True` for any tool

- [ ] **Step 1: Write the failing test**

```python
def test_no_tool_names_in_engine():
    import pathlib
    src = pathlib.Path("src/verbix/minisweagent/agents/default.py").read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
    for name in ("FOREGROUND_TOOL_NAMES", "finish", "spawn", "status", "cancel",
                 "read", "write", "edit", "bash"):
        assert f'"{name}"' not in body and f"'{name}'" not in body, name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py::test_no_tool_names_in_engine -q 2>&1 | tail -1`
Expected: FAIL (names still present)

- [ ] **Step 3: Implement: per-tool `observation_cap` field read generically by `cap_observation` call site; `finish.py` returns `ToolResult(..., is_terminal=True)`; change `_execute_tool` (`default.py:211`, today `-> tuple[str, bool]`) to return the full `ToolResult` so `is_terminal` survives, rewriting both unpack call sites (`default.py:483-511`: `obs, is_err = await ...` becomes `res = await ...` with `obs/is_err/is_terminal` read off `res`); the registry branch (`default.py:214-218`) unwraps `ToolResult.observation/.is_error` directly instead of `json.dumps` on the dataclass; `default.py` checks `is_terminal`; foreground handler returns the `ToolResult` object instead of `res.observation` string**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/agents/default.py src/verbix/tools/ src/verbix/runtime/runtime.py tests/test_tools_split.py
git commit -m "fix: engine knows no tool names, finish ends turns generically"
```

### Task 5: Runtime split by agent

**Files:**
- Create: `src/verbix/runtime/foreground.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/announcer.py`
- Modify: `src/verbix/runtime/runtime.py` (thin facade, same public names), `src/verbix/runtime/tools.py` deleted already in Task 3
- Test: existing `tests/test_runtime_e2e.py`, `tests/test_announcer_config.py` moved imports

**Interfaces:**
- Consumes: `build_registry` from Task 3, `AgentConfig`/`DefaultAgent` from `default.py:46-100`
- Produces: `build_foreground(workspace, provider) -> DefaultAgent`, `build_announcer(workspace, provider) -> DefaultAgent`, `spawn_worker(runtime, objective: str) -> str` with identical behavior to `runtime.py:627-688`; `Runtime.spawn/status/cancel` signatures unchanged

- [ ] **Step 1: Write the failing test**

```python
def test_runtime_modules_exist_with_one_job_header():
    import pathlib
    for mod in ("foreground", "workers", "announcer"):
        text = pathlib.Path(f"src/verbix/runtime/{mod}.py").read_text()
        assert text.splitlines()[0].startswith('"""')
        assert mod in text[:500].lower(), mod
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py::test_runtime_modules_exist_with_one_job_header -q 2>&1 | tail -1`
Expected: FAIL with "No such file"

- [ ] **Step 3: Implement: move foreground build (`runtime.py:386-463`), spawn flow (`runtime.py:627-688`), announcer build + `_speak_batch` + drain (`runtime.py:465-488,718-740,847-887`) verbatim into the three modules; `runtime.py` imports and delegates; keep `Runtime` public API identical; delete `src/verbix/tools/filesystem.py` + `src/verbix/runtime/tools.py` here and migrate every importer (`runtime.py:58`, `test_tools.py`, `test_agent.py:16`, `test_runtime_tools.py`, `test_worker_tools.py`, `test_ipc_and_cli.py:114,125`, `test_miniswe_agent_turn.py:252`) so this commit is whole**

- [ ] **Step 4: Run full suite to verify nothing breaks**

Run: `set -o pipefail; uv run pytest -q 2>&1 | tail -1`
Expected: PASS (all green)

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/ src/verbix/tools/ tests/
git commit -m "feat: runtime split into foreground, workers, announcer"
```

### Task 6: Dynamic-config pins plus final proof

**Files:**
- Modify: `tests/test_tools_split.py` (append), `docs/THIRD-PARTY-NOTICES` if needed (no new deps expected: none)
- Test: full suite + ruff

**Interfaces:**
- Consumes: everything from Tasks 1-5
- Produces: green suite, reviewable file tree

- [ ] **Step 1: Write the failing tests**

```python
class FakeRuntime:
    workspace = Path(".")
    async def spawn(self, objective):
        assert objective
        return "task-99"

def test_worker_with_spawn_tool_actually_spawns():
    import asyncio
    from pathlib import Path
    from verbix.tools import build_registry
    from verbix.tools.spawn import SpawnTool
    from verbix.tools.base import ToolContext
    reg = build_registry(["read", "spawn"], runtime=FakeRuntime(), workspace=Path("."))
    assert "spawn" in [d["function"]["name"] for d in reg.get_definitions()]
    res = asyncio.run(SpawnTool(FakeRuntime()).execute({"objective": "do it"}, ToolContext(workspace=Path("."))))
    assert "task-99" in res.observation


def test_sync_path_routes_through_registry():
    from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
    from verbix.minisweagent.tools.registry import ToolRegistry
    agent = DefaultAgent(model=None, env=object(), tools=ToolRegistry(), config=AgentConfig(tools=[]))
    out = agent.execute_actions({"extra": {"actions": [{"name": "bash", "arguments": {"command": "x"}}]}})
    assert out[0].get("returncode", -1) == -1


def test_worker_read_via_agent_gets_context(tmp_path):
    import asyncio
    from verbix.tools import build_registry
    from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
    (tmp_path / "note.txt").write_text("hello")
    agent = DefaultAgent(model=None, env=None, tools=build_registry(["read"], runtime=None, workspace=tmp_path), config=AgentConfig(tools=["read"]))
    res = asyncio.run(agent._execute_tool("read", {"path": "note.txt"}))
    assert res.is_error is False and "hello" in res.observation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tools_split.py -q 2>&1 | tail -1`
Expected: FAIL until Task 3-5 wiring complete (documents the dream: add-to-yaml works)

- [ ] **Step 3: Implement missing glue the test exposes (catalog entry, runtime injection) — no new files**

- [ ] **Step 4: Run full verification**

Run: `set -o pipefail; uv run pytest -q 2>&1 | tail -1 && uv run ruff check src/verbix/tools/ src/verbix/runtime/ src/verbix/minisweagent/agents/default.py src/verbix/minisweagent/models/litellm_model.py src/verbix/minisweagent/tools/registry.py`
Expected: all green

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools_split.py
git commit -m "test: dynamic tool assignment pins"
```
