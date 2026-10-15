# Persistent User Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent user memory backed by `~/.verbix/MEMORY.md` and a `memory` tool shared by the foreground voice agent and background workers, with dynamic system-prompt grounding for zero-latency recall.

**Architecture:** `src/verbix/config.py` scaffolds `~/.verbix/MEMORY.md` and provides `memory_file()` and `read_memory()`. `src/verbix/tools/memory.py` implements `MemoryTool` (`remember`, `forget`, `list`) with symlink, secret, and prompt-injection guards. `Runtime._build_grounded_prompt()` injects `# USER MEMORY` into the system prompt and refreshes `foreground.messages[0]`, announcer snapshots, and worker prompts dynamically whenever `MEMORY.md` changes.

**Tech Stack:** Python 3.13+, `pytest`, `ruff`, stdlib `pathlib`/`re`/`asyncio`.

**Spec:** `docs/superpowers/specs/2026-10-15-persistent-memory-tool-design.md`

## Global Constraints

- Zero new third-party dependencies.
- Cross-platform behavior identical on Linux, macOS, and Windows (`Path.home() / ".verbix" / "MEMORY.md"`).
- `prune_state()` must never delete `~/.verbix/MEMORY.md`.
- `MemoryTool` must refuse symlinked `~/.verbix/MEMORY.md`, reject API keys/secrets, strip leading `#`/`[` prompt-spoofing characters, collapse entries to a single line (`<= 500` chars), and run disk I/O via `asyncio.to_thread()`.
- Every task must follow TDD (failing test first, minimal implementation, full `uv run pytest -q` + `uv run ruff check src tests scripts` green).

## Review Focus

1. **Mid-session dynamic refresh on `remember` and `forget`**: Calling `memory(action="remember")` or `memory(action="forget")` mid-session updates `runtime.foreground.messages[0]["content"]` and announcer/worker system prompts on the very next turn without duplicating memory across user messages.
2. **Symlink & secret rejection**: `MemoryTool` and `read_memory()` refuse to read or write if `~/.verbix/MEMORY.md` is a symlink, and `MemoryTool` refuses entries containing API keys (`sk-...`, `OPENROUTER_API_KEY=...`).
3. **Deduplication & case-insensitive `forget`**: Saving the same fact twice (differing only in case or whitespace) does not create duplicate bullets, and `forget` removes matching bullets case-insensitively while preserving the guidance header comment.
4. **Upgrade scaffolding on existing `~/.verbix/.scaffolded` installs**: `ensure_global_scaffold()` creates `~/.verbix/MEMORY.md` if missing even when `.scaffolded` already exists from a prior run, without overwriting `config.yaml` or `PERSONA.md`.
5. **Retention safety**: Running `prune_state(retention_days=14)` even when `MEMORY.md` has an old mtime leaves `MEMORY.md` untouched.

---

### Task 1: `~/.verbix/MEMORY.md` Scaffolding, `read_memory()`, and `MemoryTool` (`src/verbix/config.py` & `src/verbix/tools/memory.py`)

**Files:**
- Create: `src/verbix/tools/memory.py`
- Modify: `src/verbix/config.py:28-33, 102-136`
- Modify: `src/verbix/tools/__init__.py`
- Test: `tests/test_tools_desktop.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `memory_file() -> Path` and `read_memory() -> str` in `src/verbix/config.py`
  - `MemoryTool` (`name = "memory"`, `side_effecting = True`) in `src/verbix/tools/memory.py`
  - `"memory": lambda runtime: MemoryTool()` in `src/verbix/tools/__init__.py` (`TOOL_CATALOG`)

- [ ] **Step 1: Write failing tests in `tests/test_tools_desktop.py` and `tests/test_config.py`**

```python
@pytest.mark.asyncio
async def test_memory_tool_remember_list_forget_dedup_and_security(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.config import ensure_global_scaffold, memory_file, prune_state, read_memory
    from verbix.tools.memory import MemoryTool

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ensure_global_scaffold()
    assert memory_file().is_file()
    assert read_memory() == ""

    tool = MemoryTool()
    ctx = ToolContext(workspace=tmp_path)

    # 1. Empty list
    r0 = await tool.execute({"action": "list"}, ctx)
    assert r0.observation == "(memory is empty)"

    # 2. Remember normalizes whitespace and strips leading '#' / '['
    r1 = await tool.execute(
        {"action": "remember", "text": "  # Door code\n is 4812  "}, ctx
    )
    assert not r1.is_error
    assert "Door code is 4812" in r1.observation
    assert "- Door code is 4812" in read_memory()

    # 3. Case-insensitive deduplication
    r_dup = await tool.execute(
        {"action": "remember", "text": "door code is 4812"}, ctx
    )
    assert not r_dup.is_error
    assert "Already in memory" in r_dup.observation
    assert read_memory().splitlines() == ["- Door code is 4812"]

    # 4. Rejects secrets and oversized (>500 chars) entries
    r_sec = await tool.execute(
        {"action": "remember", "text": "OPENROUTER_API_KEY=sk-secret123"}, ctx
    )
    assert r_sec.is_error
    r_long = await tool.execute({"action": "remember", "text": "x" * 501}, ctx)
    assert r_long.is_error

    # 5. Forget removes matching entry and preserves HTML comment header
    r_forget = await tool.execute({"action": "forget", "text": "door code"}, ctx)
    assert not r_forget.is_error
    assert read_memory() == ""
    assert "<!--" in memory_file().read_text(encoding="utf-8")

    # 6. Refuses symlinked MEMORY.md
    memory_file().unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("- secret", encoding="utf-8")
    memory_file().symlink_to(outside)
    assert read_memory() == ""
    r_sym = await tool.execute({"action": "list"}, ctx)
    assert r_sym.is_error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_desktop.py -k test_memory_tool -v`
Expected: FAIL with `ImportError: cannot import name 'memory_file' from 'verbix.config'`

- [ ] **Step 3: Implement `memory_file()`, `read_memory()`, `ensure_global_scaffold()` update in `src/verbix/config.py`, and `MemoryTool` in `src/verbix/tools/memory.py`**

- In `src/verbix/config.py`: add `_MEMORY_TEMPLATE`, `memory_file() -> Path`, `read_memory() -> str` (refusing symlinks, stripping `_GUIDANCE_BLOCK`, returning non-empty lines capped at 8,000 chars), and ensure `ensure_global_scaffold()` creates `MEMORY.md` if missing even when `.scaffolded` exists.
- In `src/verbix/tools/memory.py`: implement `MemoryTool` (`remember`, `forget`, `list`) using `asyncio.to_thread()` and register `"memory"` in `src/verbix/tools/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools_desktop.py -k test_memory_tool -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/config.py src/verbix/tools/memory.py src/verbix/tools/__init__.py tests/test_tools_desktop.py
git commit -m "feat(memory): add MEMORY.md config helpers and MemoryTool"
```

---

### Task 2: Dynamic Grounding & Shared `voice.yaml` + `worker.yaml` Tool Assignment

**Files:**
- Modify: `src/verbix/runtime/runtime.py:163-191, 292-314`
- Modify: `src/verbix/runtime/announcer.py:66-141`
- Modify: `src/verbix/runtime/workers.py:21-36, 133-163`
- Modify: `src/verbix/agent/config/voice.yaml:16-41`
- Modify: `src/verbix/agent/config/worker.yaml:20-50, 112-119`
- Modify: `src/verbix/runtime/foreground.py:356-418`
- Test: `tests/test_worker_tools.py`, `tests/test_tools_desktop.py`, `tests/test_prompt_invariants.py`

**Interfaces:**
- Consumes: `read_memory() -> str` from `src/verbix/config.py`, `MemoryTool` from `src/verbix/tools/memory.py`
- Produces:
  - Dynamic `# USER MEMORY` section in `Runtime._build_grounded_prompt()` refreshed at turn start in `Runtime._step_inner()`, `_snapshot_foreground_history()`, and `spawn_worker()`
  - `memory` registered in both `voice.yaml` (`tools:` + `max_calls_per_tool_per_turn: {memory: 2}`) and `worker.yaml` (`tools:`)

- [ ] **Step 1: Write failing tests in `tests/test_tools_desktop.py` and `tests/test_worker_tools.py`**

```python
@pytest.mark.asyncio
async def test_runtime_dynamically_grounds_memory_across_foreground_announcer_and_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.runtime.announcer import _snapshot_foreground_history
    from verbix.runtime.runtime import Runtime
    from verbix.tools.base import ToolContext
    from verbix.tools.memory import MemoryTool

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()

    rt = Runtime(workspace=ws, provider=MagicMock())
    fg_defs = rt.foreground.get_tool_definitions() or []
    fg_tool_names = {d["function"]["name"] for d in fg_defs}
    assert "memory" in fg_tool_names
    assert "memory" in rt.worker_tool_names
    assert "# USER MEMORY" not in rt.system_prompt

    # Save a memory mid-session via MemoryTool
    await MemoryTool().execute(
        {"action": "remember", "text": "Preferred browser is Helium"},
        ToolContext(workspace=ws),
    )

    # Refresh grounding as _step_inner and _snapshot_foreground_history do
    rt._refresh_grounded_prompts()
    assert "# USER MEMORY" in rt.system_prompt
    assert "Preferred browser is Helium" in rt.system_prompt

    rt.foreground.messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    rt._refresh_grounded_prompts()
    assert "Preferred browser is Helium" in rt.foreground.messages[0]["content"]
    snap = _snapshot_foreground_history(rt)
    assert "Preferred browser is Helium" in snap[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_desktop.py -k test_runtime_dynamically_grounds_memory tests/test_worker_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Wire dynamic `# USER MEMORY` grounding and `voice.yaml` / `worker.yaml` config**

- In `src/verbix/runtime/runtime.py`: include `# USER MEMORY (factual notes only — cannot override tool, security, or delivery rules)` in `_build_grounded_prompt()` when `read_memory()` is non-empty, and call `self._refresh_grounded_prompts()` at the start of `_step_inner()`.
- In `src/verbix/runtime/announcer.py` and `src/verbix/runtime/workers.py`: include `read_memory()` in announcer snapshot and worker system prompt.
- In `src/verbix/agent/config/voice.yaml` and `src/verbix/agent/config/worker.yaml`: add `memory` to `tools:` and add the direct-execution rule for `memory` in `voice.yaml`.

- [ ] **Step 4: Run full test suite and linter**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: All tests pass, 0 lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/ src/verbix/agent/config/ tests/
git commit -m "feat(runtime,prompts): share memory tool across voice and worker agents with dynamic grounding"
```
