# Tool Output Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tool results carry photos as attachments, caps touch text only, overflow spills to disk re-readable, limits come from config.

**Architecture:** Extend `ToolResult` with attachments, bind and spill text in the registry handler that owns the workspace, thread yaml limits through the two registry call sites, delete every superseded cap, then attach image parts at the engine kill sites plus every str-only consumer.

**Tech Stack:** Python 3.14, stdlib plus Pillow (new dependency, added via `uv add pillow`) for downscale, pytest, ruff. No other new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-tool-output-rework-design.md`

## Global Constraints

- Existing `ToolResult("...")` constructions keep working untouched.
- No cap may touch pixels: caps see observation text only, attachments ride alongside.
- Downscale constants: longest edge 1568 px, RGBA flattened onto white, first frame only for GIF, JPEG quality 82, metadata stripped; `MAX_IMAGE_BYTES` refuse deleted, Pillow's decompression-bomb guard is the backstop.
- Spill dir `<workspace>/.verbix/logs/`, filename `tool_<uuid4hex12>`, marker `... output truncated; full content saved to <path relative to rel_to — the workspace in production, the test dir in tests> ...`, prune older than 7 days in `spawn_worker`.
- Limits default `max_lines=2000, max_bytes=51200`, overridable per agent via yaml `tool_output:` block; non-int or non-positive values fall back to defaults.
- Full suite plus `ruff check` green before any completion claim; `set -o pipefail` on every test command.

## Review Focus

- Animated GIF read returns the first frame only, no crash, one attachment.
  Test in Task 6 step style: generate 2-frame GIF in tmp_path, read it, assert single attachment and JPEG mime.
- Transparent PNG read flattens onto white instead of crashing or black background.
  Test in Task 6 step style: generate RGBA image, read it, assert RGB JPEG bytes starting with `b"\xff\xd8"`.
- Spill write failure (unwritable workspace) degrades to capped text with no path and no exception.
  Test in Task 2 step style: read-only dir as spill_dir, assert returned text has no "saved to" marker.
- One result with over-limit text plus an image keeps the image intact while the text spills.
  Test in Task 7 step style: craft ToolResult with 200 KB text plus attachment through the handler, assert image part present and path marker present.
- Old marker-string observations still parse on the legacy sync path.
  Covered by the existing multimodal tests, re-verified in Task 9's full-suite run.

---

### Task 1: Typed attachments on ToolResult

**Files:**
- Modify: `src/verbix/tools/base.py:11-14`
- Test: `tests/test_tool_output.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `FileAttachment(data: bytes, mime: str, name: str)` frozen dataclass; `ToolResult.attachments: tuple[FileAttachment, ...] = ()` alongside unchanged `observation: str`, `is_error: bool = False`, `is_terminal: bool = False`.

- [ ] **Step 1: Write the failing test**

```python
def test_tool_result_carries_attachments_default_empty():
    from verbix.tools.base import FileAttachment, ToolResult
    res = ToolResult("hi", attachments=(FileAttachment(data=b"x", mime="image/jpeg", name="a.jpg"),))
    assert res.observation == "hi" and res.attachments[0].mime == "image/jpeg"
    assert ToolResult("hi").attachments == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_tool_result_carries_attachments_default_empty -q 2>&1 | tail -1`
Expected: FAIL with "cannot import name 'FileAttachment'"

- [ ] **Step 3: Implement `FileAttachment` and the `attachments` field in `src/verbix/tools/base.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/tools/base.py
git commit -m "feat: typed file attachments on ToolResult"
```

### Task 2: Bind, spill, prune, and limit-shaping in shared.py

**Files:**
- Modify: `src/verbix/tools/shared.py`
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `bind_output(text: str, *, max_lines: int = 2000, max_bytes: int = 51200, spill_dir: Path | None, rel_to: Path | None = None) -> str` (under limits returns text unchanged; over limits writes full text to `spill_dir/tool_<uuid4hex12>` and returns head/tail preview plus `... output truncated; full content saved to <path of the file relative to rel_to, else bare filename> ...`; `spill_dir=None` or write failure returns capped text with no marker); `prune_spills(spill_dir: Path, older_than_days: int = 7) -> None`; `resolve_output_limits(tool_output: dict | None) -> tuple[int, int]` (reads `max_lines`/`max_bytes` keys, any missing/non-int/non-positive value falls back to its default).

- [ ] **Step 1: Write the failing test**

```python
def test_over_limit_text_spills_and_rereads(tmp_path):
    from verbix.tools.shared import bind_output
    big = "line\n" * 3000
    out = bind_output(big, spill_dir=tmp_path, rel_to=tmp_path)
    assert "full content saved to" in out
    saved = out.split("saved to")[1].split("...")[0].strip()
    assert (tmp_path / saved).read_text() == big
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_over_limit_text_spills_and_rereads -q 2>&1 | tail -1`
Expected: FAIL with "cannot import name 'bind_output'"

- [ ] **Step 3: Implement `bind_output`, `prune_spills`, and `resolve_output_limits` in `src/verbix/tools/shared.py`**

One line on the approach: byte-splitting must not cut a UTF-8 char; walk chars accumulating `len(c.encode())`.

- [ ] **Step 4: Run tests to verify they pass**

```python
def test_unwritable_spill_degrades_without_path(tmp_path):
    from verbix.tools.shared import bind_output
    (tmp_path / "blocker").write_text("x")
    out = bind_output("line\n" * 3000, spill_dir=tmp_path / "blocker" / "sub")
    assert "saved to" not in out


def test_resolve_output_limits_falls_back():
    from verbix.tools.shared import resolve_output_limits
    assert resolve_output_limits(None) == (2000, 51200)
    assert resolve_output_limits({"max_lines": "lots", "max_bytes": -5}) == (2000, 51200)
    assert resolve_output_limits({"max_lines": 10}) == (10, 51200)
```

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/tools/shared.py
git commit -m "feat: text-only bind with spill to .verbix/logs"
```

### Task 3: Registry binds through output_limits

**Files:**
- Modify: `src/verbix/tools/__init__.py:60-94`
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: `bind_output`, `resolve_output_limits` from Task 2; `ToolContext` (existing).
- Produces: `build_registry(names, runtime=None, workspace=None, output_limits: tuple[int, int] | None = None)`; the handler binds each observation with those limits (default `(2000, 51200)`) spilling under `<workspace>/.verbix/logs/` with `rel_to` set to the workspace so the marker re-reads through containment.

- [ ] **Step 1: Write the failing test**

```python
def test_registry_spills_long_output_and_rereads(tmp_path):
    import asyncio
    from verbix.tools import build_registry
    reg = build_registry(["bash"], runtime=None, workspace=tmp_path)
    res = asyncio.run(reg.aexecute("bash", {"command": "seq 1 3000"}))
    assert "full content saved to" in res.observation
    saved = res.observation.split("saved to")[1].split("...")[0].strip()
    assert (tmp_path / saved).read_text().splitlines()[0] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_registry_spills_long_output_and_rereads -q 2>&1 | tail -1`
Expected: FAIL (no spill path pre-change; full text returned with no marker)

- [ ] **Step 3: Implement the `output_limits` param and handler bind in `src/verbix/tools/__init__.py:60-94`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/tools/__init__.py
git commit -m "feat: registry binds text through output limits"
```

### Task 4: Yaml drives limits at both call sites

**Files:**
- Modify: `src/verbix/agent/agents/default.py` (AgentConfig class), `src/verbix/runtime/workers.py:65-72`, `src/verbix/runtime/foreground.py:340-347`
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: `resolve_output_limits` from Task 2; `build_registry(..., output_limits=...)` from Task 3.
- Produces: `AgentConfig.tool_output: dict | None = None`; `workers.py` passes `resolve_output_limits(agent_cfg_dict.get("tool_output"))` at its `build_registry` call; `foreground.py` does the same at its call.

- [ ] **Step 1: Write the failing test**

```python
def test_agent_config_carries_tool_output():
    from verbix.agent.agents.default import AgentConfig
    assert AgentConfig(tool_output={"max_lines": 10}).tool_output == {"max_lines": 10}


def test_yaml_limits_drive_registry(tmp_path):
    import asyncio
    from verbix.tools import build_registry
    from verbix.tools.shared import resolve_output_limits
    cfg = {"tools": ["bash"], "tool_output": {"max_lines": 10, "max_bytes": 200}}
    reg = build_registry(cfg["tools"], runtime=None, workspace=tmp_path,
        output_limits=resolve_output_limits(cfg.get("tool_output")))
    res = asyncio.run(reg.aexecute("bash", {"command": "seq 1 100"}))
    assert "full content saved to" in res.observation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_agent_config_carries_tool_output tests/test_tool_output.py::test_yaml_limits_drive_registry -q 2>&1 | tail -1`
Expected: FAIL on the field test with unexpected keyword argument 'tool_output' (the wiring test passes — Tasks 2-3 built it — and stays as the regression pin for this task's call-site threading)

- [ ] **Step 3: Implement `AgentConfig.tool_output` and thread `resolve_output_limits` through both call sites**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/agent/agents/default.py src/verbix/runtime/workers.py src/verbix/runtime/foreground.py
git commit -m "feat: yaml tool_output drives registry limits"
```

### Task 5: Rewire the engine off caps, delete them, hook the prune

**Files:**
- Modify: `src/verbix/agent/agents/default.py:203-205` (kill site: exact rewrite preserving `attachments=res.attachments`, no `cap_observation` call — bind already happened in the handler), `src/verbix/agent/agents/default.py:88-93` (delete), `src/verbix/tools/spawn.py:15`, `src/verbix/tools/status.py:15`, `src/verbix/tools/cancel.py:15`, `src/verbix/tools/bash.py:12,33-41,69-73` (delete fields and truncation block), `src/verbix/tools/__init__.py:76` (delete stamp), `src/verbix/runtime/workers.py:34-36` (add prune call in `spawn_worker` body), `tests/test_agent_limits_and_caps.py:3-42` (drop the `cap_observation` import and its two cap tests; the turn/global limit tests stay)
- Test: full suite only (deletion task; neighboring tests are the gate)

**Interfaces:**
- Consumes: Tasks 2-4 (bind owns all capping now).
- Produces: `_execute_tool` returns the handler-bound text with attachments preserved; no `cap_observation`, no `observation_cap`, no `max_output_length`; `spawn_worker` calls `prune_spills(<workspace>/.verbix/logs/)` before starting the worker.

- [ ] **Step 1: Rewrite the kill site, delete the listed sites, update the cap tests, add the one-line prune call**

- [ ] **Step 2: Run the full suite to verify nothing breaks**

Run: `set -o pipefail; uv run pytest -q 2>&1 | tail -1`
Expected: PASS, count drops by exactly the 2 deleted cap tests and nothing else

- [ ] **Step 3: Commit**

```bash
git add src/verbix/agent/agents/default.py src/verbix/tools/spawn.py src/verbix/tools/status.py src/verbix/tools/cancel.py src/verbix/tools/bash.py src/verbix/tools/__init__.py src/verbix/runtime/workers.py tests/test_agent_limits_and_caps.py
git commit -m "fix: engine off caps, superseded caps deleted, spill prune hooked"
```

### Task 6: Photos ride as attachments

**Files:**
- Modify: `src/verbix/tools/read.py:17` (delete `MAX_IMAGE_BYTES` refuse), `src/verbix/tools/read.py:65-83`, `src/verbix/tools/read.py:23-26` (description), `pyproject.toml` + `uv.lock` (via `uv add pillow`)
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: `FileAttachment`, `ToolResult.attachments` from Task 1.
- Produces: image branch returns `ToolResult("[image attached: {name}]", attachments=(FileAttachment(downscaled, mime, name),))` with longest edge 1568 px, RGBA on white, first GIF frame, JPEG q82, no metadata; `MAX_IMAGE_BYTES` refuse deleted.

- [ ] **Step 1: Write the failing test**

```python
def test_large_photo_returns_small_attachment(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    img = Image.new("RGB", (3000, 2000), "blue")
    img.save(tmp_path / "big.jpg")
    res = asyncio.run(FileReadTool().execute({"path": "big.jpg"}, ToolContext(workspace=tmp_path)))
    assert len(res.attachments) == 1 and len(res.attachments[0].data) < 150 * 1024
    assert "MSWEA_MULTIMODAL" not in res.observation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_large_photo_returns_small_attachment -q 2>&1 | tail -1`
Expected: FAIL (marker string returned, no attachments; Pillow imports transitively today so the failure is on attachments, which is the behavior under test)

- [ ] **Step 3: Implement the Pillow downscale branch, run `uv add pillow`, append the description line**

- [ ] **Step 4: Run tests to verify they pass**

```python
def test_gif_read_returns_first_frame_only(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    frames = [Image.new("RGB", (100, 100), c) for c in ("red", "green")]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    res = asyncio.run(FileReadTool().execute({"path": "a.gif"}, ToolContext(workspace=tmp_path)))
    assert len(res.attachments) == 1 and res.attachments[0].mime == "image/jpeg"


def test_rgba_flattens_onto_white(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(tmp_path / "t.png")
    res = asyncio.run(FileReadTool().execute({"path": "t.png"}, ToolContext(workspace=tmp_path)))
    assert res.attachments[0].data[:2] == b"\xff\xd8"
```

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/tools/read.py pyproject.toml uv.lock
git commit -m "feat: photos ride as downscaled attachments"
```

### Task 7: Engine speaks image parts

**Files:**
- Modify: `src/verbix/agent/agents/default.py:433-440` (cancellation-repair message preserves attachments), `src/verbix/agent/agents/default.py:460-465` (tool message built via new module-level `tool_message_content(observation, attachments)` helper: plain string when no attachments, else one text part plus one `image_url` part per attachment; the `:203-205` kill-site rewrite already landed in Task 5)
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: attachments survive into the `step_stream` tool message, including the cancellation-repair path.

- [ ] **Step 1: Write the failing test**

```python
def test_tool_message_carries_image_parts():
    from verbix.agent.agents.default import tool_message_content
    from verbix.tools.base import FileAttachment
    content = tool_message_content("hi", (FileAttachment(data=b"x", mime="image/jpeg", name="a.jpg"),))
    assert content[0] == {"type": "text", "text": "hi"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert tool_message_content("hi", ()) == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_tool_message_carries_image_parts -q 2>&1 | tail -1`
Expected: FAIL with "cannot import name 'tool_message_content'"

- [ ] **Step 3: Implement the repair-path preservation and the tool-message image parts**

- [ ] **Step 4: Run tests to verify they pass**

```python
def test_over_limit_text_plus_image_keeps_image(tmp_path):
    import asyncio
    from verbix.agent.agents.default import AgentConfig, DefaultAgent, tool_message_content
    from verbix.tools import build_registry
    from verbix.tools.base import FileAttachment, ToolResult
    big = ToolResult("line\n" * 3000,
        attachments=(FileAttachment(data=b"x", mime="image/jpeg", name="a.jpg"),))
    agent = DefaultAgent(model=None, env=None,
        tools=build_registry(["bash"], runtime=None, workspace=tmp_path),
        config=AgentConfig(tools=["bash"]))
    res = asyncio.run(agent._execute_tool("bash", {"command": "seq 1 3000"}))
    content = tool_message_content(res.observation, big.attachments)
    assert any(p.get("type") == "image_url" for p in content)
    assert "full content saved to" in content[0]["text"]


def test_photo_attachments_survive_execute_tool(tmp_path):
    from PIL import Image
    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.tools import build_registry
    import asyncio
    Image.new("RGB", (3000, 2000), "blue").save(tmp_path / "big.jpg")
    agent = DefaultAgent(model=None, env=None,
        tools=build_registry(["read"], runtime=None, workspace=tmp_path),
        config=AgentConfig(tools=["read"]))
    res = asyncio.run(agent._execute_tool("read", {"path": "big.jpg"}))
    assert len(res.attachments) == 1
```

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/agent/agents/default.py
git commit -m "feat: image parts end to end in the engine"
```

### Task 8: List-safe consumers

**Files:**
- Modify: `src/verbix/core/models.py:29` (`content: str | list[dict[str, Any]]`), `src/verbix/runtime/foreground.py:90-120,195-203`, `src/verbix/runtime/worker.py:136-137,223,241-242`, `src/verbix/agent/utils/compaction.py:90,105`
- Test: `tests/test_tool_output.py` (append)

**Interfaces:**
- Consumes: list-content tool messages from Task 7.
- Produces: adapter, history proxy, and status readers pass lists through eliding image parts to `[image elided]`; compaction replaces image parts with the elision placeholder beside its str rules.

- [ ] **Step 1: Write the failing test**

```python
def test_status_after_photo_does_not_crash():
    from verbix.runtime.worker import WorkerTask
    photo_msg = {"role": "tool", "tool_call_id": "c1", "name": "read",
        "content": [{"type": "text", "text": "[image attached: a.jpg]"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,eA=="}}]}
    class FakePhotoSession:
        messages = [photo_msg]
    task = WorkerTask(id="t", description="d", session=FakePhotoSession())
    assert "image" in task.get_status(full=True).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py::test_status_after_photo_does_not_crash -q 2>&1 | tail -1`
Expected: FAIL with TypeError (list has no `.strip`/`.replace`, or pydantic ValidationError)

- [ ] **Step 3: Implement the widening, the three list guards, and the compaction elision**

- [ ] **Step 4: Run tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_tool_output.py -q 2>&1 | tail -1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool_output.py src/verbix/core/models.py src/verbix/runtime/foreground.py src/verbix/runtime/worker.py src/verbix/agent/utils/compaction.py
git commit -m "feat: list-safe consumers for image tool messages"
```

### Task 9: Final proof

**Files:**
- Modify: none (verification only)
- Test: full suite plus `ruff check`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: green suite, honest gates.

- [ ] **Step 1: Run the full suite**

Run: `set -o pipefail; uv run pytest -q 2>&1 | tail -1`
Expected: PASS, all green including the legacy marker tests

- [ ] **Step 2: Run the ruff gate over the touched areas (directory gates cover the touched files)**

Run: `uv run ruff check src/verbix/tools/ src/verbix/runtime/ src/verbix/agent/agents/default.py src/verbix/core/models.py src/verbix/agent/utils/compaction.py`
Expected: All checks passed

- [ ] **Step 3: Confirm Pillow is declared and locked**

Run: `grep -n pillow pyproject.toml uv.lock | head -3`
Expected: pinned entries shown

- [ ] **Step 4: Commit only if Steps 1-3 left working-tree fixes; otherwise record complete with no empty commit**
