# Screenshot Saving and Global Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `ScreenshotTool` to automatically archive captures to a rolling global cache (`~/.verbix/screenshots/`), auto-prune files older than 3 days, and support an optional `path` parameter for saving screenshots to the workspace.

**Architecture:** In `src/verbix/tools/screenshot.py`, update `parameters_schema` to accept optional `path: str`. In `_capture(args, context)`, save every capture to `Path.home() / ".verbix" / "screenshots"` as PNG, call `_prune_screenshots(cache_dir, max_age_days=3)`, and if `path` is passed, resolve it via `resolve_contained_path`, handle directory targets, and save PNG to the workspace. Return compressed JPEG `FileAttachment` and voice-safe observation.

**Tech Stack:** Python 3.13+, standard library (`pathlib`, `time`, `datetime`, `asyncio`, `io`), existing `Pillow` (`PIL.Image`, `PIL.ImageGrab`).

**Spec:** `docs/superpowers/specs/2026-10-14-screenshot-saving-and-cache-design.md`

## Global Constraints
- Zero new dependencies added to `pyproject.toml`.
- Cross-platform: works uniformly on Linux, macOS, and Windows.
- Workspace containment: any user-specified `path` must be validated via `resolve_contained_path` to prevent directory traversal.
- Token and payload safety: in-memory `FileAttachment` stays compressed JPEG (quality 82, max 1568px).
- Hermetic tests: unit tests must mock `ImageGrab.grab` and use `monkeypatch` / `tmp_path` for `Path.home()` and workspace.
- Full test suite and `ruff check` pass cleanly.

## Review Focus
- Directory auto-naming: does `path="screenshots/"` or an existing directory safely create `screenshot_<ts>.png` inside without `IsADirectoryError`?
- Missing parent dirs: does `path="nested/sub/shot.png"` auto-create intermediate directories without `FileNotFoundError`?
- Workspace escaping: does `path="../../outside.png"` fail gracefully with `is_error=True`?
- Cache auto-prune: does `_prune_screenshots` silently remove files older than 3 days without crashing if files are locked or missing?
- Voice observation: does the observation avoid dumping long absolute home paths into the output?

---

### Task 1: `ScreenshotTool` Saving, Global Cache & Auto-Pruning

**Files:**
- Modify: `src/verbix/tools/screenshot.py`
- Modify: `src/verbix/agent/config/worker.yaml`
- Modify: `tests/test_tools_screenshot.py`

**Interfaces:**
- Consumes: `resolve_contained_path` from `verbix.tools.shared`, `ToolContext`, `ToolResult`, `FileAttachment` from `verbix.tools.base`.
- Produces: `ScreenshotTool` accepting optional `path` in `execute(args, context)`.

- [ ] **Step 1: Write failing tests in `tests/test_tools_screenshot.py`**

Add tests:
- `test_screenshot_saves_to_global_cache`: mocks `Path.home()` to `tmp_path / "home"`, captures screen, asserts `tmp_path / "home" / ".verbix" / "screenshots"` contains a valid PNG file and observation contains `(cached)`.
- `test_screenshot_prunes_old_cache_files`: places an old screenshot (mtime > 4 days ago) in the cache directory, runs capture, asserts old screenshot is deleted and new screenshot exists.
- `test_screenshot_saves_to_explicit_workspace_file`: runs with `args={"path": "bug.png"}` and `tool_context.workspace = tmp_path`, asserts `tmp_path / "bug.png"` exists, is valid PNG, and observation contains `-> bug.png`.
- `test_screenshot_saves_to_directory_path`: runs with `args={"path": "screenshots/"}`, asserts `tmp_path / "screenshots"` directory is created and contains a `screenshot_*.png` file.
- `test_screenshot_rejects_escaping_path`: runs with `args={"path": "../../outside.png"}`, asserts `result.is_error == True` and observation mentions path escaping workspace.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools_screenshot.py -v`
Expected: FAIL on new test cases.

- [ ] **Step 3: Implement saving, global cache, and auto-pruning in `src/verbix/tools/screenshot.py`**

Implement:
1. Update `parameters_schema`:
   ```python
   {
       "type": "object",
       "properties": {
           "path": {
               "type": "string",
               "description": "Optional relative path in workspace to save the screenshot image to (e.g. 'bug.png' or 'screenshots/').",
           }
       },
       "additionalProperties": False,
   }
   ```
2. Helper `_prune_screenshots(cache_dir: Path, max_age_days: int = 3) -> None`:
   - Iterates `cache_dir.glob("screenshot_*.png")`.
   - If `time.time() - p.stat().st_mtime > max_age_days * 86400`:
     - `p.unlink(missing_ok=True)`
   - Catches `OSError` silently.
3. In `_capture(path_str: str | None, context: ToolContext | None)`:
   - Generate timestamp: `ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")`
   - Global cache:
     - `cache_dir = Path.home() / ".verbix" / "screenshots"`
     - `cache_dir.mkdir(parents=True, exist_ok=True)`
     - `cached_file = cache_dir / f"screenshot_{ts}.png"`
     - `im.save(cached_file, format="PNG")`
     - `_prune_screenshots(cache_dir, max_age_days=3)`
   - Workspace target:
     - If `path_str` and `context`:
       - `target = resolve_contained_path(path_str, context.workspace)`
       - If `target.is_dir() or path_str.endswith(("/", "\\"))`:
         `target = target / f"screenshot_{ts}.png"`
       - `target.parent.mkdir(parents=True, exist_ok=True)`
       - `im.save(target, format="PNG")`
       - Display path: relative to workspace if possible, else `target.name`
   - Observation:
     - If saved to workspace: `f"[screenshot captured: {orig_w}x{orig_h} -> {display_path}]"`
     - Else: `f"[screenshot captured: {orig_w}x{orig_h} (cached)]"`
   - Attachment:
     - In-memory compressed JPEG `(1568, 1568)`, quality 82 (unchanged).

4. Update `src/verbix/agent/config/worker.yaml`:
   - Update `screenshot` tool documentation in `format_error_template` to show optional `{"path": "screenshot.png"}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools_screenshot.py -v`
Expected: PASS (all tests pass)

- [ ] **Step 5: Run full test suite and linter**

Run: `pytest`
Run: `ruff check`
Expected: 476+ passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/tools/screenshot.py src/verbix/agent/config/worker.yaml tests/test_tools_screenshot.py
git commit -m "feat(tools): add global caching, auto-prune, and optional workspace saving to screenshot"
```

---

### Task 2: Final Review Pass & Verification

**Files:**
- All files changed in Task 1.

- [ ] **Step 1: Run full test suite and lint**
Run: `pytest -q`
Run: `ruff check`

- [ ] **Step 2: Subagent Review Pass**
Dispatch a reviewer subagent to inspect the diff against requirements, verifying zero scope creep, Ponytail compliance, and cross-platform safety.

- [ ] **Step 3: Report git diff and summary with LOC and file sizes**
Provide the final diff, file sizes, and LOC summary.
