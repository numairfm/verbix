# Screenshot Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide Verbix autonomous background workers with a dedicated, zero-dependency `screenshot` tool to capture the desktop display and attach it as a compressed JPEG for multimodal vision models.

**Architecture:** A standalone tool `ScreenshotTool` in `src/verbix/tools/screenshot.py` implementing `Tool` protocol, capturing display in thread via `PIL.ImageGrab.grab()`, scaling to max `(1568, 1568)` with `Image.LANCZOS`, encoding as JPEG quality 82, and returning `ToolResult` with `FileAttachment`. Registered in `TOOL_CATALOG` and default worker tools.

**Tech Stack:** Python 3.13+, standard library (`asyncio`, `io`, `typing`), existing `Pillow>=12.3.0` (`PIL.Image`, `PIL.ImageGrab`).

**Spec:** `docs/superpowers/specs/2026-10-14-screenshot-tool-design.md`

## Global Constraints
- Zero new third-party dependencies added to `pyproject.toml` (standard library + existing Pillow).
- Bounded scope: primary screen capture only, no arbitrary window-handle sniffing or mouse automation.
- Token and payload safety: images must be downscaled to max bounding box `(1568, 1568)` and encoded as JPEG quality 82.
- Clean error handling: headless / no-display environments or OS permission failures return `ToolResult(..., is_error=True)` without crashing.
- Hermetic tests: all unit tests must be 100% offline using mocked `ImageGrab.grab`.
- Full test suite and `ruff check` pass cleanly before completion.

## Review Focus
- Headless CI / no display: does `screenshot` catch `OSError` cleanly and report `is_error=True`?
- Aspect ratio preservation: does downscaling preserve the exact aspect ratio on 16:9, 16:10, and multi-monitor setups?
- Alpha channel handling: does RGBA transparency get blended onto a white background without Pillow `OSError: cannot write mode RGBA as JPEG`?
- Catalog binding: does `build_worker_registry()` include `screenshot`?
- Zero scope creep: does the tool stay focused strictly on display capture without adding unnecessary parameters?

---

### Task 1: `ScreenshotTool` Implementation

**Files:**
- Create: `src/verbix/tools/screenshot.py`
- Test: `tests/test_tools_screenshot.py`

**Interfaces:**
- Consumes: `FileAttachment`, `ToolContext`, `ToolResult` from `verbix.tools.base`
- Produces: `ScreenshotTool` in `src/verbix/tools/screenshot.py` with `name = "screenshot"`, `side_effecting = False`, `description = ...`, `parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}`, and `async execute(args, context) -> ToolResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_screenshot.py
import io
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pytest

from verbix.tools.base import ToolContext
from verbix.tools.screenshot import ScreenshotTool


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


@pytest.mark.asyncio
async def test_screenshot_success(tool_context: ToolContext):
    synthetic_im = Image.new("RGB", (1920, 1080), color="blue")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert "[screenshot captured: 1920x1080]" in result.observation
    assert len(result.attachments) == 1
    att = result.attachments[0]
    assert att.mime == "image/jpeg"
    assert att.name == "screenshot.jpg"

    # Verify attachment is a valid readable JPEG
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.format == "JPEG"
        assert im.size == (1568, 882)


@pytest.mark.asyncio
async def test_screenshot_resizes_4k_display(tool_context: ToolContext):
    synthetic_4k = Image.new("RGB", (3840, 2160), color="red")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_4k):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert "[screenshot captured: 3840x2160]" in result.observation
    att = result.attachments[0]
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.size == (1568, 882)


@pytest.mark.asyncio
async def test_screenshot_handles_rgba_transparency(tool_context: ToolContext):
    synthetic_rgba = Image.new("RGBA", (800, 600), color=(0, 255, 0, 128))
    with patch("PIL.ImageGrab.grab", return_value=synthetic_rgba):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    att = result.attachments[0]
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.format == "JPEG"
        assert im.size == (800, 600)


@pytest.mark.asyncio
async def test_screenshot_handles_headless_oserror(tool_context: ToolContext):
    with patch("PIL.ImageGrab.grab", side_effect=OSError("no display found")):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert result.is_error
    assert "error: could not capture screenshot: no display found" in result.observation
    assert len(result.attachments) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools_screenshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verbix.tools.screenshot'`

- [ ] **Step 3: Implement `ScreenshotTool` in `src/verbix/tools/screenshot.py`**

Implement `ScreenshotTool`:
- Properties: `name = "screenshot"`, `side_effecting = False`, `description`, `parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}`.
- `execute(args, context)`: uses `asyncio.to_thread(self._capture)`.
- `_capture()`:
  - Calls `ImageGrab.grab()`.
  - Captures `orig_w, orig_h = im.size`.
  - Converts to RGBA, blends onto white `Image.new("RGB", rgba.size, (255, 255, 255))` using alpha mask.
  - Resizes with `bg.thumbnail((1568, 1568), Image.LANCZOS)`.
  - Encodes to `io.BytesIO()` as JPEG (quality 82).
  - Returns `ToolResult(f"[screenshot captured: {orig_w}x{orig_h}]", attachments=(FileAttachment(buf.getvalue(), "image/jpeg", "screenshot.jpg"),))`.
  - Catches `Exception` and returns `ToolResult(f"error: could not capture screenshot: {e}", is_error=True)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools_screenshot.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run linter**

Run: `ruff check src/verbix/tools/screenshot.py tests/test_tools_screenshot.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/verbix/tools/screenshot.py tests/test_tools_screenshot.py
git commit -m "feat(tools): implement ScreenshotTool with JPEG compression and error recovery"
```

---

### Task 2: Catalog Registration and Worker Tool Configuration

**Files:**
- Modify: `src/verbix/tools/__init__.py:20-75`
- Modify: `src/verbix/runtime/workers.py:30-45,130-145`
- Modify: `src/verbix/agent/config/worker.yaml:30-45`
- Test: `tests/test_tools_catalog_screenshot.py`

**Interfaces:**
- Consumes: `ScreenshotTool` from `verbix.tools.screenshot`
- Produces: `screenshot` in `TOOL_CATALOG` and default worker tools in `build_worker_registry()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_catalog_screenshot.py
from pathlib import Path
from verbix.tools import TOOL_CATALOG, build_registry
from verbix.tools.screenshot import ScreenshotTool
from verbix.runtime.workers import build_worker_registry


def test_screenshot_in_tool_catalog():
    assert "screenshot" in TOOL_CATALOG
    tool = TOOL_CATALOG["screenshot"](None)
    assert isinstance(tool, ScreenshotTool)


def test_build_registry_with_screenshot(tmp_path: Path):
    registry = build_registry(["screenshot"], workspace=tmp_path)
    tool = registry.get("screenshot")
    assert tool is not None
    assert tool.name == "screenshot"


def test_build_worker_registry_includes_screenshot(tmp_path: Path):
    registry = build_worker_registry(workspace=tmp_path)
    assert registry.get("screenshot") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools_catalog_screenshot.py -v`
Expected: FAIL with `AssertionError: assert 'screenshot' in TOOL_CATALOG`

- [ ] **Step 3: Register `screenshot` in `TOOL_CATALOG`, `workers.py`, and `worker.yaml`**

- In `src/verbix/tools/__init__.py`:
  - Import `ScreenshotTool` from `verbix.tools.screenshot`.
  - Add `"ScreenshotTool"` to `__all__`.
  - Add `"screenshot": lambda runtime: ScreenshotTool()` to `TOOL_CATALOG`.
- In `src/verbix/runtime/workers.py`:
  - Add `"screenshot"` to default worker tools list in `build_worker_registry()` (line 32).
  - Add `"screenshot"` to fallback list in `spawn_worker()` (line 130).
- In `src/verbix/agent/config/worker.yaml`:
  - Add `- screenshot` to `agent.tools:`.
  - In `Execution Invariants` or `Command Execution Rules`, add guidance:
    `- Use 'screenshot' to capture the desktop display when asked to look at the screen, inspect visual layout, UI bugs, or error dialogs.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools_catalog_screenshot.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run full test suite and linter**

Run: `pytest`
Run: `ruff check`
Expected: all tests pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/tools/__init__.py src/verbix/runtime/workers.py src/verbix/agent/config/worker.yaml tests/test_tools_catalog_screenshot.py
git commit -m "feat(tools): register screenshot in catalog and worker toolset"
```

---

### Task 3: Final Whole-Branch Review & Verification

**Files:**
- All modified and created files across Tasks 1-2.

- [ ] **Step 1: Run full test suite and lint**
Run: `pytest -q`
Run: `ruff check`

- [ ] **Step 2: Subagent Review Pass**
Dispatch a reviewer subagent to inspect the diff against requirements, verifying zero scope creep, Ponytail compliance, and test hermeticity.

- [ ] **Step 3: Report git diff and summary with LOC and file sizes**
Provide the final diff, file sizes, and LOC summary.
