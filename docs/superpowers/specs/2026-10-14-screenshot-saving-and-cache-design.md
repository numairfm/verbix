# Screenshot Saving and Global Cache Design Spec

## 1. Overview
Upgrades `ScreenshotTool` in `src/verbix/tools/screenshot.py` to:
1. Accept an optional `path` parameter allowing users/workers to save screenshots into the workspace.
2. Automatically archive every capture into a global rolling cache (`~/.verbix/screenshots/`).
3. Auto-prune cached screenshots older than 3 days.
4. Retain lightweight in-memory JPEG attachments for model vision context.

## 2. Requirements & Constraints
- **Zero New Dependencies**: Uses standard library (`pathlib`, `time`, `datetime`, `asyncio`, `io`) and existing `Pillow`.
- **Cross-Platform**: 100% universal across Windows, macOS, and Linux using `Path.home()` and `pathlib`.
- **Global Archiving**: Every screenshot is saved to `~/.verbix/screenshots/screenshot_<YYYYMMDD_HHMMSS>.png` as lossless PNG.
- **Auto-Pruning**: Files in `~/.verbix/screenshots/` older than 3 days (based on `st_mtime`) are silently pruned on each capture.
- **Workspace Destination (`path`)**:
  - Validated using `resolve_contained_path(path, context.workspace)`.
  - Directory handling: if `path` ends with `/` or `\` or points to an existing directory, auto-appends `screenshot_<YYYYMMDD_HHMMSS>.png`.
  - Auto-creates parent directories with `target.parent.mkdir(parents=True, exist_ok=True)`.
  - Saves full-quality lossless PNG to the target.
- **Model Token Safety**: Returns in-memory compressed JPEG attachment `(1568, 1568)` max box, quality 82 (unchanged).
- **Voice-Safe Observation**:
  - Without path: `[screenshot captured: {orig_w}x{orig_h} (cached)]`
  - With path: `[screenshot captured: {orig_w}x{orig_h} -> {target.name}]` (or relative path in workspace).
- **Hermetic Tests**: All tests in `tests/test_tools_screenshot.py` mock `ImageGrab.grab` and use isolated temporary directories for cache and workspace.
