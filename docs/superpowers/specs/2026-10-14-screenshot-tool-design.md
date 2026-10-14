# Screenshot Tool Design Spec

## 1. Overview
The `screenshot` tool allows Verbix autonomous workers to capture the user's primary desktop display and attach it as an image for multimodal vision models.

## 2. Requirements & Constraints
- **Zero New Dependencies**: Uses standard library (`asyncio`, `io`) and existing `Pillow>=12.3.0` (`PIL.Image`, `PIL.ImageGrab`).
- **Token & Gateway Safety**: Screen captures can be up to 4K (~33MB). Must downscale to max bounding box `(1568, 1568)` using `Image.LANCZOS` and compress to JPEG (quality 82) to match [`src/verbix/tools/read.py`](file:///home/numair/dev/projects/verbix/src/verbix/tools/read.py), keeping payloads under ~300KB.
- **Cross-Platform**: Uses `PIL.ImageGrab.grab()` which supports Linux (X11/Wayland portal), macOS, and Windows.
- **Error Handling**: Gracefully handles headless environments (no display server / CI) or permission failures by returning `ToolResult("error: could not capture screenshot: <reason>", is_error=True)`.
- **Hermetic Unit Tests**: All unit tests must be 100% offline and mocked so they pass in headless CI without a display.

## 3. Tool Interface
- **Name**: `screenshot`
- **Side Effecting**: `False`
- **Parameters Schema**:
  ```json
  {
    "type": "object",
    "properties": {},
    "additionalProperties": false
  }
  ```
- **Execution**:
  - Executed in a thread via `asyncio.to_thread`.
  - Converts RGBA to RGB with white background (handles alpha transparency).
  - Downscales with `im.thumbnail((1568, 1568), Image.LANCZOS)`.
  - Saves to `io.BytesIO` as JPEG quality 82.
  - Returns `ToolResult(f"[screenshot captured: {orig_w}x{orig_h}]", attachments=(FileAttachment(buf.getvalue(), "image/jpeg", "screenshot.jpg"),))`.

## 4. Integration
- Exported and registered in `TOOL_CATALOG` in `src/verbix/tools/__init__.py`.
- Added to default worker tools in `src/verbix/runtime/workers.py`.
- Added to worker configuration in `src/verbix/agent/config/worker.yaml`.
