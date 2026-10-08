import asyncio
from pathlib import Path

import pytest

from verbix.tools.base import ToolContext, parse_lenient_json
from verbix.tools.bash import BashTool
from verbix.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool
from verbix.tools.sandbox import SandboxConfig, execute_sandboxed


def test_parse_lenient_json():
    # Standard JSON
    assert parse_lenient_json('{"key": "val"}') == {"key": "val"}

    # Markdown fenced JSON
    markdown_json = '```json\n{\n  "command": "git status"\n}\n```'
    assert parse_lenient_json(markdown_json) == {"command": "git status"}

    # Trailing/leading text around JSON block
    mixed = (
        'Here is the tool call:\n```json\n{"path": "foo.txt"}\n```\nHope that helps!'
    )
    assert parse_lenient_json(mixed) == {"path": "foo.txt"}


@pytest.mark.asyncio
async def test_filesystem_tools_and_containment(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolContext(workspace=workspace)

    write_tool = FileWriteTool()
    read_tool = FileReadTool()
    edit_tool = FileEditTool()

    # 1. Write file
    res = await write_tool.execute(
        {"path": "src/main.py", "content": "print('hello world')\nline2\nline3\n"}, ctx
    )
    assert res.is_error is False
    assert (workspace / "src/main.py").exists()

    # 2. Read file with line range
    read_res = await read_tool.execute(
        {"path": "src/main.py", "start_line": 1, "end_line": 2}, ctx
    )
    assert read_res.is_error is False
    assert "1: print('hello world')" in read_res.observation
    assert "2: line2" in read_res.observation
    assert "line3" not in read_res.observation

    # 3. Edit file
    edit_res = await edit_tool.execute(
        {"path": "src/main.py", "old_str": "hello world", "new_str": "verbix"}, ctx
    )
    assert edit_res.is_error is False
    assert "print('verbix')" in (workspace / "src/main.py").read_text()

    # 4. Containment rejection
    escape_res = await write_tool.execute(
        {"path": "../outside.txt", "content": "malicious"}, ctx
    )
    assert escape_res.is_error is True
    assert "escapes configured workspace" in escape_res.observation


@pytest.mark.asyncio
async def test_bash_tool_execution_and_cancellation(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path)
    tool = BashTool()

    # 1. Basic command
    res = await tool.execute({"command": "echo 'verbix core'"}, ctx)
    assert res.is_error is False
    assert res.observation.strip() == "verbix core"

    # 2. Cancellation kills process group
    task = asyncio.create_task(tool.execute({"command": "sleep 10"}, ctx))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_bubblewrap_sandbox_isolation(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = SandboxConfig(workspace=workspace, allow_network=False)

    # 1. Test filesystem write inside workspace
    res = await execute_sandboxed(
        "echo 'sandboxed' > output.txt && cat output.txt", config
    )
    assert res.exit_code == 0
    assert res.stdout.strip() == "sandboxed"
    assert (workspace / "output.txt").read_text().strip() == "sandboxed"

    # 2. Test filesystem isolation (cannot write to /root or /tmp outside sandbox)
    escape_res = await execute_sandboxed(
        "touch /root/escape.txt 2>/dev/null || echo 'denied'", config
    )
    assert "denied" in escape_res.stdout

    # 3. Test network isolation (default-deny)
    net_res = await execute_sandboxed("ping -c 1 8.8.8.8 2>&1 || echo 'no-net'", config)
    assert (
        "no-net" in net_res.stdout
        or "Network is unreachable" in net_res.stdout
        or net_res.exit_code != 0
    )


@pytest.mark.asyncio
async def test_read_lists_directories_with_pagination(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "b.txt").write_text("b")
    (workspace / "a.txt").write_text("a")
    ctx = ToolContext(workspace=workspace)
    read_tool = FileReadTool()

    page1 = await read_tool.execute({"path": ".", "start_line": 1, "end_line": 2}, ctx)
    assert page1.is_error is False
    assert "a.txt" in page1.observation
    assert "b.txt" in page1.observation
    assert "sub/" not in page1.observation

    page2 = await read_tool.execute({"path": ".", "start_line": 3, "end_line": 3}, ctx)
    assert page2.is_error is False
    assert "sub/" in page2.observation
    assert "a.txt" not in page2.observation


@pytest.mark.asyncio
async def test_read_empty_directory(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "empty").mkdir(parents=True)
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": "empty"}, ctx)
    assert res.is_error is False
    assert "empty" in res.observation


@pytest.mark.asyncio
async def test_read_returns_images_as_viewable_content(tmp_path: Path):
    import base64

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    (workspace / "tiny.png").write_bytes(png)
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": "tiny.png"}, ctx)
    assert res.is_error is False
    assert "MSWEA_MULTIMODAL_CONTENT" in res.observation
    assert "data:image/png;base64," in res.observation


@pytest.mark.asyncio
async def test_read_rejects_oversized_images(tmp_path: Path, monkeypatch):
    from verbix.tools import filesystem as fs

    monkeypatch.setattr(fs, "MAX_IMAGE_BYTES", 4)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "big.png").write_bytes(b"12345")
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": "big.png"}, ctx)
    assert res.is_error is True
    assert "too large" in res.observation


def test_multimodal_expansion_is_on_by_default():
    from verbix.minisweagent.models.litellm_model import LitellmModelConfig
    from verbix.minisweagent.models.utils.actions_toolcall import (
        format_toolcall_observation_messages,
    )

    cfg = LitellmModelConfig(model_name="test/model")
    assert cfg.multimodal_regex, "image expansion must be enabled by default"
    tag = (
        "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
        "data:image/png;base64,AAA</MSWEA_MULTIMODAL_CONTENT>"
    )
    msgs = format_toolcall_observation_messages(
        actions=[{"tool_call_id": "c1"}],
        outputs=[{"output": tag, "returncode": 0, "exception_info": ""}],
        observation_template="{{ output.output }}",
        multimodal_regex=cfg.multimodal_regex,
    )
    parts = msgs[0]["content"]
    assert isinstance(parts, list)
    assert any(p.get("type") == "image_url" for p in parts)


@pytest.mark.asyncio
async def test_read_states_its_own_completeness(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "f.txt").write_text("one\ntwo\n")
    (workspace / "d").mkdir()
    (workspace / "d" / "a.txt").write_text("a")
    ctx = ToolContext(workspace=workspace)
    read_tool = FileReadTool()

    file_res = await read_tool.execute({"path": "f.txt", "start_line": 1, "end_line": 1}, ctx)
    assert "showing lines 1-1 of 2" in file_res.observation

    dir_res = await read_tool.execute({"path": "d"}, ctx)
    assert "showing entries 1-1 of 1" in dir_res.observation
