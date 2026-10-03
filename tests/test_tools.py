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
