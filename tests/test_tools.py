import asyncio
from pathlib import Path

import pytest

from verbix.tools.base import ToolContext, parse_lenient_json
from verbix.tools.bash import BashTool
from verbix.tools.edit import FileEditTool
from verbix.tools.read import FileReadTool
from verbix.tools.sandbox import SandboxConfig, execute_sandboxed
from verbix.tools.write import FileWriteTool


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
    assert res.observation == "[image attached: tiny.png]"
    assert len(res.attachments) == 1
    assert res.attachments[0].mime == "image/jpeg"
    assert res.attachments[0].data[:2] == b"\xff\xd8"
    assert "MSWEA_MULTIMODAL" not in res.observation


@pytest.mark.asyncio
async def test_read_downscales_large_images_instead_of_refusing(tmp_path: Path):
    from PIL import Image

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Image.new("RGB", (3000, 2000), "blue").save(workspace / "big.jpg")
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": "big.jpg"}, ctx)
    assert res.is_error is False
    assert len(res.attachments) == 1
    assert len(res.attachments[0].data) < 150 * 1024
    assert "MSWEA_MULTIMODAL" not in res.observation


def test_multimodal_expansion_is_on_by_default():
    from verbix.agent.models.litellm_model import LitellmModelConfig
    from verbix.agent.models.utils.actions_toolcall import (
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


@pytest.mark.asyncio
async def test_edit_empty_old_str(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("content")
    ctx = ToolContext(workspace=tmp_path)
    read_tool = FileReadTool()
    await read_tool.execute({"path": "hello.txt"}, ctx)

    edit_tool = FileEditTool()
    res = await edit_tool.execute({"path": "hello.txt", "old_str": "", "new_str": "new"}, ctx)
    assert res.is_error is True
    assert "old_str cannot be empty" in res.observation


@pytest.mark.asyncio
async def test_read_workspace_auth_json_denied(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / ".verbix").mkdir(parents=True)
    (workspace / ".verbix" / "auth.json").write_text('{"openrouter": {"key": "sk-x"}}')
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": ".verbix/auth.json"}, ctx)
    assert res.is_error is True
    assert "secret" in res.observation.lower()


@pytest.mark.asyncio
async def test_read_home_auth_json_denied_regardless_of_spelling(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home_auth = tmp_path / "home" / ".verbix" / "auth.json"
    home_auth.parent.mkdir(parents=True)
    home_auth.write_text('{"openrouter": {"key": "sk-x"}}')
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": str(home_auth)}, ctx)
    assert res.is_error is True
    assert "secret" in res.observation.lower()


@pytest.mark.asyncio
async def test_read_dotenv_denied(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("OPENROUTER_API_KEY=sk-x\n")
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": ".env"}, ctx)
    assert res.is_error is True
    assert "secret" in res.observation.lower()


@pytest.mark.asyncio
async def test_write_and_edit_secret_files_denied(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / ".verbix").mkdir(parents=True)
    (workspace / ".verbix" / "auth.json").write_text("{}")
    ctx = ToolContext(workspace=workspace)
    w = await FileWriteTool().execute(
        {"path": ".verbix/auth.json", "content": "{}"}, ctx
    )
    assert w.is_error is True
    assert "secret" in w.observation.lower()
    (workspace / ".verbix" / "auth.yaml").write_text("k: v\n")
    e = await FileEditTool().execute(
        {"path": ".verbix/auth.yaml", "old_str": "k", "new_str": "v2"}, ctx
    )
    assert e.is_error is True
    assert "secret" in e.observation.lower()


@pytest.mark.asyncio
async def test_legitimate_verbix_paths_still_allowed(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / ".verbix" / "logs").mkdir(parents=True)
    (workspace / ".verbix" / "logs" / "app.log").write_text("hello\n")
    ctx = ToolContext(workspace=workspace)
    res = await FileReadTool().execute({"path": ".verbix/logs/app.log"}, ctx)
    assert res.is_error is False
    assert "hello" in res.observation


@pytest.mark.asyncio
async def test_bash_tool_is_always_sandboxed_allowlists_env_and_masks_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools.sandbox import probe_sandbox

    status = probe_sandbox()
    if not status.available:
        pytest.skip(f"Sandbox unavailable: {status.reason}")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".env").write_text("OPENROUTER_API_KEY=sk-dotenv-secret\n", encoding="utf-8")
    (workspace / ".verbix").mkdir()
    (workspace / ".verbix" / "auth.json").write_text(
        '{"openrouter":{"key":"sk-auth-secret"}}', encoding="utf-8"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-secret")
    monkeypatch.setenv("VERBIX_API_KEY", "sk-verbix-secret")
    monkeypatch.setenv("CUSTOM_UNLISTED_VAR", "should-not-appear")

    tool = BashTool()
    assert "bubblewrap" in tool.description.lower()
    ctx = ToolContext(workspace=workspace)

    # 1. Host env vars outside the allowlist (PATH, LANG, TERM) must not leak into child env
    env_res = await tool.execute({"command": "env"}, ctx)
    assert "sk-env-secret" not in env_res.observation
    assert "sk-verbix-secret" not in env_res.observation
    assert "should-not-appear" not in env_res.observation

    # 2. Workspace .env and .verbix/auth.json must be masked as /dev/null inside bwrap
    secret_res = await tool.execute({"command": "cat .env .verbix/auth.json"}, ctx)
    assert secret_res.observation.strip() == ""

    # 3. Cannot write outside workspace
    esc_res = await tool.execute({"command": "touch /usr/escape_test"}, ctx)
    assert esc_res.is_error is True


def test_worker_registry_omits_bash_when_sandbox_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from verbix.runtime.workers import build_worker_registry
    from verbix.tools.sandbox import SandboxStatus

    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=False, backend=None, reason="no sandbox on host", capabilities=""
        ),
    )
    reg = build_worker_registry(workspace=tmp_path)
    names = {d["function"]["name"] for d in reg.get_definitions()}
    assert "bash" not in names
    assert "read" in names


@pytest.mark.asyncio
async def test_spawn_worker_omits_bash_from_worker_config_when_sandbox_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from verbix.runtime.foreground import DEFAULT_WORKER_PROMPT
    from verbix.runtime.ledger import TaskLedger
    from verbix.runtime.worker import WorkerTask
    from verbix.runtime.workers import spawn_worker
    from verbix.tools.sandbox import SandboxStatus

    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=False, backend=None, reason="no sandbox on host", capabilities=""
        ),
    )
    monkeypatch.setattr(WorkerTask, "start", lambda self: None)
    runtime = SimpleNamespace(
        workspace=tmp_path,
        _task_counter=0,
        worker_system_prompt=DEFAULT_WORKER_PROMPT,
        worker_model=None,
        provider=MagicMock(),
        worker_echo=False,
        tasks={},
        _environments=[],
        ledger=TaskLedger(),
        _broadcast_event=lambda *a: None,
    )
    task_id = await spawn_worker(runtime, "inspect files")
    worker_agent = runtime.tasks[task_id].session
    assert "bash" not in worker_agent.config.tools
    tool_defs = worker_agent.get_tool_definitions() or []
    assert "bash" not in {d["function"]["name"] for d in tool_defs}


