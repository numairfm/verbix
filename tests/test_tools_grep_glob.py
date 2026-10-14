from __future__ import annotations

from pathlib import Path

import pytest

from verbix.tools import build_registry
from verbix.tools.base import ToolContext
from verbix.tools.glob import GlobTool
from verbix.tools.grep import GrepTool


@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "main.py").write_text("def hello():\n    print('Hello World')\n")
    (ws / "utils.py").write_text("def helper():\n    return 'help'\n")
    sub = ws / "sub"
    sub.mkdir()
    (sub / "test.py").write_text("def test_hello():\n    pass\n")
    (sub / "notes.txt").write_text("some notes here\n")
    return ws


@pytest.mark.asyncio
async def test_grep_basic_search(sample_workspace: Path):
    tool = GrepTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "hello"}, ctx)

    assert res.is_error is False
    assert "main.py:1:def hello():" in res.observation
    assert "sub/test.py:1:def test_hello():" in res.observation


@pytest.mark.asyncio
async def test_grep_case_sensitivity(sample_workspace: Path):
    tool = GrepTool()
    ctx = ToolContext(workspace=sample_workspace)

    # Case-insensitive by default
    res = await tool.execute({"pattern": "HELLO"}, ctx)
    assert res.is_error is False
    assert "main.py" in res.observation

    # Case-sensitive when specified
    res_sensitive = await tool.execute({"pattern": "HELLO", "case_sensitive": True}, ctx)
    assert res_sensitive.is_error is False
    assert "No matches found." in res_sensitive.observation


@pytest.mark.asyncio
async def test_grep_with_glob_filter(sample_workspace: Path):
    tool = GrepTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "here", "glob": "*.txt"}, ctx)

    assert res.is_error is False
    assert "notes.txt" in res.observation
    assert "main.py" not in res.observation


@pytest.mark.asyncio
async def test_grep_no_matches(sample_workspace: Path):
    tool = GrepTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "nonexistent_pattern_xyz"}, ctx)

    assert res.is_error is False
    assert "No matches found." in res.observation


@pytest.mark.asyncio
async def test_grep_refuses_escaping_workspace(sample_workspace: Path):
    tool = GrepTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "foo", "path": "../"}, ctx)

    assert res.is_error is True
    assert "escapes" in res.observation.lower()


@pytest.mark.asyncio
async def test_glob_basic_pattern(sample_workspace: Path):
    tool = GlobTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "*.py"}, ctx)

    assert res.is_error is False
    lines = res.observation.strip().splitlines()
    assert "main.py" in lines
    assert "utils.py" in lines
    assert "notes.txt" not in lines


@pytest.mark.asyncio
async def test_glob_recursive_pattern(sample_workspace: Path):
    tool = GlobTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "**/*.py"}, ctx)

    assert res.is_error is False
    lines = res.observation.strip().splitlines()
    assert any("sub/test.py" in l or "sub\\test.py" in l for l in lines)


@pytest.mark.asyncio
async def test_glob_no_matches(sample_workspace: Path):
    tool = GlobTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "*.nonexistent"}, ctx)

    assert res.is_error is False
    assert "No files matched pattern." in res.observation


@pytest.mark.asyncio
async def test_glob_refuses_escaping_workspace(sample_workspace: Path):
    tool = GlobTool()
    ctx = ToolContext(workspace=sample_workspace)
    res = await tool.execute({"pattern": "*.py", "path": "../"}, ctx)

    assert res.is_error is True
    assert "escapes" in res.observation.lower()


def test_registry_builds_grep_and_glob(sample_workspace: Path):
    registry = build_registry(["grep", "glob"], workspace=sample_workspace)
    defs = registry.get_definitions()
    names = {d["function"]["name"] for d in defs}
    assert "grep" in names
    assert "glob" in names


@pytest.mark.asyncio
async def test_grep_and_glob_stdlib_fallback_honors_gitignore_and_skips_binary(
    sample_workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("verbix.tools.grep.find_ripgrep_binary", lambda: None)
    monkeypatch.setattr("verbix.tools.glob.find_ripgrep_binary", lambda: None)

    (sample_workspace / ".gitignore").write_text("ignored_dir/\n*.log\n!keep.log\n", encoding="utf-8")
    ignored = sample_workspace / "ignored_dir"
    ignored.mkdir()
    (ignored / "secret.py").write_text("def hello_secret(): pass\n", encoding="utf-8")
    (sample_workspace / "drop.log").write_text("hello drop\n", encoding="utf-8")
    (sample_workspace / "keep.log").write_text("hello keep\n", encoding="utf-8")
    (sample_workspace / "binary.py").write_bytes(b"\x00\x01\x02hello binary\n")
    (sample_workspace / "latin1.py").write_bytes(b"def hello_latin(): # caf\xe9\n")
    (sample_workspace / "auth.json").write_text('{"key": "hello_secret_auth"}\n', encoding="utf-8")
    (sample_workspace / ".env").write_text("SECRET=hello_dotenv\n", encoding="utf-8")
    try:
        (sample_workspace / "escape_link.py").symlink_to(sample_workspace / ".env")
    except OSError:
        pass

    # Nested .gitignore inside sub/
    (sample_workspace / "sub" / ".gitignore").write_text("*.tmp\n!keep_sub.tmp\n", encoding="utf-8")
    (sample_workspace / "sub" / "drop_sub.tmp").write_text("hello drop sub\n", encoding="utf-8")
    (sample_workspace / "sub" / "keep_sub.tmp").write_text("hello keep sub\n", encoding="utf-8")

    ctx = ToolContext(workspace=sample_workspace)

    glob_res = await GlobTool().execute({"pattern": "**/*"}, ctx)
    assert glob_res.is_error is False
    glob_lines = glob_res.observation.splitlines()
    assert "sub/test.py" in glob_lines
    assert "keep.log" in glob_lines
    assert "sub/keep_sub.tmp" in glob_lines
    assert "drop.log" not in glob_lines
    assert "sub/drop_sub.tmp" not in glob_lines
    assert "auth.json" not in glob_lines
    assert "escape_link.py" not in glob_lines
    assert not any("ignored_dir" in line for line in glob_lines)

    grep_res = await GrepTool().execute({"pattern": "hello"}, ctx)
    assert grep_res.is_error is False
    assert "main.py:1:def hello():" in grep_res.observation
    assert "sub/test.py:1:def test_hello():" in grep_res.observation
    assert "keep.log:1:hello keep" in grep_res.observation
    assert "sub/keep_sub.tmp:1:hello keep sub" in grep_res.observation
    assert "latin1.py:1:def hello_latin():" in grep_res.observation
    assert "drop.log" not in grep_res.observation
    assert "drop_sub.tmp" not in grep_res.observation
    assert "secret.py" not in grep_res.observation
    assert "binary.py" not in grep_res.observation
    assert "hello_secret_auth" not in grep_res.observation
    assert "hello_dotenv" not in grep_res.observation


@pytest.mark.asyncio
async def test_grep_and_glob_rg_fast_path_excludes_secret_files(sample_workspace: Path):
    (sample_workspace / "auth.json").write_text('{"key": "hello_rg_auth_secret"}\n', encoding="utf-8")
    (sample_workspace / ".env").write_text("KEY=hello_rg_dotenv_secret\n", encoding="utf-8")
    ctx = ToolContext(workspace=sample_workspace)

    glob_res = await GlobTool().execute({"pattern": "*"}, ctx)
    assert "auth.json" not in glob_res.observation
    assert ".env" not in glob_res.observation

    grep_res = await GrepTool().execute({"pattern": "hello_rg_"}, ctx)
    assert "hello_rg_auth_secret" not in grep_res.observation
    assert "hello_rg_dotenv_secret" not in grep_res.observation

    grep_dotenv = await GrepTool().execute({"pattern": "hello_rg_", "glob": ".env"}, ctx)
    assert "hello_rg_dotenv_secret" not in grep_dotenv.observation


