from pathlib import Path


def test_split_tools_importable_from_new_paths():
    from verbix.tools.edit import FileEditTool
    from verbix.tools.read import FileReadTool
    from verbix.tools.shared import resolve_contained_path
    from verbix.tools.write import FileWriteTool
    assert FileReadTool().name == "read"
    assert FileWriteTool().name == "write"
    assert FileEditTool().name == "edit"
    assert callable(resolve_contained_path)


def test_engine_offers_no_tools_when_registry_empty():
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.agent.voice import VoiceAgent, VoiceConfig
    agent = VoiceAgent(model=None, env=object(), tools=ToolRegistry(), config=VoiceConfig(tools=[]))
    assert agent.get_tool_definitions() is None


def test_unknown_tool_name_raises_loudly():
    import pytest

    from verbix.tools import build_registry
    with pytest.raises(KeyError, match="raed"):
        build_registry(["raed"], runtime=None)


def test_no_tool_names_in_engine():
    import pathlib
    for path in ("src/verbix/agent/voice.py", "src/verbix/agent/worker.py"):
        src = pathlib.Path(path).read_text(encoding="utf-8")
        body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
        for name in ("FOREGROUND_TOOL_NAMES", "finish", "spawn", "status", "cancel",
                     "read", "write", "edit", "bash"):
            assert f'"{name}"' not in body and f"'{name}'" not in body, f"{name} in {path}"


def test_runtime_modules_exist_with_one_job_header():
    import pathlib
    for mod in ("foreground", "workers", "announcer"):
        text = pathlib.Path(f"src/verbix/runtime/{mod}.py").read_text()
        assert text.splitlines()[0].startswith('"""')
        assert mod in text[:500].lower(), mod


class FakeRuntime:
    workspace = Path(".")
    async def spawn(self, objective):
        assert objective
        return "task-99"


def test_worker_with_spawn_tool_actually_spawns():
    import asyncio

    from verbix.tools import build_registry
    from verbix.tools.base import ToolContext
    from verbix.tools.spawn import SpawnTool
    reg = build_registry(["read", "spawn"], runtime=FakeRuntime(), workspace=Path("."))
    assert "spawn" in [d["function"]["name"] for d in reg.get_definitions()]
    res = asyncio.run(SpawnTool(FakeRuntime()).execute({"objective": "do it"}, ToolContext(workspace=Path("."))))
    assert "task-99" in res.observation


def test_worker_read_via_agent_gets_context(tmp_path):
    import asyncio

    from verbix.agent.voice import VoiceAgent, VoiceConfig
    from verbix.tools import build_registry
    (tmp_path / "note.txt").write_text("hello")
    agent = VoiceAgent(model=None, env=None, tools=build_registry(["read"], runtime=None, workspace=tmp_path), config=VoiceConfig(tools=["read"]))
    res = asyncio.run(agent._execute_tool("read", {"path": "note.txt"}))
    assert res.is_error is False and "hello" in res.observation
