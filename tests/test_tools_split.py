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
    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.agent.tools.registry import ToolRegistry
    agent = DefaultAgent(model=None, env=object(), tools=ToolRegistry(), config=AgentConfig(tools=[]))
    assert agent.get_tool_definitions() is None


def test_unknown_tool_name_raises_loudly():
    import pytest

    from verbix.tools import build_registry
    with pytest.raises(KeyError, match="raed"):
        build_registry(["raed"], runtime=None)


def test_no_tool_names_in_engine():
    import pathlib
    src = pathlib.Path("src/verbix/agent/agents/default.py").read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
    for name in ("FOREGROUND_TOOL_NAMES", "finish", "spawn", "status", "cancel",
                 "read", "write", "edit", "bash"):
        assert f'"{name}"' not in body and f"'{name}'" not in body, name


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


def test_sync_path_routes_through_registry():
    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.agent.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register("bash", {"function": {"name": "bash"}}, lambda args: "routed-marker-xyz")
    agent = DefaultAgent(model=None, env=object(), tools=reg, config=AgentConfig(tools=[]))
    out = agent.execute_actions({"extra": {"actions": [{"name": "bash", "arguments": {"command": "x"}}]}})
    assert out[0].get("returncode", -1) == -1
    assert "routed-marker-xyz" in str(out[0])


def test_worker_read_via_agent_gets_context(tmp_path):
    import asyncio

    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.tools import build_registry
    (tmp_path / "note.txt").write_text("hello")
    agent = DefaultAgent(model=None, env=None, tools=build_registry(["read"], runtime=None, workspace=tmp_path), config=AgentConfig(tools=["read"]))
    res = asyncio.run(agent._execute_tool("read", {"path": "note.txt"}))
    assert res.is_error is False and "hello" in res.observation
