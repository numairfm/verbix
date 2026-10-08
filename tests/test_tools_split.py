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
