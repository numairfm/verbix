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
