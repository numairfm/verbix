"""Worker registry contract: the background agent's tools."""

from verbix.runtime.tools import build_worker_registry


def test_worker_registry_has_five_main_tools():
    registry = build_worker_registry()
    names = {d["function"]["name"] for d in registry.get_definitions()}
    assert names == {"read", "write", "edit", "bash", "finish"}
    for name in names:
        assert name in registry._handlers
