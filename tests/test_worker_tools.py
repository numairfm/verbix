"""Worker registry contract: the background agent's tools."""

from verbix.runtime.workers import build_worker_registry


def test_worker_registry_has_main_tools():
    registry = build_worker_registry()
    names = {d["function"]["name"] for d in registry.get_definitions()}
    assert names == {
        "read",
        "write",
        "edit",
        "grep",
        "glob",
        "web_search",
        "web_fetch",
        "screenshot",
        "bash",
    }
    for name in names:
        assert name in registry._handlers
