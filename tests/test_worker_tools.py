"""Worker registry contract: the background agent's tools."""

from verbix.runtime.workers import build_worker_registry
from verbix.tools.sandbox import probe_sandbox


def test_worker_registry_has_main_tools():
    registry = build_worker_registry()
    names = {d["function"]["name"] for d in registry.get_definitions()}
    expected = {
        "read",
        "write",
        "edit",
        "grep",
        "glob",
        "web_search",
        "web_fetch",
        "screenshot",
        "clipboard",
        "media",
        "app",
        "open",
        "system",
        "timer",
        "memory",
    }
    if probe_sandbox().available:
        expected.add("bash")
    assert names == expected
    for name in names:
        assert name in registry._handlers
