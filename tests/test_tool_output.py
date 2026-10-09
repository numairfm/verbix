def test_tool_result_carries_attachments_default_empty():
    from verbix.tools.base import FileAttachment, ToolResult
    res = ToolResult("hi", attachments=(FileAttachment(data=b"x", mime="image/jpeg", name="a.jpg"),))
    assert res.observation == "hi" and res.attachments[0].mime == "image/jpeg"
    assert ToolResult("hi").attachments == ()


def test_over_limit_text_spills_and_rereads(tmp_path):
    from verbix.tools.shared import bind_output
    big = "line\n" * 3000
    out = bind_output(big, spill_dir=tmp_path, rel_to=tmp_path)
    assert "full content saved to" in out
    saved = out.split("saved to")[1].split("...")[0].strip()
    assert (tmp_path / saved).read_text() == big


def test_unwritable_spill_degrades_without_path(tmp_path):
    from verbix.tools.shared import bind_output
    (tmp_path / "blocker").write_text("x")
    out = bind_output("line\n" * 3000, spill_dir=tmp_path / "blocker" / "sub")
    assert "saved to" not in out


def test_resolve_output_limits_falls_back():
    from verbix.tools.shared import resolve_output_limits
    assert resolve_output_limits(None) == (2000, 51200)
    assert resolve_output_limits({"max_lines": "lots", "max_bytes": -5}) == (2000, 51200)
    assert resolve_output_limits({"max_lines": 10}) == (10, 51200)


def test_registry_spills_long_output_and_rereads(tmp_path):
    import asyncio
    from verbix.tools import build_registry
    reg = build_registry(["bash"], runtime=None, workspace=tmp_path)
    res = asyncio.run(reg.aexecute("bash", {"command": "seq 1 3000"}))
    assert "full content saved to" in res.observation
    saved = res.observation.split("saved to")[1].split("...")[0].strip()
    assert (tmp_path / saved).read_text().splitlines()[0] == "1"
