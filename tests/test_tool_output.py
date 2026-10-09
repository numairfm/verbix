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


def test_agent_config_carries_tool_output():
    from verbix.agent.agents.default import AgentConfig
    assert AgentConfig(tool_output={"max_lines": 10}).tool_output == {"max_lines": 10}


def test_yaml_limits_drive_registry(tmp_path):
    import asyncio
    from verbix.tools import build_registry
    from verbix.tools.shared import resolve_output_limits
    cfg = {"tools": ["bash"], "tool_output": {"max_lines": 10, "max_bytes": 200}}
    reg = build_registry(cfg["tools"], runtime=None, workspace=tmp_path,
        output_limits=resolve_output_limits(cfg.get("tool_output")))
    res = asyncio.run(reg.aexecute("bash", {"command": "seq 1 100"}))
    assert "full content saved to" in res.observation


def test_prune_spills_removes_only_old_files(tmp_path):
    import os
    import time

    from verbix.tools.shared import prune_spills
    spill = tmp_path / "logs"
    spill.mkdir()
    old = spill / "tool_old"
    fresh = spill / "tool_fresh"
    old.write_text("old")
    fresh.write_text("fresh")
    aged = time.time() - 8 * 86400
    os.utime(old, (aged, aged))
    prune_spills(spill)
    assert not old.exists()
    assert fresh.exists()
    prune_spills(tmp_path / "missing")  # missing dir is a no-op


def test_large_photo_returns_small_attachment(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    img = Image.new("RGB", (3000, 2000), "blue")
    img.save(tmp_path / "big.jpg")
    res = asyncio.run(FileReadTool().execute({"path": "big.jpg"}, ToolContext(workspace=tmp_path)))
    assert len(res.attachments) == 1 and len(res.attachments[0].data) < 150 * 1024
    assert "MSWEA_MULTIMODAL" not in res.observation


def test_gif_read_returns_first_frame_only(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    frames = [Image.new("RGB", (100, 100), c) for c in ("red", "green")]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    res = asyncio.run(FileReadTool().execute({"path": "a.gif"}, ToolContext(workspace=tmp_path)))
    assert len(res.attachments) == 1 and res.attachments[0].mime == "image/jpeg"


def test_rgba_flattens_onto_white(tmp_path):
    from PIL import Image
    from verbix.tools.base import ToolContext
    from verbix.tools.read import FileReadTool
    import asyncio
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(tmp_path / "t.png")
    res = asyncio.run(FileReadTool().execute({"path": "t.png"}, ToolContext(workspace=tmp_path)))
    assert res.attachments[0].data[:2] == b"\xff\xd8"
