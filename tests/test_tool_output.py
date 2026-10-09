def test_tool_result_carries_attachments_default_empty():
    from verbix.tools.base import FileAttachment, ToolResult
    res = ToolResult("hi", attachments=(FileAttachment(data=b"x", mime="image/jpeg", name="a.jpg"),))
    assert res.observation == "hi" and res.attachments[0].mime == "image/jpeg"
    assert ToolResult("hi").attachments == ()
