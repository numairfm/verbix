def test_split_tools_importable_from_new_paths():
    from verbix.tools.edit import FileEditTool
    from verbix.tools.read import FileReadTool
    from verbix.tools.shared import resolve_contained_path
    from verbix.tools.write import FileWriteTool
    assert FileReadTool().name == "read"
    assert FileWriteTool().name == "write"
    assert FileEditTool().name == "edit"
    assert callable(resolve_contained_path)
