from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.runtime.tools import CancelTool, SpawnTool, StatusTool
from verbix.tools.base import ToolContext


@pytest.fixture
def dummy_runtime():
    runtime = MagicMock()
    runtime.workspace = MagicMock()
    runtime.spawn = AsyncMock(return_value="task-1")
    runtime.status = MagicMock(return_value="Active tasks:\n- task-1: 'build binary' (running)")
    runtime.cancel = AsyncMock(return_value="Cancelled task-1 ('build binary').")
    return runtime


@pytest.mark.asyncio
async def test_spawn_tool_execution(dummy_runtime):
    tool = SpawnTool(dummy_runtime)
    ctx = ToolContext(dummy_runtime.workspace)

    assert tool.parameters_schema["required"] == ["objective"]
    assert "objective" in tool.parameters_schema["properties"]

    # Missing objective argument
    err_res = await tool.execute({}, ctx)
    assert err_res.is_error is True
    assert "error: 'objective' is required" in err_res.observation

    # Valid objective argument
    res = await tool.execute({"objective": "build binary"}, ctx)
    assert res.is_error is False
    assert res.observation == "task-1 accepted. Confirm starting in one short sentence and call no further tools."
    dummy_runtime.spawn.assert_awaited_once_with("build binary")


@pytest.mark.asyncio
async def test_status_tool_execution(dummy_runtime):
    tool = StatusTool(dummy_runtime)
    ctx = ToolContext(dummy_runtime.workspace)

    # Status without task_id
    res = await tool.execute({}, ctx)
    assert res.is_error is False
    assert "task-1" in res.observation
    dummy_runtime.status.assert_called_with(task_id=None, full=False)

    # Status with task_id and full=True
    dummy_runtime.status.return_value = "Detailed task-1 log"
    res_full = await tool.execute({"task_id": "task-1", "full": True}, ctx)
    assert res_full.is_error is False
    assert res_full.observation == "Detailed task-1 log"
    dummy_runtime.status.assert_called_with(task_id="task-1", full=True)


@pytest.mark.asyncio
async def test_cancel_tool_execution(dummy_runtime):
    tool = CancelTool(dummy_runtime)
    ctx = ToolContext(dummy_runtime.workspace)

    # Cancel without task_id
    res = await tool.execute({}, ctx)
    assert res.is_error is False
    assert "Cancelled" in res.observation
    dummy_runtime.cancel.assert_awaited_once_with(task_id=None)

    # Cancel specific task
    dummy_runtime.cancel.reset_mock()
    await tool.execute({"task_id": "task-2"}, ctx)
    dummy_runtime.cancel.assert_awaited_once_with(task_id="task-2")
