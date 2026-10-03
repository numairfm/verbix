from __future__ import annotations

from pathlib import Path

import pytest

from verbix.ipc.client import VerbixClient
from verbix.ipc.server import VerbixServer
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime


@pytest.mark.asyncio
async def test_ipc_server_and_client_uds(tmp_path: Path):
    socket_path = tmp_path / "verbix.sock"
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    server = VerbixServer(runtime=runtime, socket_path=socket_path)
    await server.start()
    assert socket_path.exists()

    client = VerbixClient(socket_path=socket_path)
    await client.connect()

    # 1. Ping
    pong = await client.ping()
    assert pong is True

    # 2. Status
    status = await client.get_status()
    assert "no active" in status.lower()

    # 3. Spawn a task on runtime and cancel via IPC
    task_id = await runtime.spawn_task(title="Build Task", prompt="build")
    status_with_task = await client.get_status(task_id=task_id)
    assert task_id in status_with_task

    cancelled = await client.cancel_task(task_id)
    assert cancelled is True
    cancelled_task = runtime.get_task(task_id)
    assert cancelled_task is not None
    assert cancelled_task.status == "cancelled"

    # 4. Toggle via IPC
    toggle_res = await client.toggle()
    assert isinstance(toggle_res, dict)

    await client.close()
    await server.stop()
    assert not socket_path.exists()
