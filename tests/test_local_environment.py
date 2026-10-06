import asyncio
import os
import subprocess
import pytest
from verbix.minisweagent.environments.local import LocalEnvironment

@pytest.mark.asyncio
async def test_local_env_kills_process_group_on_cancel():
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    marker = f"verbix_kill_test_{os.getpid()}"

    t = asyncio.create_task(env.aexecute({"command": f"exec -a {marker} sleep 10"}))
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    await asyncio.sleep(0.1)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.strip()
    assert not alive
