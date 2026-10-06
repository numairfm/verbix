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
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, check=False).stdout.strip()
    assert not alive

@pytest.mark.asyncio
async def test_aexecute_daemon_holding_stdout_survives_fast():
    import time
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    marker = f"verbix_daemon_test_{os.getpid()}"

    t0 = time.monotonic()
    try:
        res = await env.aexecute({"command": f"exec -a {marker} sleep 30 & echo done"})
    finally:
        pass
    dt = time.monotonic() - t0
    assert res["returncode"] == 0
    assert "done" in res["output"]
    assert dt < 5  # must not block to full timeout on the pipe-holding grandchild

    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, check=False).stdout.strip()
    assert alive  # intentional daemon survives normal completion
    subprocess.run(["pkill", "-f", marker], capture_output=True, check=False)
