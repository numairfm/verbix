import pytest

from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.minisweagent.exceptions import Submitted


def test_local_environment_echo(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    res = env.execute({"command": "echo 'hello world'"})
    assert res["returncode"] == 0
    assert "hello world" in res["output"]

def test_local_environment_submission(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    with pytest.raises(Submitted) as exc_info:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nDone!"})
    assert "Done!" in exc_info.value.messages[0]["content"]
