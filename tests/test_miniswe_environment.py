from verbix.agent.environments.local import LocalEnvironment


def test_local_environment_echo(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    res = env.execute({"command": "echo 'hello world'"})
    assert res["returncode"] == 0
    assert "hello world" in res["output"]

def test_local_environment_submission_token_is_not_intercepted(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    # The command should execute normally without raising Submitted
    res = env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho 'Done!'"})
    assert res["returncode"] == 0
    assert "Done!" in res["output"]



def test_local_environment_submission_inline_is_not_intercepted(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    res = env.execute({"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT: All 72 tests passed'"})
    assert res["returncode"] == 0
    assert "All 72 tests passed" in res["output"]


