from pathlib import Path

import yaml

from verbix.runtime.runtime import DEFAULT_WORKER_PROMPT


def test_worker_config_exists_and_swe_worker_removed():
    config_dir = Path("src/verbix/minisweagent/config")
    worker_yaml = config_dir / "worker.yaml"
    swe_worker_yaml = config_dir / "swe_worker.yaml"

    assert worker_yaml.is_file(), "worker.yaml must exist"
    assert not swe_worker_yaml.exists(), "swe_worker.yaml must be removed"


def test_worker_prompt_contains_zero_examples_and_has_invariants():
    config_dir = Path("src/verbix/minisweagent/config")
    worker_yaml = config_dir / "worker.yaml"
    assert worker_yaml.is_file()
    raw_text = worker_yaml.read_text(encoding="utf-8")
    worker_data = yaml.safe_load(raw_text)
    prompt = worker_data["agent"]["instance_template"] + " " + worker_data["agent"]["system_template"]

    # Assert zero command examples
    forbidden_examples = ["npm test", "cargo test", "nl -ba", "sed -n", "pytest"]
    for ex in forbidden_examples:
        assert ex not in prompt, f"Forbidden example found in worker prompt: {ex}"

    # Assert tools include bash and finish
    tools = worker_data["agent"].get("tools", [])
    assert "bash" in tools, "worker must have bash tool"
    assert "finish" in tools, "worker must have finish tool"

    # Assert execution invariants
    assert "Discover" in prompt
    assert "Execute" in prompt
    assert "Verify" in prompt
    assert "finish" in prompt

    # Assert zero benchmark submission token references anywhere in worker.yaml
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in raw_text, (
        "worker.yaml must not contain legacy benchmark token COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )


def test_default_worker_prompt_invariants():
    assert "Discover" in DEFAULT_WORKER_PROMPT
    assert "execute" in DEFAULT_WORKER_PROMPT.lower()
    assert "verify" in DEFAULT_WORKER_PROMPT.lower()
    assert "finish" in DEFAULT_WORKER_PROMPT.lower()
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in DEFAULT_WORKER_PROMPT


def test_companion_prompt_invariants():
    config_dir = Path("src/verbix/minisweagent/config")
    voice_data = yaml.safe_load((config_dir / "voice.yaml").read_text(encoding="utf-8"))
    sys_prompt = voice_data["agent"]["system_template"]

    assert "never guess" in sys_prompt.lower() or "do not guess" in sys_prompt.lower()
    assert "objective" in sys_prompt.lower()
    assert "spawn" in sys_prompt.lower()
    assert "persona" in sys_prompt.lower() or "spoken" in sys_prompt.lower()


def test_default_verbix_prompt_invariants():
    from verbix.runtime.runtime import DEFAULT_VERBIX_PROMPT

    assert "never guess" in DEFAULT_VERBIX_PROMPT.lower() or "do not guess" in DEFAULT_VERBIX_PROMPT.lower()
    assert "objective" in DEFAULT_VERBIX_PROMPT.lower()
    assert "spawn" in DEFAULT_VERBIX_PROMPT.lower()
