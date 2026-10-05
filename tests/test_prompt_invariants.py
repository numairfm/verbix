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
    worker_data = yaml.safe_load(worker_yaml.read_text(encoding="utf-8"))
    prompt = worker_data["agent"]["instance_template"] + " " + worker_data["agent"]["system_template"]

    # Assert zero command examples
    forbidden_examples = ["npm test", "cargo test", "nl -ba", "sed -n", "pytest"]
    for ex in forbidden_examples:
        assert ex not in prompt, f"Forbidden example found in worker prompt: {ex}"

    # Assert general execution invariants
    assert "Discover" in prompt
    assert "Execute" in prompt
    assert "Verify" in prompt
    assert "Submit" in prompt
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in prompt


def test_default_worker_prompt_invariants():
    assert "Discover" in DEFAULT_WORKER_PROMPT
    assert "execute" in DEFAULT_WORKER_PROMPT.lower()
    assert "verify" in DEFAULT_WORKER_PROMPT.lower()
    assert "submit" in DEFAULT_WORKER_PROMPT.lower()
