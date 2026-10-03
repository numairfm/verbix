from __future__ import annotations

import asyncio
import os
from pathlib import Path

from verbix.cli import load_env
from verbix.core.events import TextDelta
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        return

    model_name = "google/gemma-4-26b-a4b-it"
    workspace = Path("scratch/test_workspace_30b").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"VALIDATING 30B-CLASS MODEL: {model_name}")
    print("Provider: Google Vertex (via OpenRouter routing)")
    print(f"Workspace: {workspace}")
    print("=" * 70)

    provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        api_key=api_key,
        max_tokens=1024,
        default_extra_body={"provider": {"order": ["Google"]}},
    )

    runtime = TaskRuntime(workspace=workspace, provider=provider, max_worker_turns=10)
    runtime.create_receptionist()

    # Turn 1: Request task spawn
    user_prompt = "Create a file named gemma_test.txt containing 'Gemma-4 Validated' in the workspace."
    print(f"\nUser Turn 1: {user_prompt}")

    reply = ""
    async for event in runtime.step_receptionist(user_input=user_prompt):
        if isinstance(event, TextDelta):
            reply += event.text

    print(f"Receptionist: {reply.strip()}")

    # Check spawned tasks
    tasks = runtime.list_all_tasks()
    print(f"\nTasks spawned: {len(tasks)}")
    if not tasks:
        print("FAIL: Receptionist did not spawn a background task.")
        return

    task = tasks[0]
    print(f"Task ID: {task.id}, Title: '{task.title}', Status: {task.status}")
    assert task.workspace == workspace.resolve()

    # Await background worker
    print(f"Awaiting background worker '{task.id}'...")
    bg_task = runtime._worker_tasks.get(task.id)
    if bg_task:
        try:
            await asyncio.wait_for(bg_task, timeout=60.0)
        except TimeoutError:
            print("FAIL: Background worker timed out.")
            return

    finished_task = runtime.get_task(task.id)
    assert finished_task is not None
    print(f"Worker status: {finished_task.status}, steps: {finished_task.step_count}")
    if finished_task.result:
        print(f"Speech summary: {finished_task.result.speech_summary}")
        print(f"Details: {finished_task.result.details[:150]}...")

    # Check file on disk
    target_file = workspace / "gemma_test.txt"
    if target_file.exists():
        content = target_file.read_text().strip()
        print(f"PASS: Target file exists with content: {content!r}")
        target_file.unlink(missing_ok=True)
    else:
        print(
            f"NOTE: Target file was not found on disk. Success={finished_task.result.success if finished_task.result else False}"
        )

    # Turn 2: Verify Receptionist consumes completion from history
    user_prompt_2 = "What did the background task accomplish?"
    print(f"\nUser Turn 2: {user_prompt_2}")

    reply_2 = ""
    async for event in runtime.step_receptionist(user_input=user_prompt_2):
        if isinstance(event, TextDelta):
            reply_2 += event.text

    print(f"Receptionist: {reply_2.strip()}")
    print("\n" + "=" * 70)
    print("30B MODEL (google/gemma-4-26b-a4b-it) VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
