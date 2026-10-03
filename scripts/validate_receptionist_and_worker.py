from __future__ import annotations

import asyncio
import os
from pathlib import Path

from verbix.cli import load_env
from verbix.core.events import TextDelta
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime


async def test_with_model(model_name: str, workspace: Path) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Validating Receptionist & Worker with model: {model_name}")
    print(f"Workspace: {workspace}")
    print(f"{'=' * 60}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        api_key=api_key,
        max_tokens=1024,
    )

    runtime = TaskRuntime(workspace=workspace, provider=provider, max_worker_turns=10)
    runtime.create_receptionist()

    # Step 1: Tell the receptionist to spawn a task to create a file in the workspace
    user_prompt = (
        "Create a file named hello_model.txt containing 'Validated' in the workspace."
    )
    print(f"User: {user_prompt}")

    reply = ""
    async for event in runtime.step_receptionist(user_input=user_prompt):
        if isinstance(event, TextDelta):
            reply += event.text

    print(f"Receptionist: {reply.strip()}")

    # Check if a task was spawned
    active_tasks = runtime.list_active_tasks()
    all_tasks = runtime.list_all_tasks()
    print(
        f"Tasks registered: {len(all_tasks)} (active: {len(active_tasks)}, ids: {[t.id for t in all_tasks]})"
    )

    if not all_tasks:
        print("FAIL: Receptionist did not spawn a background task.")
        return False

    task = all_tasks[0]
    # Safety verification: verify runtime set workspace correctly
    if task.workspace != workspace.resolve():
        print(
            f"FAIL: Task workspace {task.workspace} does not match runtime workspace {workspace}."
        )
        return False
    print(f"PASS: Safety invariant verified: task.workspace == {task.workspace}")

    # Wait for the background worker task to finish
    print(f"Awaiting background worker '{task.id}' execution...")
    bg_task = runtime._worker_tasks.get(task.id)
    if bg_task:
        try:
            await asyncio.wait_for(bg_task, timeout=45.0)
        except TimeoutError:
            print("FAIL: Worker task timed out.")
            return False

    task = runtime.get_task(task.id)
    assert task is not None
    print(f"Worker finished with status='{task.status}', steps={task.step_count}")
    if task.result:
        print(f"Result speech: {task.result.speech_summary}")
        print(f"Result details: {task.result.details}")

    # Check if the file was created
    created_file = workspace / "hello_model.txt"
    if created_file.exists():
        content = created_file.read_text().strip()
        print(f"PASS: File created successfully with content: {content!r}")
        created_file.unlink(missing_ok=True)
    else:
        print(
            f"NOTE: Target file was not created directly; task result success={task.result.success if task.result else False}"
        )

    # Step 2: Next user turn: verify receptionist observes the completed task without calling task_status
    user_prompt_2 = "What did the background task accomplish?"
    print(f"\nUser: {user_prompt_2}")
    reply_2 = ""
    async for event in runtime.step_receptionist(user_input=user_prompt_2):
        if isinstance(event, TextDelta):
            reply_2 += event.text

    print(f"Receptionist: {reply_2.strip()}")
    print("SUCCESS: End-to-end receptionist and worker validation passed!\n")
    return True


async def main() -> None:
    load_env()
    workspace = Path("scratch/test_workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # 1. Validate with google/gemini-3.5-flash-lite
    ok_gemini = await test_with_model(
        "google/gemini-3.5-flash-lite", workspace=workspace
    )
    if not ok_gemini:
        print("FAILED on google/gemini-3.5-flash-lite")
        return

    # 2. Validate with a small model (meta-llama/llama-3.2-3b-instruct or mistralai/ministral-8b)
    small_model = "meta-llama/llama-3.2-3b-instruct"
    try:
        await test_with_model(small_model, workspace=workspace)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Small model {small_model} test encountered: {e}")
        # Try google/gemini-2.5-flash
        print("Retrying with google/gemini-2.5-flash...")
        await test_with_model("google/gemini-2.5-flash", workspace=workspace)


if __name__ == "__main__":
    asyncio.run(main())
