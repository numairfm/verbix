from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from verbix.cli import load_env
from verbix.core.events import TextDelta, ToolCallStarted, TurnComplete
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime


async def run_scenario_turn(
    runtime: TaskRuntime, user_text: str
) -> tuple[str, list[dict[str, Any]]]:
    reply = ""
    tool_calls: list[dict[str, Any]] = []
    async for event in runtime.step_receptionist(user_input=user_text):
        if isinstance(event, TextDelta):
            reply += event.text
        elif isinstance(event, ToolCallStarted):
            tool_calls.append({"name": event.tool_name, "args": event.arguments})
        elif isinstance(event, TurnComplete) and not reply and event.final_text:
            reply = event.final_text
    return reply.strip(), tool_calls


async def test_scenarios_for_model(
    model_name: str, api_key: str, default_extra_body: dict[str, Any] | None = None
) -> dict[str, bool]:
    print(f"\n{'=' * 75}\nTESTING ROBUSTNESS SCENARIO SUITE: {model_name}\n{'=' * 75}")
    ws = Path("scratch/robustness_ws").resolve()
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)

    provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        api_key=api_key,
        max_tokens=512,
        default_extra_body=default_extra_body,
    )
    runtime = TaskRuntime(workspace=ws, provider=provider)
    runtime.create_receptionist()

    results: dict[str, bool] = {}

    # Scenario 1: Ambiguous request
    print("1. Scenario: Ambiguous Request ('Can you make it better?')")
    reply, tools = await run_scenario_turn(runtime, "Can you make it better?")
    print(f"   Reply: {reply[:100]}...")
    # PASS if it asks for clarification and does NOT blindly spawn a task with arbitrary commands
    sc1_pass = len(tools) == 0 and (
        "what" in reply.lower()
        or "clarif" in reply.lower()
        or "how" in reply.lower()
        or "better" in reply.lower()
        or "mean" in reply.lower()
    )
    results["Ambiguous request"] = sc1_pass
    print(f"   Result: {'PASS' if sc1_pass else 'FAIL'}")

    # Scenario 2: Concurrent tasks request
    print("\n2. Scenario: Concurrent Tasks Request ('Run tests and check git status')")
    reply, tools = await run_scenario_turn(
        runtime,
        "Run pytest in the background and also check git status in another task.",
    )
    print(f"   Tools called: {[t['name'] for t in tools]}")
    sc2_pass = (
        any(t["name"] == "task_spawn" for t in tools)
        or len(runtime.list_all_tasks()) >= 1
    )
    results["Concurrent tasks"] = sc2_pass
    print(f"   Result: {'PASS' if sc2_pass else 'FAIL'}")

    # Ensure at least one task is active for mid-task status / steer / cancel
    active_tasks = runtime.list_active_tasks()
    if not active_tasks:
        await runtime.spawn_task("test task", "sleep 60")

    active_id = runtime.list_active_tasks()[0].id

    # Scenario 3: Status mid-task
    print(f"\n3. Scenario: Status Mid-Task ('What is the status of {active_id}?')")
    reply, tools = await run_scenario_turn(
        runtime, f"What is the current status of {active_id}?"
    )
    print(f"   Reply: {reply[:120]}...")
    sc3_pass = (
        "running" in reply.lower()
        or active_id in reply
        or any(t["name"] == "task_status" for t in tools)
    )
    results["Status mid-task"] = sc3_pass
    print(f"   Result: {'PASS' if sc3_pass else 'FAIL'}")

    # Scenario 4: Steer
    print(f"\n4. Scenario: Steer Active Task ('On {active_id}, focus on unit tests')")
    reply, tools = await run_scenario_turn(
        runtime, f"On {active_id}, steer it to focus strictly on unit tests."
    )
    print(f"   Tools called: {[t['name'] for t in tools]}")
    sc4_pass = any(t["name"] == "task_steer" for t in tools)
    results["Steer task"] = sc4_pass
    print(f"   Result: {'PASS' if sc4_pass else 'FAIL'}")

    # Scenario 5: Cancel
    if not runtime.list_active_tasks():
        cancel_target = await runtime.spawn_task(
            "task to cancel", "keep running and waiting"
        )
    else:
        cancel_target = runtime.list_active_tasks()[0].id

    print(f"\n5. Scenario: Cancel Task ('Cancel {cancel_target}')")
    reply, tools = await run_scenario_turn(runtime, f"Cancel {cancel_target}.")
    print(f"   Reply: {reply[:100]}...")
    print(f"   Tools called: {[t['name'] for t in tools]}")
    task_after = runtime.get_task(cancel_target)
    sc5_pass = any(t["name"] == "task_cancel" for t in tools) or (
        task_after is not None and not task_after.is_running
    )
    results["Cancel task"] = sc5_pass
    print(f"   Result: {'PASS' if sc5_pass else 'FAIL'}")

    # Scenario 6: "Thank you" after completion
    print(
        "\n6. Scenario: 'Thank you' after completion ('Thank you, that's all for now.')"
    )
    reply, tools = await run_scenario_turn(runtime, "Thank you, that's all for now.")
    print(f"   Reply: {reply[:100]}...")
    # PASS if polite conversational response and NO spurious tools called
    sc6_pass = len(tools) == 0 and len(reply) > 0
    results["Thank you / farewell"] = sc6_pass
    print(f"   Result: {'PASS' if sc6_pass else 'FAIL'}")

    await provider.aclose()
    return results


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        return

    models = [
        {
            "name": "google/gemini-3.5-flash-lite",
            "extra": None,
            "label": "Gemini 3.5 Flash-Lite",
        },
        {
            "name": "google/gemma-4-26b-a4b-it",
            "extra": {"provider": {"order": ["Google"]}},
            "label": "Gemma 4 26B",
        },
        {
            "name": "google/gemini-2.5-pro",
            "extra": None,
            "label": "Gemini 2.5 Pro (Strong Model)",
        },
    ]

    all_results: dict[str, dict[str, bool]] = {}

    for m in models:
        res = await test_scenarios_for_model(
            m["name"], api_key, default_extra_body=m["extra"]
        )
        all_results[m["label"]] = res

    # Summary table
    scenarios = [
        "Ambiguous request",
        "Concurrent tasks",
        "Status mid-task",
        "Steer task",
        "Cancel task",
        "Thank you / farewell",
    ]

    print("\n" + "=" * 80)
    print("ROBUSTNESS SCENARIO SUITE RESULTS")
    print("=" * 80)
    header = f"{'Scenario':<25} | " + " | ".join(f"{m['label']:<15}" for m in models)
    print(header)
    print("-" * 80)
    for sc in scenarios:
        row = f"{sc:<25} | "
        for m in models:
            passed = all_results[m["label"]].get(sc, False)
            status = "PASS" if passed else "FAIL"
            row += f"{status:<15} | "
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
