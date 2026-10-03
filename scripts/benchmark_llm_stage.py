from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2

from verbix.cli import load_env
from verbix.core.models import CanonicalMessage
from verbix.provider.client import ProviderClient
from verbix.provider.errors import ProviderError
from verbix.provider.models import (
    TextDelta,
    ToolDefinition,
    ToolFunction,
    Usage,
)
from verbix.runtime.tasks import TaskRuntime


def calc_stats(values: list[float]) -> tuple[float, float]:
    """Returns (p50, p95) for a list of numeric values."""
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]
    return p50, p95


async def measure_endpoint_rtt(
    base_url: str, api_key: str | None, runs: int = 5
) -> tuple[float, float]:
    """Measures raw network round-trip time (RTT) to the endpoint in milliseconds."""
    client = ProviderClient(base_url=base_url, model="probe", api_key=api_key)
    await client.warm()
    http_client = client._get_client()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    rtts: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await http_client.get(f"{base_url}/models", headers=headers)
        rtt = (time.perf_counter() - t0) * 1000.0
        rtts.append(rtt)
        await asyncio.sleep(0.05)
    await client.aclose()
    return calc_stats(rtts)


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        return

    print("=" * 80)
    print("REPRESENTATIVE INTERLEAVED LLM STAGE LATENCY BENCHMARK")
    print("=" * 80)

    # 1. Raw Network RTT measurement
    print("Measuring raw network RTT to https://openrouter.ai/api/v1...")
    rtt_p50, rtt_p95 = await measure_endpoint_rtt(
        "https://openrouter.ai/api/v1", api_key=api_key, runs=5
    )
    print(f"Network RTT: p50 = {rtt_p50:5.1f} ms | p95 = {rtt_p95:5.1f} ms\n")

    # 2. Extract Real Receptionist System Prompt & Tool Schemas
    dummy_provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1", model="mock", api_key=api_key
    )
    runtime = TaskRuntime(workspace=Path.cwd(), provider=dummy_provider)
    receptionist_session = runtime.create_receptionist()
    real_system_prompt = receptionist_session.system_prompt
    real_tools = [
        ToolDefinition(
            function=ToolFunction(
                name=t.name,
                description=t.description,
                parameters=t.parameters_schema,
            )
        )
        for t in runtime.get_receptionist_tools()
    ]

    user_prompt = "The build finished successfully in 12 seconds."
    messages = [
        CanonicalMessage(role="system", content=real_system_prompt),
        CanonicalMessage(role="user", content=user_prompt),
    ]

    # 3. Configurations to test (interleaved)
    configs = [
        {
            "id": "flash-lite-minimal",
            "name": "Gemini 3.5 Flash Lite (effort=minimal, reused conn)",
            "model": "google/gemini-3.5-flash-lite",
            "reasoning_effort": "minimal",
            "extra_body": None,
        },
        {
            "id": "flash-lite-low",
            "name": "Gemini 3.5 Flash Lite (effort=low, reused conn)",
            "model": "google/gemini-3.5-flash-lite",
            "reasoning_effort": "low",
            "extra_body": None,
        },
        {
            "id": "gemma-4-none",
            "name": "Gemma 4 26B (Vertex, effort=none, reused conn)",
            "model": "google/gemma-4-26b-a4b-it",
            "reasoning_effort": "none",
            "extra_body": {"provider": {"order": ["Google"]}},
        },
        {
            "id": "gemma-4-low",
            "name": "Gemma 4 26B (Vertex, effort=low, reused conn)",
            "model": "google/gemma-4-26b-a4b-it",
            "reasoning_effort": "low",
            "extra_body": {"provider": {"order": ["Google"]}},
        },
    ]

    # Pre-create persistent clients and warm connections
    clients: dict[str, ProviderClient] = {}
    for cfg in configs:
        c = ProviderClient(
            base_url="https://openrouter.ai/api/v1",
            model=cfg["model"],
            api_key=api_key,
            max_tokens=256,
            default_extra_body=cfg["extra_body"],
        )
        await c.warm()
        clients[cfg["id"]] = c

    # Result collectors
    records: dict[str, list[dict[str, Any]]] = {cfg["id"]: [] for cfg in configs}

    total_rounds = 10
    print(
        f"Executing {total_rounds} interleaved rounds across {len(configs)} configurations:"
    )
    print(
        "Configurations are interleaved in round-robin order to eliminate time-of-day/caching bias.\n"
    )

    for round_idx in range(1, total_rounds + 1):
        print(f"--- Round {round_idx:2d}/{total_rounds} ---")
        for cfg in configs:
            cid = cfg["id"]
            client = clients[cid]
            ts_iso = datetime.now(UTC).isoformat()

            t0 = time.perf_counter()
            t_first_token: float | None = None
            t_first_sentence: float | None = None
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            reasoning_tokens = 0

            try:
                async for event in client.stream(
                    messages,
                    tools=real_tools,
                    reasoning_effort=cfg["reasoning_effort"],
                ):
                    now = time.perf_counter()
                    if isinstance(event, TextDelta) and event.text:
                        if t_first_token is None:
                            t_first_token = now
                        accumulated_text += event.text
                        if t_first_sentence is None and any(
                            p in accumulated_text
                            for p in (". ", "! ", "? ", ".\n", "!\n")
                        ):
                            t_first_sentence = now
                    elif isinstance(event, Usage):
                        prompt_tokens = event.prompt_tokens
                        completion_tokens = event.completion_tokens
                        reasoning_tokens = event.reasoning_tokens

                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                if t_first_sentence is None:
                    t_first_sentence = now

                ttft = (t_first_token - t0) * 1000.0
                ttfs = (t_first_sentence - t0) * 1000.0

                record = {
                    "timestamp": ts_iso,
                    "ttft": ttft,
                    "ttfs": ttfs,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "text": accumulated_text.strip(),
                }
                records[cid].append(record)

                print(
                    f"  [{ts_iso[11:19]}] {cid:<18} | TTFT={ttft:6.1f}ms | TTFS={ttfs:6.1f}ms | "
                    f"rtok={reasoning_tokens:3d} | ptok={prompt_tokens:3d} | '{accumulated_text.strip()[:30]}...'"
                )
            except (
                httpx2.HTTPError,
                ProviderError,
                RuntimeError,
                OSError,
                ValueError,
            ) as e:
                print(f"  [{ts_iso[11:19]}] {cid:<18} | RETRYING after: {e}")
                await asyncio.sleep(1.0)
                try:
                    t0 = time.perf_counter()
                    t_first_token = None
                    t_first_sentence = None
                    accumulated_text = ""
                    async for event in client.stream(
                        messages,
                        tools=real_tools,
                        reasoning_effort=cfg["reasoning_effort"],
                    ):
                        now = time.perf_counter()
                        if isinstance(event, TextDelta) and event.text:
                            if t_first_token is None:
                                t_first_token = now
                            accumulated_text += event.text
                            if t_first_sentence is None and any(
                                p in accumulated_text
                                for p in (". ", "! ", "? ", ".\n", "!\n")
                            ):
                                t_first_sentence = now
                        elif isinstance(event, Usage):
                            prompt_tokens = event.prompt_tokens
                            completion_tokens = event.completion_tokens
                            reasoning_tokens = event.reasoning_tokens
                    now = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = now
                    if t_first_sentence is None:
                        t_first_sentence = now
                    ttft = (t_first_token - t0) * 1000.0
                    ttfs = (t_first_sentence - t0) * 1000.0
                    records[cid].append(
                        {
                            "timestamp": ts_iso,
                            "ttft": ttft,
                            "ttfs": ttfs,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "text": accumulated_text.strip(),
                        }
                    )
                    print(
                        f"  [{ts_iso[11:19]}] {cid:<18} | TTFT={ttft:6.1f}ms | TTFS={ttfs:6.1f}ms | "
                        f"rtok={reasoning_tokens:3d} | ptok={prompt_tokens:3d} | '{accumulated_text.strip()[:30]}...'"
                    )
                except (
                    httpx2.HTTPError,
                    ProviderError,
                    RuntimeError,
                    OSError,
                    ValueError,
                ) as retry_err:
                    print(
                        f"  [{ts_iso[11:19]}] {cid:<18} | ERROR after retry: {retry_err}"
                    )

            await asyncio.sleep(0.3)  # slight spacing between calls

    # Clean up clients
    for c in clients.values():
        await c.aclose()

    # 4. Summary Table strictly reporting p50 and p95
    print("\n" + "=" * 96)
    print("REPRESENTATIVE INTERLEAVED BENCHMARK SUMMARY (p50 / p95)")
    print("=" * 96)
    print(
        f"{'Configuration':<45} | {'TTFT p50 / p95':<20} | {'TTFS p50 / p95':<20} | {'Reasoning Tok':<14} | {'Prompt Tok'}"
    )
    print("-" * 96)
    for cfg in configs:
        cid = cfg["id"]
        recs = records[cid]
        if not recs:
            print(f"{cfg['name']:<45} | NO DATA")
            continue
        ttft_p50, ttft_p95 = calc_stats([r["ttft"] for r in recs])
        ttfs_p50, ttfs_p95 = calc_stats([r["ttfs"] for r in recs])
        rt_p50, rt_p95 = calc_stats([r["reasoning_tokens"] for r in recs])
        pt_p50, _ = calc_stats([r["prompt_tokens"] for r in recs])

        ttft_str = f"{ttft_p50:5.1f} / {ttft_p95:5.1f} ms"
        ttfs_str = f"{ttfs_p50:5.1f} / {ttfs_p95:5.1f} ms"
        rt_str = f"{int(rt_p50)} / {int(rt_p95)}"
        pt_str = f"{int(pt_p50)}"
        print(
            f"{cfg['name']:<45} | {ttft_str:<20} | {ttfs_str:<20} | {rt_str:<14} | {pt_str}"
        )
    print("=" * 96)
    print(
        f"Network RTT to openrouter.ai: p50 = {rtt_p50:5.1f} ms | p95 = {rtt_p95:5.1f} ms"
    )
    print("=" * 96)


if __name__ == "__main__":
    asyncio.run(main())
