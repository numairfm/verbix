from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from verbix.cli import load_env
from verbix.core.models import CanonicalMessage
from verbix.provider.client import ProviderClient
from verbix.provider.models import TextDelta, Usage


def calc_stats(values: list[float]) -> tuple[float, float]:
    """Returns (p50, p95) for a list of numeric values."""
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]
    return p50, p95


async def benchmark_config(
    name: str,
    model: str,
    api_key: str,
    reasoning_effort: str | None = None,
    extra_body: dict[str, Any] | None = None,
    runs: int = 10,
) -> dict[str, Any]:
    print(f"\n{'=' * 70}\nRunning benchmark: {name} ({runs} runs)\n{'=' * 70}")
    client = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model=model,
        api_key=api_key,
        max_tokens=256,
    )

    system_prompt = (
        "You are Verbix Receptionist, an orchestration assistant. "
        "Reply concisely and directly in one short conversational sentence."
    )
    user_prompt = "The build finished successfully in 12 seconds."

    messages = [
        CanonicalMessage(role="system", content=system_prompt),
        CanonicalMessage(role="user", content=user_prompt),
    ]

    ttft_list: list[float] = []
    ttfs_list: list[float] = []
    prompt_tokens_list: list[float] = []
    completion_tokens_list: list[float] = []
    reasoning_tokens_list: list[float] = []
    sample_text = ""

    for run_idx in range(1, runs + 1):
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
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            ):
                now = time.perf_counter()
                if isinstance(event, TextDelta) and event.text:
                    if t_first_token is None:
                        t_first_token = now
                    accumulated_text += event.text
                    if t_first_sentence is None and any(
                        p in accumulated_text for p in (". ", "! ", "? ", ".\n", "!\n")
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

            ttft_list.append(ttft)
            ttfs_list.append(ttfs)
            prompt_tokens_list.append(float(prompt_tokens))
            completion_tokens_list.append(float(completion_tokens))
            reasoning_tokens_list.append(float(reasoning_tokens))
            sample_text = accumulated_text.strip()

            print(
                f"  Run {run_idx:2d}: TTFT={ttft:6.1f}ms | TTFS={ttfs:6.1f}ms | "
                f"prompt_tok={prompt_tokens:2d} | comp_tok={completion_tokens:2d} | "
                f"reasoning_tok={reasoning_tokens:2d} -> '{sample_text[:35]}...'"
            )
        except (RuntimeError, OSError, ValueError) as e:
            print(f"  Run {run_idx:2d}: ERROR: {e}")
            return {
                "name": name,
                "error": str(e),
            }

        await asyncio.sleep(0.2)  # avoid immediate rate limit spikes

    ttft_p50, ttft_p95 = calc_stats(ttft_list)
    ttfs_p50, ttfs_p95 = calc_stats(ttfs_list)
    pt_p50, pt_p95 = calc_stats(prompt_tokens_list)
    ct_p50, ct_p95 = calc_stats(completion_tokens_list)
    rt_p50, rt_p95 = calc_stats(reasoning_tokens_list)

    return {
        "name": name,
        "model": model,
        "ttft_p50": ttft_p50,
        "ttft_p95": ttft_p95,
        "ttfs_p50": ttfs_p50,
        "ttfs_p95": ttfs_p95,
        "prompt_tokens_p50": pt_p50,
        "prompt_tokens_p95": pt_p95,
        "completion_tokens_p50": ct_p50,
        "completion_tokens_p95": ct_p95,
        "reasoning_tokens_p50": rt_p50,
        "reasoning_tokens_p95": rt_p95,
        "sample_output": sample_text,
    }


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        return

    results: list[dict[str, Any]] = []

    # 1. Gemini 3.5 Flash Lite with OpenRouter Default Routing (effort="low")
    res1 = await benchmark_config(
        name="Gemini 3.5 Flash Lite (OpenRouter Default, effort=low)",
        model="google/gemini-3.5-flash-lite",
        api_key=api_key,
        reasoning_effort="low",
    )
    results.append(res1)

    # 2. Gemini 3.5 Flash Lite with Vertex Provider Routing (effort="low")
    res2 = await benchmark_config(
        name="Gemini 3.5 Flash Lite (Google Vertex, effort=low)",
        model="google/gemini-3.5-flash-lite",
        api_key=api_key,
        reasoning_effort="low",
        extra_body={"provider": {"order": ["Google"]}},
    )
    results.append(res2)

    # 3. Gemma 4 26B-A4B-IT (Vertex Provider, reasoning="none")
    res3 = await benchmark_config(
        name="Gemma 4 26B (Google Vertex, effort=none)",
        model="google/gemma-4-26b-a4b-it",
        api_key=api_key,
        reasoning_effort="none",
        extra_body={"provider": {"order": ["Google"]}},
    )
    results.append(res3)

    # 4. Gemma 4 26B-A4B-IT (Vertex Provider, reasoning="low")
    res4 = await benchmark_config(
        name="Gemma 4 26B (Google Vertex, effort=low)",
        model="google/gemma-4-26b-a4b-it",
        api_key=api_key,
        reasoning_effort="low",
        extra_body={"provider": {"order": ["Google"]}},
    )
    results.append(res4)

    # Summary Table strictly using p50 and p95
    print("\n" + "=" * 90)
    print("LLM STAGE ISOLATION BENCHMARK SUMMARY (p50 / p95)")
    print("=" * 90)
    print(
        f"{'Configuration':<42} | {'TTFT p50 / p95':<20} | {'TTFS p50 / p95':<20} | {'Reasoning Tok':<14} | {'Prompt Tok'}"
    )
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<42} | ERROR: {r['error']}")
            continue
        ttft_str = f"{r['ttft_p50']:5.1f} / {r['ttft_p95']:5.1f} ms"
        ttfs_str = f"{r['ttfs_p50']:5.1f} / {r['ttfs_p95']:5.1f} ms"
        rt_str = f"{int(r['reasoning_tokens_p50'])} / {int(r['reasoning_tokens_p95'])}"
        pt_str = f"{int(r['prompt_tokens_p50'])}"
        print(
            f"{r['name']:<42} | {ttft_str:<20} | {ttfs_str:<20} | {rt_str:<14} | {pt_str}"
        )
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
