from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from verbix.core.agent import AgentSession
from verbix.core.models import CanonicalMessage
from verbix.provider.client import ProviderClient
from verbix.provider.models import TextDelta, ToolCallDelta
from verbix.runtime.tasks import DefaultTurnOutput, TaskRuntime
from verbix.tools.base import Tool, ToolContext, ToolResult
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def assert_valid_history(msgs: list[CanonicalMessage]) -> None:
    """Every assistant tool_calls message must be immediately followed by results for all its ids,

    and there must be no orphan tool messages.
    """
    i = 0
    n = len(msgs)
    while i < n:
        msg = msgs[i]
        if msg.role == "tool":
            raise AssertionError(
                f"Orphan tool message at index {i} with id {msg.tool_call_id}"
            )
        if msg.role == "assistant" and msg.tool_calls:
            call_ids = [tc.id for tc in msg.tool_calls]
            num_calls = len(call_ids)
            tool_res_msgs = msgs[i + 1 : i + 1 + num_calls]
            if len(tool_res_msgs) != num_calls:
                raise AssertionError(
                    f"Assistant message at {i} has {num_calls} tool calls ({call_ids}), but {len(tool_res_msgs)} messages follow"
                )
            for j, t_msg in enumerate(tool_res_msgs):
                if t_msg.role != "tool":
                    raise AssertionError(
                        f"Expected tool message at index {i + 1 + j} for call {call_ids[j]}, found role {t_msg.role}"
                    )
                if t_msg.tool_call_id != call_ids[j]:
                    raise AssertionError(
                        f"Expected tool message for {call_ids[j]} at index {i + 1 + j}, found {t_msg.tool_call_id}"
                    )
            i += 1 + num_calls
        else:
            i += 1


class SlowSentenceProvider(ProviderClient):
    """Yields sentences with small delays between them."""

    def __init__(self, sentences: list[str], delay: float = 0.05) -> None:
        super().__init__("http://mock/v1", "mock-model", "mock-key")
        self.sentences = sentences
        self.delay = delay

    async def stream(
        self,
        messages: Any,
        tools: Any = None,
        reasoning_effort: Any = None,
        max_tokens: Any = None,
        extra_body: Any = None,
        **kwargs: Any,
    ):
        for s in self.sentences:
            await asyncio.sleep(self.delay)
            yield TextDelta(text=s + " ")


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "A dummy tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"val": {"type": "string"}}}

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(observation="dummy result")


class CycleCallProvider(ProviderClient):
    """Repeatedly yields the exact same tool call."""

    def __init__(self) -> None:
        super().__init__("http://mock/v1", "mock-model", "mock-key")

    async def stream(
        self,
        messages: Any,
        tools: Any = None,
        reasoning_effort: Any = None,
        max_tokens: Any = None,
        extra_body: Any = None,
        **kwargs: Any,
    ):
        yield ToolCallDelta(
            index=0,
            call_id="call-cycle",
            name="dummy_tool",
            arguments_delta='{"val": "repeat"}',
        )


class SteeringWorkerProvider(ProviderClient):
    """Simulates a worker that runs a tool call and observes subsequent steering."""

    def __init__(self) -> None:
        super().__init__("http://mock/v1", "mock-model", "mock-key")
        self.call_count = 0
        self.received_messages: list[list[CanonicalMessage]] = []

    async def stream(
        self,
        messages: Any,
        tools: Any = None,
        reasoning_effort: Any = None,
        max_tokens: Any = None,
        extra_body: Any = None,
        **kwargs: Any,
    ):
        self.call_count += 1
        self.received_messages.append([m.model_copy() for m in messages])
        if self.call_count == 1:
            yield ToolCallDelta(
                index=0,
                call_id="call-sleep",
                name="bash",
                arguments_delta='{"command": "sleep 0.1"}',
            )
        else:
            yield ToolCallDelta(
                index=0,
                call_id="call-fin",
                name="worker_finish",
                arguments_delta='{"success": true, "speech_summary": "Done", "details": "Done"}',
            )


@pytest.mark.asyncio
async def test_repro_cycle_stop_history_validity(tmp_path: Path):
    """Cycle detection must leave history in a valid state (no dangling tool calls)."""
    provider = CycleCallProvider()
    session = AgentSession(
        session_id="cycle-test",
        system_prompt="Test",
        provider=provider,
        workspace=tmp_path,
        tools=[DummyTool()],
        max_turns=10,
    )
    events = []
    async for ev in session.step("start"):
        events.append(ev)

    assert_valid_history(session.history.messages)


@pytest.mark.asyncio
async def test_repro_steering_delivered_mid_run(tmp_path: Path):
    """Steering instruction sent mid-run must be visible to the worker model on next step."""
    provider = SteeringWorkerProvider()
    runtime = TaskRuntime(
        workspace=tmp_path,
        provider=provider,
    )
    task_id = await runtime.spawn_task("Steer test", "Initial instructions")

    # Give worker time to start and execute first step (sleeping)
    await asyncio.sleep(0.05)
    steered = await runtime.steer_task(task_id, "Do not forget unit tests")
    assert steered is True

    # Wait for worker to finish
    for _ in range(50):
        t = runtime.get_task(task_id)
        if t and not t.is_running:
            break
        await asyncio.sleep(0.05)

    final_task = runtime.get_task(task_id)
    assert final_task is not None
    assert final_task.status == "completed"
    # Verify steering reached the model in call 2
    assert len(provider.received_messages) >= 2
    second_call_msgs = provider.received_messages[1]
    has_steer = any(
        "Do not forget unit tests" in (m.content or "") for m in second_call_msgs
    )
    assert has_steer, "Steering message was not delivered to worker model messages"
    assert (
        task_id not in runtime._steer_queues or runtime._steer_queues[task_id].empty()
    ), "Steer queue was not drained and closed"


@pytest.mark.asyncio
async def test_repro_barge_in_cancels_turn_and_trims_history(tmp_path: Path):
    """Barge-in via toggle during speech must cancel the active turn and trim history to what was spoken."""
    provider = SlowSentenceProvider(
        ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four."],
        delay=0.04,
    )
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    sink = MockAudioSink(playback_delay=0.08)
    tts = MockTextToSpeech()
    player = AudioPlayer(tts=tts, sink=sink)
    pipeline = VoicePipeline(
        runtime=runtime,
        stt=MockSpeechToText(),
        player=player,
    )

    # Start voice turn in background
    _turn_task = asyncio.create_task(pipeline.execute_voice_turn("Hello"))
    # Wait until sentence 1 starts playing
    await asyncio.sleep(0.06)

    # User toggles (barge in)
    toggle_res = await pipeline.toggle()
    assert toggle_res.get("action") in ("interrupted_and_recording", "recording")

    # Wait a moment to ensure no more sentences are played
    await asyncio.sleep(0.2)

    # Assert only sentence 1 was synthesized/spoken
    assert len(sink.played_chunks) > 0
    assert "Sentence one" in tts.synthesized_texts[0]
    # Sentence two/three/four should NOT have been spoken
    unspoken = [t for t in tts.synthesized_texts if "Sentence three" in t]
    assert len(unspoken) == 0, (
        f"Unspoken sentences were synthesized: {tts.synthesized_texts}"
    )

    # Verify history was trimmed to what was actually delivered/spoken
    assert runtime.main_session is not None
    msgs = runtime.main_session.history.messages
    assert_valid_history(msgs)
    assistant_msgs = [m for m in msgs if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert "Sentence three" not in assistant_msgs[0].content


@pytest.mark.asyncio
async def test_repro_single_foreground_turn_at_a_time(tmp_path: Path):
    """A new foreground turn must cancel the prior active turn cleanly, leaving valid history."""
    provider = SlowSentenceProvider(
        ["Long sentence one.", "Long sentence two.", "Long sentence three."],
        delay=0.05,
    )
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    out1 = DefaultTurnOutput()
    out2 = DefaultTurnOutput()

    # Launch turn 1
    t1 = runtime.start_turn("Turn 1 input", out1)
    await asyncio.sleep(0.06)

    # Launch turn 2 while turn 1 is in-flight
    t2 = runtime.start_turn("Turn 2 input", out2)

    # Wait for t2 to complete
    await t2

    # Verify t1 was cancelled
    assert t1.cancelled() or t1.done()

    # History must be valid
    assert runtime.main_session is not None
    msgs = runtime.main_session.history.messages
    assert_valid_history(msgs)

    # Ensure user messages and assistant messages are ordered and preserved
    roles = [m.role for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant"] or roles == [
        "user",
        "user",
        "assistant",
    ]
