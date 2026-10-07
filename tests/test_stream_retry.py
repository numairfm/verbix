"""Stream-retry tests: transient failures retry silently before anything is
spoken; anything else (or failure after speech) surfaces immediately."""

import litellm.exceptions
import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.minisweagent.models.litellm_model import LitellmModel


def _text_chunk(content, finish=None):
    return {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}


def _model(monkeypatch, script):
    """Script: list per attempt of ('stream', [chunks...]) or ('raise', exc)."""
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        kind, payload = script[min(len(calls) - 1, len(script) - 1)]

        async def gen():
            if kind == "raise":
                raise payload
            for c in payload:
                yield c

        return gen()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    return LitellmModel(model_name="test/model"), calls


def _conn_error():
    return litellm.exceptions.APIConnectionError(
        "Connection closed.", llm_provider="openrouter", model="test/model"
    )


@pytest.mark.asyncio
async def test_transient_blips_retry_silently_before_speech(monkeypatch):
    model, calls = _model(
        monkeypatch,
        [
            ("raise", _conn_error()),
            ("raise", _conn_error()),
            ("stream", [_text_chunk("hi"), _text_chunk("", "stop")]),
        ],
    )
    events = [e async for e in model.query_stream([{"role": "user", "content": "hi"}])]
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].final_text == "hi"
    assert len(calls) == 3
    assert not any(isinstance(e, TurnComplete) and "Sorry" in (e.final_text or "") for e in events)


@pytest.mark.asyncio
async def test_persistent_failure_surfaces_after_bounded_attempts(monkeypatch):
    model, calls = _model(monkeypatch, [("raise", _conn_error())])
    with pytest.raises(litellm.exceptions.APIConnectionError):
        [e async for e in model.query_stream([{"role": "user", "content": "hi"}])]
    assert 2 <= len(calls) <= 4


@pytest.mark.asyncio
async def test_failure_after_speech_is_never_retried(monkeypatch):
    async def gen_then_die():
        yield _text_chunk("half ")
        raise _conn_error()

    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return gen_then_die()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    model = LitellmModel(model_name="test/model")
    seen = []
    with pytest.raises(litellm.exceptions.APIConnectionError):
        async for e in model.query_stream([{"role": "user", "content": "hi"}]):
            seen.append(e)
    assert len(calls) == 1
    assert any(isinstance(e, TextDelta) and e.text == "half " for e in seen)


@pytest.mark.asyncio
async def test_auth_failure_fails_fast_with_no_retry(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        raise litellm.exceptions.AuthenticationError(
            "bad key", llm_provider="openrouter", model="test/model"
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    model = LitellmModel(model_name="test/model")
    with pytest.raises(litellm.exceptions.AuthenticationError):
        [e async for e in model.query_stream([{"role": "user", "content": "hi"}])]
    assert len(calls) == 1
