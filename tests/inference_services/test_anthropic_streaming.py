"""Exercise the real Anthropic SDK over an in-memory HTTP transport; no API calls."""

import asyncio
import base64
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic as RealAsyncAnthropic

from edsl import Cache
from edsl.inference_services.services.anthropic_service import AnthropicService
from edsl.language_models import LanguageModel
from edsl.language_models.exceptions import LanguageModelBadResponseError


def event(kind, **payload):
    return (
        f"event: {kind}\ndata: " + json.dumps(dict(type=kind, **payload)) + "\n\n"
    ).encode()


def complete_events(stop_reason="end_turn"):
    """Include a signed thinking block and two text blocks, not just text deltas."""
    return [
        event(
            "message_start",
            message={
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        ),
        event(
            "content_block_start",
            index=0,
            content_block={"type": "thinking", "thinking": "", "signature": ""},
        ),
        event(
            "content_block_delta",
            index=0,
            delta={"type": "thinking_delta", "thinking": "private reasoning"},
        ),
        event(
            "content_block_delta",
            index=0,
            delta={"type": "signature_delta", "signature": "opaque-signature"},
        ),
        event("content_block_stop", index=0),
        event(
            "content_block_start", index=1, content_block={"type": "text", "text": ""}
        ),
        event(
            "content_block_delta",
            index=1,
            delta={"type": "text_delta", "text": "first"},
        ),
        event("content_block_stop", index=1),
        event(
            "content_block_start", index=2, content_block={"type": "text", "text": ""}
        ),
        event(
            "content_block_delta",
            index=2,
            delta={"type": "text_delta", "text": "second"},
        ),
        event("content_block_stop", index=2),
        event(
            "message_delta",
            delta={"stop_reason": stop_reason, "stop_sequence": None},
            usage={"output_tokens": 17},
        ),
        event("message_stop"),
    ]


class BodyStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, fail=False, wait=False):
        self.chunks = chunks
        self.fail = fail
        self.wait = wait
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        self.started.set()
        if self.fail:
            raise httpx.ReadError("injected broken stream")
        if self.wait:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def install_transport(monkeypatch, *, chunks=None, fail=False, wait=False, status=200):
    wire = BodyStream(
        complete_events() if chunks is None else chunks, fail=fail, wait=wait
    )
    record = SimpleNamespace(wire=wire, requests=[], clients=[])

    async def handle(request):
        record.requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(
                status,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "injected provider failure",
                    },
                },
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    def factory(**kwargs):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        # Disable SDK retries only in this test fixture so request counts are decisive.
        client = RealAsyncAnthropic(
            api_key="offline-placeholder", http_client=http_client, max_retries=0
        )
        record.clients.append(client)
        return client

    monkeypatch.setattr(
        "edsl.inference_services.services.anthropic_service.AsyncAnthropic", factory
    )
    monkeypatch.setattr("edsl.coop.Coop.report_error", AsyncMock())
    return record


def model(name="claude-opus-4-6", **kwargs):
    result = AnthropicService.create_model(name)(skip_api_key_check=True, **kwargs)
    result._api_token = "offline-placeholder"
    return result


def assert_closed(record):
    assert record.wire.closed
    assert record.clients and all(client.is_closed() for client in record.clients)


@pytest.mark.parametrize("limit", [1000, 20000, 64000, 128000])
def test_always_streams_and_preserves_the_complete_response(monkeypatch, limit):
    record = install_transport(monkeypatch)
    thinking = {"type": "adaptive"}
    effort = {"effort": "max"}
    m = model(
        max_tokens=limit, thinking=thinking, output_config=effort, temperature=0.2
    )
    parameters = deepcopy(m.parameters)
    result = asyncio.run(m.async_execute_model_call("hello", "system"))
    assert len(record.requests) == 1
    request = record.requests[0]
    assert request["stream"] is True
    assert request["max_tokens"] == limit
    assert request["thinking"] == thinking
    assert request["output_config"] == effort
    assert request["system"] == "system"
    assert request["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert result["id"] == "msg_offline"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 11
    assert result["usage"]["output_tokens"] == 17
    assert [block["type"] for block in result["content"]] == [
        "thinking",
        "text",
        "text",
    ]
    assert result["content"][0]["signature"] == "opaque-signature"
    assert result["content"][0]["thinking"] == "private reasoning"
    assert [
        block["text"] for block in result["content"] if block["type"] == "text"
    ] == ["first", "second"]
    assert m.parameters == parameters
    assert_closed(record)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("claude-opus-4-5-20251124", 0.2),
        ("claude-sonnet-4-6", 1.0),
        ("claude-opus-4-7", None),
        ("claude-sonnet-5", None),
    ],
)
def test_temperature_policy_is_unchanged(monkeypatch, name, expected):
    record = install_transport(monkeypatch)
    m = model(name, max_tokens=64000, temperature=0.2)
    asyncio.run(m.async_execute_model_call("hello"))
    request = record.requests[0]
    if expected is None:
        assert "temperature" not in request
    else:
        assert request["temperature"] == expected
    assert "thinking" not in request and "output_config" not in request
    assert m.parameters["temperature"] == 0.2
    assert_closed(record)


@pytest.mark.parametrize("count", [0, 1, 12])
def test_early_eof_is_not_a_completed_response(monkeypatch, count):
    record = install_transport(monkeypatch, chunks=complete_events()[:count])
    with pytest.raises(LanguageModelBadResponseError, match="message_stop"):
        asyncio.run(model(max_tokens=64000).async_execute_model_call("hello"))
    assert len(record.requests) == 1
    assert_closed(record)


def test_broken_stream_propagates_without_replaying(monkeypatch):
    record = install_transport(monkeypatch, chunks=complete_events()[:7], fail=True)
    with pytest.raises(httpx.ReadError, match="injected broken stream"):
        asyncio.run(model(max_tokens=64000).async_execute_model_call("hello"))
    assert len(record.requests) == 1
    assert_closed(record)


def test_provider_rejection_closes_the_client(monkeypatch):
    record = install_transport(monkeypatch, status=400)
    from anthropic import BadRequestError

    with pytest.raises(BadRequestError):
        asyncio.run(model(max_tokens=64000).async_execute_model_call("hello"))
    assert len(record.requests) == 1
    assert record.clients and all(client.is_closed() for client in record.clients)


def test_timeout_uses_edsl_outer_deadline_and_does_not_cache_partial_output(
    monkeypatch,
):
    record = install_transport(monkeypatch, chunks=complete_events()[:7], wait=True)
    m = model(max_tokens=64000)
    monkeypatch.setattr(type(m), "_compute_timeout", lambda self, files_list=None: 0.03)
    cache = Cache()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            m._async_get_intended_model_call_outcome(
                user_prompt="hello", system_prompt="", cache=cache
            )
        )
    assert not cache.data
    assert len(record.requests) == 1
    assert_closed(record)


def test_cancellation_closes_resources_and_does_not_cache_partial_output(monkeypatch):
    record = install_transport(monkeypatch, chunks=complete_events()[:7], wait=True)
    m = model(max_tokens=64000)
    cache = Cache()

    async def cancel():
        task = asyncio.create_task(
            m._async_get_intended_model_call_outcome(
                user_prompt="hello", system_prompt="", cache=cache
            )
        )
        try:
            await asyncio.wait_for(record.wire.started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(cancel())
    assert not cache.data
    assert len(record.requests) == 1
    assert_closed(record)


def test_completed_max_token_response_retains_stop_reason(monkeypatch):
    record = install_transport(monkeypatch, chunks=complete_events("max_tokens"))
    result = asyncio.run(model(max_tokens=64000).async_execute_model_call("hello"))
    assert result["stop_reason"] == "max_tokens"
    assert result["usage"]["output_tokens"] == 17
    assert_closed(record)


def test_serialized_configuration_reaches_the_adapter_unchanged(monkeypatch):
    record = install_transport(monkeypatch)
    original = model(
        max_tokens=64000, thinking={"type": "adaptive"}, output_config={"effort": "max"}
    )
    restored = LanguageModel.from_dict(original.to_dict())
    restored._api_token = "offline-placeholder"
    asyncio.run(restored.async_execute_model_call("hello"))
    for key in ("max_tokens", "thinking", "output_config"):
        assert record.requests[0][key] == original.parameters[key]
    assert_closed(record)


def test_attachments_and_system_prompt_survive_streaming(monkeypatch):
    record = install_transport(monkeypatch)
    files = [
        SimpleNamespace(
            suffix="txt",
            filename="note.txt",
            mime_type="text/plain",
            base64_string=base64.b64encode(b"fixture text").decode(),
        ),
        SimpleNamespace(
            suffix="pdf",
            filename="file.pdf",
            mime_type="application/pdf",
            base64_string="JVBERg==",
        ),
        SimpleNamespace(
            suffix="png",
            filename="file.png",
            mime_type="image/png",
            base64_string="iVBORw==",
        ),
    ]
    asyncio.run(
        model(max_tokens=64000).async_execute_model_call(
            "hello", "unchanged system", files_list=files
        )
    )
    request = record.requests[0]
    assert request["system"] == "unchanged system"
    content = request["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "text", "document", "image"]
    assert "fixture text" in content[1]["text"]
    assert content[2]["source"]["data"] == "JVBERg=="
    assert content[3]["source"]["data"] == "iVBORw=="
    assert_closed(record)


def test_successful_final_message_is_cached_and_reused(monkeypatch):
    record = install_transport(monkeypatch)
    m = model(max_tokens=64000)
    # Cost pricing is outside this transport test and may require a remote catalog.
    monkeypatch.setattr(
        type(m),
        "cost",
        lambda self, response: SimpleNamespace(
            input_tokens=11,
            output_tokens=17,
            input_price_per_million_tokens=0,
            output_price_per_million_tokens=0,
            total_cost=0,
            thinking_tokens=None,
        ),
    )
    cache = Cache()

    async def twice():
        first = await m._async_get_intended_model_call_outcome(
            user_prompt="hello", system_prompt="", cache=cache
        )
        second = await m._async_get_intended_model_call_outcome(
            user_prompt="hello", system_prompt="", cache=cache
        )
        return first, second

    first, second = asyncio.run(twice())
    assert not first.cache_used and second.cache_used
    assert first.response == second.response
    assert first.response["stop_reason"] == "end_turn"
    assert len(cache.data) == 1
    assert len(record.requests) == 1
    assert_closed(record)


def test_eof_after_stop_reason_still_cannot_cache_a_partial_response(monkeypatch):
    # message_delta already supplies end_turn, but message_stop never arrives.
    record = install_transport(monkeypatch, chunks=complete_events()[:-1])
    m = model(max_tokens=64000)
    cache = Cache()
    with pytest.raises(LanguageModelBadResponseError, match="message_stop"):
        asyncio.run(
            m._async_get_intended_model_call_outcome(
                user_prompt="hello", system_prompt="", cache=cache
            )
        )
    assert not cache.data
    assert len(record.requests) == 1
    assert_closed(record)
