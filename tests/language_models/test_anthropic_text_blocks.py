"""All visible Anthropic blocks must reach text extraction, in order."""
import json
from copy import deepcopy

import pytest

from edsl.inference_services.services.anthropic_service import AnthropicService
from edsl.language_models.raw_response_handler import RawResponseHandler
from edsl.language_models.exceptions import LanguageModelBadResponseError


def handler():
    return RawResponseHandler(
        ["content", 0, "text"],
        usage_sequence=["usage"],
        reasoning_sequence=["content"],
        inference_service="anthropic",
    )


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("first", ["text", "thinking", "tool_use"])
def test_all_text_blocks_are_kept_regardless_of_first_block(first, encoded):
    blocks = [
        {"type": "text", "text": "first"},
        {"type": "thinking", "thinking": "hidden", "signature": "opaque"},
        {"type": "text", "text": "second"},
    ]
    if first == "thinking":
        blocks.insert(
            0, {"type": "thinking", "thinking": "earlier", "signature": "untouched"}
        )
    elif first == "tool_use":
        blocks.insert(
            0,
            {
                "type": "tool_use",
                "id": "tool1",
                "name": "noop",
                "input": {"text": "not answer"},
            },
        )
    raw = {
        "content": blocks,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    original = deepcopy(raw)
    source = json.dumps(raw) if encoded else raw
    assert handler().get_generated_token_string(source) == "first\n\nsecond"
    assert raw == original
    assert handler().get_usage_dict(source) == raw["usage"]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "thinking", "thinking": "never answer", "signature": "opaque"}],
        [
            {
                "type": "tool_use",
                "id": "call",
                "name": "noop",
                "input": {"text": "never answer"},
            }
        ],
    ],
)
def test_non_text_content_is_not_exposed_as_an_answer(content):
    assert handler().get_generated_token_string({"content": content}) == ""


@pytest.mark.parametrize("text", [None, 42, {}, ["wrong"]])
def test_malformed_text_blocks_are_visible_failures(text):
    raw = {
        "content": [{"type": "text", "text": "first"}, {"type": "text", "text": text}]
    }
    with pytest.raises(LanguageModelBadResponseError) as error:
        handler().get_generated_token_string(raw)
    assert error.value.response_json == raw


def test_missing_text_field_is_not_silently_discarded():
    with pytest.raises(LanguageModelBadResponseError):
        handler().get_generated_token_string({"content": [{"type": "text"}]})


def test_whitespace_matches_existing_outer_trim_and_block_separator():
    assert (
        handler().get_generated_token_string(
            {
                "content": [
                    {"type": "text", "text": "  first "},
                    {"type": "text", "text": " second  "},
                ]
            }
        )
        == "first \n\n second"
    )
    assert (
        handler().get_generated_token_string(
            {"content": [{"type": "text", "text": "  first  "}]}
        )
        == "first"
    )


def test_non_anthropic_service_keeps_its_configured_path():
    raw = {
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    }
    for service in [None, "openai", "test"]:
        other = RawResponseHandler(["content", 0, "text"], inference_service=service)
        assert other.get_generated_token_string(raw) == "first"


def test_plain_text_legacy_response_stays_plain_text():
    assert handler().get_generated_token_string("plain response") == "plain response"


def test_real_model_free_text_parsing_receives_all_blocks():
    model = AnthropicService.create_model("claude-opus-4-6")(skip_api_key_check=True)
    raw = {
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    original = deepcopy(raw)
    parsed = model.parse_response(raw, is_free_text=True)
    assert parsed.answer == "first\n\nsecond"
    assert raw == original  # This does not relabel max_tokens as clean completion.
