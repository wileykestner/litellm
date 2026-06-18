"""
Test that AnthropicStreamWrapper maps an OpenAI-compatible ``reasoning_content``
stream (DeepSeek via Fireworks, vLLM/SGLang reasoning parsers, OpenRouter, ...)
to Anthropic ``thinking_delta`` events instead of emitting empty ``text_delta``
events for every reasoning chunk.

DeepSeek models send model thinking in the non-standard ``delta.reasoning_content``
field *before* ``delta.content``. The translator routes ``reasoning_content`` to
``thinking_delta`` (transformation.py), and the classifier opens a ``thinking``
block when a chunk carries ``reasoning_content`` without ``thinking_blocks``.
This test wires a full chunk sequence through the wrapper and asserts the
observable SSE has thinking deltas followed by text deltas with no empty
``text_delta("")`` events in between, which is the symptom end users
(Claude Code, Anthropic SDK ``.stream()``) hit when the routing regresses.
"""

import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices


def _make_chunk(delta: Delta, finish_reason: str = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=delta,
            logprobs=None,
        )
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


def _collect_events_sync(wrapper: AnthropicStreamWrapper) -> List[dict]:
    events = []
    for event in wrapper:
        events.append(event)
    return events


async def _collect_events_async(wrapper: AnthropicStreamWrapper) -> List[dict]:
    events = []
    async for event in wrapper:
        events.append(event)
    return events


def _content_block_deltas(events: List[dict]) -> List[dict]:
    return [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "content_block_delta"
        and isinstance(e.get("delta"), dict)
    ]


def _content_block_starts(events: List[dict]) -> List[dict]:
    return [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "content_block_start"
        and isinstance(e.get("content_block"), dict)
    ]


def _assert_reasoning_then_text_no_empty_text_deltas(events: List[dict]) -> None:
    deltas = _content_block_deltas(events)
    assert deltas, f"expected content_block_delta events; got {events}"

    block_starts = _content_block_starts(events)
    block_types = [e["content_block"].get("type") for e in block_starts]
    assert "thinking" in block_types, (
        f"reasoning_content must open a thinking content block; got block types "
        f"{block_types}"
    )
    assert (
        "text" in block_types
    ), f"content must open a text content block; got block types {block_types}"

    thinking_deltas = [d for d in deltas if d["delta"].get("type") == "thinking_delta"]
    text_deltas = [d for d in deltas if d["delta"].get("type") == "text_delta"]

    assert thinking_deltas, (
        f"expected at least one thinking_delta; deltas: "
        f"{[d['delta'] for d in deltas]}"
    )
    assert (
        "".join(d["delta"].get("thinking", "") for d in thinking_deltas)
        == "We must say HELLO"
    ), (
        f"thinking_delta text should reconstruct the reasoning stream; "
        f"got {[d['delta'] for d in thinking_deltas]}"
    )

    assert text_deltas, (
        f"expected at least one text_delta; deltas: " f"{[d['delta'] for d in deltas]}"
    )
    assert "".join(d["delta"].get("text", "") for d in text_deltas) == "HELLO", (
        f"text_delta text should reconstruct the content stream; "
        f"got {[d['delta'] for d in text_deltas]}"
    )

    empty_text_deltas = [d for d in text_deltas if not d["delta"].get("text")]
    assert not empty_text_deltas, (
        f"no empty text_delta events should be emitted; got "
        f"{[d['delta'] for d in empty_text_deltas]}"
    )

    first_text_idx = deltas.index(text_deltas[0])
    last_thinking_idx = deltas.index(thinking_deltas[-1])
    assert last_thinking_idx < first_text_idx, (
        "all thinking_delta events must precede the first text_delta; "
        f"last thinking at {last_thinking_idx}, first text at {first_text_idx}"
    )


def _build_deepseek_stream() -> List[MagicMock]:
    chunks = [
        _make_chunk(
            Delta(
                content=None, reasoning_content="We", role="assistant", tool_calls=None
            )
        ),
        _make_chunk(
            Delta(
                content=None,
                reasoning_content=" must say HELLO",
                role="assistant",
                tool_calls=None,
            )
        ),
        _make_chunk(
            Delta(
                content="HE", reasoning_content=None, role="assistant", tool_calls=None
            )
        ),
        _make_chunk(
            Delta(
                content="LLO", reasoning_content=None, role="assistant", tool_calls=None
            )
        ),
        _make_chunk(
            Delta(
                content=None, reasoning_content=None, role="assistant", tool_calls=None
            ),
            finish_reason="stop",
        ),
    ]
    return chunks


@pytest.mark.asyncio
async def test_async_stream_reasoning_content_then_content():
    """
    Async path: a reasoning_content-first stream (DeepSeek/Fireworks style) must
    produce thinking_delta events carrying the reasoning, then text_delta events
    carrying the content, with no empty text_delta events in between.
    """

    async def mock_stream():
        for c in _build_deepseek_stream():
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="accounts/fireworks/models/deepseek-v4-flash-fw",
    )

    events = await _collect_events_async(wrapper)
    _assert_reasoning_then_text_no_empty_text_deltas(events)


def test_sync_stream_reasoning_content_then_content():
    """
    Sync path: a reasoning_content-first stream must produce thinking_delta then
    text_delta events with no empty text_delta events, mirroring the async path.
    """
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_build_deepseek_stream()),
        model="accounts/fireworks/models/deepseek-v4-flash-fw",
    )

    events = _collect_events_sync(wrapper)
    _assert_reasoning_then_text_no_empty_text_deltas(events)


def test_sync_stream_text_only_has_no_thinking_delta():
    """
    Regression guard: a stream with no reasoning_content (e.g. GPT-4o, or a
    direct Anthropic Claude call routed through the adapter) must still emit
    text_delta events and must not emit any thinking_delta events.
    """
    chunks = [
        _make_chunk(Delta(content="Hi", role="assistant", tool_calls=None)),
        _make_chunk(Delta(content=" there", role="assistant", tool_calls=None)),
        _make_chunk(
            Delta(content=None, role="assistant", tool_calls=None),
            finish_reason="stop",
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(chunks),
        model="gpt-4o",
    )

    events = _collect_events_sync(wrapper)
    deltas = _content_block_deltas(events)

    text_deltas = [d for d in deltas if d["delta"].get("type") == "text_delta"]
    thinking_deltas = [d for d in deltas if d["delta"].get("type") == "thinking_delta"]

    assert not thinking_deltas, (
        f"no thinking_delta expected for a non-reasoning model; "
        f"got {[d['delta'] for d in thinking_deltas]}"
    )
    assert "".join(d["delta"].get("text", "") for d in text_deltas) == "Hi there"
