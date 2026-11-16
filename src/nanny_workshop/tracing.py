"""OpenInference-flavored spans for Phoenix: agent runs, tool calls and cache hits.

Everything here goes through the OpenTelemetry API, which is a no-op until a tracer
provider is registered (``start_phoenix()`` does that). Code that never starts Phoenix
behaves exactly as if this module didn't exist.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value


def to_text(value: Any) -> str:
    """Render a span input/output value: strings as-is, everything else as JSON."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


@contextmanager
def span(name: str, kind: str, **attributes: Any) -> Iterator[Span]:
    """Open a span as the current span, so spans started inside it become its children.

    The tracer is looked up on every call (not cached at import) so it always follows
    whichever provider is currently registered.
    """
    attrs = {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind}
    attrs.update({k: v for k, v in attributes.items() if v is not None})
    with trace.get_tracer("nanny_workshop").start_as_current_span(name, attributes=attrs) as s:
        yield s


def mark_error(s: Span, description: str) -> None:
    s.set_status(Status(StatusCode.ERROR, description))


def tool_span(name: str, args: dict):
    params = to_text(args)
    return span(
        name,
        TOOL,
        **{
            SpanAttributes.TOOL_NAME: name,
            SpanAttributes.TOOL_PARAMETERS: params,
            SpanAttributes.INPUT_VALUE: params,
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
        },
    )


def agent_span(name: str, user_input: str):
    return span(name, AGENT, **{SpanAttributes.INPUT_VALUE: user_input})


def set_output(s: Span, value: Any) -> None:
    if value is not None:
        s.set_attribute(SpanAttributes.OUTPUT_VALUE, to_text(value))


def _best_effort(fn):
    """Tracing must never break the call it describes: log and carry on."""

    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("tracing: %s failed", fn.__name__, exc_info=True)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_best_effort
def record_cached_completion(model: str, messages: list[dict], content: str) -> None:
    """A chat completion served from the on-disk cache: no API call, so no instrumentor
    span. Emit one ourselves so the call doesn't silently vanish from Phoenix."""
    attrs = {
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.INPUT_VALUE: to_text(messages),
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.OUTPUT_VALUE: content,
        "cache.hit": True,
    }
    for i, m in enumerate(messages):
        base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}"
        attrs[f"{base}.{MessageAttributes.MESSAGE_ROLE}"] = str(m.get("role", ""))
        attrs[f"{base}.{MessageAttributes.MESSAGE_CONTENT}"] = to_text(m.get("content", ""))
    out = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    attrs[f"{out}.{MessageAttributes.MESSAGE_ROLE}"] = "assistant"
    attrs[f"{out}.{MessageAttributes.MESSAGE_CONTENT}"] = content
    with span("ChatCompletion (cached)", LLM, **attrs):
        pass


@_best_effort
def record_cached_embedding(model: str, text: str) -> None:
    """An embedding served from the on-disk cache (see record_cached_completion)."""
    with span(
        "Embedding (cached)",
        EMBEDDING,
        **{
            SpanAttributes.EMBEDDING_MODEL_NAME: model,
            SpanAttributes.INPUT_VALUE: text,
            "cache.hit": True,
        },
    ):
        pass
