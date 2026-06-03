"""Bridge BAML function calls into OpenTelemetry spans for Phoenix.

BAML uses its own Rust HTTP client to reach OpenAI, so openinference's
openai auto-instrumentation never sees BAML calls. This module wraps the
generated `baml_client.sync_client.b` so each call attaches a BAML
`Collector`, then converts the captured `FunctionLog` into an
OpenInference-flavored LLM span that Phoenix can render.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from baml_py import Collector
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer("nanny_workshop.baml")
_TRACED_MARKER = "_nanny_workshop_traced"


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _emit_span_from_log(function_name: str, log: Any) -> None:
    """Convert a BAML FunctionLog into one OTel LLM span."""
    call = getattr(log, "selected_call", None) or (log.calls[0] if log.calls else None)
    if call is None:
        return

    try:
        req = call.http_request.body.json() if call.http_request else {}
    except Exception:
        req = {}
    try:
        resp = call.http_response.body.json() if call.http_response else {}
    except Exception:
        resp = {}

    timing = call.timing
    start_ns = timing.start_time_utc_ms * 1_000_000
    end_ns = (timing.start_time_utc_ms + timing.duration_ms) * 1_000_000

    span = _tracer.start_span(function_name, start_time=start_ns)
    try:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        if call.provider:
            span.set_attribute(SpanAttributes.LLM_PROVIDER, call.provider)
        model = req.get("model") or resp.get("model")
        if model:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(model))

        usage = log.usage
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, in_tok)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, out_tok)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, in_tok + out_tok)

        for i, m in enumerate(req.get("messages", []) or []):
            base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}"
            span.set_attribute(f"{base}.{MessageAttributes.MESSAGE_ROLE}", m.get("role", ""))
            span.set_attribute(
                f"{base}.{MessageAttributes.MESSAGE_CONTENT}",
                _flatten_content(m.get("content", "")),
            )

        for i, ch in enumerate(resp.get("choices", []) or []):
            msg = ch.get("message", {}) or {}
            base = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}"
            span.set_attribute(
                f"{base}.{MessageAttributes.MESSAGE_ROLE}",
                msg.get("role", "assistant"),
            )
            span.set_attribute(
                f"{base}.{MessageAttributes.MESSAGE_CONTENT}",
                _flatten_content(msg.get("content", "")),
            )
            finish = ch.get("finish_reason")
            if finish:
                span.set_attribute(SpanAttributes.LLM_FINISH_REASON, finish)

        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(req.get("messages", [])))
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            log.raw_llm_response or _flatten_content(
                (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            ),
        )
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
    finally:
        span.end(end_time=end_ns)


class _TracedBamlFn:
    def __init__(self, fn, name: str):
        self._fn = fn
        self._name = name

    def __call__(self, *args, **kwargs):
        user_opts = dict(kwargs.pop("baml_options", None) or {})
        collector = Collector(name=f"otel_{self._name}")
        existing = user_opts.get("collector")
        if existing is None:
            user_opts["collector"] = collector
        elif isinstance(existing, list):
            user_opts["collector"] = [*existing, collector]
        else:
            user_opts["collector"] = [existing, collector]
        kwargs["baml_options"] = user_opts

        try:
            return self._fn(*args, **kwargs)
        finally:
            try:
                log = collector.last
                if log is not None:
                    _emit_span_from_log(self._name, log)
            except Exception:  # noqa: BLE001
                logger.exception("BAML→OTel span emission failed for %s", self._name)


class _TracedBaml:
    """Proxy for the BAML sync client; wraps every function call with an OTel span."""

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, _TRACED_MARKER, True)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if callable(attr) and not name.startswith("_"):
            return _TracedBamlFn(attr, name)
        return attr


def install_baml_otel_bridge() -> None:
    """Replace ``baml_client.sync_client.b`` with a tracing proxy (idempotent)."""
    import baml_client.sync_client as mod

    if getattr(mod.b, _TRACED_MARKER, False):
        return
    mod.b = _TracedBaml(mod.b)
