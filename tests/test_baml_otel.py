"""Verify the BAML → OTel bridge emits OpenInference-flavored LLM spans."""

import os

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_spans(monkeypatch):
    """Install a fresh global TracerProvider feeding an InMemorySpanExporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", type("o", (), {"_done": False})(), raising=False)
    trace._TRACER_PROVIDER = provider
    return exporter


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
def test_baml_call_emits_openinference_llm_span(in_memory_spans):
    from nanny_workshop.baml_otel import install_baml_otel_bridge

    install_baml_otel_bridge()

    from baml_client.sync_client import b

    result = b.EchoNannyName("trace-test")
    assert result.strip().lower() == "trace-test"

    spans = in_memory_spans.get_finished_spans()
    assert spans, "expected at least one span from the BAML call"

    span = next((s for s in spans if s.name == "EchoNannyName"), None)
    assert span is not None, f"no EchoNannyName span; got {[s.name for s in spans]}"

    attrs = dict(span.attributes)
    assert attrs.get("openinference.span.kind") == "LLM"
    assert attrs.get("llm.provider") == "openai"
    assert "gpt-4o-mini" in str(attrs.get("llm.model_name", ""))
    assert attrs.get("llm.token_count.prompt", 0) > 0
    assert attrs.get("llm.token_count.completion", 0) > 0
    assert "trace-test" in str(attrs.get("output.value", ""))
