"""Agent runs, tool calls and cache hits must show up in Phoenix as one connected trace.

Without these spans Phoenix shows only disconnected LLM calls: no tool calls, no tree
per agent run, and nothing at all for responses served from the on-disk cache.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from nanny_workshop.agent import react_run, run_team
from nanny_workshop.openai_client import CachedOpenAI


@pytest.fixture
def spans(monkeypatch):
    """Install a fresh global TracerProvider feeding an InMemorySpanExporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider, raising=False)
    return exporter


def _step(name, args=None, thought="t", final=None):
    step = MagicMock()
    step.thought = thought
    step.tool_call.name = name
    step.tool_call.args = args or {}
    step.final_answer = final
    return step


def _scripted_decide(steps):
    """A decide_fn that emits an LLM span (like the BAML bridge does) and replays `steps`."""
    it = iter(steps)

    def decide(user_message, history):
        with trace.get_tracer("test").start_as_current_span("DecideWithTools"):
            return next(it)

    return decide


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_react_run_traces_each_tool_call_as_a_tool_span(spans):
    react_run(
        user_message="Book Maria for Thursday",
        decide_fn=_scripted_decide([
            _step("check_availability", {"nanny_id": "n_01", "day": "Thursday"}),
            _step("finish", final="Maria is free."),
        ]),
        tools={"check_availability": lambda nanny_id, day: True},
    )

    [tool] = _by_name(spans, "check_availability")
    attrs = dict(tool.attributes)
    assert attrs["openinference.span.kind"] == "TOOL"
    assert attrs["tool.name"] == "check_availability"
    assert json.loads(attrs["tool.parameters"]) == {"nanny_id": "n_01", "day": "Thursday"}
    assert json.loads(attrs["input.value"]) == {"nanny_id": "n_01", "day": "Thursday"}
    assert attrs["output.value"] == "true"


def test_react_run_puts_llm_and_tool_spans_under_one_agent_span(spans):
    react_run(
        user_message="Find me a nanny",
        decide_fn=_scripted_decide([
            _step("search_nannies", {"query": "CPR"}),
            _step("finish", final="Here are two."),
        ]),
        tools={"search_nannies": lambda query: [{"id": "n_01"}]},
    )

    [agent] = _by_name(spans, "react_run")
    attrs = dict(agent.attributes)
    assert attrs["openinference.span.kind"] == "AGENT"
    assert attrs["input.value"] == "Find me a nanny"
    assert attrs["output.value"] == "Here are two."
    assert agent.parent is None

    children = [s for s in spans.get_finished_spans() if s is not agent]
    assert [s.name for s in children] == ["DecideWithTools", "search_nannies", "DecideWithTools"]
    assert all(s.parent.span_id == agent.context.span_id for s in children)
    assert {s.context.trace_id for s in children} == {agent.context.trace_id}


def test_failing_tool_span_is_marked_error_and_run_continues(spans):
    def broken(query):
        raise RuntimeError("chroma down")

    result = react_run(
        user_message="Find me a nanny",
        decide_fn=_scripted_decide([
            _step("search_nannies", {"query": "CPR"}),
            _step("finish", final="Sorry."),
        ]),
        tools={"search_nannies": broken},
    )

    assert result.steps[0].observation == "<<error: RuntimeError: chroma down>>"
    assert result.final_answer == "Sorry."
    [tool] = _by_name(spans, "search_nannies")
    assert tool.status.status_code == StatusCode.ERROR
    assert dict(tool.attributes)["output.value"] == "<<error: RuntimeError: chroma down>>"


def test_unknown_tool_still_shows_up_as_an_error_span(spans):
    react_run(
        user_message="hi",
        decide_fn=_scripted_decide([_step("teleport", {"to": "mars"}), _step("finish", final="ok")]),
        tools={},
    )

    [tool] = _by_name(spans, "teleport")
    assert tool.status.status_code == StatusCode.ERROR
    assert "unknown tool" in dict(tool.attributes)["output.value"]


def test_run_team_groups_planner_researcher_executor_in_one_trace(spans):
    tracer = trace.get_tracer("test")

    def plan_fn(user_query):
        with tracer.start_as_current_span("PlanAgent"):
            return MagicMock(steps=[MagicMock(agent="researcher", task="r"), MagicMock(agent="executor", task="e")])

    def research_fn(task, evidence_context):
        with tracer.start_as_current_span("ResearcherAgent"):
            return MagicMock(source="policy", claim="c", excerpt="x")

    def executor_fn(task, findings):
        with tracer.start_as_current_span("ExecutorAgent"):
            return "final reply"

    run_team(
        user_query="What's the cancellation fee?",
        plan_fn=plan_fn,
        research_fn=research_fn,
        executor_fn=executor_fn,
        evidence_for_research=lambda task: "evidence",
    )

    [team] = _by_name(spans, "run_team")
    attrs = dict(team.attributes)
    assert attrs["openinference.span.kind"] == "AGENT"
    assert attrs["input.value"] == "What's the cancellation fee?"
    assert attrs["output.value"] == "final reply"
    children = [s for s in spans.get_finished_spans() if s is not team]
    assert [s.name for s in children] == ["PlanAgent", "ResearcherAgent", "ExecutorAgent"]
    assert all(s.parent.span_id == team.context.span_id for s in children)


@patch("nanny_workshop.openai_client.OpenAI")
def test_cached_completion_still_emits_an_llm_span(mock_openai_cls, tmp_path, spans):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))]
    )
    mock_openai_cls.return_value = mock_client
    client = CachedOpenAI(cache_dir=tmp_path, api_key="test")
    messages = [{"role": "user", "content": "hi"}]

    client.complete(model="gpt-6-luna", messages=messages)  # real call: the OpenAI instrumentor's job
    assert _by_name(spans, "ChatCompletion (cached)") == []

    assert client.complete(model="gpt-6-luna", messages=messages) == "hello"  # cache hit

    [span] = _by_name(spans, "ChatCompletion (cached)")
    attrs = dict(span.attributes)
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "gpt-6-luna"
    assert attrs["cache.hit"] is True
    assert json.loads(attrs["input.value"]) == messages
    assert attrs["output.value"] == "hello"


@patch("nanny_workshop.openai_client.OpenAI")
def test_cached_embedding_still_emits_an_embedding_span(mock_openai_cls, tmp_path, spans):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
    mock_openai_cls.return_value = mock_client
    client = CachedOpenAI(cache_dir=tmp_path, api_key="test")

    client.embed(model="text-embedding-3-small", text="CPR nanny")
    assert client.embed(model="text-embedding-3-small", text="CPR nanny") == [0.1, 0.2]

    [span] = _by_name(spans, "Embedding (cached)")
    attrs = dict(span.attributes)
    assert attrs["openinference.span.kind"] == "EMBEDDING"
    assert attrs["embedding.model_name"] == "text-embedding-3-small"
    assert attrs["cache.hit"] is True
    assert attrs["input.value"] == "CPR nanny"


def test_start_phoenix_does_not_set_collector_endpoint_before_launch(monkeypatch):
    """Setting PHOENIX_COLLECTOR_ENDPOINT before px.launch_app() makes Phoenix print a
    misleading 'traces will be sent to the collector and not this app' warning."""
    import phoenix
    import phoenix.otel

    from nanny_workshop import phoenix_setup

    for var in ("PHOENIX_PORT", "PHOENIX_GRPC_PORT", "PHOENIX_COLLECTOR_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(phoenix_setup, "_started", {})
    monkeypatch.setattr(phoenix_setup, "_phoenix_alive", lambda port: False)
    monkeypatch.setattr(phoenix_setup, "_install_baml_bridge", lambda: None)
    monkeypatch.setattr(phoenix.otel, "register", lambda **kw: None)

    import os

    seen = {}

    def fake_launch_app():
        seen["collector_at_launch"] = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        return MagicMock(url="http://localhost:6006/")

    monkeypatch.setattr(phoenix, "launch_app", fake_launch_app)

    url, _ = phoenix_setup.start_phoenix()

    assert url == "http://localhost:6006/"
    assert seen["collector_at_launch"] is None


@patch("nanny_workshop.openai_client.OpenAI")
def test_cache_hits_still_return_when_tracing_breaks(mock_openai_cls, tmp_path, monkeypatch):
    """Tracing is best-effort: a broken span must never break a cached call."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))]
    )
    mock_client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1])])
    mock_openai_cls.return_value = mock_client
    client = CachedOpenAI(cache_dir=tmp_path, api_key="test")
    messages = [{"role": "user", "content": "hi"}]
    client.complete(model="gpt-6-luna", messages=messages)
    client.embed(model="text-embedding-3-small", text="x")

    from nanny_workshop import tracing

    def boom(*a, **kw):
        raise RuntimeError("tracing exploded")

    monkeypatch.setattr(tracing, "span", boom)

    assert client.complete(model="gpt-6-luna", messages=messages) == "hello"
    assert client.embed(model="text-embedding-3-small", text="x") == [0.1]
