from unittest.mock import patch, MagicMock

import pytest

from nanny_workshop.agent import react_run, AgentTrace


def test_react_run_terminates_on_finish():
    """The loop must stop when the model returns tool_call.name == 'finish'."""

    def fake_decide(user_message, history):
        step = MagicMock()
        step.thought = "I'm done."
        step.tool_call.name = "finish"
        step.tool_call.args = {}
        step.final_answer = "Here is your answer."
        return step

    trace = react_run(
        user_message="hi",
        decide_fn=fake_decide,
        tools={},
        max_steps=5,
    )
    assert isinstance(trace, AgentTrace)
    assert trace.final_answer == "Here is your answer."
    assert len(trace.steps) == 1


def test_react_run_respects_max_steps():
    """Even if the model never finishes, we stop after max_steps."""

    def never_finish(user_message, history):
        step = MagicMock()
        step.thought = "thinking"
        step.tool_call.name = "search_nannies"
        step.tool_call.args = {"query": "x"}
        step.final_answer = None
        return step

    tools = {"search_nannies": lambda **kw: [{"name": "stub"}]}

    trace = react_run(
        user_message="hi",
        decide_fn=never_finish,
        tools=tools,
        max_steps=3,
    )
    assert len(trace.steps) == 3
    assert trace.final_answer is None


def test_react_run_dispatches_to_tool():
    """When tool_call.name is a known tool, react_run must call it with the args."""
    calls = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        return [{"name": "Alex"}]

    step1 = MagicMock()
    step1.thought = "searching"
    step1.tool_call.name = "search_nannies"
    step1.tool_call.args = {"query": "CPR"}
    step1.final_answer = None

    step2 = MagicMock()
    step2.thought = "done"
    step2.tool_call.name = "finish"
    step2.tool_call.args = {}
    step2.final_answer = "Found Alex."

    decisions = [step1, step2]
    idx = {"i": 0}

    def decide(user_message, history):
        step = decisions[idx["i"]]
        idx["i"] += 1
        return step

    trace = react_run(
        user_message="find a nanny",
        decide_fn=decide,
        tools={"search_nannies": fake_search},
        max_steps=5,
    )
    assert calls == [{"query": "CPR"}]
    assert trace.final_answer == "Found Alex."


def test_memory_store_persists_across_instances(tmp_path):
    from nanny_workshop.agent import MemoryStore

    store_path = tmp_path / "memory.json"
    s1 = MemoryStore(path=store_path)
    s1.set(parent_id="p_01", key="preferences", value={"dogs": "ok", "language": "Spanish"})
    s1.save()

    s2 = MemoryStore(path=store_path)
    pref = s2.get(parent_id="p_01", key="preferences")
    assert pref == {"dogs": "ok", "language": "Spanish"}


def test_memory_store_returns_default_for_missing(tmp_path):
    from nanny_workshop.agent import MemoryStore

    s = MemoryStore(path=tmp_path / "m.json")
    assert s.get(parent_id="p_99", key="anything") is None
    assert s.get(parent_id="p_99", key="anything", default={"x": 1}) == {"x": 1}


def test_run_team_invokes_planner_researcher_executor():
    """run_team calls plan_fn once, researcher_fn per research step, executor_fn per exec step."""
    plan_steps = [
        MagicMock(agent="researcher", task="look up policy"),
        MagicMock(agent="executor", task="draft reply"),
    ]
    plan = MagicMock(goal="answer the parent", steps=plan_steps)
    plan_calls = []
    research_calls = []
    exec_calls = []

    def plan_fn(user_query):
        plan_calls.append(user_query)
        return plan

    def research_fn(task, evidence_context):
        research_calls.append(task)
        return MagicMock(source="policies", excerpt="cancellation excerpt", claim="...")

    def executor_fn(task, findings):
        exec_calls.append(task)
        return "Drafted reply."

    from nanny_workshop.agent import run_team

    result = run_team(
        user_query="Cancel my booking",
        plan_fn=plan_fn,
        research_fn=research_fn,
        executor_fn=executor_fn,
        evidence_for_research=lambda task: "evidence",
    )
    assert plan_calls == ["Cancel my booking"]
    assert research_calls == ["look up policy"]
    assert exec_calls == ["draft reply"]
    assert "Drafted reply." in result["reply"]
    assert len(result["findings"]) == 1
