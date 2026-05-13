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
