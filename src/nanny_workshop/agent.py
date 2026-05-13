"""ReAct loop, multi-agent orchestrator, memory, guardrails for the booking agent."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentStepRecord:
    thought: str
    tool_name: str
    tool_args: dict
    observation: Any = None
    final_answer: str | None = None


@dataclass
class AgentTrace:
    user_message: str
    steps: list[AgentStepRecord] = field(default_factory=list)
    final_answer: str | None = None


def react_run(
    user_message: str,
    decide_fn: Callable,
    tools: dict[str, Callable],
    max_steps: int = 6,
) -> AgentTrace:
    """Run a ReAct loop until the model returns finish or we hit max_steps.

    decide_fn(user_message, history) -> AgentStep-like object with
        .thought (str)
        .tool_call.name (str)
        .tool_call.args (dict[str, str])
        .final_answer (str | None)

    tools: dict mapping tool name to a callable. The callable receives **args (string-keyed)
    and returns whatever observation should be recorded.

    Returns an AgentTrace.
    """
    trace = AgentTrace(user_message=user_message)
    history_lines: list[str] = []

    for _ in range(max_steps):
        history = "\n".join(history_lines)
        step = decide_fn(user_message=user_message, history=history)

        name = step.tool_call.name
        args = dict(step.tool_call.args or {})
        record = AgentStepRecord(thought=step.thought, tool_name=name, tool_args=args)

        if name == "finish":
            record.final_answer = step.final_answer
            trace.steps.append(record)
            trace.final_answer = step.final_answer
            return trace

        tool = tools.get(name)
        if tool is None:
            record.observation = f"<<error: unknown tool {name!r}>>"
        else:
            try:
                record.observation = tool(**args)
            except Exception as e:  # noqa: BLE001
                record.observation = f"<<error: {type(e).__name__}: {e}>>"

        trace.steps.append(record)
        history_lines.append(f"Thought: {record.thought}")
        history_lines.append(f"Action: {name}({args})")
        history_lines.append(f"Observation: {record.observation}")

    return trace
