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


import json
from pathlib import Path


class MemoryStore:
    """JSON-backed long-term memory keyed by parent_id then by arbitrary string keys.

    In-memory state is loaded from disk on construction. Call save() to persist.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if self._path.exists():
            self._data = json.loads(self._path.read_text())
        else:
            self._data = {}

    def get(self, parent_id: str, key: str, default=None):
        return self._data.get(parent_id, {}).get(key, default)

    def set(self, parent_id: str, key: str, value):
        self._data.setdefault(parent_id, {})[key] = value

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


def run_team(
    user_query: str,
    plan_fn: Callable,
    research_fn: Callable,
    executor_fn: Callable,
    evidence_for_research: Callable[[str], str],
    max_research_steps: int = 4,
) -> dict:
    """Run the Planner → Researcher → Executor pipeline.

    plan_fn(user_query) -> Plan (object with .goal, .steps)
    research_fn(task, evidence_context) -> ResearchFinding
    executor_fn(task, findings) -> str
    evidence_for_research(task) -> str (controller decides where to fetch evidence)
    """
    plan = plan_fn(user_query=user_query)
    findings: list = []
    reply = ""

    research_count = 0
    for step in plan.steps:
        if step.agent == "researcher":
            if research_count >= max_research_steps:
                continue
            evidence = evidence_for_research(step.task)
            finding = research_fn(task=step.task, evidence_context=evidence)
            findings.append(finding)
            research_count += 1
        elif step.agent == "executor":
            finding_summaries = "\n".join(
                f"- [{f.source}] {f.claim}: {f.excerpt}" for f in findings
            ) or "(no findings yet)"
            reply = executor_fn(task=step.task, findings=finding_summaries)

    return {"plan": plan, "findings": findings, "reply": reply}
