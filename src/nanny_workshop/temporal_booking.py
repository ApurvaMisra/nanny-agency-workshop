"""Durable booking agent: Temporal workflow + activities (Notebook 4).

The Notebook-2 ReAct loop lives in ``BookingWorkflow`` (deterministic, durable). The
LLM decision and the tool calls are Temporal *activities* — they run outside the
workflow sandbox, in a thread pool, and their results are checkpointed so the workflow
can pause for human approval (or survive a worker restart) without re-running them.

Sandbox rule: keep top-level imports light. Heavy/IO imports (baml_client, agent_tools,
json, pathlib, time) go INSIDE the activity bodies, which run outside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import activity, workflow

TASK_QUEUE = "nanny-booking"

# Where activities append one line per real execution. Reading this back after a run
# is how the notebook demonstrates exactly-once execution.
ACTIVITY_LOG = "activity_log.jsonl"


@dataclass
class Decision:
    """Serializable view of one ReAct step (a flattened BAML AgentStep)."""

    thought: str
    tool_name: str
    tool_args: dict
    final_answer: Optional[str] = None


@dataclass
class Approval:
    approved: bool
    note: str = ""


@dataclass
class BookingResult:
    user_message: str
    steps: list = field(default_factory=list)   # list[dict]: {thought, tool_name, tool_args, observation}
    drafted_email: str = ""
    outcome: str = "done_no_draft"              # one of: sent | rejected | done_no_draft
    approval_note: str = ""


def _log_activity(name: str) -> None:
    """Append one JSONL record marking a real activity execution."""
    import json
    import time
    from pathlib import Path

    log_dir = Path(__file__).resolve().parents[2] / ".cache" / "temporal"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / ACTIVITY_LOG).open("a") as fh:
        fh.write(json.dumps({"activity": name, "ts": time.time()}) + "\n")


@activity.defn
def decide_activity(user_message: str, history: str) -> Decision:
    """Ask the BAML agent for the next step (one LLM call)."""
    _log_activity("decide_activity")
    from baml_client.sync_client import b

    step = b.DecideOneTool(user_message=user_message, history=history)
    return Decision(
        thought=step.thought,
        tool_name=step.tool_call.name,
        tool_args={k: str(v) for k, v in dict(step.tool_call.args or {}).items()},
        final_answer=step.final_answer,
    )


@activity.defn
def run_tool_activity(tool_name: str, tool_args: dict) -> str:
    """Dispatch a single tool call and return a string observation."""
    _log_activity(f"run_tool_activity:{tool_name}")
    from nanny_workshop import agent_tools

    try:
        if tool_name == "search_nannies":
            return str(agent_tools.search_nannies(**tool_args))
        if tool_name == "get_policy":
            return agent_tools.get_policy(**tool_args)
        if tool_name == "check_availability":
            return str(agent_tools.check_availability(**tool_args))
        if tool_name == "draft_email":
            return agent_tools.draft_email(**tool_args)
        return f"<<error: unknown tool {tool_name!r}>>"
    except Exception as e:  # noqa: BLE001
        return f"<<error: {type(e).__name__}: {e}>>"


@activity.defn
def send_email_activity(drafted_email: str) -> str:
    """Exactly-once side effect: 'send' the approved booking email (stub — no real send)."""
    _log_activity("send_email_activity")
    # In production this would call an email provider. Here we just confirm.
    preview = drafted_email.strip().splitlines()[0] if drafted_email.strip() else "(empty)"
    return f"SENT: {preview}"


@workflow.defn
class BookingWorkflow:
    """Durable ReAct booking loop with a human-approval gate.

    Mirrors Notebook 2's react_run, but the orchestration is durable and the
    'send email' side effect is gated behind a human ``approve`` signal.
    """

    def __init__(self) -> None:
        self._status: str = "drafting"
        self._decision: Optional[Approval] = None

    @workflow.run
    async def run(self, user_message: str, max_steps: int = 6) -> BookingResult:
        result = BookingResult(user_message=user_message)
        history_lines: list[str] = []
        draft: Optional[str] = None
        timeout = timedelta(seconds=60)

        for _ in range(max_steps):
            history = "\n".join(history_lines)
            decision = await workflow.execute_activity(
                decide_activity, args=[user_message, history], start_to_close_timeout=timeout
            )
            if decision.tool_name == "finish":
                result.steps.append(
                    {"thought": decision.thought, "tool_name": "finish", "tool_args": {}, "observation": None}
                )
                break

            observation = await workflow.execute_activity(
                run_tool_activity,
                args=[decision.tool_name, decision.tool_args],
                start_to_close_timeout=timeout,
            )
            result.steps.append(
                {
                    "thought": decision.thought,
                    "tool_name": decision.tool_name,
                    "tool_args": decision.tool_args,
                    "observation": observation,
                }
            )
            history_lines.append(f"Thought: {decision.thought}")
            history_lines.append(f"Action: {decision.tool_name}({decision.tool_args})")
            history_lines.append(f"Observation: {observation}")

            if decision.tool_name == "draft_email":
                draft = observation
                break

        if draft is None:
            result.outcome = "done_no_draft"
            self._status = "done_no_draft"
            return result

        result.drafted_email = draft

        # Durably pause until a human sends the approve signal — seconds or days.
        self._status = "awaiting_approval"
        await workflow.wait_condition(lambda: self._decision is not None)

        result.approval_note = self._decision.note
        if self._decision.approved:
            await workflow.execute_activity(
                send_email_activity, args=[draft], start_to_close_timeout=timeout
            )
            result.outcome = "sent"
            self._status = "sent"
        else:
            result.outcome = "rejected"
            self._status = "rejected"
        return result

    @workflow.signal
    def approve(self, approved: bool, note: str = "") -> None:
        self._decision = Approval(approved=approved, note=note)

    @workflow.query
    def status(self) -> str:
        return self._status
