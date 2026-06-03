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
