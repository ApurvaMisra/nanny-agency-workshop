"""Unit-test BookingWorkflow against Temporal's in-memory test environment.

Activities are mocked (no BAML, no network, no API key), so this runs in CI. We assert
the workflow drives the ReAct loop, pauses for approval, and gates the send.
"""

import uuid

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from nanny_workshop.temporal_booking import (
    Decision,
    BookingWorkflow,
    TASK_QUEUE,
)

# --- Mocks: same activity NAMES as the real ones, async so no executor needed. ---

_SENT: list = []


@activity.defn(name="decide_activity")
async def mock_decide(user_message: str, history: str) -> Decision:
    # First call: draft. Second call (not reached because we break on draft): finish.
    if "Action: draft_email" in history:
        return Decision(thought="done", tool_name="finish", tool_args={}, final_answer="ok")
    return Decision(
        thought="I'll draft the booking email",
        tool_name="draft_email",
        tool_args={"parent_id": "p_01", "nanny_id": "n_01", "day": "thu", "hours": "6"},
    )


@activity.defn(name="run_tool_activity")
async def mock_run_tool(tool_name: str, tool_args: dict) -> str:
    return "Dear family, your booking with Maria for Thursday is confirmed."


@activity.defn(name="send_email_activity")
async def mock_send(drafted_email: str) -> str:
    _SENT.append(drafted_email)
    return "SENT: ok"


async def _run(env: WorkflowEnvironment, approve: bool):
    _SENT.clear()
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[BookingWorkflow],
        activities=[mock_decide, mock_run_tool, mock_send],
    ):
        handle = await env.client.start_workflow(
            BookingWorkflow.run,
            "Book Maria for Thursday, 6 hours.",
            id=f"wf-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )
        # Wait until the workflow is paused awaiting approval.
        while await handle.query(BookingWorkflow.status) != "awaiting_approval":
            pass
        await handle.signal(BookingWorkflow.approve, args=[approve, "ok" if approve else "no budget"])
        return await handle.result()


@pytest.mark.asyncio
async def test_booking_workflow_approve_sends_email():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, approve=True)
    assert result.outcome == "sent"
    assert result.drafted_email.startswith("Dear family")
    assert len(_SENT) == 1


@pytest.mark.asyncio
async def test_booking_workflow_reject_does_not_send():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, approve=False)
    assert result.outcome == "rejected"
    assert result.approval_note == "no budget"
    assert len(_SENT) == 0
