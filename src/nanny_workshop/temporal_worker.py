"""Temporal worker for the durable booking agent (Notebook 4).

Run from the repo root so baml_client + .env resolve:

    uv run python -m nanny_workshop.temporal_worker

Prints ``WORKER READY`` on stdout once polling, so the notebook can wait for it.
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.worker import Worker

from nanny_workshop.temporal_booking import (
    BookingWorkflow,
    TASK_QUEUE,
    decide_activity,
    run_tool_activity,
    send_email_activity,
)


async def main() -> None:
    client = await Client.connect("localhost:7233")
    with ThreadPoolExecutor(max_workers=8) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[BookingWorkflow],
            activities=[decide_activity, run_tool_activity, send_email_activity],
            activity_executor=executor,
        )
        print("WORKER READY", flush=True)
        await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
