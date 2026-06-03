"""Generate notebooks/04_durable_agent.ipynb."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Notebook 4 — Durable Booking Agent with Temporal

Notebook 2's booking agent runs entirely in the kernel. If the process dies — or the
agent needs to **wait for a human to approve** sending a booking email — all progress is
lost, and re-running re-charges expensive LLM calls and risks sending duplicate emails.

**Temporal** fixes this by splitting the agent into:
- a **durable workflow** — the ReAct orchestration loop (deterministic, checkpointed), and
- **activities** — the LLM decision and the tool calls (run in a thread pool, exactly-once).

The workflow can durably **pause for a human approval** (a Temporal *signal*) — for seconds
or days — and survive a worker restart. We'll book a nanny, pause for the manager's
approval, then send only on approval.""")

md("""## 1. Setup

This notebook needs **two** things beyond an OpenAI key:

1. The **`temporal` CLI** — install with `brew install temporal` (macOS) or see
   <https://docs.temporal.io/cli#install>.
2. A running dev server: in a separate terminal run **`temporal server start-dev`**
   (Temporal Web UI at <http://localhost:8233>, gRPC at `localhost:7233`).

The cell below checks both.""")

code('''import os, sys, shutil, socket
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Set OPENAI_API_KEY in .env"

assert shutil.which("temporal"), (
    "The `temporal` CLI is not installed. Install it (brew install temporal) and run "
    "`temporal server start-dev` in a separate terminal."
)
with socket.socket() as s:
    ok = s.connect_ex(("localhost", 7233)) == 0
assert ok, "No Temporal server on localhost:7233 — run `temporal server start-dev`."
print("OK — temporal CLI found and server is up. Web UI: http://localhost:8233")''')

md("""## 2. From Notebook 2's loop to a Temporal workflow

The durable agent lives in `nanny_workshop/temporal_booking.py`. The ReAct loop is the
**workflow**; the LLM decision and tools are **activities**:

```
BookingWorkflow.run(user_message):
    loop:  decide_activity            # b.DecideAllTools (LLM)
           run_tool_activity(...)     # search / check_availability / draft_email
           break when a draft is ready
    await wait_condition(approved?)   # <-- durable human-approval gate
    if approved: send_email_activity  # exactly-once side effect
```

Signals carry the human decision; a query exposes the current status.""")

code('''import inspect
from nanny_workshop import temporal_booking
print(inspect.getsource(temporal_booking.BookingWorkflow.run))''')

md("""## 3. Start a worker and launch a booking

The **worker** is a separate process that executes the workflow + activities. We start it
as a subprocess and wait for its `WORKER READY` marker.""")

code('''import subprocess, sys, time, asyncio, uuid
from temporalio.client import Client
from nanny_workshop.temporal_booking import BookingWorkflow, TASK_QUEUE

def start_worker():
    p = subprocess.Popen(
        [sys.executable, "-m", "nanny_workshop.temporal_worker"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in p.stdout:                 # block until ready
        print("worker:", line.strip())
        if "WORKER READY" in line:
            return p
    # stdout closed without the marker -> the worker died during startup.
    p.wait()
    raise RuntimeError("worker exited before becoming ready (see output above)")

worker = start_worker()
client = await Client.connect("localhost:7233")
print("connected")''')

code('''# Clear the activity-execution log so we can prove exactly-once later.
log_path = ROOT / ".cache" / "temporal" / "activity_log.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text("")

handle = await client.start_workflow(
    BookingWorkflow.run,
    "I want to book Maria (n_01) for Saturday, 6 hours. I'm family p_01.",
    id=f"booking-{uuid.uuid4()}",
    task_queue=TASK_QUEUE,
)
print("workflow started:", handle.id)
print("Watch it live:", f"http://localhost:8233/namespaces/default/workflows/{handle.id}")''')

md("""## 4. The agent drafts, then **pauses for approval**

We poll the workflow's `status` query until it reaches `awaiting_approval`. The workflow is
now durably suspended — no kernel, no thread is holding its state; it's persisted in
Temporal. Look at the Web UI: the run is "Running" but waiting.""")

code('''while await handle.query(BookingWorkflow.status) != "awaiting_approval":
    await asyncio.sleep(0.5)
print("status:", await handle.query(BookingWorkflow.status))
print("Activities run so far:")
print(log_path.read_text())''')

md("""## 5. The manager approves → email is sent (exactly once)

We act as the manager UI and send the `approve` signal. The workflow wakes, runs
`send_email_activity` once, and completes.""")

code('''await handle.signal(BookingWorkflow.approve, args=[True, "approved by manager"])
result = await handle.result()
print("outcome:", result.outcome)
print("drafted email (first line):", result.drafted_email.splitlines()[0] if result.drafted_email else "")
print()
print("Final activity log — note each LLM/tool call appears exactly once:")
print(log_path.read_text())''')

md("""## 6. The reject path

A second booking, rejected — no email is sent.""")

code('''h2 = await client.start_workflow(
    BookingWorkflow.run,
    "Book Maria (n_01) for Saturday, 6 hours, family p_01.",
    id=f"booking-{uuid.uuid4()}",
    task_queue=TASK_QUEUE,
)
while await h2.query(BookingWorkflow.status) != "awaiting_approval":
    await asyncio.sleep(0.5)
await h2.signal(BookingWorkflow.approve, args=[False, "over budget this month"])
r2 = await h2.result()
print("outcome:", r2.outcome, "| note:", r2.approval_note)''')

md("""## 7. Bonus — the durable wait survives a worker restart

Start a booking, let it reach `awaiting_approval`, **kill and restart the worker**, then
approve. The pending approval was never in the worker's memory — it's persisted in
Temporal — so it still completes.""")

code('''h3 = await client.start_workflow(
    BookingWorkflow.run,
    "Book Maria (n_01) for Saturday, 6 hours, family p_01.",
    id=f"booking-{uuid.uuid4()}",
    task_queue=TASK_QUEUE,
)
while await h3.query(BookingWorkflow.status) != "awaiting_approval":
    await asyncio.sleep(0.5)
print("awaiting approval; restarting worker...")
worker.terminate(); worker.wait()
worker = start_worker()                       # fresh process
await h3.signal(BookingWorkflow.approve, args=[True, "approved after restart"])
r3 = await h3.result()
print("outcome after worker restart:", r3.outcome)''')

md("""## 8. Recap

- The ReAct loop became a **durable workflow**; LLM + tool calls became **exactly-once
  activities**.
- The workflow **durably paused** for a human `approve` signal and survived a worker restart.
- `send_email_activity` (the side effect) ran **only after approval, exactly once**.

**Next steps (not built here):** put a *timeout* on the wait (`workflow.wait_condition(...,
timeout=timedelta(hours=24))`) to auto-escalate; add activity **retry policies** and
**timeouts**; make `send_email_activity` **idempotent** (an activity interrupted mid-run is
retried at-least-once).""")

code('''# Teardown
worker.terminate(); worker.wait()
print("worker stopped")''')

nb.cells = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_durable_agent.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
