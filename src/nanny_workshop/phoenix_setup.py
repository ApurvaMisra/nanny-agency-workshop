"""Phoenix observability setup for agent notebooks."""

import os
import socket
from typing import Callable

_started: dict = {}  # module-level cache so re-runs return the same session


def _find_free_port(start: int, attempts: int = 30) -> int:
    """Find a free TCP port, preferring `start` and walking forward."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in {start}..{start + attempts - 1}")


def start_phoenix() -> tuple[str, Callable[[], None]]:
    """Launch Phoenix in-process. Returns (ui_url, stop_callable).

    Idempotent within a Python process: a second call returns the same session.
    If port 6006 (HTTP UI) or 4317 (gRPC OTel collector) is busy, walks forward
    to find free ones. Re-registers OTel each call.
    """
    if "url" in _started:
        # Re-register OTel against the already-running session and return cached URL.
        from phoenix.otel import register
        register(project_name="nanny-agency", auto_instrument=True)
        return _started["url"], _started["stop"]

    # Find free ports BEFORE importing phoenix, then export as env vars so
    # phoenix uses them for both the FastAPI server and the gRPC collector.
    ui_port = _find_free_port(start=6006)
    grpc_port = _find_free_port(start=4317)
    os.environ["PHOENIX_PORT"] = str(ui_port)
    os.environ["PHOENIX_GRPC_PORT"] = str(grpc_port)
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"http://localhost:{ui_port}"

    import phoenix as px
    from phoenix.otel import register

    session = px.launch_app()
    register(project_name="nanny-agency", auto_instrument=True)

    _started["session"] = session
    _started["url"] = session.url
    _started["stop"] = lambda: None  # Phoenix doesn't expose clean shutdown

    return _started["url"], _started["stop"]
