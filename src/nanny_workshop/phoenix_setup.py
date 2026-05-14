"""Phoenix observability setup for agent notebooks."""

import socket
from typing import Callable

_started: dict = {}  # module-level cache so re-runs return the same session


def _find_free_port(start: int = 6006, attempts: int = 20) -> int:
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
    If port 6006 is busy (a prior session that didn't clean up, or another app),
    walks forward to find a free port. Re-registers OTel each call (cheap).
    """
    import phoenix as px
    from phoenix.otel import register

    if "url" not in _started:
        port = _find_free_port(start=6006)
        session = px.launch_app(port=port)
        _started["session"] = session
        _started["url"] = session.url

    register(project_name="nanny-agency", auto_instrument=True)

    def stop() -> None:
        # Phoenix doesn't expose a clean shutdown; no-op.
        pass

    return _started["url"], stop
