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
    Finds free ports for both HTTP (6006) and gRPC (4317) so it survives stale
    leftover Phoenix processes. Configures the OTel exporter to point at the
    server we just started — otherwise traces silently go to the default
    localhost:4317 and never reach our Phoenix.
    """
    if "url" in _started:
        from phoenix.otel import register
        register(
            project_name="nanny-agency",
            endpoint=_started["http_endpoint"],
            protocol="http/protobuf",
            auto_instrument=True,
        )
        return _started["url"], _started["stop"]

    ui_port = _find_free_port(start=6006)
    grpc_port = _find_free_port(start=4317)
    os.environ["PHOENIX_PORT"] = str(ui_port)
    os.environ["PHOENIX_GRPC_PORT"] = str(grpc_port)

    # The OTel collector endpoint Phoenix listens on for trace ingestion.
    # We use HTTP/protobuf to keep things simple (one port to think about).
    http_endpoint = f"http://localhost:{ui_port}/v1/traces"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"http://localhost:{ui_port}"

    import phoenix as px
    from phoenix.otel import register

    session = px.launch_app()

    # Pass endpoint+protocol explicitly so the exporter targets THIS Phoenix,
    # not the default localhost:4317 (which won't exist when we walked past 4317).
    register(
        project_name="nanny-agency",
        endpoint=http_endpoint,
        protocol="http/protobuf",
        auto_instrument=True,
    )

    _started["session"] = session
    _started["url"] = session.url
    _started["http_endpoint"] = http_endpoint
    _started["stop"] = lambda: None

    return _started["url"], _started["stop"]
