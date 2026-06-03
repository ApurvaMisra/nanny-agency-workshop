"""Phoenix observability setup for agent notebooks."""

import logging
import os
import socket
import urllib.request
from typing import Callable

logger = logging.getLogger(__name__)

_DEFAULT_UI_PORT = 6006
_DEFAULT_GRPC_PORT = 4317
_PROJECT = "nanny-agency"

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


def _phoenix_alive(port: int) -> bool:
    """True if a Phoenix server is already answering on this port."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/arize_phoenix_version", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_phoenix() -> tuple[str, Callable[[], None]]:
    """Make Phoenix observe this kernel's LLM calls. Returns (ui_url, stop_callable).

    Reuse-or-launch, anchored to the default port 6006 so the UI URL is always
    predictable:

    * If a Phoenix is *already* serving on 6006 (e.g. launched by another notebook
      kernel), we REUSE it — point this kernel's OTel exporter at it rather than
      launching a second instance on 6007 that nobody would think to open. This is
      the common cause of "I ran the notebook but Phoenix shows no traces": the
      traces went to a different port than the one in the browser.
    * Otherwise we launch a fresh in-process Phoenix on 6006.

    Idempotent within a process (the OTel TracerProvider is set-once globally, so
    we register only on the first call).

    Also installs the BAML->OTel bridge. This is essential, not optional: BAML
    talks to OpenAI through its own Rust HTTP client, so openinference's OpenAI
    auto-instrumentation never sees BAML calls and Phoenix would otherwise show
    no LLM spans at all. The bridge is what actually makes traces appear.
    """
    if "url" in _started:
        # OTel TracerProvider is set-once globally; re-registering is a silent no-op
        # ("Overriding of current TracerProvider is not allowed"). Skip it.
        _install_baml_bridge()
        return _started["url"], _started["stop"]

    from phoenix.otel import register

    if _phoenix_alive(_DEFAULT_UI_PORT):
        # Reuse the Phoenix already on 6006 instead of launching a competitor.
        # Export traces over OTLP-HTTP to its /v1/traces (served on the UI port),
        # so we don't have to guess which gRPC port that instance is using.
        url = f"http://localhost:{_DEFAULT_UI_PORT}/"
        logger.info("Reusing Phoenix already running at %s", url)
        register(
            project_name=_PROJECT,
            endpoint=f"http://localhost:{_DEFAULT_UI_PORT}/v1/traces",
            protocol="http/protobuf",
            auto_instrument=True,
        )
        session = None
    else:
        # Nothing on 6006 → launch fresh. Only walk the UI port if 6006 is held by
        # something that is NOT Phoenix (rare); that keeps the URL predictable.
        ui_port = _find_free_port(start=_DEFAULT_UI_PORT)
        grpc_port = _find_free_port(start=_DEFAULT_GRPC_PORT)
        os.environ["PHOENIX_PORT"] = str(ui_port)
        os.environ["PHOENIX_GRPC_PORT"] = str(grpc_port)
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"http://localhost:{ui_port}"

        import phoenix as px

        session = px.launch_app()
        # Export over OTLP-HTTP to the port we just launched on, so trace delivery
        # never depends on the default gRPC :4317 being the one we're using.
        register(
            project_name=_PROJECT,
            endpoint=f"http://localhost:{ui_port}/v1/traces",
            protocol="http/protobuf",
            auto_instrument=True,
        )
        url = session.url
        if ui_port != _DEFAULT_UI_PORT:
            logger.warning(
                "Port %s was busy (not Phoenix); launched Phoenix on %s instead. "
                "Open %s — NOT localhost:%s — to see traces.",
                _DEFAULT_UI_PORT, ui_port, url, _DEFAULT_UI_PORT,
            )

    _started["session"] = session
    _started["url"] = url
    _started["stop"] = lambda: None  # Phoenix doesn't expose clean shutdown

    _install_baml_bridge()
    return _started["url"], _started["stop"]


def _install_baml_bridge() -> None:
    """Route BAML calls through OTel so Phoenix shows them as LLM spans.

    Failure here means Phoenix will be silently empty for BAML calls — the exact
    symptom this bridge exists to cure — so warn loudly rather than swallow it.
    """
    try:
        from nanny_workshop.baml_otel import install_baml_otel_bridge

        install_baml_otel_bridge()
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "BAML->OTel bridge failed to install; BAML LLM calls will NOT appear "
            "in Phoenix. Traces from non-BAML OpenAI calls are unaffected.",
            exc_info=True,
        )
