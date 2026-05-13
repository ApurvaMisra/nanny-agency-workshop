"""Phoenix observability setup for agent notebooks."""

from typing import Callable


def start_phoenix() -> tuple[str, Callable[[], None]]:
    """Launch Phoenix in-process. Returns (ui_url, stop_callable).

    Idempotent: calling twice reuses the existing session.
    """
    import phoenix as px
    from phoenix.otel import register

    session = px.launch_app()
    tracer_provider = register(project_name="nanny-agency", auto_instrument=True)

    def stop() -> None:
        # Phoenix doesn't expose a clean shutdown; this is a no-op for now.
        # Kept for future-proofing the API.
        pass

    return session.url, stop
