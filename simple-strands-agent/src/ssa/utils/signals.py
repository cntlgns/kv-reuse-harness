"""Graceful shutdown signal handling for SSA agent runs."""

import logging
import signal

LOG = logging.getLogger(__name__)


class GracefulExit(KeyboardInterrupt):
    """Raised to trigger a controlled shutdown that hits finally blocks."""
    pass


def _handle_term(signum, frame):
    try:
        LOG.warning(f"Received signal {signum}; initiating graceful shutdown...")
    except Exception:
        pass
    raise GracefulExit()


def install_signal_handlers():
    """Install handlers for SIGTERM, SIGINT, and SIGHUP to enable graceful shutdown."""
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        signal.signal(sig, _handle_term)
    try:
        signal.siginterrupt(signal.SIGTERM, False)
    except Exception:
        pass
