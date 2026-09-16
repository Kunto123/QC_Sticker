from __future__ import annotations

import atexit
import logging
import signal
import threading

logger = logging.getLogger(__name__)

_shutdown_lock = threading.Lock()
_shutdown_hooks_installed = False
_shutdown_workers_stopped = False


def shutdown_workers(*, reason: str = "shutdown") -> None:
    """Stop process workers once, in a safe order.

    PLC worker is stopped before the push worker so clamp release has the
    highest chance to run before any other background teardown.
    """
    global _shutdown_workers_stopped
    with _shutdown_lock:
        if _shutdown_workers_stopped:
            return
        _shutdown_workers_stopped = True

    logger.info("[shutdown] stopping workers (%s)", reason)

    try:
        from backend.app.core.container import plc_worker, push_worker
    except Exception as exc:  # noqa: BLE001
        logger.warning("[shutdown] failed to import workers: %s", exc)
        return

    if plc_worker is not None:
        try:
            plc_worker.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("[shutdown] plc worker stop failed: %s", exc, exc_info=True)

    try:
        push_worker.stop()
    except Exception as exc:  # noqa: BLE001
        logger.error("[shutdown] push worker stop failed: %s", exc, exc_info=True)


def install_process_shutdown_handlers() -> None:
    """Install atexit and signal handlers for graceful worker shutdown."""
    global _shutdown_hooks_installed
    with _shutdown_lock:
        if _shutdown_hooks_installed:
            return
        _shutdown_hooks_installed = True

    atexit.register(shutdown_workers, reason="atexit")

    def _handle_signal(signum, _frame) -> None:
        logger.warning("[shutdown] received signal %s", signum)
        shutdown_workers(reason=f"signal:{signum}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signal.signal(sigterm, _handle_signal)
