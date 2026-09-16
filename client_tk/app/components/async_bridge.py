"""async_bridge.py — non-blocking API calls for Tkinter.

Usage
-----
    from client_tk.app.components.async_bridge import run_async

    def on_done(result, error):
        if error:
            show_error(str(error))
        else:
            populate_table(result)

    run_async(root_widget, api_client.list_templates, callback=on_done)

``run_async`` submits *fn* to a background daemon thread. The worker never
touches Tk directly. Instead, the owning widget polls for completion on the Tk
main thread and then delivers the result to *callback*.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


_POLL_INTERVAL_MS = 16


def run_async(
    widget,
    fn: Callable[[], Any],
    *,
    callback: Callable[[Any, Exception | None], None] | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
) -> threading.Thread:
    """Run *fn(*args, **kwargs)* in a background thread.

    Parameters
    ----------
    widget:
        Any live Tkinter widget used to schedule the callback via
        ``widget.after(0, ...)``.
    fn:
        The callable to execute off the main thread (e.g. an API call).
    callback:
        ``callback(result, error)`` — called on the Tk main thread once *fn*
        completes.  *error* is ``None`` on success, an ``Exception`` on
        failure.  *result* is ``None`` on failure.
    args / kwargs:
        Forwarded to *fn*.
    """
    _kwargs = kwargs or {}
    result_box: dict[str, Any] = {"result": None, "error": None}
    completed = threading.Event()

    def _poll_completion() -> None:
        try:
            if not widget.winfo_exists():
                return
        except Exception:
            return

        if not completed.is_set():
            try:
                widget.after(_POLL_INTERVAL_MS, _poll_completion)
            except Exception:
                return
            return

        if callback is None:
            return

        try:
            callback(result_box["result"], result_box["error"])
        except Exception:
            import traceback

            traceback.print_exc()
            return

    def _worker():
        try:
            result_box["result"] = fn(*args, **_kwargs)
        except Exception as exc:  # noqa: BLE001
            result_box["error"] = exc
        finally:
            completed.set()

    if callback is not None:
        try:
            widget.after(_POLL_INTERVAL_MS, _poll_completion)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
