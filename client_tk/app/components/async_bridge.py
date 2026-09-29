"""async_bridge.py — pemanggilan API non-blocking untuk Tkinter.

Usage
-----
    from client_tk.app.components.async_bridge import run_async

    def on_done(result, error):
        if error:
            show_error(str(error))
        else:
            populate_table(result)

    run_async(root_widget, api_client.list_templates, callback=on_done)

``run_async`` mengirim *fn* ke thread daemon background. Worker tidak pernah
menyentuh Tk secara langsung. Sebagai gantinya, widget pemilik melakukan poll
untuk selesainya proses di thread utama Tk lalu mengantarkan hasilnya ke *callback*.
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
    """Jalankan *fn(*args, **kwargs)* di thread background.

    Parameters
    ----------
    widget:
        Widget Tkinter hidup mana pun yang dipakai untuk menjadwalkan callback lewat
        ``widget.after(0, ...)``.
    fn:
        Callable yang dieksekusi di luar thread utama (mis. pemanggilan API).
    callback:
        ``callback(result, error)`` — dipanggil di thread utama Tk setelah *fn*
        selesai. *error* adalah ``None`` kalau berhasil, ``Exception`` kalau
        gagal. *result* adalah ``None`` kalau gagal.
    args / kwargs:
        Diteruskan ke *fn*.
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
