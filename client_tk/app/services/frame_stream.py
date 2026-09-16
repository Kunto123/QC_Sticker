from __future__ import annotations

# Deprecated remote-only compatibility path.
# Local-only desktop mode no longer selects this transport by default.
# ---------------------------------------------------------------------------
# WebSocket frame streaming client (replaces FrameUploadService for sessions)
# ---------------------------------------------------------------------------
# Architecture:
#   • A single daemon thread runs an asyncio event loop.
#   • Two async tasks share a connection: _send_loop and _recv_loop.
#   • _send_loop calls get_frame() at each tick and sends the raw JPEG over
#     the socket.  CameraCaptureService.get_latest_frame() already returns the
#     newest frame so the send loop naturally implements a "latest only" policy.
#   • _recv_loop forwards every JSON text frame to on_result / on_error.
#   • Either task finishing (disconnect, stop flag) cancels the other.
# ---------------------------------------------------------------------------

import asyncio
import json
import struct
import threading
import time
from collections.abc import Callable

import cv2

from shared.contracts.streaming import (
    BINARY_HEADER_SIZE,
    MSG_AUTH,
    MSG_AUTH_OK,
    MSG_ERROR,
    MSG_PING,
    MSG_PONG,
    MSG_RESULT,
)

_PNG_ENCODE = True  # Gunakan PNG lossless untuk inference WebSocket


class FrameStreamService:
    """Persistent WebSocket sender — drop-in replacement for FrameUploadService
    on the session streaming path.

    start() accepts the same ``get_frame`` / ``on_result`` / ``on_error``
    callbacks as FrameUploadService so OperatorScreen can swap them without
    any change to the result-handling logic.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        stream_url: str,
        token: str,
        session_id: str,
        get_frame: Callable[[], object],
        on_result: Callable[[dict], None],
        on_error: Callable[[str], None],
        fps: float = 20.0,
    ) -> None:
        """Open a WebSocket connection and begin streaming frames.

        Parameters
        ----------
        stream_url:
            Full WebSocket URL, e.g. ``ws://127.0.0.1:8101``.
        token:
            Bearer token obtained from HTTP login (same token used for HTTP API).
        session_id:
            Active inspection session ID obtained from HTTP session start.
        get_frame:
            Callable returning the latest BGR numpy frame (or None).
            Must be thread-safe; called from the async event loop thread.
        on_result:
            Callback invoked with each parsed JSON result dict from the server.
            Called from the background thread — use threading.Lock if needed.
        on_error:
            Callback invoked with a string error message.
        fps:
            Target send rate.  Frames are sent at most this often; if inference
            is slower the server drops older frames automatically.
        """
        self.stop()
        self._running = True
        args = (stream_url, token, session_id, get_frame, on_result, on_error, max(1.0, float(fps)))
        self._thread = threading.Thread(
            target=self._run, args=args, daemon=True, name="ws-frame-stream",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the streaming loop to stop and wait briefly for it to exit."""
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._thread = None

    # ------------------------------------------------------------------
    # Internal — thread entry point
    # ------------------------------------------------------------------

    def _run(self, stream_url, token, session_id, get_frame, on_result, on_error, fps) -> None:
        try:
            asyncio.run(
                self._async_run(stream_url, token, session_id, get_frame, on_result, on_error, fps)
            )
        except Exception as exc:  # noqa: BLE001
            on_error(f"stream stopped: {exc}")

    # ------------------------------------------------------------------
    # Internal — async entry point
    # ------------------------------------------------------------------

    async def _async_run(
        self,
        stream_url: str,
        token: str,
        session_id: str,
        get_frame: Callable,
        on_result: Callable,
        on_error: Callable,
        fps: float,
    ) -> None:
        import websockets  # lazy so missing dep gives a clear import error

        try:
            async with websockets.connect(
                stream_url,
                open_timeout=10,
                close_timeout=5,
            ) as ws:
                # Auth handshake
                await ws.send(json.dumps({
                    "type": MSG_AUTH,
                    "token": token,
                    "session_id": session_id,
                }))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    on_error("stream auth timeout")
                    return

                try:
                    resp = json.loads(raw)
                except Exception:  # noqa: BLE001
                    on_error("stream: invalid auth response")
                    return

                if resp.get("type") != MSG_AUTH_OK:
                    on_error(f"stream auth failed: {resp.get('message', 'unknown')}")
                    return

                send_task = asyncio.create_task(
                    self._send_loop(ws, get_frame, 1.0 / fps)
                )
                recv_task = asyncio.create_task(
                    self._recv_loop(ws, on_result, on_error)
                )

                _done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except Exception as exc:  # noqa: BLE001
            if self._running:
                on_error(f"stream: {exc}")

    # ------------------------------------------------------------------
    # Internal — send loop
    # ------------------------------------------------------------------

    async def _send_loop(self, ws, get_frame: Callable, interval: float) -> None:
        """Encode latest frame and send as binary; sleep for remainder of interval."""
        seq = 0
        while self._running:
            tick = time.perf_counter()
            frame = get_frame()
            if frame is not None:
                ok, encoded = cv2.imencode(".png", frame)
                if ok:
                    seq = (seq + 1) & 0xFFFF_FFFF
                    await ws.send(struct.pack(">I", seq) + encoded.tobytes())
            elapsed = time.perf_counter() - tick
            sleep_s = interval - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Internal — receive loop
    # ------------------------------------------------------------------

    async def _recv_loop(self, ws, on_result: Callable, on_error: Callable) -> None:
        """Forward JSON frames from server to callbacks."""
        async for message in ws:
            if not self._running:
                break
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except Exception:  # noqa: BLE001
                continue
            msg_type = payload.get("type")
            if msg_type == MSG_ERROR:
                on_error(f"ws: {payload.get('message', 'server_error')}")
            elif msg_type in (MSG_RESULT, MSG_PONG) or msg_type is None:
                # Pass anything that looks like a result dict straight through.
                # on_result callbacks check for the keys they care about, so
                # a pong or unknown type is ignored downstream harmlessly.
                if msg_type == MSG_RESULT:
                    on_result(payload)
            # Ignore MSG_PONG and unknown control frames silently.
