from __future__ import annotations

# Deprecated remote-only compatibility path.
# Local-only desktop mode does not start this sidecar.
# ---------------------------------------------------------------------------
# WebSocket Streaming Sidecar
# ---------------------------------------------------------------------------
# Runs on a separate port from the Flask HTTP server so the WSGI stack is
# completely untouched.  A single daemon thread owns an asyncio event loop
# that hosts the websockets server.
#
# Protocol (see shared/contracts/streaming.py for full spec):
#   1. Client connects.
#   2. Client sends JSON AUTH frame with Bearer token + session_id.
#   3. Server validates token (same TokenStore as HTTP) and session existence.
#   4. Server sends AUTH_OK; binary JPEG frames may now flow.
#   5. Binary frame format: [4 bytes BE uint32 seq][JPEG bytes]
#   6. Server processes each frame with InspectionSessionService.process_frame_decoded()
#      in a thread-pool executor (never blocks the async loop).
#   7. Server sends JSON RESULT frame back on the same connection.
#   8. Drop-oldest policy: only the latest pending frame is kept per session.
# ---------------------------------------------------------------------------

import asyncio
import json
import logging
import struct
import threading
from typing import Any

import cv2
import numpy as np

from shared.contracts.streaming import (
    BINARY_HEADER_SIZE,
    MSG_AUTH,
    MSG_AUTH_OK,
    MSG_ERROR,
    MSG_PING,
    MSG_PONG,
    MSG_RESULT,
)

logger = logging.getLogger(__name__)

_server_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Lazy service accessors (avoids circular import at module load time)
# ---------------------------------------------------------------------------

def _get_token_store():
    from backend.app.core.container import token_store  # noqa: PLC0415
    return token_store


def _get_session_service():
    from backend.app.core.container import inspection_session_service  # noqa: PLC0415
    return inspection_session_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(jpeg_bytes: bytes):
    """Decode raw JPEG bytes to a BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid_jpeg_payload")
    return frame


async def _send_json(websocket, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

async def _handle_connection(websocket, *_args) -> None:
    """Entry point for each new WebSocket connection."""
    token_store = _get_token_store()
    session_service = _get_session_service()

    # ------------------------------------------------------------------
    # Phase 1 — auth handshake (10 s timeout)
    # ------------------------------------------------------------------
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
    except asyncio.TimeoutError:
        await websocket.close(1008, "auth_timeout")
        return
    except Exception:  # noqa: BLE001
        return

    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        await _send_json(websocket, {"type": MSG_ERROR, "message": "invalid_json"})
        await websocket.close(1008, "invalid_json")
        return

    if msg.get("type") != MSG_AUTH:
        await _send_json(websocket, {"type": MSG_ERROR, "message": "expected_auth_frame"})
        await websocket.close(1008, "expected_auth_frame")
        return

    token = str(msg.get("token") or "").strip()
    session_id = str(msg.get("session_id") or "").strip()

    user = token_store.get(token)
    if user is None:
        await _send_json(websocket, {"type": MSG_ERROR, "message": "invalid_token"})
        await websocket.close(4003, "unauthorized")
        return

    if not session_service.has_session(session_id):
        await _send_json(websocket, {"type": MSG_ERROR, "message": "session_not_found"})
        await websocket.close(4004, "session_not_found")
        return

    await _send_json(websocket, {"type": MSG_AUTH_OK, "session_id": session_id})
    logger.debug("[ws] auth ok user=%s session=%s", user.username, session_id[:8])

    # ------------------------------------------------------------------
    # Phase 2 — frame pump
    # Each incoming binary frame goes into a size-1 queue (drop-oldest).
    # A separate inference coroutine drains the queue sequentially so
    # inference never overlaps and backlogged frames are automatically
    # discarded.
    # ------------------------------------------------------------------
    frame_queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue(maxsize=1)
    loop = asyncio.get_running_loop()

    async def _recv_loop() -> None:
        try:
            async for message in websocket:
                if isinstance(message, bytes) and len(message) > BINARY_HEADER_SIZE:
                    seq = struct.unpack(">I", message[:BINARY_HEADER_SIZE])[0]
                    jpeg = message[BINARY_HEADER_SIZE:]
                    # Drop any waiting frame — keep only the latest.
                    if not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put((seq, jpeg))
                elif isinstance(message, str):
                    try:
                        ctrl = json.loads(message)
                        if ctrl.get("type") == MSG_PING:
                            await _send_json(websocket, {"type": MSG_PONG})
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            # Unblock the inference loop with a sentinel.
            _drain_and_put(frame_queue, None)

    def _drain_and_put(q: asyncio.Queue, item) -> None:
        """Put item into a full queue by draining first (called from async context)."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def _run_inference(seq: int, jpeg_bytes: bytes) -> dict:
        """Blocking inference call — runs in thread-pool executor."""
        import time
        decode_start = time.perf_counter()
        frame = _decode_jpeg(jpeg_bytes)
        decode_ms = round((time.perf_counter() - decode_start) * 1000.0, 2)
        return session_service.process_frame_decoded(
            session_id,
            frame=frame,
            decode_ms=decode_ms,
            response_mode="compact",
            username=user.username,
            user_id=user.id,
        )

    async def _infer_loop() -> None:
        while True:
            item = await frame_queue.get()
            if item is None:
                break
            seq, jpeg_bytes = item
            try:
                result = await loop.run_in_executor(None, _run_inference, seq, jpeg_bytes)
                payload: dict[str, Any] = {"type": MSG_RESULT, "frame_seq": seq, **result}
                await _send_json(websocket, payload)
            except ValueError as exc:
                await _send_json(websocket, {
                    "type": MSG_ERROR, "frame_seq": seq, "message": str(exc),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ws] inference error session=%s seq=%d: %s", session_id[:8], seq, exc)
                await _send_json(websocket, {
                    "type": MSG_ERROR, "frame_seq": seq, "message": "inference_error",
                })

    recv_task = asyncio.ensure_future(_recv_loop())
    infer_task = asyncio.ensure_future(_infer_loop())

    _done, pending = await asyncio.wait(
        [recv_task, infer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    logger.debug("[ws] connection closed user=%s session=%s", user.username, session_id[:8])


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start(host: str, port: int) -> None:
    """Start the WebSocket streaming server in a daemon background thread.

    Safe to call multiple times — no-op if the thread is already alive.
    The server runs until the process exits (daemon thread).
    """
    global _server_thread  # noqa: PLW0603

    if _server_thread is not None and _server_thread.is_alive():
        return

    def _thread_main() -> None:
        import websockets  # lazy so missing dep gives a clear error at startup

        async def _run() -> None:
            async with websockets.serve(_handle_connection, host, port):
                logger.info("[ws] streaming server listening on ws://%s:%d", host, port)
                await asyncio.get_running_loop().create_future()  # run forever

        asyncio.run(_run())

    _server_thread = threading.Thread(target=_thread_main, daemon=True, name="ws-stream")
    _server_thread.start()
    logger.info("[ws] streaming server thread started (ws://%s:%d)", host, port)
