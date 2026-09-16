from __future__ import annotations

# ---------------------------------------------------------------------------
# WebSocket Streaming Protocol Contract
# ---------------------------------------------------------------------------
# Transport boundary:
#   HTTP  → login, start/stop session, deployment/template lookup, CRUD.
#   WS    → realtime frame streaming only (connect after HTTP session_start).
#
# Connection lifecycle:
#   1. Client opens ws://host:stream_port (after obtaining token via HTTP login
#      and session_id via HTTP /inspection/sessions/start).
#   2. Client sends a text AUTH frame (JSON).
#   3. Server replies AUTH_OK or ERROR then closes.
#   4. Client sends binary FRAME messages; server replies with text RESULT.
#   5. Either side may send PING/PONG for keepalive.
#   6. Either side closes the connection to end the stream.
#
# Binary frame format (client → server):
#   Bytes 0-3  : frame sequence number, big-endian uint32.
#   Bytes 4+   : raw JPEG payload (no base64 encoding).
#
# Text frame format (JSON, both directions):
#   {"type": <MSG_*>, ...fields...}
# ---------------------------------------------------------------------------

# -- Control message type constants --

MSG_AUTH = "auth"
"""Client → server. Payload: {"type":"auth","token":"...","session_id":"..."}"""

MSG_AUTH_OK = "auth_ok"
"""Server → client. Payload: {"type":"auth_ok","session_id":"..."}"""

MSG_ERROR = "error"
"""Server → client. Payload: {"type":"error","message":"...","frame_seq":N|None}"""

MSG_RESULT = "result"
"""Server → client. Full inspection result dict with added fields:
   "type": "result", "frame_seq": N
   Plus all fields from InspectionSessionService.process_frame_decoded():
     session, roi, part_ready, sticker_detection, validation, event_state,
     counters, timings, overlay_image_b64, device_backend, device_fallback_reason, …
"""

MSG_PING = "ping"
"""Client → server. Payload: {"type":"ping"}"""

MSG_PONG = "pong"
"""Server → client. Payload: {"type":"pong"}"""

# -- Binary frame header --

BINARY_HEADER_SIZE = 4
"""Bytes reserved for the big-endian uint32 frame sequence number prefix."""
