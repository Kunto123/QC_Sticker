"""Test clamp gate retry mechanism.

Plan #9 — Scenario:
- enqueue_part_ready dipanggil saat reclamp interval masih aktif
  -> plc_part_ready_triggered di-reset
- Setelah interval lewat, retry berhasil -> gate terbuka
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.app.models.session_state import SessionState
from backend.app.services.inspection_session import InspectionSessionService
from shared.contracts.enums import SessionStatus
from shared.contracts.templates import (
    CameraDefaults,
    InspectionTemplate,
    PartReadyConfig,
    PersistenceConfig,
    RoiGeometry,
    StickerRule,
    VisionConfig,
)


def _make_session_state(session_id: str = "test-sess") -> SessionState:
    return SessionState(
        session_id=session_id,
        client_id="test-client",
        camera_index=0,
        template=InspectionTemplate(
            id=1,
            version_id=1,
            version_number=1,
            name="Test",
            description="",
            is_active=True,
            camera=CameraDefaults(),
            part_ready_roi=RoiGeometry(),
            sticker_roi=RoiGeometry(),
            vision=VisionConfig(),
            part_ready=PartReadyConfig(),
            sticker=StickerRule(part_name="P", expected_class="K"),
            persistence=PersistenceConfig(),
        ),
        status=SessionStatus.IDLE,
    )


def _make_service(**kwargs) -> InspectionSessionService:
    """Create service with mocked dependencies for unit testing."""
    with patch.object(InspectionSessionService, "__init__", lambda self, **kw: None):
        svc = InspectionSessionService.__new__(InspectionSessionService)
    svc._sessions = {}
    svc._lock = __import__("threading").RLock()
    if "plc_worker" in kwargs:
        # Caller supplies (and configures) their own worker — may be None
        svc._plc_worker = kwargs["plc_worker"]
    else:
        svc._plc_worker = MagicMock()
        svc._plc_worker.clamp_engaged.return_value = True
    svc._plc_clamp_feedback_enabled = kwargs.get("feedback_enabled", True)
    svc._plc_clamp_feedback_fallback_delay_ms = kwargs.get(
        "fallback_delay_ms", 100.0
    )
    svc._plc_clamp_feedback_timeout_ms = kwargs.get("timeout_ms", 5000.0)
    svc._phase_next_part_delay_ms = kwargs.get("phase_next_part_ms", 200.0)
    svc._phase_sticker_install_delay_ms = 0
    return svc


class TestClampGateRetry:
    """Test bahwa clamp gate retry bekerja setelah reclamp interval lewat."""

    def test_enqueue_resets_when_reclamp_blocked(self):
        """Fallback path: PLC masih IDLE (command diblock reclamp interval) -> flag di-reset."""
        plc_worker = MagicMock()
        plc_worker.clamp_engaged.return_value = False  # PLC IDLE — command diblock
        svc = _make_service(
            plc_worker=plc_worker,
            feedback_enabled=False,  # retry-reset hanya ada di fallback path
            fallback_delay_ms=100.0,
        )

        state = _make_session_state()
        state.plc_part_ready_triggered = True
        state.plc_clamp_requested_at = time.time() - 0.2  # 200ms ago (> fallback delay)
        svc._sessions["s1"] = state

        result = svc._clamp_gate_status(
            state, part_ready_settled=True, now_s=time.time()
        )

        assert result[0] is False
        assert state.plc_part_ready_triggered is False  # harus di-reset

    def test_gate_opens_after_reclamp_interval_elapsed(self):
        """Setelah reclamp interval lewat, gate harus terbuka."""
        plc_worker = MagicMock()
        plc_worker.clamp_engaged.return_value = True
        svc = _make_service(
            plc_worker=plc_worker,
            feedback_enabled=True,
        )

        state = _make_session_state()
        state.plc_clamp_requested_at = time.time() - 4.0  # 4 detik ago
        svc._sessions["s1"] = state

        svc.plc_min_reclamp_interval_ms = 3000
        ready, info = svc._clamp_gate_status(
            state, part_ready_settled=True, now_s=time.time()
        )

        assert ready is True, f"Expected gate ready=True, got {ready} ({info})"
        assert info["status"] == "clamped"

    def test_fallback_when_plc_feedback_false(self):
        """Fallback path: feedback_ready=False saat fallback delay elapsed -> flag di-reset untuk retry."""
        plc_worker = MagicMock()
        plc_worker.clamp_engaged.return_value = False  # PLC tidak clamp
        svc = _make_service(
            plc_worker=plc_worker,
            feedback_enabled=False,  # fallback path (time-based)
            fallback_delay_ms=50.0,  # 50ms fallback
        )

        state = _make_session_state()
        state.plc_part_ready_triggered = True
        state.plc_clamp_requested_at = time.time() - 0.2  # 200ms ago
        svc._sessions["s1"] = state

        svc.plc_min_reclamp_interval_ms = 0  # tidak ada block
        ready, info = svc._clamp_gate_status(
            state, part_ready_settled=True, now_s=time.time()
        )

        # Fallback delay 50ms sudah lewat (elapsed 200ms), tapi PLC masih IDLE -> flag di-reset
        assert state.plc_part_ready_triggered is False

    def test_gate_disabled_when_no_plc_worker(self):
        """Tanpa PLC worker, gate selalu terbuka."""
        svc = _make_service(plc_worker=None)
        state = _make_session_state()

        ready, info = svc._clamp_gate_status(
            state, part_ready_settled=True, now_s=time.time()
        )
        assert ready is True
        assert info["status"] == "disabled"

    def test_gate_waiting_when_part_not_settled(self):
        """Kalau part belum settled, gate tertutup."""
        svc = _make_service(plc_worker=MagicMock())
        state = _make_session_state()

        ready, info = svc._clamp_gate_status(
            state, part_ready_settled=False, now_s=time.time()
        )
        assert ready is False
        assert info["status"] == "waiting_part_ready"
