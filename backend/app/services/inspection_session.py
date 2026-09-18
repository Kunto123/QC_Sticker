from __future__ import annotations

import base64
import concurrent.futures
import copy
import cv2
import logging
import os
import threading
import time
from time import monotonic
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from backend.app.core.config import AppConfig
from backend.app.core.json_safety import to_jsonable
from backend.app.models.session_state import SessionState
from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
from backend.app.repositories.profiles_repository import ProfilesRepository
from backend.app.repositories.reject_log_repository import RejectLogRepository
from backend.app.services.operator_state_machine import OperatorInspectionStateMachine
from backend.app.services.part_ready_detector import evaluate_color_profile_match, evaluate_hsv_black_ratio, _hsv_bounds
from backend.app.services.sticker_inference import StickerInferenceService
from backend.app.services.template_runtime import TemplateRuntimeService
from backend.app.services.text_tilt import estimate_white_text_tilt
from shared.contracts.enums import DecisionCode, InspectionEventState, RejectReasonCode, SessionStatus
from shared.contracts.templates import RoiGeometry, normalize_mode
from backend.app.services.evaluators.registry import get_evaluator


logger = logging.getLogger(__name__)


KNOWN_REJECT_CODES = (
    RejectReasonCode.OUT_OF_ANGLE.value,
    RejectReasonCode.WRONG_TYPE.value,
    RejectReasonCode.COMMIT_TIMEOUT.value,
)
MAX_RECENT_EVENTS = 8
COMMIT_STABLE_FRAMES = 1
COMMIT_COOLDOWN_MS = 800
PRESENCE_MIN_AREA_RATIO = 0.01
PRESENCE_MIN_STD = 8.0
PRESENCE_MIN_MEAN = 6.0
ROI_CLASS_VALIDATOR_MODES = {
    "ml_roi_class",
    "ml_roi_classification",
    "roi_class",
    "roi_partial",
}


def _decode_image(image_b64: str):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image payload.")
    return image


_OVERLAY_JPEG_QUALITY = int(os.getenv("QC_SUITE_OVERLAY_JPEG_QUALITY", "80"))
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, max(40, min(100, _OVERLAY_JPEG_QUALITY))]


def _encode_image(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image, _ENCODE_PARAMS)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _apply_rotation(frame: np.ndarray, rotation_degrees: float) -> np.ndarray:
    """Apply free rotation to frame. Supports any angle."""
    if not frame.size:
        return frame
    rotation_degrees = float(rotation_degrees) % 360.0
    if rotation_degrees == 0.0:
        return frame
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -rotation_degrees, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(frame, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)


def _rotate_frame(frame: np.ndarray, rotation_degrees: float) -> np.ndarray:
    """Alias for _apply_rotation (backward compat)."""
    return _apply_rotation(frame, rotation_degrees)


def _empty_reject_breakdown() -> dict[str, int]:
    return {code: 0 for code in KNOWN_REJECT_CODES}


def _round_bbox(position: dict[str, Any] | None) -> dict[str, float] | None:
    if not position:
        return None
    return {
        "x1": round(float(position.get("x1", 0.0)), 2),
        "y1": round(float(position.get("y1", 0.0)), 2),
        "x2": round(float(position.get("x2", 0.0)), 2),
        "y2": round(float(position.get("y2", 0.0)), 2),
    }


def _estimate_tilt_from_roi(roi_frame, expected_tilt_degrees: float, config: Any | None = None) -> dict[str, Any]:
    """Estimate sticker rotation using edge detection pipeline.
    Always delegates to estimate_sticker_rotation (alias: estimate_white_text_tilt).
    """
    return estimate_white_text_tilt(roi_frame, expected_tilt_degrees, config)


class InspectionSessionService:
    def __init__(
        self,
        template_runtime: TemplateRuntimeService,
        profiles_repo: ProfilesRepository,
        results_repo: InspectionResultsRepository,
        sticker_inference: StickerInferenceService,
        app_config: AppConfig | None = None,
        plc_worker=None,
        reject_log_repo: RejectLogRepository | None = None,
    ) -> None:
        self._template_runtime = template_runtime
        self._profiles_repo = profiles_repo
        self._results_repo = results_repo
        self._sticker_inference = sticker_inference
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self._accept_holdover_ms: int = (
            max(0, int(app_config.accept_holdover_ms))
            if app_config is not None
            else 2000
        )
        # System-wide settle default: used when a template's part_ready_settle_ms is None.
        self._default_settle_ms: int = (
            max(0, int(app_config.part_ready_settle_ms_default))
            if app_config is not None
            else 0
        )
        self._plc_clamp_feedback_enabled = (
            bool(getattr(app_config, "plc_clamp_feedback_enabled", False))
            if app_config is not None
            else False
        )
        self._inference_cache_grace_ms: int = (
            max(0, int(app_config.inference_cache_grace_ms))
            if app_config is not None
            else 300
        )
        self._plc_clamp_feedback_timeout_ms = (
            max(0, int(getattr(app_config, "plc_clamp_feedback_timeout_ms", 1500)))
            if app_config is not None
            else 1500
        )
        self._plc_clamp_feedback_fallback_delay_ms = (
            max(0, int(getattr(app_config, "plc_clamp_feedback_fallback_delay_ms", 300)))
            if app_config is not None
            else 300
        )
        self._phase_sticker_install_delay_ms = (
            max(0, int(getattr(app_config, "phase_sticker_install_delay_ms", 0)))
            if app_config is not None
            else 0
        )
        self._phase_next_part_delay_ms = (
            max(0, int(getattr(app_config, "phase_next_part_delay_ms", 2000)))
            if app_config is not None
            else 2000
        )
        self._plc_worker = plc_worker
        self._reject_log_repo = reject_log_repo
        self._operator_state_machine = OperatorInspectionStateMachine()
        self._max_consecutive_rejects: int = (
            max(0, int(app_config.max_consecutive_rejects))
            if app_config is not None
            else 0
        )
        self._idle_timeout_s: int = (
            max(0, int(app_config.session_idle_timeout_s))
            if app_config is not None
            else 300
        )
        self._part_ready_release_ms: int = (
            max(0, int(app_config.part_ready_release_ms_default))
            if app_config is not None
            else 300
        )
# Inspection policy settings
        self._hard_reject_reasons: set[str] = set(
            r.strip().upper()
            for r in app_config.inspect_hard_reject_reasons.split(",")
            if r.strip()
        ) if app_config is not None and app_config.inspect_hard_reject_reasons else set()
        self._reject_timeout_ms: int = (
            max(0, int(app_config.reject_timeout_ms))
            if app_config is not None else 15000
        )
        self._commit_grace_ms: int = (
            max(0, int(app_config.commit_grace_ms))
            if app_config is not None else 1500
        )
        self._accept_stable_frames: int = (
            max(1, int(app_config.accept_stable_frames))
            if app_config is not None else 2
        )
        self._accept_stable_ms: int = (
            max(0, int(app_config.accept_stable_ms))
            if app_config is not None else 200
        )
        self._hard_reject_stable_frames: int = (
            max(1, int(app_config.hard_reject_stable_frames))
            if app_config is not None else 3
        )
        self._hard_reject_stable_ms: int = (
            max(0, int(app_config.hard_reject_stable_ms))
            if app_config is not None else 500
        )
        self._camera_rotation_degrees: float = (
            float(app_config.camera_default_rotation_degrees)
            if app_config is not None else 0.0
        )
        # Inference cache TTL (ms) — how long cached inference is considered fresh
        self._inference_cache_ttl_ms: int = (
            max(100, int(app_config.inference_cache_ttl_ms))
            if app_config is not None else 10000
        )
        self._inference_timeout_s: float = (
            max(1.0, float(getattr(app_config, "inference_timeout_s", 5.0)))
            if app_config is not None else 5.0
        )
        # NG JSONL logger (daily rolling, no images)
        from backend.app.services.ng_cache_logger import NgCacheLogger
        _ng_log_dir = getattr(app_config, "ng_log_dir", None) if app_config is not None else None
        self._ng_logger = NgCacheLogger(_ng_log_dir, retention_days=30) if _ng_log_dir else None
        # Register PLC state change callback
        if self._plc_worker is not None:
            self._plc_worker.set_on_state_change_callback(self._on_plc_state_change)
            # Item 1: register actuation result callback for ACK/NACK reconciliation
            self._plc_worker.set_on_actuation_result_callback(self._on_actuation_result)

        # Item 1: pending actuation results awaiting ACK/NACK
        # Maps event_id -> {decision, result_id, status}
        self._pending_actuations: dict[str, dict] = {}

    def update_timing_settings(self, data: dict) -> None:
        """Override timing/inspection settings at runtime (called after Machine Settings save).

        Accepts a flat dict matching TimingConfig schema, or a nested 'timing' key.
        Falls back to current value if key is missing (safe for partial updates).
        """
        timing = data.get("timing", data) if isinstance(data, dict) else {}

        self._phase_next_part_delay_ms = max(0, int(timing.get(
            "phase_next_part_delay_ms", self._phase_next_part_delay_ms)))
        self._phase_sticker_install_delay_ms = max(0, int(timing.get(
            "phase_sticker_install_delay_ms", self._phase_sticker_install_delay_ms)))
        self._accept_stable_frames = max(1, int(timing.get(
            "accept_stable_frames", self._accept_stable_frames)))
        self._accept_stable_ms = max(0, int(timing.get(
            "accept_stable_ms", self._accept_stable_ms)))
        self._hard_reject_stable_frames = max(1, int(timing.get(
            "hard_reject_stable_frames", self._hard_reject_stable_frames)))
        self._hard_reject_stable_ms = max(0, int(timing.get(
            "hard_reject_stable_ms", self._hard_reject_stable_ms)))
        self._commit_grace_ms = max(0, int(timing.get(
            "commit_grace_ms", self._commit_grace_ms)))
        self._reject_timeout_ms = max(0, int(timing.get(
            "reject_timeout_ms", self._reject_timeout_ms)))
        self._part_ready_release_ms = max(0, int(timing.get(
            "part_ready_release_ms", self._part_ready_release_ms)))
        self._default_settle_ms = max(0, int(timing.get(
            "part_ready_settle_ms_default", self._default_settle_ms)))
        self._inference_cache_grace_ms = max(0, int(timing.get(
            "inference_cache_grace_ms", self._inference_cache_grace_ms)))
        self._accept_holdover_ms = max(0, int(timing.get(
            "accept_holdover_ms", self._accept_holdover_ms)))
        self._inference_cache_ttl_ms = max(100, int(timing.get(
            "inference_cache_ttl_ms", self._inference_cache_ttl_ms)))
        self._idle_timeout_s = max(0, int(timing.get(
            "session_idle_timeout_s", self._idle_timeout_s)))
        self._max_consecutive_rejects = max(0, int(timing.get(
            "max_consecutive_rejects", self._max_consecutive_rejects)))
        logger.info(
            "[inspection-session] timing settings updated from machine-settings (%d fields)",
            len(timing),
        )

    def _on_plc_state_change(self, old_state: str, new_state: str) -> None:
        """Callback dari PLC worker saat state berubah.
        Reset clamp gate saat PLC kembali ke IDLE (manual release).
        NOTE: settle_frame_count dan consecutive_reject_count TIDAK di-reset di sini
        supaya part berikutnya bisa langsung infer tanpa settling ulang.
        Hanya di-reset ketika part benar-benar leave (process_frame frame-by-frame).
        Thread-safe: called from PLC worker thread, acquires session lock.
        """
        if new_state == "IDLE" and old_state != "IDLE":
            cooldown_until = time.time() + (self._phase_next_part_delay_ms / 1000.0)
            with self._lock:
                for state in self._sessions.values():
                    # Reset clamp gate only — keep settle/reject state intact
                    state.plc_part_ready_triggered = False
                    state.inspection_result_cache = None
                    state.operator_state = "IDLE"
                    state.part_ready_latched = False
                    state.part_ready_latched_at = None
                    state.part_ready_unsettled_at = None
                    state.consecutive_part_ready_frames = 0          # ← reset untuk part berikutnya
                    state.part_ready_settle_started_at = None
                    state.last_inference_ms = 0
                    # Unlock PLC cycle
                    if self._plc_worker is not None:
                        self._plc_worker.unlock_cycle(reason=f"plc_idle_after_{old_state}")
                    state.manual_release_cooldown_until = cooldown_until
                    state.plc_clamp_requested_at = 0.0
                    state.plc_clamp_ready_at = 0.0
                    state.plc_clamp_timeout = False
                    state.plc_clamp_event_id = None
                    state.operator_sticker_delay_started_at = 0.0
                    state.operator_sticker_ready_at = 0.0
                    state.part_ready_settled_at = None  # reset timeout tracker for new cycle
                    # Reset accept-cycle counters — mencegah commit instan dari akumulasi selama hold
                    state.accept_cycle_started_at = None
                    state.policy_stable_frames = 0
                    state.policy_stable_started_at = None
                    state.policy_holdover_expires_at = None
                    state.inference_accept_count = 0
                    state.inference_last_counted_generation = -1
                    state.inference_accept_first_ts = 0.0
            logger.info(
                "[inspection] PLC returned IDLE - next-part delay %dms started",
                self._phase_next_part_delay_ms,
            )

        # Saat PLC mulai clamp baru (IDLE → CLAMPING): ini adalah siklus fisik baru.
        # Clear inference cache only
        if new_state == "CLAMPING" and old_state == "IDLE":
            with self._lock:
                for state in self._sessions.values():
                    state.inference_result_cache = None  # flush stale cache
                    state.inference_result_ts = 0.0       # paksa re-infer
            logger.info("[inspection] PLC CLAMPING — inference cache cleared for new cycle")

    def _on_actuation_result(self, event_id: str | None, decision: str, ok: bool, reason: str = "") -> None:
        """Item 1: handle actuation ACK/NACK from PLC worker.

        On ACK (ok=True): mark pending result as "actuated".
        On NACK (ok=False): mark pending result as "actuation_failed".
        """
        if not event_id:
            return
        pending = self._pending_actuations.pop(event_id, None)
        if pending is None:
            return
        result_id = pending.get("result_id")
        if result_id is None:
            return
        _new_status = "actuated" if ok else "actuation_failed"
        try:
            if self._results_repo is not None and hasattr(self._results_repo, "update_result"):
                self._results_repo.update_result(result_id, {
                    "actuation_status": _new_status,
                    "actuation_error": None if ok else reason,
                })
        except Exception as exc:
            logger.warning("[inspection] failed to update actuation status for result %s: %s", result_id, exc)
        logger.info(
            "[inspection] actuation %s for event %s result %s (decision=%s)",
            _new_status, event_id[:8] if event_id else "?", result_id, decision,
        )

    def _reset_clamp_gate(self, state: SessionState) -> None:
        state.plc_part_ready_triggered = False
        state.plc_clamp_requested_at = 0.0
        state.plc_clamp_ready_at = 0.0
        state.plc_clamp_timeout = False
        state.plc_clamp_event_id = None
        state.operator_sticker_delay_started_at = 0.0
        state.operator_sticker_ready_at = 0.0

    def _clamp_gate_status(self, state: SessionState, *, part_ready_settled: bool, now_s: float) -> tuple[bool, dict[str, Any]]:
        if self._plc_worker is None:
            return True, {
                "enabled": False,
                "feedback_enabled": False,
                "status": "disabled",
                "ready": True,
            }
        if not part_ready_settled:
            return False, {
                "enabled": True,
                "feedback_enabled": self._plc_clamp_feedback_enabled,
                "status": "waiting_part_ready",
                "ready": False,
            }

        cooldown_until = float(getattr(state, "manual_release_cooldown_until", 0.0) or 0.0)
        if cooldown_until > now_s:
            return False, {
                "enabled": True,
                "feedback_enabled": self._plc_clamp_feedback_enabled,
                "status": "next_part_delay",
                "ready": False,
                "remaining_ms": round((cooldown_until - now_s) * 1000.0, 1),
                "delay_ms": self._phase_next_part_delay_ms,
            }

        requested_at = float(getattr(state, "plc_clamp_requested_at", 0.0) or 0.0)
        elapsed_ms = max(0.0, (now_s - requested_at) * 1000.0) if requested_at > 0 else 0.0
        feedback_ready = False
        try:
            feedback_ready = bool(self._plc_worker.clamp_engaged())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[inspection] clamp feedback read failed: %s", exc)

        if self._plc_clamp_feedback_enabled:
            if feedback_ready:
                if not state.plc_clamp_ready_at:
                    state.plc_clamp_ready_at = now_s
                return True, {
                    "enabled": True,
                    "feedback_enabled": True,
                    "status": "clamped",
                    "ready": True,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "feedback": True,
                }
            timeout_ms = float(self._plc_clamp_feedback_timeout_ms)
            if timeout_ms > 0 and elapsed_ms >= timeout_ms:
                state.plc_clamp_timeout = True
                return False, {
                    "enabled": True,
                    "feedback_enabled": True,
                    "status": "feedback_timeout",
                    "ready": False,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "timeout_ms": self._plc_clamp_feedback_timeout_ms,
                    "feedback": False,
                }
            return False, {
                "enabled": True,
                "feedback_enabled": True,
                "status": "wait_feedback",
                "ready": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "timeout_ms": self._plc_clamp_feedback_timeout_ms,
                "feedback": False,
            }

        delay_ms = float(self._plc_clamp_feedback_fallback_delay_ms)
        if requested_at <= 0 or elapsed_ms < delay_ms:
            return False, {
                "enabled": True,
                "feedback_enabled": False,
                "status": "clamping",
                "ready": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "fallback_delay_ms": self._plc_clamp_feedback_fallback_delay_ms,
            }
        if not feedback_ready:
            # PLC masih IDLE — enqueue_part_ready diblock (reclamp interval / cycle lock).
            # Reset agar session retry di frame berikutnya.
            state.plc_part_ready_triggered = False
            state.plc_clamp_requested_at = 0.0
            state.plc_clamp_ready_at = 0.0
            return False, {
                "enabled": True,
                "feedback_enabled": False,
                "status": "clamping",
                "ready": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "fallback_delay_ms": self._plc_clamp_feedback_fallback_delay_ms,
            }
        if not state.plc_clamp_ready_at:
            state.plc_clamp_ready_at = now_s
        return True, {
            "enabled": True,
            "feedback_enabled": False,
            "status": "clamped",
            "ready": True,
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback_delay_ms": self._plc_clamp_feedback_fallback_delay_ms,
        }

    def _operator_phase_status(
        self,
        state: SessionState,
        *,
        raw_part_ready: bool,
        part_ready_settled: bool,
        clamp_ready: bool,
        now_s: float,
    ) -> tuple[bool, dict[str, Any]]:
        base_payload = {
            "sticker_install_delay_ms": self._phase_sticker_install_delay_ms,
            "next_part_delay_ms": self._phase_next_part_delay_ms,
        }
        if not raw_part_ready:
            state.operator_sticker_delay_started_at = 0.0
            state.operator_sticker_ready_at = 0.0
            return False, {**base_payload, "status": "waiting_part_ready", "ready": False}
        if not part_ready_settled:
            state.operator_sticker_delay_started_at = 0.0
            state.operator_sticker_ready_at = 0.0
            return False, {**base_payload, "status": "part_ready_settling", "ready": False}
        if not clamp_ready:
            state.operator_sticker_delay_started_at = 0.0
            state.operator_sticker_ready_at = 0.0
            return False, {**base_payload, "status": "waiting_clamp", "ready": False}

        delay_ms = float(self._phase_sticker_install_delay_ms)
        if delay_ms <= 0:
            if not state.operator_sticker_ready_at:
                state.operator_sticker_ready_at = now_s
            return True, {
                **base_payload,
                "status": "ready",
                "ready": True,
                "elapsed_ms": 0.0,
            }

        started_at = float(getattr(state, "operator_sticker_delay_started_at", 0.0) or 0.0)
        if started_at <= 0:
            started_at = now_s
            state.operator_sticker_delay_started_at = started_at
            state.operator_sticker_ready_at = 0.0
        elapsed_ms = max(0.0, (now_s - started_at) * 1000.0)
        if elapsed_ms < delay_ms:
            return False, {
                **base_payload,
                "status": "sticker_install_delay",
                "ready": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "remaining_ms": round(delay_ms - elapsed_ms, 1),
            }

        if not state.operator_sticker_ready_at:
            state.operator_sticker_ready_at = now_s
        return True, {
            **base_payload,
            "status": "ready",
            "ready": True,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def start_session(
        self,
        *,
        client_id: str,
        camera_index: int,
        camera_rotation_degrees: float = 0.0,
        template_version_id: int,
        line_id: str | None = None,
        station_id: str | None = None,
    ) -> dict[str, Any]:
        template = self._template_runtime.resolve_template_by_version(template_version_id)
        # Apply camera rotation override from session creation payload
        _cam_rotation = float(camera_rotation_degrees or 0)
        if _cam_rotation != 0.0:
            template = copy.deepcopy(template)
            template.camera.rotation_degrees = _cam_rotation
        session_id = uuid.uuid4().hex
        state = SessionState(
            session_id=session_id,
            client_id=client_id,
            camera_index=int(camera_index),
            template=template,
            status=SessionStatus.RUNNING,
            line_id=line_id,
            station_id=station_id,
            session_reject_breakdown=_empty_reject_breakdown(),
            inference_interval_ms=0,
            inference_cache_ttl_ms=self._inference_cache_ttl_ms,
        )
        with self._lock:
            self._sessions[session_id] = state
        return self._session_payload(state)

    def update_roi(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        state = self._require_session(session_id)
        allowed = ("x", "y", "w", "h", "width", "height")
        before_part_ready_signature = self._roi_signature(
            self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        )

        legacy_updates = {key: updates[key] for key in allowed if key in updates}
        legacy_nested = updates.get("roi") if isinstance(updates.get("roi"), dict) else None
        if legacy_nested:
            legacy_updates.update({key: legacy_nested[key] for key in allowed if key in legacy_nested})
        if legacy_updates:
            state.part_ready_roi_override.update(legacy_updates)
            state.sticker_roi_override.update(legacy_updates)

        part_ready_updates = updates.get("part_ready_roi")
        if isinstance(part_ready_updates, dict):
            state.part_ready_roi_override.update(
                {key: part_ready_updates[key] for key in allowed if key in part_ready_updates}
            )

        sticker_updates = updates.get("sticker_roi")
        if isinstance(sticker_updates, dict):
            state.sticker_roi_override.update(
                {key: sticker_updates[key] for key in allowed if key in sticker_updates}
            )

        after_part_ready_signature = self._roi_signature(
            self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        )
        if after_part_ready_signature != before_part_ready_signature:
            state.part_ready_ratio_history.clear()
            state.part_ready_ema_ratio = -1.0
            state.hsv_adaptive_lower = None
            state.hsv_adaptive_upper = None
        return self._session_payload(state)

    def get_latest_preview(self) -> dict[str, Any] | None:
        with self._lock:
            for state in self._sessions.values():
                if state.last_overlay_b64:
                    return {
                        "overlay_image_b64": state.last_overlay_b64,
                        "session_id": state.session_id,
                        "frame_index": state.frame_index,
                    }
        return None

    def stop_session(self, session_id: str) -> dict[str, Any]:
        state = self._require_session(session_id)
        state.status = SessionStatus.STOPPED
        with self._lock:
            self._sessions.pop(session_id, None)
        # Shutdown background inference thread cleanly
        if state._inference_executor is not None:
            state._inference_executor.shutdown(wait=False)
            state._inference_executor = None
        return self._session_payload(state)

    def has_session(self, session_id: str) -> bool:
        """Return True if a session with this id is currently active."""
        with self._lock:
            return session_id in self._sessions

    def manual_release(self, session_id: str, *, reason: str = "manual_operator") -> dict[str, Any]:
        """Operator-triggered NEUTRAL release.

        Unclamps the current part and resets the clamping/inspection cycle WITHOUT
        committing any result. No accept/reject is logged and counters are unchanged.
        Used when a part is still being inspected (no ACCEPT yet) but the operator
        wants to remove it (e.g. obviously wrong part, mis-loaded part).
        """
        state = self._require_session(session_id)
        # Reset the cycle under lock — PLC callback thread also accesses session state.
        with self._lock:
            state.part_ready_latched = False
            state.part_ready_latched_at = None
            state.part_ready_unsettled_at = None
            state.part_ready_settled_at = None
            state.consecutive_part_ready_frames = 0
            state.plc_part_ready_triggered = False
            state.current_event_committed = False
            state.current_event_id = None
            state.current_event_key = None
            state.current_presence = False
            state.current_event_started_at = None
            state.current_event_stable_frames = 0
            state.cooldown_until = None
            state.policy_stable_frames = 0
            state.last_policy_key = ""
            state.policy_stable_started_at = None
            state.policy_holdover_expires_at = None
            state.accept_cycle_started_at = None
            state.inference_accept_count = 0
            state.inference_last_counted_generation = -1
            state.inference_accept_first_ts = 0.0
            # Blackout to prevent immediate re-clamp of the same still-present part.
            if self._phase_next_part_delay_ms > 0:
                state.manual_release_cooldown_until = max(
                    float(getattr(state, "manual_release_cooldown_until", 0.0) or 0.0),
                    time.time() + (self._phase_next_part_delay_ms / 1000.0),
                )
        # Unclamp via PLC — neutral, no decision recorded.
        if self._plc_worker is not None:
            try:
                self._plc_worker.force_release(reason=reason)
            except Exception as exc:  # noqa: BLE001
                logger.error("[inspection] manual_release force_release failed: %s", exc)
        return self._session_payload(state)

    def process_frame(
        self,
        session_id: str,
        *,
        image_b64: str,
        response_mode: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        decode_started = time.perf_counter()
        frame = _decode_image(image_b64)
        decode_ms = round((time.perf_counter() - decode_started) * 1000.0, 2)
        return self.process_frame_decoded(
            session_id,
            frame=frame,
            decode_ms=decode_ms,
            response_mode=response_mode,
            username=username,
            user_id=user_id,
        )

    def process_frame_decoded(
        self,
        session_id: str,
        *,
        frame,
        decode_ms: float = 0.0,
        response_mode: str | None = None,
        username: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Process a pre-decoded numpy BGR frame for the given session.

        Called by ``process_frame`` (which decodes base64 first) and by the
        WebSocket streaming handler (which receives raw JPEG bytes directly).
        ``decode_ms`` carries the caller's decode timing so it is reflected in
        the returned timings payload.
        """
        total_started = time.perf_counter()
        timings: dict[str, float] = {"decode_ms": float(decode_ms)}

        def _elapsed_ms(started_at: float) -> float:
            return round((time.perf_counter() - started_at) * 1000.0, 2)

        state = self._require_session(session_id)
        state.frame_index += 1
        state.last_activity_at = time.time()

        # Apply camera rotation from template config
        _rotation = float(getattr(state.template.camera, "rotation_degrees", None)
                          or self._camera_rotation_degrees)
        if _rotation != 0.0:
            frame = _apply_rotation(frame, _rotation)

        part_ready_started = time.perf_counter()
        validator_mode = str(getattr(state.template.sticker, "validator_mode", "") or "").strip().lower()
        is_component_counter = normalize_mode(validator_mode) == "counter"
        is_sticker = validator_mode in ("sticker", "ml_detection")
        _part_ready_source = "sensor"
        if is_component_counter:
            _part_ready_source = str((state.template.criteria or {}).get("part_ready_source") or "sensor").strip().lower()
        _use_sensor_stub = is_component_counter and _part_ready_source != "camera_roi"

        if _use_sensor_stub:
            # Component Counter mode with sensor input: ignore camera-based part ready ROI
            part_ready_frame = None
            part_ready_roi_meta = {}
            part_ready = {"part_ready": True, "status": "sensor_input", "match_ratio": 1.0}
            presence = {"present": True, "area_ratio": 1.0, "mean": 255.0, "std": 0.0}
        else:
            part_ready_frame, part_ready_roi_meta = self._crop_stage_roi(
                frame,
                state.template.part_ready_roi,
                state.part_ready_roi_override,
            )
        if not _use_sensor_stub:
            if state.part_ready_latched:
                part_ready = {"part_ready": True, "status": "latched", "match_ratio": 1.0}
            else:
                part_ready = self._evaluate_part_ready(part_ready_frame, state)
            presence = self._detect_part_presence(part_ready_frame)
            timings["part_ready_eval_ms"] = _elapsed_ms(part_ready_started)

        roi_crop_started = time.perf_counter()
        sticker_frame, sticker_roi_meta = self._crop_stage_roi(
            frame,
            state.template.sticker_roi,
            state.sticker_roi_override,
        )
        timings["sticker_roi_crop_ms"] = _elapsed_ms(roi_crop_started)
        # Blackout gate: after commit, force "part not found" for phase_next_part_delay_ms
        # to give operator time to swap part. Re-arms automatically when timer expires.
        _now_s = time.time()
        _blackout_until = float(getattr(state, "manual_release_cooldown_until", 0.0) or 0.0)
        _in_blackout = _now_s < _blackout_until
        if _in_blackout:
            # Force all cycle state to idle
            state.part_ready_latched = False
            state.consecutive_part_ready_frames = 0
            state.plc_part_ready_triggered = False
            # Build and return minimal "part not found" response — skip all processing
            phase_remaining_ms = round((_blackout_until - _now_s) * 1000.0, 1)
            return {
                "session": self._session_payload(state),
                "presence": presence,
                "part_ready": {
                    **part_ready,
                    "status": "part_not_ready",
                    "part_ready": False,
                    "reject_reason_code": None,
                    "effective_part_ready": False,
                    "part_ready_latched": False,
                    "latch_status": "inactive",
                },
                "event_state": InspectionEventState.IDLE.value,
                "operator_state": "blackout_reset",
                "clamp": {
                    "enabled": True,
                    "feedback_enabled": False,
                    "status": "idle",
                    "ready": True,
                },
                "phase": {
                    "status": "post_commit_reset",
                    "remaining_ms": phase_remaining_ms,
                    "ready": False,
                },
                "inference_gate": {
                    "raw_part_ready": False,
                    "raw_status": "part_not_ready",
                    "raw_match_ratio": 0.0,
                    "part_ready_latched": False,
                    "effective_part_ready": False,
                    "clamp_ready": True,
                    "phase_ready": False,
                    "can_infer": False,
                    "block_reason": "blackout_reset",
                },
                "sticker_detection": {},
                # Blackout = idle reset window, no inspection ran → neutral decision
                # (None renders as "WAITING" on the client, not a spurious reject).
                "validation": {
                    "decision": None,
                    "reject_reason_code": None,
                    "validation_details": {"mode": normalize_mode(validator_mode)},
                },
                "inspection_policy": {"action": "pending", "commit_allowed": False},
                "count_committed": False,
                "count_source": None,
                "counters": self._counter_payload(state),
                "last_committed_result": state.last_committed_result,
                "recent_events": list(state.recent_events),
                "timings": {**timings, "event_state_ms": 0.0, "total_ms": _elapsed_ms(total_started)},
            }
# ------------------------------------------------------------------
        # Settle — frame-count based
        # Tunggu N frame berturut-turut di atas threshold sebelum clamp engage.
        # Jika sudah latched, abaikan raw state — tetap settled.
        # ------------------------------------------------------------------
        # Prefer ms-based settle if configured; fall back to system default (part_ready_settle_ms_default)
        # settle_ms == 0 means immediate settle (no wait)
        _settle_ms = getattr(state.template.sticker, "part_ready_settle_ms", None)
        if _settle_ms is not None:
            if _settle_ms == 0:
                _settle_frames = 0  # immediate settle
            else:
                _settle_frames = max(1, int(_settle_ms / 100.0))
        else:
            # Use system default settle_ms (from config), convert to frames
            _settle_ms = self._default_settle_ms
            if _settle_ms == 0:
                _settle_frames = 0
            else:
                _settle_frames = max(1, int(_settle_ms / 100.0))
        _settle_now = datetime.now(UTC)
        _raw_part_ready = part_ready.get("part_ready", False)

        if state.part_ready_latched:
            # Sudah engaged — tetap settled tanpa melihat raw part_ready
            part_ready_settled = True
            settle_remaining_ms = 0.0
        elif _settle_frames == 0:
            # settle_ms == 0: immediate settle, no frame counting needed
            part_ready_settled = True
            settle_remaining_ms = 0.0
        elif _raw_part_ready and presence.get("present", False):
            state.consecutive_part_ready_frames += 1
            part_ready_settled = state.consecutive_part_ready_frames >= _settle_frames
            settle_remaining_ms = (
                max(0.0, float(_settle_frames - state.consecutive_part_ready_frames) * 100.0)
                if not part_ready_settled else 0.0
            )
        else:
            state.consecutive_part_ready_frames = 0
            part_ready_settled = False
            settle_remaining_ms = 0.0
            self._reset_clamp_gate(state)

        # ── Add settle metadata to part_ready payload for UI ──
        # Report the original ms value from template (or derived from frames)
        _report_settle_ms = _settle_ms if _settle_ms is not None else _settle_frames * 100.0
        part_ready["part_ready_settled"] = part_ready_settled
        part_ready["part_ready_settle_ms"] = _report_settle_ms
        part_ready["part_ready_settle_remaining_ms"] = round(settle_remaining_ms, 1)

        # ── Timeout tracker: record when part first becomes settled ──
        if part_ready_settled and state.part_ready_settled_at is None:
            state.part_ready_settled_at = _settle_now

        # ------------------------------------------------------------------
        # Part-Ready Latch Logic
        # Latch protects inference gate from brief part_ready drops during
        # inspection.  It MUST release when presence is truly gone for
        # _part_ready_release_ms so that the next cycle can start clean.
        # ------------------------------------------------------------------
        _now_dt = _settle_now

        if part_ready_settled and not state.part_ready_latched:
            state.part_ready_latched = True
            state.part_ready_latched_at = _now_dt
            state.part_ready_unsettled_at = None

        # Release latch when presence is gone long enough
        if state.part_ready_latched and not presence.get("present", False):
            if state.part_ready_unsettled_at is None:
                state.part_ready_unsettled_at = _now_dt
            _unsettled_ms = (
                (_now_dt - state.part_ready_unsettled_at).total_seconds() * 1000.0
            )
            if _unsettled_ms >= self._part_ready_release_ms:
                state.part_ready_latched = False
                state.part_ready_latched_at = None
                state.part_ready_unsettled_at = None
                self._reset_clamp_gate(state)
        elif state.part_ready_latched and presence.get("present", False):
            # Presence back — reset unsettled timer
            state.part_ready_unsettled_at = None

        # Trigger PLC clamp hold on the first frame where part is settled.
        # But check release/next-part cooldown before re-clamping.
        _cooldown_until = float(getattr(state, "manual_release_cooldown_until", 0.0))
        _now_s = time.time()
        if part_ready_settled and not state.plc_part_ready_triggered:
            if _cooldown_until > 0 and _now_s < _cooldown_until:
                logger.debug("[inspection] cooldown active, skip clamp re-trigger")
            else:
                state.plc_part_ready_triggered = True
                state.plc_clamp_requested_at = _now_s
                state.plc_clamp_ready_at = 0.0
                state.plc_clamp_timeout = False
                state.plc_clamp_event_id = state.current_event_id
                if self._plc_worker is not None:
                    try:
                        self._plc_worker.enqueue_part_ready(event_id=state.current_event_id)
                    except Exception as exc:
                        logger.warning("[inspection] enqueue_part_ready failed: %s", exc)
        clamp_ready_for_inference, clamp_payload = self._clamp_gate_status(
            state,
            part_ready_settled=bool(part_ready_settled),
            now_s=_now_s,
        )
        phase_ready_for_inference, phase_payload = self._operator_phase_status(
            state,
            raw_part_ready=bool(_raw_part_ready or state.part_ready_latched),
            part_ready_settled=bool(part_ready_settled),
            clamp_ready=bool(clamp_ready_for_inference),
            now_s=_now_s,
        )

        # effective_part_ready gate for downstream logic / UI.
        # When latch is active, we report effective_part_ready=True even if raw
        # briefly dropped, so sticker inference is not blocked by noise.
        _latch_ready = state.part_ready_latched and part_ready_settled
        if _raw_part_ready and part_ready_settled:
            effective_part_ready_val = True
            _effective_block_reason = None
        elif _latch_ready:
            # Latched + settled but raw briefly down — still allow inference
            effective_part_ready_val = True
            _effective_block_reason = None
        elif _raw_part_ready and not part_ready_settled:
            effective_part_ready_val = False
            _effective_block_reason = "settling"
        elif not _raw_part_ready and presence.get("present", False):
            effective_part_ready_val = False
            _effective_block_reason = "part_not_ready"
        else:
            effective_part_ready_val = False
            _effective_block_reason = "no_part"

        # Build effective_part_ready dict for backward compatibility
        if effective_part_ready_val:
            effective_part_ready = {**part_ready, "part_ready": True}
        else:
            effective_part_ready = {
                **part_ready,
                "part_ready": False,
                "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
                "status": _effective_block_reason or "part_not_ready",
            }

        # ── Build inference_gate top-level response ──
        _can_infer = (
            effective_part_ready_val
            and clamp_ready_for_inference
            and phase_ready_for_inference
        )
        _gate_block_reason = None
        if not effective_part_ready_val:
            _gate_block_reason = _effective_block_reason or "part_not_ready"
        elif not clamp_ready_for_inference:
            _gate_block_reason = str(clamp_payload.get("status") or "clamping")
        elif not phase_ready_for_inference:
            _gate_block_reason = str(phase_payload.get("status") or "operator_phase_delay")

        inference_gate = {
            "raw_part_ready": _raw_part_ready,
            "raw_status": str(part_ready.get("status") or "unknown"),
            "raw_match_ratio": part_ready.get("match_ratio"),
            "part_ready_latched": state.part_ready_latched,
            "effective_part_ready": effective_part_ready_val,
            "clamp_ready": clamp_ready_for_inference,
            "phase_ready": phase_ready_for_inference,
            "can_infer": _can_infer,
            "block_reason": _gate_block_reason,
        }

        # Augment part_ready dict with latch + gate info for UI
        part_ready["effective_part_ready"] = effective_part_ready_val
        part_ready["part_ready_latched"] = state.part_ready_latched
        part_ready["latch_status"] = (
            "latched" if state.part_ready_latched
            else "released" if not state.part_ready_latched and state.part_ready_latched_at is not None
            else "inactive"
        )

        # Effective presence for operator state machine: use raw presence.
        # Latch only protects the inference gate, not removal detection.
        _effective_present = bool(presence.get("present", False))
        operator_state_decision = self._operator_state_machine.update(
            state,
            part_ready=bool(effective_part_ready_val),
            present=_effective_present,          # ← ganti dari presence.get("present", False)
            settled=bool(part_ready_settled and clamp_ready_for_inference and phase_ready_for_inference),
        )

        if operator_state_decision.use_cached_result and state.inspection_result_cache:
            cached_payload = dict(state.inspection_result_cache)
            cached_timings = dict(cached_payload.get("timings") or {})
            cached_timings.update(
                {
                    **timings,
                    "inference_ms": 0.0,
                    "inference_skipped": True,
                    "operator_state": operator_state_decision.state.value,
                    "total_ms": _elapsed_ms(total_started),
                }
            )
            cached_payload.update(
                {
                    "session": self._session_payload(state),
                    "presence": presence,
                    "part_ready": part_ready,
                    "event_state": InspectionEventState.COOLDOWN.value,
                    "operator_state": operator_state_decision.state.value,
                    "clamp": clamp_payload,
                    "phase": phase_payload,
                    "count_committed": False,
                    "count_source": None,
                    "counters": self._counter_payload(state),
                    "last_committed_result": state.last_committed_result,
                    "recent_events": list(state.recent_events),
                    "timings": cached_timings,
                }
            )
            state.latest_result = to_jsonable(cached_payload)
            return cached_payload

        _effective_pr_ready = bool(operator_state_decision.run_inspection)
        detections: list[dict[str, Any]] = []
        inference_ms = 0.0
        stage_timings: dict[str, Any] = {}

        # ── Async background inference (frame skip + TTL cache) ──
        # Submit inference every 3rd frame when not busy.
        # Use cached result if fresh (<500ms), otherwise compose without bbox.
        if _effective_pr_ready:
            state.inference_frame_counter += 1
            # Inference timeout guard: reset stuck busy flag
            if (
                state.inference_thread_busy
                and state.inference_submit_at > 0
                and (monotonic() - state.inference_submit_at) > self._inference_timeout_s
            ):
                logger.warning(
                    "[inference] timeout after %.0fs — resetting busy flag + clearing stale cache for session %s",
                    self._inference_timeout_s, state.session_id[:8],
                )
                state.inference_result_cache = None
                state.inference_result_ts = 0.0
                state.inference_thread_busy = False
                state.inference_submit_at = 0.0
            _should_submit = (
                state.inference_frame_counter % 1 == 0
                and not state.inference_thread_busy
            )
            if _should_submit:
                state.inference_thread_busy = True
                state.inference_submit_at = monotonic()
                # Lazy-init executor per session
                if state._inference_executor is None:
                    state._inference_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"qc-inference-{state.session_id[:8]}",
                    )
                try:
                    # Component counter: run inference on FULL frame so detections
                    # carry absolute pixel coords for assign-by-coordinate.
                    _inf_frame = frame.copy() if is_component_counter else sticker_frame.copy()
                    _future = state._inference_executor.submit(
                        self._run_sticker_inference_sync,
                        _inf_frame,
                        state,
                    )
                    _future.add_done_callback(
                        lambda f, s=state: self._on_inference_done(f, s)
                    )
                except Exception as exc:
                    logger.warning("[inference] submit failed: %s", exc)
                    state.inference_thread_busy = False

            # Use cached result if fresh enough
            _cache_ttl_s = state.inference_cache_ttl_ms / 1000.0
            _age = monotonic() - state.inference_result_ts
            if state.inference_result_cache is not None and _age <= _cache_ttl_s:
                _cached = state.inference_result_cache
                detections = list(_cached.get("detections") or [])
                inference_ms = float(_cached.get("inference_ms", 0.0))
                stage_timings = dict(_cached.get("timings") or {})
                for key, value in stage_timings.items():
                    try:
                        timings[f"inference_{key}"] = float(value)
                    except (TypeError, ValueError):
                        continue
                inference_payload = _cached
            else:
                # Stale or no cache — compose without bbox
                inference_payload = {
                    "backend": "async_cache",
                    "model_path": state.template.vision.model_path,
                    "meta_path": state.template.vision.model_meta_path,
                    "class_names": state.template.vision.classes,
                    "fallback_reason": None,
                    "raw_detection_count": 0,
                    "allowed_labels_filter": [],
                    "anchor": None,
                    "ocr": None,
                    "geometry": None,
                    "device_mode": None,
                    "effective_device": None,
                    "device_backend": None,
                    "device_fallback_reason": None,
                    "gpu_available": None,
                    "timings": {},
                    "inference_ms": 0.0,
                }
            sticker_detection = self._build_sticker_detection_payload(
                detections,
                skipped=False,
                backend=str(inference_payload.get("backend") or "unknown"),
                model_path=inference_payload.get("model_path"),
                meta_path=inference_payload.get("meta_path"),
                class_names=inference_payload.get("class_names") or [],
                fallback_reason=inference_payload.get("fallback_reason"),
                raw_detection_count=inference_payload.get("raw_detection_count"),
                allowed_labels_filter=inference_payload.get("allowed_labels_filter"),
                anchor=inference_payload.get("anchor"),
                ocr=inference_payload.get("ocr"),
                geometry=inference_payload.get("geometry"),
                stage_timings=stage_timings,
            )
            sticker_detection.update(
                {
                    "device_mode": inference_payload.get("device_mode"),
                    "effective_device": inference_payload.get("effective_device"),
                    "device_backend": inference_payload.get("device_backend"),
                    "device_fallback_reason": inference_payload.get("device_fallback_reason"),
                    "gpu_available": inference_payload.get("gpu_available"),
                }
            )
            # Cache valid inference result for hand-obstruction handling
            _detected_cls = sticker_detection.get("detected_class")
            if _detected_cls and not sticker_detection.get("skipped", False):
                state.last_valid_inference = {
                    "detected_class": _detected_cls,
                    "confidence": sticker_detection.get("confidence"),
                    "bbox": sticker_detection.get("bbox"),
                    "inference_payload": inference_payload,
                    "sticker_detection": sticker_detection,
                }
                state.last_valid_inference_ts = monotonic()
        else:
            # Part not ready — check if we have a fresh cached inference result
            # (e.g., hand obstructing during commit wait). Use it if within grace window.
            _cache_age_ms = (monotonic() - state.last_valid_inference_ts) * 1000.0 if state.last_valid_inference_ts > 0 else float("inf")
            if state.last_valid_inference is not None and _cache_age_ms < self._inference_cache_grace_ms:
                logger.debug(
                    "[inference] using cached result (age=%.0fms, class=%s) due to part_ready drop",
                    _cache_age_ms, state.last_valid_inference.get("detected_class"),
                )
                _cached_inf = state.last_valid_inference
                sticker_detection = self._build_sticker_detection_payload(
                    _cached_inf["inference_payload"].get("detections") or [],
                    skipped=False,
                    backend=str(_cached_inf["inference_payload"].get("backend") or "cached"),
                    model_path=_cached_inf["inference_payload"].get("model_path"),
                    meta_path=_cached_inf["inference_payload"].get("meta_path"),
                    class_names=_cached_inf["inference_payload"].get("class_names") or [],
                    fallback_reason=_cached_inf["inference_payload"].get("fallback_reason"),
                    raw_detection_count=_cached_inf["inference_payload"].get("raw_detection_count"),
                    allowed_labels_filter=_cached_inf["inference_payload"].get("allowed_labels_filter"),
                    anchor=_cached_inf["inference_payload"].get("anchor"),
                    ocr=_cached_inf["inference_payload"].get("ocr"),
                    geometry=_cached_inf["inference_payload"].get("geometry"),
                )
                sticker_detection["from_cache"] = True
                sticker_detection["cache_age_ms"] = round(_cache_age_ms, 1)
            else:
                _skip_reason = (
                "part_ready_settling"
                if _raw_part_ready and not part_ready_settled
                else str(clamp_payload.get("status") or "clamping")
                if _raw_part_ready and part_ready_settled and not clamp_ready_for_inference
                else str(phase_payload.get("status") or "operator_phase_delay")
                if _raw_part_ready and part_ready_settled and clamp_ready_for_inference and not phase_ready_for_inference
                else (part_ready.get("reject_reason_code") or "part_not_ready")
            )
            sticker_detection = self._build_sticker_detection_payload(
                [],
                skipped=True,
                reason=_skip_reason,
                backend="skipped",
                model_path=state.template.vision.model_path,
                meta_path=state.template.vision.model_meta_path,
                class_names=state.template.vision.classes,
            )
        timings["inference_ms"] = round(inference_ms, 2)

        # Component counter: assign detections to ROIs by coordinate
        if is_component_counter and state.template.component_rois:
            _fh = int(frame.shape[0]) if frame is not None else 0
            _fw = int(frame.shape[1]) if frame is not None else 0
            self._assign_detections_to_component_rois(
                detections, state.template.component_rois, _fw, _fh
            )

        validation_started = time.perf_counter()
        validation = self._validate_sticker(
            roi_frame=sticker_frame,
            state=state,
            detections=detections,
            detection_payload=sticker_detection,
            part_ready_payload=effective_part_ready,
            username=username,
            user_id=user_id,
            full_frame=frame,  # needed for defect mode (crop ROIs)
        )
        timings["validation_ms"] = _elapsed_ms(validation_started)
        validation_details = validation.get("validation_details") or {}
        if validation_details:
            sticker_detection["selected_candidate"] = validation_details.get("selected_candidate")
            sticker_detection["candidate_source"] = validation_details.get("candidate_source")
            sticker_detection["matching_candidate_count"] = validation_details.get("matching_candidate_count")

        event_state_started = time.perf_counter()

        # ── Inspection Policy Commit Gate ──
        # Determine whether this frame's validation result is allowed to commit.
        # - ACCEPT: commit only after stability threshold (consecutive frames + elapsed ms).
        # - REJECT with hard reason (OUT_OF_ANGLE, WRONG_TYPE): commit only after stability threshold.
        # - REJECT with non-hard reason (NOT_FOUND, gap, low conf, etc.): never auto-commit.
        #   These stay as pending so the system keeps inferring until ACCEPT (or COMMIT_TIMEOUT).
        _now_policy = datetime.now(UTC)
        _decision = str(validation.get("decision") or "").strip().upper()
        _reason = str(validation.get("reject_reason_code") or "").strip()
        _detected = str(validation.get("detected_class") or "").strip()
        _expected = str(validation.get("expected_class") or "").strip()
        _policy_key = f"{_decision}|{_reason}|{_detected}|{_expected}"

        _hard_reject_reasons = self._hard_reject_reasons  # set from config
        _is_accept = _decision == DecisionCode.ACCEPT.value
        _is_accept_candidate = (_decision == "ACCEPT_CANDIDATE")
        _is_hard_reject = (
            _decision == DecisionCode.REJECT.value
            and _reason in _hard_reject_reasons
        )
        _is_non_hard_reject = (
            _decision == DecisionCode.REJECT.value
            and _reason not in _hard_reject_reasons
        )

        # Track stability
        _was_accept = state.last_policy_key.split("|")[0] == "ACCEPT"

        # Start holdover window when transitioning ACCEPT → non-ACCEPT
        if _was_accept and not _is_accept and self._accept_holdover_ms > 0:
            if state.policy_holdover_expires_at is None:
                state.policy_holdover_expires_at = _now_policy + timedelta(
                    milliseconds=self._accept_holdover_ms
                )

        _in_holdover = (
            state.policy_holdover_expires_at is not None
            and _now_policy < state.policy_holdover_expires_at
            and not _is_accept
        )

        # Hard reject must always cancel holdover — otherwise the holdover
        # window would mask a genuine WRONG_TYPE/OUT_OF_ANGLE and let it
        # credit accept counters (false-accept risk).
        if _is_hard_reject and _in_holdover:
            state.policy_holdover_expires_at = None
            _in_holdover = False

        _effective_is_accept = _is_accept or _in_holdover
        _is_stabilizing = _is_accept_candidate  # all OK but not yet N-consecutive
        if _is_accept:                           # real detection came back
            state.policy_holdover_expires_at = None  # cancel holdover on re-detection

        if _is_non_hard_reject:
            # Non-hard reject is pure noise — do NOT touch any stability counters.
            # We must not increment policy_stable_frames (would falsely accumulate
            # toward hard_reject_stable_frames threshold) and must not reset
            # last_policy_key (would break an existing accept streak).
            pass
        elif _is_stabilizing:
            # Accept candidate (component counter: all OK but not yet N-consecutive).
            # Increment stability counters but don't allow commit yet.
            state.policy_stable_frames += 1
        elif _policy_key == state.last_policy_key:
            state.policy_stable_frames += 1
        elif _in_holdover:
            # During holdover: don't reset counters, treat gap as noise
            state.policy_stable_frames += 1
        else:
            state.last_policy_key = _policy_key
            state.policy_stable_frames = 1
            state.policy_stable_started_at = _now_policy
            state.policy_holdover_expires_at = None

        # ── Inference-generation-based accept gate ──
        # Only count a frame as a "new accept reading" when the underlying
        # inference result has actually changed (generation counter advanced).
        # This prevents the same cached result from being counted as multiple
        # stable frames on slow PCs where inference takes >500ms.
        if _is_non_hard_reject:
            # Non-hard reject is pure noise — do NOT touch any accept counters.
            # The system must keep inferring; non-hard reject should not break an
            # existing accept streak.
            pass
        elif _effective_is_accept or _is_stabilizing:
            if state.inference_result_generation > state.inference_last_counted_generation:
                # New inference result since last counted — record it
                state.inference_last_counted_generation = state.inference_result_generation
                _now_s = time.time()
                # Check if accept_window has expired since first accept reading.
                # Skip window check when accept_stable_ms=0 (no time threshold).
                if state.inference_accept_first_ts > 0 and self._accept_stable_ms > 0:
                    _accept_window_ms = (_now_s - state.inference_accept_first_ts) * 1000.0
                    if _accept_window_ms > self._accept_stable_ms * 3:
                        # Window expired — reset and start fresh
                        state.inference_accept_count = 1
                        state.inference_accept_first_ts = _now_s
                    else:
                        state.inference_accept_count += 1
                elif state.inference_accept_first_ts > 0:
                    # accept_stable_ms=0: count every generation, no time window gate
                    state.inference_accept_count += 1
                else:
                    state.inference_accept_count = 1
                    state.inference_accept_first_ts = _now_s
            # else: same generation — don't double-count
        else:
            # Known hard-reject reason (not non-hard, not accept) — reset counters
            # This breaks an existing accept streak (hard reject blocks commit).
            state.inference_accept_count = 0
            state.inference_accept_first_ts = 0.0
            state.inference_last_counted_generation = -1

        # ── Cycle-level grace timer ──
        # Tracks the first moment ACCEPT was seen in this clamping cycle.
        # Unlike policy_stable_started_at, this does NOT reset when holdover expires —
        # it persists across all detection gaps until the cycle resets.
        if _effective_is_accept:
            if state.accept_cycle_started_at is None:
                state.accept_cycle_started_at = _now_policy
        elif _is_non_hard_reject:
            # Non-hard reject is pure noise — do NOT reset cycle timer.
            # The grace period must keep running from the last real ACCEPT.
            pass
        elif not _is_accept and not _in_holdover:
            # True non-accept (holdover fully expired) — reset cycle timer
            state.accept_cycle_started_at = None

        _accept_cycle_elapsed_ms = 0.0
        if state.accept_cycle_started_at is not None:
            _accept_cycle_elapsed_ms = (
                (_now_policy - state.accept_cycle_started_at).total_seconds() * 1000.0
            )

        _stable_elapsed_ms = 0.0
        if state.policy_stable_started_at is not None:
            _stable_elapsed_ms = (
                (_now_policy - state.policy_stable_started_at).total_seconds() * 1000.0
            )

        # Determine commit_allowed with grace period + stability

        _commit_allowed = False
        _plc_fault = False
        _policy_action = "pending"
        _pending_reason = ""

        if _effective_is_accept:
            # Accept: commit only after grace period + stability
            # Both policy_stable_frames AND inference_accept_count must meet thresholds.
            # inference_accept_count only increments when a NEW inference result arrives,
            # so repeated reads of the same cached result don't count as extra stable frames.
            _grace_ok = _accept_cycle_elapsed_ms >= self._commit_grace_ms
            _frames_ok = state.policy_stable_frames >= self._accept_stable_frames
            _inference_ok = state.inference_accept_count >= self._accept_stable_frames
            _ms_ok = _stable_elapsed_ms >= self._accept_stable_ms
            if _grace_ok and _frames_ok and _inference_ok and _ms_ok:
                _commit_allowed = True
                _policy_action = "accept_commit"
            else:
                _policy_action = "pending"
                _parts = []
                if not _grace_ok:
                    _parts.append(f"grace({_accept_cycle_elapsed_ms:.0f}/{self._commit_grace_ms}ms)")
                if not _frames_ok:
                    _parts.append(f"stable_frames({state.policy_stable_frames}/{self._accept_stable_frames})")
                if not _inference_ok:
                    _parts.append(f"inference_count({state.inference_accept_count}/{self._accept_stable_frames})")
                if not _ms_ok:
                    _parts.append(f"stable_ms({_stable_elapsed_ms:.0f}/{self._accept_stable_ms}ms)")
                _pending_reason = f"accept_stabilizing({', '.join(_parts)})"

        elif _is_accept_candidate:
            # Component counter: all ROIs OK but not yet N-consecutive.
            # Keep scanning — don't commit, don't reject.
            _policy_action = "pending"
            _pending_reason = "component_stabilizing"

        elif _is_hard_reject:
            # Hard reject (OUT_OF_ANGLE, WRONG_TYPE): commit only after grace + higher stability
            # EXCEPTION: For sticker mode, do NOT auto-commit hard reject — wait for timeout reject instead
            if is_sticker:
                _policy_action = "pending"
                _pending_reason = "sticker_hard_reject_awaiting_timeout"
            else:
                _grace_ok = _stable_elapsed_ms >= self._commit_grace_ms
                _frames_ok = state.policy_stable_frames >= self._hard_reject_stable_frames
                _ms_ok = _stable_elapsed_ms >= self._hard_reject_stable_ms
                if _grace_ok and _frames_ok and _ms_ok:
                    _commit_allowed = True
                    _policy_action = "hard_reject_commit"
                else:
                    _policy_action = "pending"
                    _parts = []
                    if not _grace_ok:
                        _parts.append(f"grace({_stable_elapsed_ms:.0f}/{self._commit_grace_ms}ms)")
                    if not _frames_ok:
                        _parts.append(f"stable_frames({state.policy_stable_frames}/{self._hard_reject_stable_frames})")
                    if not _ms_ok:
                        _parts.append(f"stable_ms({_stable_elapsed_ms:.0f}/{self._hard_reject_stable_ms}ms)")
                    _pending_reason = f"hard_reject_stabilizing({', '.join(_parts)})"

        else:
            # Non-hard reject (NOT_FOUND, gap, low conf, etc.) — never auto-commit.
            # Keep inferring until ACCEPT (or COMMIT_TIMEOUT safety-net below).
            _policy_action = "pending"
            _pending_reason = f"non_hard_reject:{_reason}"

        # ── Timeout reject ──
        # Jika part sudah settled tapi tidak ada accept-commit dalam waktu reject_timeout_ms
        # For sticker mode: also allow timeout reject for hard rejects (OUT_OF_ANGLE, WRONG_TYPE)
        _allow_timeout_for_hard_reject = is_sticker and _is_hard_reject
        if (
            not _commit_allowed
            and (not _is_hard_reject or _allow_timeout_for_hard_reject)
            and state.part_ready_settled_at is not None
            and self._reject_timeout_ms > 0
        ):
            _settled_elapsed_ms = (datetime.now(UTC) - state.part_ready_settled_at).total_seconds() * 1000.0
            if _settled_elapsed_ms >= self._reject_timeout_ms:
                _is_hard_reject = True
                _reason = RejectReasonCode.COMMIT_TIMEOUT.value
                _decision = DecisionCode.REJECT.value
                _commit_allowed = True  # Allow commit untuk reject timeout
                _policy_action = "timeout_reject"
                validation["decision"] = DecisionCode.REJECT.value
                validation["reject_reason_code"] = RejectReasonCode.COMMIT_TIMEOUT.value

        # ── Item 1: Pre-commit PLC health gate ──
        # If PLC is enabled (not dry-run) and worker exists, require healthy link before commit.
        # This prevents persisting ACCEPT/REJECT while no relay can fire.
        if (
            _commit_allowed
            and self._plc_worker is not None
            and not self._plc_worker._dry_run
        ):
            if not self._plc_worker.is_healthy():
                _plc_fault = True
                _commit_allowed = False
                _policy_action = "plc_fault"
                _pending_reason = "plc_unhealthy_commit_blocked"

        # ── Item 2: below_min_confidence gate from part_ready ──
        # If part_ready reported that match_ratio is below min_match_ratio,
        # block commit even if inference says ACCEPT. This prevents committing
        # results when the part_ready confidence is too low.
        if _commit_allowed and part_ready.get("below_min_confidence"):
            _commit_allowed = False
            _policy_action = "low_confidence_pending"
            _pending_reason = "part_ready_below_min_confidence"

        # Build inspection_policy response
        inspection_policy = {
            "action": _policy_action,
            "commit_allowed": _commit_allowed,
            "hard_reject": _is_hard_reject,
            "plc_fault": _plc_fault,
            "pending_reason": _pending_reason,
            "stable_elapsed_ms": round(_stable_elapsed_ms, 1),
            "stable_frames": state.policy_stable_frames,
        }

        # Reset stability counters on commit
        if _commit_allowed:
            state.policy_stable_frames = 0
            state.last_policy_key = ""
            state.policy_stable_started_at = None
            state.policy_holdover_expires_at = None
            state.accept_cycle_started_at = None

        # Event state advance — policy gate is the sole commit authority.
        # Event machine provides dedup (anti-double-count) + event_id.
        event_state, event_id, count_committed = self._advance_event_state(
            state=state,
            validation=validation,
            part_ready_payload=effective_part_ready,
            presence=presence,
            now=_now_policy,
            commit_allowed=_commit_allowed,
        )

        # count_committed is already authoritative: only True when policy allows
        # AND event is fresh (not already committed / COOLDOWN).

        timings["event_state_ms"] = _elapsed_ms(event_state_started)

        persistence_started = time.perf_counter()
        db_write = {"written": False, "reason": "not_committed"}
        if count_committed:
            if self._phase_next_part_delay_ms > 0:
                state.manual_release_cooldown_until = max(
                    float(getattr(state, "manual_release_cooldown_until", 0.0) or 0.0),
                    time.time() + (self._phase_next_part_delay_ms / 1000.0),
                )
            # Full cycle reset — deterministic, does not depend on presence gap
            state.part_ready_latched = False
            state.part_ready_latched_at = None
            state.part_ready_unsettled_at = None
            state.part_ready_settled_at = None  # ← reset timeout timer for new cycle
            state.consecutive_part_ready_frames = 0
            state.plc_part_ready_triggered = False
            state.current_event_committed = False
            state.current_event_id = None
            state.current_event_key = None
            state.current_presence = False
            state.current_event_started_at = None
            state.current_event_stable_frames = 0
            state.cooldown_until = None
            state.policy_stable_frames = 0
            state.policy_stable_started_at = None
            state.accept_cycle_started_at = None
            state.inference_accept_count = 0
            state.inference_last_counted_generation = -1
            state.inference_accept_first_ts = 0.0
            # Reset ratio history — prevent stale ratios from contaminating next cycle
            state.part_ready_ratio_history.clear()
            state.part_ready_ema_ratio = -1.0
            # Reset adaptive HSV thresholds — start fresh for next cycle
            state.hsv_adaptive_lower = None
            state.hsv_adaptive_upper = None
            # Notify PLC worker of inspection decision
            # Only commit to PLC for accept or hard reject (not non-hard reject)
            decision = validation.get("decision", "")
            if (
                self._plc_worker is not None
                and decision in ("ACCEPT", "REJECT")
                and _commit_allowed
            ):
                try:
                    self._plc_worker.notify_decision(decision, event_id=event_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("[inspection] plc_worker.notify_decision failed: %s", exc)

            # ── COMMIT_TIMEOUT: PLC fires, but NO DB write ──
            # Timeout reject triggers NG buzzer + solenoid hold (via PLC),
            # but no inspection row is written — cycle resets locally.
            _is_timeout_reject = (
                _reason == RejectReasonCode.COMMIT_TIMEOUT.value
                and validation.get("decision") == DecisionCode.REJECT.value
            )
            if _is_timeout_reject:
                db_write = {"written": False, "reason": "timeout_reject_no_db"}
            else:
                # ── PLC commit interlock ──────────────────────────────────────
                # If PLC is enabled (not dry-run) and link is unhealthy, block
                # DB persist to prevent ACCEPT/REJECT rows with no relay action.
                if (
                    self._plc_worker is not None
                    and not self._plc_worker._dry_run
                    and not getattr(self._plc_worker, "_stopped", False)
                ):
                    if not self._plc_worker.is_healthy():
                        if _commit_allowed:
                            logger.warning(
                                "[inspection] PLC link unhealthy — blocking commit for session %s",
                                state.session_id[:8],
                            )
                        _commit_allowed = False
                        _plc_fault = True
                        _policy_action = "pending"
                        _pending_reason = "PLC_FAULT"

                db_write = self._maybe_persist(
                    validation,
                    state,
                    part_ready=part_ready,
                    part_ready_roi_meta=part_ready_roi_meta,
                    sticker_roi_meta=sticker_roi_meta,
                    sticker_detection=sticker_detection,
                    event_id=event_id,
                )
            # Item 1: register pending actuation for ACK/NACK reconciliation
            if db_write.get("written") and event_id:
                _result_id = db_write.get("result_id")
                self._pending_actuations[event_id] = {
                    "decision": validation.get("decision"),
                    "result_id": _result_id,
                    "status": "pending",
                }
            # Component-count REJECT → also log to daily JSONL
            _val_details = validation.get("validation_details") or {}
            if (
                _val_details.get("mode") == "component_count"
                and validation.get("decision") == "REJECT"
                and self._ng_logger is not None
            ):
                self._ng_logger.log_ng(
                    session_id=state.session_id,
                    event_id=event_id,
                    mode="component_count",
                    part_type=state.template.sticker.part_name,
                    reject_reason=validation.get("reject_reason_code"),
                    per_roi=_val_details.get("component_rois", []),
                )
            self._register_committed_result(
                state=state,
                validation=validation,
                part_ready_payload=part_ready,
                sticker_detection=sticker_detection,
                db_write=db_write,
                event_id=event_id,
                part_ready_roi_meta=part_ready_roi_meta,
                sticker_roi_meta=sticker_roi_meta,
                committed_at=datetime.now(UTC),
            )
        timings["persistence_ms"] = _elapsed_ms(persistence_started)

        normalized_response_mode = str(response_mode or "").strip().lower()
        # "stream": skip overlay compose and encode entirely — client renders locally.
        # "compact"/"minimal"/"overlay": encode overlay but skip heavy preview images.
        # (default) "full": encode everything.
        stream_response = normalized_response_mode in {"stream"}
        compact_response = stream_response or normalized_response_mode in {"compact", "minimal", "overlay"}

        overlay_started = time.perf_counter()
        overlay_image_b64 = ""
        if not stream_response:
            overlay = self._compose_overlay(
                full_frame=frame,
                part_ready_roi_meta=part_ready_roi_meta,
                sticker_roi_meta=sticker_roi_meta,
                detections=detections,
                validation=validation,
                part_ready=part_ready,
                event_state=event_state,
                state=state,
            )
            overlay_image_b64 = _encode_image(overlay)
            state.last_overlay_b64 = overlay_image_b64
        timings["overlay_compose_ms"] = _elapsed_ms(overlay_started)

        encode_started = time.perf_counter()
        preview_image_b64 = None
        part_ready_preview_image_b64 = None
        sticker_preview_image_b64 = None
        if not compact_response:
            preview_image_b64 = _encode_image(sticker_frame)
            part_ready_preview_image_b64 = _encode_image(part_ready_frame)
            sticker_preview_image_b64 = preview_image_b64
        timings["encode_ms"] = _elapsed_ms(encode_started)

        timings_payload = {
            **timings,
            "inference_skipped": not _effective_pr_ready,
            "compact_response": compact_response,
            "total_ms": _elapsed_ms(total_started),
        }

        payload = {
            "session": self._session_payload(state),
            "roi": sticker_roi_meta,
            "part_ready_roi_meta": part_ready_roi_meta,
            "sticker_roi_meta": sticker_roi_meta,
            "detections": detections,
            "sticker_detection": sticker_detection,
            "presence": presence,
            "part_ready": part_ready,
            "validation": validation,
            "event_state": event_state,
            "operator_state": state.operator_state,
            "clamp": clamp_payload,
            "phase": phase_payload,
            "event_id": event_id,
            "count_committed": count_committed,
            "count_source": "session" if count_committed else None,
            "counters": self._counter_payload(state),
            "last_committed_result": state.last_committed_result,
            "recent_events": list(state.recent_events),
            "db_write": db_write,
            "timings": timings_payload,
            "response_mode": "compact" if compact_response else "full",
            "overlay_image_b64": overlay_image_b64,
            "preview_image_b64": preview_image_b64,
            "part_ready_preview_image_b64": part_ready_preview_image_b64,
            "sticker_preview_image_b64": sticker_preview_image_b64,
            "inference_gate": inference_gate,
            "inspection_policy": inspection_policy,
        }
        if count_committed:
            payload["operator_state"] = "RESULT"
            self._operator_state_machine.mark_result(state, payload)
            payload["operator_state"] = state.operator_state
        state.latest_result = to_jsonable(payload)
        return payload

    def _require_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
        if not state:
            raise ValueError("Inspection session not found.")
        # Idle timeout: auto-end session if no frames received for too long
        if (hasattr(self, '_idle_timeout_s') and self._idle_timeout_s > 0):
            last = float(getattr(state, 'last_activity_at', 0.0) or 0.0)
            if last > 0 and (time.time() - last) > self._idle_timeout_s:
                logger.info(
                    "[inspection] session %s idle for %.0fs > timeout %.0fs — auto-ending",
                    session_id, time.time() - last, self._idle_timeout_s,
                )
                state.status = SessionStatus.STOPPED
                # Shutdown background inference executor to prevent thread leak
                if state._inference_executor is not None:
                    try:
                        state._inference_executor.shutdown(wait=False)
                    except Exception:  # noqa: BLE001
                        pass
                    state._inference_executor = None
                with self._lock:
                    self._sessions.pop(session_id, None)
                raise ValueError("Inspection session stopped: idle timeout reached.")
        return state

    def _merged_roi_payload(self, base: RoiGeometry, override: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "x": base.x,
            "y": base.y,
            "w": base.w,
            "h": base.h,
            "width": base.width,
            "height": None,
        }
        payload.update(override)
        return payload

    def _session_payload(self, state: SessionState) -> dict[str, Any]:
        part_ready_roi = self._merged_roi_payload(state.template.part_ready_roi, state.part_ready_roi_override)
        sticker_roi = self._merged_roi_payload(state.template.sticker_roi, state.sticker_roi_override)
        return {
            "session_id": state.session_id,
            "client_id": state.client_id,
            "camera_index": state.camera_index,
            "template_version_id": state.template.version_id,
            "line_id": state.line_id,
            "station_id": state.station_id,
            "status": state.status.value,
            "operator_state": state.operator_state,
            "template_name": state.template.name,
            "part_ready_roi": part_ready_roi,
            "sticker_roi": sticker_roi,
            "roi": sticker_roi,
        }

    @staticmethod
    def _roi_signature(roi_payload: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            round(float(roi_payload.get("x", 0.0) or 0.0), 6),
            round(float(roi_payload.get("y", 0.0) or 0.0), 6),
            round(float(roi_payload.get("w", 1.0) or 1.0), 6),
            round(float(roi_payload.get("h", 1.0) or 1.0), 6),
        )

    def _counter_payload(self, state: SessionState) -> dict[str, Any]:
        breakdown = _empty_reject_breakdown()
        breakdown.update(state.session_reject_breakdown)
        return {
            "scope": "session",
            "session_total": state.session_total,
            "session_accept": state.session_accept,
            "session_reject": state.session_reject,
            "session_reject_breakdown": breakdown,
        }

    def _crop_stage_roi(self, frame, base_roi: RoiGeometry, override: dict[str, Any]):
        roi = self._merged_roi_payload(base_roi, override)
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, int(float(roi.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(roi.get("y", 0.0)) * height)))
        roi_w = max(1, int(float(roi.get("w", 1.0)) * width))
        roi_h = max(1, int(float(roi.get("h", 1.0)) * height))
        x2 = min(width, x + roi_w)
        y2 = min(height, y + roi_h)
        cropped = frame[y:y2, x:x2]
        rotation = float(roi.get("rotation", 0.0) or 0.0)
        if abs(rotation) > 0.1:
            ch, cw = cropped.shape[:2]
            center = (cw / 2, ch / 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            cropped = cv2.warpAffine(
                cropped, M, (cw, ch),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        meta = {"x": x, "y": y, "width": x2 - x, "height": y2 - y}
        return cropped, meta

    def _on_inference_done(
        self, future: concurrent.futures.Future, state: SessionState
    ) -> None:
        """Callback when async inference completes. Thread-safe write to cache."""
        try:
            result = future.result()
            state.inference_result_cache = result
            state.inference_result_ts = monotonic()
            state.inference_result_generation += 1
        except Exception as exc:
            logger.warning("[inference-thread] callback error: %s", exc)
        finally:
            state.inference_thread_busy = False

    def _run_sticker_inference_sync(
        self, frame: np.ndarray, state: SessionState
    ) -> dict:
        """Wrapper for background thread — calls sync inference and returns serializable dict."""
        try:
            return self._run_sticker_inference(frame, state)
        except Exception as exc:
            logger.warning("[inference-thread] error: %s", exc)
            return {"detections": [], "timings": {}}

    def _run_sticker_inference(self, frame, state: SessionState) -> dict[str, Any]:
        return self._sticker_inference.predict(
            frame,
            state.template.vision,
            expected_class=state.template.sticker.expected_class,
            sticker_rule=state.template.sticker,
        )

    def _build_sticker_detection_payload(
        self,
        detections: list[dict[str, Any]],
        *,
        skipped: bool,
        reason: str | None = None,
        backend: str | None = None,
        model_path: str | None = None,
        meta_path: str | None = None,
        class_names: list[str] | None = None,
        fallback_reason: str | None = None,
        raw_detection_count: int | None = None,
        allowed_labels_filter: list[str] | None = None,
        anchor: dict[str, Any] | None = None,
        ocr: dict[str, Any] | None = None,
        geometry: dict[str, Any] | None = None,
        stage_timings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best = max(detections, key=lambda item: float(item.get("confidence") or 0.0), default=None)
        return {
            "status": "skipped" if skipped else "ok",
            "reason": reason,
            "backend": backend,
            "model_path": model_path,
            "meta_path": meta_path,
            "class_names": list(class_names or []),
            "fallback_reason": fallback_reason,
            "count": len(detections),
            "raw_detection_count": raw_detection_count,
            "allowed_labels_filter": allowed_labels_filter,
            "anchor": dict(anchor or {}),
            "ocr": dict(ocr or {}),
            "geometry": dict(geometry or {}),
            "stage_timings": dict(stage_timings or {}),
            "items": detections,
            "best": best,
        }

    def _detect_part_presence(self, frame) -> dict[str, Any]:
        if frame.size == 0:
            return {"present": False, "score": 0.0, "reason": "empty_roi", "area_ratio": 0.0}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
        area_ratio = float(largest_area / max(1, frame.shape[0] * frame.shape[1]))
        present = (
            area_ratio >= PRESENCE_MIN_AREA_RATIO
            or std_intensity >= PRESENCE_MIN_STD
            or mean_intensity >= PRESENCE_MIN_MEAN
        )
        reason = "texture" if std_intensity >= PRESENCE_MIN_STD else "brightness"
        score = min(1.0, max(area_ratio * 8.0, std_intensity / 32.0, mean_intensity / 255.0))
        return {
            "present": present,
            "score": round(score, 4),
            "reason": reason if present else "idle",
            "area_ratio": round(area_ratio, 6),
        }

    def _evaluate_part_ready(self, frame, state: SessionState) -> dict[str, Any]:
        config = state.template.part_ready
        if not config.enabled:
            return {
                "enabled": False,
                "part_ready": True,
                "part_ready_confidence": 1.0,
                "decision": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "status": "disabled",
                "match_ratio": None,
                "mean_distance": None,
                "color_profile_id": None,
                "gap_score": None,
            }
        method = str(getattr(config, "method", "gap_template_match") or "gap_template_match").strip().lower()
        if method == "mean_std_threshold":
            result = self._evaluate_part_ready_mean_std(frame, state, config)
        elif method == "gap_template_match":
            result = self._evaluate_part_ready_gap(frame, state, config)
        else:
            # Item 2: fail-closed on unsupported method — don't silently substitute gap
            logger.error(
                "[inspection] unsupported part_ready method '%s' for template %s — failing closed",
                method, state.template.id,
            )
            result = {
                "enabled": True,
                "part_ready": False,
                "part_ready_confidence": 0.0,
                "decision": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.ERROR.value,
                "status": f"UNSUPPORTED_PART_READY_METHOD:{method}",
                "match_ratio": None,
                "mean_distance": None,
                "color_profile_id": None,
                "gap_score": None,
            }

        # Universal min_match_ratio gate: apply floor confidence to ALL methods
        # If match_ratio is below min_match_ratio, mark it but DON'T change part_ready
        # (part_ready controls whether inference runs; we still want inference to run
        # so the operator sees results, but commit gate should check this flag)
        _min_conf = float(getattr(config, "min_match_ratio", 0.5) or 0.5)
        _match_ratio = result.get("match_ratio")
        if _match_ratio is not None and result.get("part_ready", False) and _match_ratio < _min_conf:
            result["below_min_confidence"] = True
            # Override status to signal low confidence but keep part_ready=True
            # so inference can still run. The commit gate will check this flag.
            result["part_ready_confidence"] = _match_ratio
            result["status"] = "below_min_confidence"

        return result

    def _evaluate_part_ready_gap(self, frame, state: SessionState, config) -> dict[str, Any]:
        """Gap detection via template matching against reference patch."""
        from backend.app.services.gap_detector import load_ref_patch, match_gap, get_ref_path

        ref_path = getattr(config, "gap_ref_path", None)
        threshold = float(getattr(config, "gap_match_threshold", 0.85) or 0.85)

        # Load reference patch (cached in state if available)
        cache_key = f"_gap_ref_{state.template.id}_{ref_path}"
        ref_patch = state.gap_ref_cache.get(cache_key)
        if ref_patch is None:
            ref_patch = load_ref_patch(ref_path, state.template.id)
            if ref_patch is not None:
                state.gap_ref_cache[cache_key] = ref_patch

        if ref_patch is None:
            return {
                "enabled": True,
                "part_ready": False,
                "part_ready_confidence": 0.0,
                "decision": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
                "status": "no_reference",
                "match_ratio": 0.0,
                "mean_distance": None,
                "color_profile_id": None,
                "gap_score": 0.0,
                "gap_method": "template_match",
            }

        # frame sudah di-crop ke part_ready_roi oleh _crop_stage_roi
        # — gunakan full dimensi frame sebagai area pencarian gap
        _fh, _fw = frame.shape[:2]
        roi = {"x": 0, "y": 0, "w": _fw, "h": _fh}

        result = match_gap(frame, roi, ref_patch, threshold)
        score = result["score"]
        ready = result["match"]

        return {
            "enabled": True,
            "part_ready": ready,
            "part_ready_confidence": score,
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": score,
            "mean_distance": None,
            "color_profile_id": None,
            "gap_score": score,
            "gap_method": "template_match",
            "gap_location": result.get("location", (0, 0)),
        }

    def _evaluate_part_ready_color(self, frame, state: SessionState, config) -> dict[str, Any]:
        """Legacy color profile match."""
        if not config.color_profile_id:
            return {
                "enabled": False,
                "part_ready": True,
                "part_ready_confidence": 1.0,
                "decision": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "status": "skipped",
                "match_ratio": None,
                "mean_distance": None,
                "color_profile_id": None,
                "gap_score": None,
            }
        record = self._profiles_repo.get(config.color_profile_id)
        if not record:
            return {
                "enabled": True,
                "part_ready": True,
                "part_ready_confidence": 1.0,
                "decision": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "status": "missing_profile_bypass",
                "match_ratio": 1.0,
                "mean_distance": None,
                "color_profile_id": config.color_profile_id,
                "gap_score": None,
            }
        evaluation = evaluate_color_profile_match(frame, config=config, profile=record["profile"])
        raw_ratio = float(evaluation["match_ratio"])
        # EMA smoothing — more responsive to current conditions than simple average
        _ema_alpha = max(0.0, min(1.0, float(getattr(config, "ema_alpha", 0.3) or 0.3)))
        if state.part_ready_ema_ratio < 0.0:
            # First reading in cycle — initialize EMA with raw value
            state.part_ready_ema_ratio = raw_ratio
        else:
            state.part_ready_ema_ratio = round(
                _ema_alpha * raw_ratio + (1.0 - _ema_alpha) * state.part_ready_ema_ratio, 6
            )
        smoothed_ratio = state.part_ready_ema_ratio
        resolved_min = float(evaluation["min_match_ratio"])
        ready = smoothed_ratio >= resolved_min
        return {
            **evaluation,
            "part_ready": ready,
            "part_ready_confidence": smoothed_ratio,
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": smoothed_ratio,
            "raw_match_ratio": raw_ratio,
            "min_match_ratio": resolved_min,
            "gap_score": None,
        }

    def _evaluate_part_ready_hsv(self, frame, state: SessionState, config) -> dict[str, Any]:
        """Legacy HSV black ratio with optional adaptive threshold."""
        # ── Adaptive HSV threshold ──
        # When enabled, slowly adjust hsv_lower/hsv_upper toward the current
        # frame's actual HSV distribution so the threshold tracks lighting drift.
        _hsv_adaptive = bool(getattr(config, "hsv_adaptive", False))
        _adaptive_alpha = max(0.0, min(1.0, float(getattr(config, "hsv_adaptive_alpha", 0.1) or 0.1)))
        _adaptive_min_ratio = float(getattr(config, "hsv_adaptive_min_ratio", 0.85) or 0.85)
        if _hsv_adaptive and frame is not None and getattr(frame, "size", 0) > 0:
            try:
                from backend.app.services.part_ready_detector import compute_hsv_reference_from_roi
                _live_ref = compute_hsv_reference_from_roi(frame)
                _live_lower = _live_ref["hsv_lower"]
                _live_upper = _live_ref["hsv_upper"]
                if state.hsv_adaptive_lower is None:
                    # First frame — initialize from config defaults
                    state.hsv_adaptive_lower = list(_hsv_bounds(getattr(config, "hsv_lower", None), (0, 0, 0)))
                    state.hsv_adaptive_upper = list(_hsv_bounds(getattr(config, "hsv_upper", None), (180, 255, 80)))
                else:
                    # EMA update toward live reference
                    state.hsv_adaptive_lower = [
                        round((1.0 - _adaptive_alpha) * state.hsv_adaptive_lower[i] + _adaptive_alpha * _live_lower[i], 1)
                        for i in range(3)
                    ]
                    state.hsv_adaptive_upper = [
                        round((1.0 - _adaptive_alpha) * state.hsv_adaptive_upper[i] + _adaptive_alpha * _live_upper[i], 1)
                        for i in range(3)
                    ]
                # Only apply adaptive thresholds when current ratio is above minimum
                # (prevents adapting to a wrong/no-part frame)
                _raw_for_check = evaluate_hsv_black_ratio(frame, config)
                if float(_raw_for_check["match_ratio"]) >= _adaptive_min_ratio:
                    # Use adaptive bounds directly for re-evaluation
                    import cv2 as _cv2
                    import numpy as _np
                    _hsv = _cv2.cvtColor(frame, _cv2.COLOR_BGR2HSV)
                    _lower = _np.array(state.hsv_adaptive_lower, dtype=_np.uint8)
                    _upper = _np.array(state.hsv_adaptive_upper, dtype=_np.uint8)
                    _mask = _cv2.inRange(_hsv, _lower, _upper)
                    _adaptive_ratio = float(_np.count_nonzero(_mask) / max(1, _mask.size))
                    evaluation = {
                        "enabled": True,
                        "method": "hsv_black_ratio",
                        "part_ready": _adaptive_ratio >= _adaptive_min_ratio,
                        "part_ready_confidence": round(_adaptive_ratio, 6),
                        "decision": DecisionCode.ACCEPT.value if _adaptive_ratio >= _adaptive_min_ratio else DecisionCode.REJECT.value,
                        "reject_reason_code": None if _adaptive_ratio >= _adaptive_min_ratio else RejectReasonCode.PART_NOT_READY.value,
                        "status": "ready" if _adaptive_ratio >= _adaptive_min_ratio else "not_ready",
                        "match_ratio": round(_adaptive_ratio, 6),
                        "raw_match_ratio": round(_adaptive_ratio, 6),
                        "mean_distance": None,
                        "distance_threshold": None,
                        "min_match_ratio": _adaptive_min_ratio,
                        "color_profile_id": getattr(config, "color_profile_id", None),
                        "colorspace": "HSV",
                        "hsv_lower": list(state.hsv_adaptive_lower),
                        "hsv_upper": list(state.hsv_adaptive_upper),
                        "hsv_adaptive": True,
                    }
                else:
                    evaluation = _raw_for_check
                    evaluation["hsv_adaptive"] = True
                    evaluation["hsv_lower"] = list(state.hsv_adaptive_lower)
                    evaluation["hsv_upper"] = list(state.hsv_adaptive_upper)
            except Exception:
                # Fallback to static evaluation on any error
                evaluation = evaluate_hsv_black_ratio(frame, config)
                evaluation["hsv_adaptive"] = False
        else:
            evaluation = evaluate_hsv_black_ratio(frame, config)
            evaluation["hsv_adaptive"] = False

        raw_ratio = float(evaluation["match_ratio"])
        # EMA smoothing — more responsive to current conditions than simple average
        _ema_alpha = max(0.0, min(1.0, float(getattr(config, "ema_alpha", 0.3) or 0.3)))
        if state.part_ready_ema_ratio < 0.0:
            state.part_ready_ema_ratio = raw_ratio
        else:
            state.part_ready_ema_ratio = round(
                _ema_alpha * raw_ratio + (1.0 - _ema_alpha) * state.part_ready_ema_ratio, 6
            )
        smoothed_ratio = state.part_ready_ema_ratio
        resolved_min = float(evaluation["min_match_ratio"])
        ready = smoothed_ratio >= resolved_min
        evaluation.update({
            "part_ready": ready,
            "part_ready_confidence": smoothed_ratio,
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": smoothed_ratio,
            "raw_match_ratio": raw_ratio,
        })
        return evaluation

    def _evaluate_part_ready_mean_std(self, frame, state: SessionState, config) -> dict[str, Any]:
        """"Mean + Std threshold classification.

        Classifies the ROI into empty / part_normal / sticker based on
        grayscale mean and standard deviation, with EMA smoothing.
        """
        from backend.app.services.part_ready_detector import evaluate_mean_std_threshold

        evaluation = evaluate_mean_std_threshold(frame, config)

        # EMA smoothing on the classification confidence (match_ratio)
        # Use sentinel -1.0 to distinguish "not yet initialized" from valid 0.0
        _ema_alpha = max(0.0, min(1.0, float(getattr(config, "ema_alpha", 0.3) or 0.3)))
        raw_ratio = float(evaluation["match_ratio"])
        if state.part_ready_ema_ratio < 0.0:
            state.part_ready_ema_ratio = raw_ratio
        else:
            state.part_ready_ema_ratio = round(
                _ema_alpha * raw_ratio + (1.0 - _ema_alpha) * state.part_ready_ema_ratio, 6
            )
        smoothed_ratio = state.part_ready_ema_ratio

        # Decision uses the same smoothed ratio as displayed
        # (raw_part_ready from classifier is kept as reference, but effective decision
        # for the gateway is based on smoothed confidence)
        _raw_ready = bool(evaluation["part_ready"])
        # Only consider ready if the smoothed match_ratio is above the method threshold
        # AND the raw classifier said ready (structural classification still required)
        ready = _raw_ready
        evaluation.update({
            "part_ready": ready,
            "part_ready_confidence": float(smoothed_ratio),
            "decision": DecisionCode.ACCEPT.value if ready else DecisionCode.REJECT.value,
            "reject_reason_code": None if ready else RejectReasonCode.PART_NOT_READY.value,
            "status": "ready" if ready else "not_ready",
            "match_ratio": smoothed_ratio,
            "raw_match_ratio": raw_ratio,
        })
        return evaluation

    def _normalize_label(self, value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _normalize_tilt_180(angle: float | None) -> float | None:
        if angle is None:
            return None
        normalized = float(angle)
        while normalized > 90.0:
            normalized -= 180.0
        while normalized < -90.0:
            normalized += 180.0
        if abs(normalized) == 0:
            normalized = 0.0
        return round(normalized, 2)

    @staticmethod
    def _normalize_code(value: Any) -> str:
        return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())

    @staticmethod
    def _ocr_validation_fields(detection_payload: dict[str, Any]) -> dict[str, Any]:
        return {}
        anchor = detection_payload.get("anchor") or {}
        geometry = detection_payload.get("geometry") or {}
        return {
            "text_bbox": anchor.get("text_bbox"),
            "dot_bbox": anchor.get("dot_bbox"),
            "dot_position": geometry.get("dot_position") or anchor.get("dot_position"),
            "anchor_offset": geometry.get("anchor_offset"),
            "pose_angle": geometry.get("pose_angle"),
        }

    def _validate_ocr_anchor(
        self,
        *,
        state: SessionState,
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
        line_id: str,
        thresholds: dict[str, Any],
        detection_context: dict[str, Any],
        max_tilt_degrees_value: float | None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        anchor = detection_payload.get("anchor") or {}
        ocr = detection_payload.get("ocr") or {}
        geometry = detection_payload.get("geometry") or {}

        anchor_min_confidence = (
            thresholds["min_roi_confidence"]
            if getattr(sticker, "anchor_min_confidence", None) is None
            else float(getattr(sticker, "anchor_min_confidence") or 0.0)
        )
        dot_min_confidence = (
            anchor_min_confidence
            if getattr(sticker, "dot_min_confidence", None) is None
            else float(getattr(sticker, "dot_min_confidence") or 0.0)
        )
        ocr_min_confidence = (
            self._default_ocr_min_confidence
            if getattr(sticker, "ocr_min_confidence", None) is None
            else float(getattr(sticker, "ocr_min_confidence") or 0.0)
        )
        offset_limit_x = (
            getattr(sticker, "max_anchor_offset_x", None)
            if getattr(sticker, "max_anchor_offset_x", None) is not None
            else sticker.max_offset_x
        )
        offset_limit_y = (
            getattr(sticker, "max_anchor_offset_y", None)
            if getattr(sticker, "max_anchor_offset_y", None) is not None
            else sticker.max_offset_y
        )
        offset_x = None
        offset_y = None
        anchor_offset = geometry.get("anchor_offset") or {}
        if anchor_offset:
            offset_x = float(anchor_offset.get("x", 0.0))
            offset_y = float(anchor_offset.get("y", 0.0))

        reject_reason = None
        if anchor.get("text_anchor") is None or anchor.get("center_dot") is None:
            reject_reason = RejectReasonCode.ANCHOR_NOT_FOUND.value
        elif float(anchor.get("text_confidence") or 0.0) < anchor_min_confidence:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif float(anchor.get("dot_confidence") or 0.0) < dot_min_confidence:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif str(ocr.get("status") or "") != "ok":
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif ocr.get("confidence") is not None and float(ocr.get("confidence") or 0.0) < ocr_min_confidence:
            reject_reason = RejectReasonCode.LOW_OCR_CONF.value
        elif not bool(ocr.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TEXT.value
        elif offset_x is None or offset_y is None:
            # Dot tidak terdeteksi — skip OUT_OF_POSITION (tidak ada dot)
            # Langsung cek OUT_OF_ANGLE jika enabled
            pass  # lanjut ke cek angle di bawah
        elif (
            thresholds["tilt_gate_enabled"]
            and max_tilt_degrees_value is not None
            and geometry.get("pose_deviation") is not None
            and float(geometry.get("pose_deviation") or 0.0) > max_tilt_degrees_value
        ):
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value

        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        detected_text = ocr.get("canonical_text") or ocr.get("text")
        selected_candidate = {
            "label": detected_text,
            "normalized_label": self._normalize_label(detected_text),
            "confidence": ocr.get("confidence"),
            "class_confidence": ocr.get("confidence"),
            "bbox": anchor.get("text_bbox"),
            "center": geometry.get("dot_position") or anchor.get("dot_position"),
            "offset": {"x": round(offset_x or 0.0, 2), "y": round(offset_y or 0.0, 2)} if offset_x is not None and offset_y is not None else None,
            "match_expected": bool(ocr.get("match_expected")),
            "source": "ocr_anchor",
        }
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": detected_text,
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": ocr.get("confidence"),
            "data2": anchor.get("dot_confidence"),
            "position": geometry.get("dot_position") or {},
            "offset": selected_candidate.get("offset") or {},
            "candidate_source": "ocr_anchor",
        }
        return {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": ocr.get("confidence"),
            "targets": [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": detected_text,
            "expected_class": sticker.expected_class,
            "sticker_confidence": ocr.get("confidence"),
            "sticker_bbox": anchor.get("text_bbox"),
            "sticker_backend": detection_context["backend"],
            "sticker_tilt_angle": geometry.get("pose_angle"),
            "sticker_tilt_expected": thresholds["expected_tilt_degrees"],
            "sticker_tilt_deviation": geometry.get("pose_deviation"),
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": "ocr_anchor",
                "selected_candidate": selected_candidate,
                "candidate_count": int(detection_payload.get("count") or 0),
                "matching_candidate_count": 1 if ocr.get("match_expected") else 0,
                "expected_center": geometry.get("expected_dot_position"),
                "tilt": {
                    "status": geometry.get("status"),
                    "angle_degrees": geometry.get("pose_angle"),
                    "expected_tilt_degrees": thresholds["expected_tilt_degrees"],
                    "deviation_degrees": geometry.get("pose_deviation"),
                    "source": "anchor_geometry",
                },
                "thresholds": {
                    **thresholds,
                    "anchor_min_confidence": anchor_min_confidence,
                    "dot_min_confidence": dot_min_confidence,
                    "ocr_min_confidence": ocr_min_confidence,
                    "max_anchor_offset_x": None if offset_limit_x is None else float(offset_limit_x),
                    "max_anchor_offset_y": None if offset_limit_y is None else float(offset_limit_y),
                },
                "ocr": ocr,
                "anchor": anchor,
                "geometry": geometry,
                "model": detection_context,
            },
        }

    def _validate_sticker_ocr_only(
        self,
        *,
        roi_frame,
        state: SessionState,
        detections: list[dict[str, Any]],
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
        line_id: str,
        thresholds: dict[str, Any],
        detection_context: dict[str, Any],
        max_tilt_degrees_value: float | None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        candidates, expected_center = self._build_validation_candidate_summaries(
            detections,
            roi_frame,
            sticker.expected_class,
            sticker,
        )
        selected_candidate, candidate_source = self._select_validation_candidate(candidates)
        matching_candidate_count = sum(1 for item in candidates if item.get("match_expected"))

        tilt_info = dict(detection_payload.get("tilt_info") or {})
        geometry = detection_payload.get("geometry") or {}
        if not tilt_info:
            tilt_info = {
                "status": geometry.get("status"),
                "angle_degrees": geometry.get("pose_angle"),
                "expected_tilt_degrees": thresholds["expected_tilt_degrees"],
                "deviation_degrees": geometry.get("pose_deviation"),
                "source": "sticker_only_geometry",
            }
        raw_angle = tilt_info.get("angle_degrees")
        normalized_angle = self._normalize_tilt_180(None if raw_angle is None else float(raw_angle))
        expected_tilt = float(thresholds["expected_tilt_degrees"] or 0.0)
        normalized_deviation = None
        if normalized_angle is not None:
            normalized_deviation = round(abs(float(normalized_angle) - expected_tilt), 2)
        tilt_info["normalized_angle_degrees"] = normalized_angle
        tilt_info["normalized_deviation_degrees"] = normalized_deviation

        anchor_offset = geometry.get("anchor_offset") or {}
        if not anchor_offset and selected_candidate is not None:
            anchor_offset = selected_candidate.get("offset") or {}
        offset_x = float(anchor_offset.get("x", 0.0)) if anchor_offset else 0.0
        offset_y = float(anchor_offset.get("y", 0.0)) if anchor_offset else 0.0
        offset_limit_x = (
            getattr(sticker, "max_anchor_offset_x", None)
            if getattr(sticker, "max_anchor_offset_x", None) is not None
            else thresholds["max_offset_x"]
        )
        offset_limit_y = (
            getattr(sticker, "max_anchor_offset_y", None)
            if getattr(sticker, "max_anchor_offset_y", None) is not None
            else thresholds["max_offset_y"]
        )

        ocr_payload = detection_payload.get("ocr") or {}
        unique_code = str(detection_payload.get("unique_code") or "").strip()
        reject_reason = None
        if selected_candidate is None:
            reject_reason = RejectReasonCode.NOT_FOUND.value
        elif float(selected_candidate.get("confidence") or 0.0) < thresholds["min_roi_confidence"]:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif not bool(selected_candidate.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TYPE.value
        elif thresholds["min_class_confidence"] is not None and float(selected_candidate.get("class_confidence") or 0.0) < float(thresholds["min_class_confidence"]):
            reject_reason = RejectReasonCode.LOW_CLASS_CONF.value
        elif thresholds["tilt_gate_enabled"] and max_tilt_degrees_value is not None and normalized_deviation is not None and normalized_deviation > max_tilt_degrees_value:
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif offset_limit_x is not None and abs(offset_x) > float(offset_limit_x):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif offset_limit_y is not None and abs(offset_y) > float(offset_limit_y):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value

        # Only allow hard reject reasons; suppress all others
        _hard_rejects = {
            RejectReasonCode.WRONG_TYPE.value,
            RejectReasonCode.OUT_OF_ANGLE.value,
            RejectReasonCode.COMMIT_TIMEOUT.value,
        }
        if reject_reason is not None and reject_reason not in _hard_rejects:
            reject_reason = None

        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        bbox = dict((selected_candidate or {}).get("bbox") or {}) or None
        selected_label = None if selected_candidate is None else selected_candidate.get("label")
        confidence = None if selected_candidate is None else selected_candidate.get("confidence")
        selected_center = dict((selected_candidate or {}).get("center") or {})
        selected_offset = {"x": round(offset_x, 2), "y": round(offset_y, 2)} if selected_candidate is not None else {}
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": selected_label,
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": confidence,
            "data2": None if selected_candidate is None else selected_candidate.get("class_confidence"),
            "position": selected_center,
            "offset": selected_offset,
            "candidate_source": candidate_source,
        }
        result = {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": confidence,
            "targets": [] if selected_candidate is None else [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": selected_label,
            "expected_class": sticker.expected_class,
            "unique_code": unique_code,
            "sticker_confidence": confidence,
            "sticker_bbox": bbox,
            "sticker_backend": detection_context["backend"],
            "sticker_tilt_angle": normalized_angle,
            "sticker_tilt_expected": expected_tilt,
            "sticker_tilt_deviation": normalized_deviation,
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": "sticker_only" if candidate_source == "expected_class" else candidate_source,
                "selected_candidate": selected_candidate,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "tilt": tilt_info,
                "thresholds": {
                    **thresholds,
                    "max_anchor_offset_x": None if offset_limit_x is None else float(offset_limit_x),
                    "max_anchor_offset_y": None if offset_limit_y is None else float(offset_limit_y),
                },
                "candidates": candidates,
                "ocr": ocr_payload,
                "anchor": detection_payload.get("anchor") or {},
                "geometry": geometry,
                "unique_code": unique_code,
                "model": detection_context,
            },
        }
        for key, value in self._ocr_validation_fields(detection_payload).items():
            result[key] = value
        return result

    def _build_validation_candidate_summaries(
        self,
        detections: list[dict[str, Any]],
        roi_frame,
        expected_class: str,
        sticker_config=None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        roi_h, roi_w = roi_frame.shape[:2]
        cx = float((sticker_config.expected_center_x if sticker_config and sticker_config.expected_center_x is not None else None) or 0.5)
        cy = float((sticker_config.expected_center_y if sticker_config and sticker_config.expected_center_y is not None else None) or 0.5)
        expected_center = {
            "x": round(cx * roi_w, 2),
            "y": round(cy * roi_h, 2),
        }
        expected_label = self._normalize_label(expected_class)
        candidates: list[dict[str, Any]] = []
        for index, det in enumerate(detections):
            pos = det.get("position") or {}
            center_x = (float(pos.get("x1", 0.0)) + float(pos.get("x2", 0.0))) / 2.0
            center_y = (float(pos.get("y1", 0.0)) + float(pos.get("y2", 0.0))) / 2.0
            offset_x = center_x - expected_center["x"]
            offset_y = center_y - expected_center["y"]
            candidates.append(
                {
                    "index": index,
                    "label": str(det.get("label") or ""),
                    "normalized_label": self._normalize_label(det.get("label")),
                    "confidence": round(float(det.get("confidence") or 0.0), 4),
                    "class_confidence": round(float(det.get("class_confidence") or 0.0), 4),
                    "bbox": _round_bbox(pos),
                    "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                    "offset": {"x": round(offset_x, 2), "y": round(offset_y, 2)},
                    "match_expected": self._normalize_label(det.get("label")) == expected_label,
                }
            )
        return candidates, expected_center

    def _select_validation_candidate(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str]:
        if not candidates:
            return None, "none"
        expected_matches = [item for item in candidates if item.get("match_expected")]
        if expected_matches:
            selected = max(
                expected_matches,
                key=lambda item: (
                    float(item.get("confidence") or 0.0),
                    float(item.get("class_confidence") or 0.0),
                ),
            )
            return selected, "expected_class"
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                float(item.get("class_confidence") or 0.0),
            ),
        )
        return selected, "highest_confidence"

    def _validate_sticker(
        self,
        *,
        roi_frame,
        state: SessionState,
        detections: list[dict[str, Any]],
        detection_payload: dict[str, Any],
        part_ready_payload: dict[str, Any],
        username: str | None,
        user_id: int | None,
        full_frame=None,
    ) -> dict[str, Any]:
        sticker = state.template.sticker
        validator_mode = str(getattr(sticker, "validator_mode", "ml_detection") or "ml_detection").strip().lower()

        # Map legacy validator_mode values to evaluator mode names
        _mode_map = {
            "ml_detection": "sticker",
            "component_count": "counter",
            "ml_roi_class": "sticker",
            "ml_roi_classification": "sticker",
            "roi_class": "sticker",
            "roi_partial": "sticker",
            "defect": "defect",
        }
        evaluator_mode = _mode_map.get(validator_mode, "sticker")

        # If a dedicated evaluator exists, use it instead of inline logic
        if evaluator_mode != "sticker":
            from backend.app.services.evaluators.registry import get_evaluator as _get_eval
            from backend.app.services.evaluators.base import EvalContext
            # Build mode-specific EvalContext
            if evaluator_mode == "counter":
                _criteria = state.template.criteria if state.template.criteria else {
                    "component_rois": [
                        {
                            "name": cr.name,
                            "classes": [
                                {
                                    "class_name": ct.class_name,
                                    "count": ct.count,
                                    "min_count": ct.min_count,
                                    "max_count": ct.max_count,
                                }
                                for ct in cr.classes
                            ],
                            "strict_foreign_class": cr.strict_foreign_class,
                        }
                        for cr in state.template.component_rois
                    ],
                }
                _frame = None
            elif evaluator_mode == "defect":
                _criteria = state.template.criteria if state.template.criteria else {"rois": []}
                # Frame required for defect (crop ROIs per camera geometry)
                _frame = full_frame
            else:
                _criteria = {}
                _frame = None

            _ctx = EvalContext(
                detections=detections,
                frame=_frame,
                criteria=_criteria,
                state=state,
                additional={
                    "accept_stable_frames": self._accept_stable_frames,
                },
            )
            try:
                _eval = _get_eval(evaluator_mode)
                _decision = _eval.evaluate(_ctx)
            except NotImplementedError:
                # If evaluator raises NotImplemented, fall through to old logic
                _decision = None

            if _decision is not None:
                # Convert Decision back to old dict format
                _is_accept = _decision.accept
                _reason = _decision.reason_code
                _details = _decision.details

                _decision_str = DecisionCode.ACCEPT.value if _is_accept else DecisionCode.REJECT.value
                result = {
                    "decision": _decision_str,
                    "decision_code": _decision_str,
                    "reject_reason_code": _reason if not _is_accept else None,
                    "validation_details": _details,
                }
                # Counter mode: early return — evaluator handles stability internally
                if _details.get("mode") == "counter":
                    return result

        position_gate_enabled = validator_mode not in ROI_CLASS_VALIDATOR_MODES
        line_id = state.line_id
        expected_tilt_degrees = float(getattr(sticker, "expected_tilt_degrees", 0.0) or 0.0)
        max_tilt_degrees = getattr(sticker, "max_tilt_degrees", None)
        max_tilt_degrees_value = None if max_tilt_degrees is None else float(max_tilt_degrees)
        tilt_gate_enabled = bool(getattr(sticker, "tilt_gate_enabled", False))
        tilt_info = _estimate_tilt_from_roi(roi_frame, expected_tilt_degrees, sticker)
        thresholds = {
            "min_roi_confidence": float(sticker.min_roi_confidence or 0.0),
            "min_class_confidence": (
                None if sticker.min_class_confidence is None else float(sticker.min_class_confidence)
            ),
            "max_offset_x": None if sticker.max_offset_x is None else float(sticker.max_offset_x),
            "max_offset_y": None if sticker.max_offset_y is None else float(sticker.max_offset_y),
            "validator_mode": validator_mode,
            "position_gate_enabled": position_gate_enabled,
            "tilt_gate_enabled": tilt_gate_enabled,
            "max_tilt_degrees": max_tilt_degrees_value,
            "expected_tilt_degrees": expected_tilt_degrees,
        }
        detection_context = {
            "backend": detection_payload.get("backend"),
            "model_path": detection_payload.get("model_path"),
            "meta_path": detection_payload.get("meta_path"),
            "class_names": list(detection_payload.get("class_names") or []),
            "fallback_reason": detection_payload.get("fallback_reason"),
        }
        if not sticker.enabled:
            return {
                "decision": DecisionCode.ACCEPT.value,
                "decision_code": DecisionCode.ACCEPT.value,
                "reject_reason_code": None,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "disabled",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(detections),
                    "matching_candidate_count": 0,
                    "expected_center": None,
                    "tilt": tilt_info,
                    "thresholds": thresholds,
                },
            }
#         if not part_ready_payload.get("part_ready", False):
#             return {
#                 "decision": DecisionCode.REJECT.value,
#                 "decision_code": DecisionCode.REJECT.value,
#                 "reject_reason_code": RejectReasonCode.PART_NOT_READY.value,
#                 "part_name": sticker.part_name,
#                 "line_id": line_id,
#                 "station_id": state.station_id,
#                 # Contract: data1 = part_ready confidence, data2 = sticker confidence
#                 "data1": part_ready_payload.get("part_ready_confidence"),
#                 "data2": None,
#                 "targets": [],
#                 "operator_user_id": user_id,
#                 "mp_check": username,
#                 "detected_class": None,
#                 "expected_class": sticker.expected_class,
#                 "sticker_confidence": None,
#                 "sticker_bbox": None,
#                 "sticker_backend": detection_context["backend"],
#                 "sticker_tilt_angle": tilt_info.get("angle_degrees"),
#                 "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
#                 "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
#                 "sticker_tilt_threshold": max_tilt_degrees_value,
#                 "validation_details": {
#                     "status": "part_not_ready",
#                     "candidate_source": "none",
#                     "selected_candidate": None,
#                     "candidate_count": len(detections),
#                     "matching_candidate_count": 0,
#                     "expected_center": None,
#                     "tilt": tilt_info,
#                     "thresholds": thresholds,
#                 },
#             }
        # NOTE: Do NOT return early on no detections — let flow continue to
        # _validate_sticker which handles selected_candidate=None as NOT_FOUND,
        # and the commit gate will keep it as pending (never auto-commit).
        # Returning here bypasses the commit gate and causes immediate REJECT.
        #
        # if not detections:
        #     return {
        #         "decision": DecisionCode.REJECT.value,
        #         "decision_code": DecisionCode.REJECT.value,
        #         "reject_reason_code": RejectReasonCode.NOT_FOUND.value,
        #         "part_name": sticker.part_name,
        #         "line_id": line_id,
        #         "station_id": state.station_id,
        #         "data1": part_ready_payload.get("part_ready_confidence"),
        #         "data2": None,
        #         "targets": [],
        #         "operator_user_id": user_id,
        #         "mp_check": username,
        #         "detected_class": None,
        #         "expected_class": sticker.expected_class,
        #         "sticker_confidence": None,
        #         "sticker_bbox": None,
        #         "sticker_backend": detection_context["backend"],
        #         "sticker_tilt_angle": tilt_info.get("angle_degrees"),
        #         "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
        #         "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
        #         "sticker_tilt_threshold": max_tilt_degrees_value,
        #         "validation_details": {
        #             "status": "inferring",
        #             "candidate_source": "none",
        #             "selected_candidate": None,
        #             "candidate_count": 0,
        #             "matching_candidate_count": 0,
        #             "expected_center": None,
        #             "tilt": tilt_info,
        #             "thresholds": thresholds,
        #         },
        #     }

        candidates, expected_center = self._build_validation_candidate_summaries(
            detections,
            roi_frame,
            sticker.expected_class,
            sticker,
        )
        selected_candidate, candidate_source = self._select_validation_candidate(candidates)
        matching_candidate_count = sum(1 for item in candidates if item.get("match_expected"))
        if selected_candidate is None:
            return {
                # Detections exist but none usable → not a final result.
                # Keep inferring (non-hard reject → pending) instead of ACCEPT.
                "decision": DecisionCode.REJECT.value,
                "decision_code": DecisionCode.REJECT.value,
                "reject_reason_code": RejectReasonCode.NOT_FOUND.value,
                "part_name": sticker.part_name,
                "line_id": line_id,
                "station_id": state.station_id,
                # Contract: data1 = part_ready confidence, data2 = sticker confidence
                "data1": part_ready_payload.get("part_ready_confidence"),
                "data2": None,
                "targets": [],
                "operator_user_id": user_id,
                "mp_check": username,
                "detected_class": None,
                "expected_class": sticker.expected_class,
                "sticker_confidence": None,
                "sticker_bbox": None,
                "sticker_backend": detection_context["backend"],
                "sticker_tilt_angle": tilt_info.get("angle_degrees"),
                "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
                "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
                "sticker_tilt_threshold": max_tilt_degrees_value,
                "validation_details": {
                    "status": "inferring",
                    "candidate_source": "none",
                    "selected_candidate": None,
                    "candidate_count": len(candidates),
                    "matching_candidate_count": matching_candidate_count,
                    "expected_center": expected_center,
                    "tilt": tilt_info,
                    "thresholds": thresholds,
                    "candidates": candidates,
                },
            }

        offset_x = float((selected_candidate.get("offset") or {}).get("x", 0.0))
        offset_y = float((selected_candidate.get("offset") or {}).get("y", 0.0))
        reject_reason = None
        if float(selected_candidate.get("confidence") or 0.0) < thresholds["min_roi_confidence"]:
            reject_reason = RejectReasonCode.LOW_ROI_CONF.value
        elif not bool(selected_candidate.get("match_expected")):
            reject_reason = RejectReasonCode.WRONG_TYPE.value
        elif thresholds["min_class_confidence"] is not None and float(selected_candidate.get("class_confidence") or 0.0) < float(thresholds["min_class_confidence"]):
            reject_reason = RejectReasonCode.LOW_CLASS_CONF.value
        elif tilt_gate_enabled and max_tilt_degrees_value is not None and tilt_info.get("angle_degrees") is not None and float(tilt_info.get("deviation_degrees") or 0.0) > max_tilt_degrees_value:
            reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif tilt_gate_enabled and max_tilt_degrees_value is not None:
            # Also check if text-band edge sticks out of ROI (even if angle is OK)
            roi_check = (tilt_info or {}).get("text_band_roi_check") or {}
            if roi_check and not roi_check.get("is_inside", True):
                reject_reason = RejectReasonCode.OUT_OF_ANGLE.value
        elif position_gate_enabled and thresholds["max_offset_x"] is not None and abs(offset_x) > float(thresholds["max_offset_x"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        elif position_gate_enabled and thresholds["max_offset_y"] is not None and abs(offset_y) > float(thresholds["max_offset_y"]):
            reject_reason = RejectReasonCode.OUT_OF_POSITION.value
        decision = DecisionCode.ACCEPT.value if reject_reason is None else DecisionCode.REJECT.value
        status = "accepted" if reject_reason is None else reject_reason.lower()
        target = {
            "target_id": "target-1",
            "part_name": sticker.part_name,
            "expected_class": sticker.expected_class,
            "detected_class": selected_candidate.get("label"),
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "data1": selected_candidate.get("confidence"),
            "data2": selected_candidate.get("class_confidence"),
            "position": dict(selected_candidate.get("center") or {}),
            "offset": {"x": round(offset_x, 2), "y": round(offset_y, 2)},
            "candidate_source": candidate_source,
        }
        return {
            "decision": decision,
            "decision_code": decision,
            "reject_reason_code": reject_reason,
            "part_name": sticker.part_name,
            "line_id": line_id,
            "station_id": state.station_id,
            # Contract: data1 = part_ready confidence, data2 = sticker confidence
            "data1": part_ready_payload.get("part_ready_confidence"),
            "data2": selected_candidate.get("confidence"),
            "targets": [target],
            "operator_user_id": user_id,
            "mp_check": username,
            "detected_class": selected_candidate.get("label"),
            "expected_class": sticker.expected_class,
            "sticker_confidence": selected_candidate.get("confidence"),
            "sticker_bbox": dict(selected_candidate.get("bbox") or {}) or None,
            "sticker_backend": detection_context["backend"],
            "sticker_tilt_angle": tilt_info.get("angle_degrees"),
            "sticker_tilt_expected": tilt_info.get("expected_tilt_degrees"),
            "sticker_tilt_deviation": tilt_info.get("deviation_degrees"),
            "sticker_tilt_threshold": max_tilt_degrees_value,
            "validation_details": {
                "status": status,
                "candidate_source": candidate_source,
                "selected_candidate": selected_candidate,
                "candidate_count": len(candidates),
                "matching_candidate_count": matching_candidate_count,
                "expected_center": expected_center,
                "tilt": tilt_info,
                "thresholds": thresholds,
                "candidates": candidates,
                "model": detection_context,
            },
        }

    def _advance_event_state(
        self,
        *,
        state: SessionState,
        validation: dict[str, Any],
        part_ready_payload: dict[str, Any],
        presence: dict[str, Any],
        now: datetime,
        commit_allowed: bool = False,
    ) -> tuple[str, str | None, bool]:
        if not presence.get("present", False):
            state.current_presence = False
            state.current_event_id = None
            state.current_event_key = None
            state.current_event_started_at = None
    def _advance_event_state(
        self,
        *,
        state: SessionState,
        validation: dict[str, Any],
        part_ready_payload: dict[str, Any],
        presence: dict[str, Any],
        now: datetime,
        commit_allowed: bool = False,
    ) -> tuple[str, str | None, bool]:
        if not presence.get("present", False):
            state.current_presence = False
            state.current_event_id = None
            state.current_event_key = ""
            state.current_event_started_at = None
            state.current_event_stable_frames = 0
            state.current_event_committed = False
            return InspectionEventState.IDLE.value, None, False

        _event_key = (
            f"{part_ready_payload.get('part_ready')}::"
            f"{validation.get('decision')}::"
            f"{validation.get('reject_reason_code') or 'OK'}::"
            f"{validation.get('part_name') or '-'}"
        )
        if _event_key == state.current_event_key and state.current_event_id is not None:
            # Same event continuing
            if state.current_event_committed:
                return InspectionEventState.COOLDOWN.value, state.current_event_id, False
            state.current_event_stable_frames += 1
        else:
            # Outcome changed (e.g. REJECT → ACCEPT after config fix) → new event.
            state.event_sequence += 1
            state.current_event_id = f"evt-{state.event_sequence:05d}"
            state.current_event_key = _event_key
            state.current_event_started_at = now
            state.current_event_stable_frames = 1
            state.current_event_committed = False

        if not part_ready_payload.get("part_ready", False):
            return InspectionEventState.PART_DETECTED.value, state.current_event_id, False

        # Commit authority: policy gate is the sole timing authority.
        # commit_allowed=True means grace + stable_frames + inference count all passed.
        commit_ready = commit_allowed

        if commit_ready:
            # Check consecutive reject threshold
            decision = str(validation.get("decision") or "").strip().upper()
            reject_reason = str(validation.get("reject_reason_code") or "").strip().upper()
            is_timeout_reject = reject_reason == RejectReasonCode.COMMIT_TIMEOUT.value
            if (
                decision == DecisionCode.REJECT.value
                and self._max_consecutive_rejects > 0
                and not is_timeout_reject
            ):
                # Increment consecutive reject counter
                state.consecutive_reject_count = int(getattr(state, "consecutive_reject_count", 0)) + 1
                if state.consecutive_reject_count < self._max_consecutive_rejects:
                    # Not enough consecutive rejects yet — don't commit, keep inferring
                    logger.info(
                        "[inspection] reject count %d/%d — delaying commit",
                        state.consecutive_reject_count, self._max_consecutive_rejects,
                    )
                    return InspectionEventState.DECISION_PENDING.value, state.current_event_id, False
                else:
                    # Reached threshold — commit reject and reset counter
                    state.consecutive_reject_count = 0
            elif decision == DecisionCode.ACCEPT.value or is_timeout_reject:
                # Accept and COMMIT_TIMEOUT always commit immediately — no debounce needed.
                # COMMIT_TIMEOUT is guaranteed valid (part settled for reject_timeout_ms without accept).
                state.consecutive_reject_count = 0

            state.current_event_committed = True
            state.cooldown_until = now + timedelta(milliseconds=COMMIT_COOLDOWN_MS)
            return InspectionEventState.DECISION_COMMITTED.value, state.current_event_id, True

        if state.current_event_stable_frames == 1:
            return InspectionEventState.PART_READY.value, state.current_event_id, False
        return InspectionEventState.DECISION_PENDING.value, state.current_event_id, False

    def _register_committed_result(
        self,
        *,
        state: SessionState,
        validation: dict[str, Any],
        part_ready_payload: dict[str, Any],
        sticker_detection: dict[str, Any],
        db_write: dict[str, Any],
        event_id: str | None,
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
        committed_at: datetime,
    ) -> None:
        self._increment_session_counters(state, validation)
        committed_payload = {
            "event_id": event_id,
            "committed_at": committed_at.isoformat(),
            "validation": dict(validation),
            "part_ready": dict(part_ready_payload),
            "sticker_detection": dict(sticker_detection),
            "part_ready_roi_meta": dict(part_ready_roi_meta),
            "sticker_roi_meta": dict(sticker_roi_meta),
            "db_write": dict(db_write),
            "count_source": "session",
        }
        state.last_committed_result = committed_payload
        state.recent_events.insert(
            0,
            {
                "event_id": event_id,
                "committed_at": committed_payload["committed_at"],
                "decision": validation.get("decision"),
                "reject_reason_code": validation.get("reject_reason_code"),
                "part_name": validation.get("part_name"),
                "line_id": validation.get("line_id"),
                "station_id": validation.get("station_id"),
                "db_written": bool(db_write.get("written")),
            },
        )
        del state.recent_events[MAX_RECENT_EVENTS:]

    def _increment_session_counters(self, state: SessionState, validation: dict[str, Any]) -> None:
        if validation.get("decision") == DecisionCode.ACCEPT.value:
            state.session_total += 1
            state.session_accept += 1
            return
        # REJECT: tracked in reject counters only, does not increment session_total
        state.session_reject += 1
        reject_reason = str(validation.get("reject_reason_code") or RejectReasonCode.ERROR.value)
        state.session_reject_breakdown.setdefault(reject_reason, 0)
        state.session_reject_breakdown[reject_reason] += 1

    def _compose_overlay(
        self,
        *,
        full_frame,
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
        detections: list[dict[str, Any]],
        validation: dict[str, Any],
        part_ready: dict[str, Any],
        event_state: str,
        state: SessionState,
    ):
        overlay = full_frame.copy()
        decision = validation.get("decision")
        reject_reason = validation.get("reject_reason_code") or "OK"
        decision_color = (0, 180, 0) if decision == DecisionCode.ACCEPT.value else (0, 0, 220)

        cv2.rectangle(
            overlay,
            (int(part_ready_roi_meta["x"]), int(part_ready_roi_meta["y"])),
            (
                int(part_ready_roi_meta["x"] + part_ready_roi_meta["width"]),
                int(part_ready_roi_meta["y"] + part_ready_roi_meta["height"]),
            ),
            (50, 180, 255),
            2,
        )
        cv2.putText(
            overlay,
            "PART READY ROI",
            (int(part_ready_roi_meta["x"]), max(16, int(part_ready_roi_meta["y"]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 180, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            overlay,
            (int(sticker_roi_meta["x"]), int(sticker_roi_meta["y"])),
            (
                int(sticker_roi_meta["x"] + sticker_roi_meta["width"]),
                int(sticker_roi_meta["y"] + sticker_roi_meta["height"]),
            ),
            (255, 200, 0),
            2,
        )
        cv2.putText(
            overlay,
            "STICKER ROI",
            (int(sticker_roi_meta["x"]), max(16, int(sticker_roi_meta["y"]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            f"{decision} / {reject_reason}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            decision_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            (
                f"part_ready={part_ready.get('part_ready')} "
                f"ratio={part_ready.get('match_ratio', part_ready.get('part_ready_confidence', '-'))}"
            ),
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"state={event_state} total={state.session_total} acc={state.session_accept} rej={state.session_reject}",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        anchor_offset = validation.get("anchor_offset") or {}
        if anchor_offset:
            cv2.putText(
                overlay,
                f"anchor dx={anchor_offset.get('x', '-')} dy={anchor_offset.get('y', '-')}",
                (12, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for det in detections:
            pos = det["position"]
            x1 = int(sticker_roi_meta["x"] + float(pos["x1"]))
            y1 = int(sticker_roi_meta["y"] + float(pos["y1"]))
            x2 = int(sticker_roi_meta["x"] + float(pos["x2"]))
            y2 = int(sticker_roi_meta["y"] + float(pos["y2"]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"{det['label']} {det['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        sticker = state.template.sticker
        cx_ratio = float(sticker.expected_center_x if sticker.expected_center_x is not None else 0.5)
        cy_ratio = float(sticker.expected_center_y if sticker.expected_center_y is not None else 0.5)
        exp_x = int(sticker_roi_meta["x"] + cx_ratio * sticker_roi_meta["width"])
        exp_y = int(sticker_roi_meta["y"] + cy_ratio * sticker_roi_meta["height"])
        arm = 18
        cv2.line(overlay, (exp_x - arm, exp_y), (exp_x + arm, exp_y), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, (exp_x, exp_y - arm), (exp_x, exp_y + arm), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (exp_x, exp_y), 5, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "EXP",
            (exp_x + arm + 4, exp_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def _maybe_persist(
        self,
        validation: dict[str, Any],
        state: SessionState,
        *,
        part_ready: dict[str, Any],
        part_ready_roi_meta: dict[str, Any],
        sticker_roi_meta: dict[str, Any],
        sticker_detection: dict[str, Any],
        event_id: str | None,
    ) -> dict[str, Any]:
        decision = str(validation.get("decision") or "").strip().upper()
        if decision != DecisionCode.ACCEPT.value:
            reject_log_written = False
            reject_entry = None
            if self._reject_log_repo is not None:
                try:
                    reject_entry = self._reject_log_repo.log_reject(
                        {
                            "session_id": state.session_id,
                            "event_id": event_id or state.current_event_id,
                            "template_version_id": state.template.version_id,
                            "line_id": validation.get("line_id") or state.line_id,
                            "station_id": validation.get("station_id") or state.station_id,
                            "part_name": validation.get("part_name"),
                            "decision_code": decision or DecisionCode.REJECT.value,
                            "reject_reason_code": validation.get("reject_reason_code") or RejectReasonCode.ERROR.value,
                            "operator_user_id": validation.get("operator_user_id"),
                            "mp_check": validation.get("mp_check"),
                            "validation_details": validation.get("validation_details"),
                            "part_ready": part_ready,
                            "sticker_detection": sticker_detection,
                            "part_ready_roi_meta": dict(part_ready_roi_meta),
                            "sticker_roi_meta": dict(sticker_roi_meta),
                        }
                    )
                    reject_log_written = True
                except Exception as exc:  # noqa: BLE001
                    logger.error("[inspection] reject log write failed: %s", exc, exc_info=True)
            return {
                "written": False,
                "reason": "reject_logged" if reject_log_written else "reject_log_error",
                "reject_log_written": reject_log_written,
                "reject_log_entry": reject_entry,
            }

        if not state.template.persistence.write_to_db:
            return {"written": False, "reason": "disabled"}
        persist_key = (
            f"{state.current_event_id or 'event'}:"
            f"{validation.get('decision')}:"
            f"{validation.get('reject_reason_code') or 'OK'}:"
            f"{validation.get('part_name')}"
        )
        if state.last_persisted_key == persist_key:
            return {"written": False, "reason": "duplicate_event"}
        record = self._results_repo.create_result(
            {
                "template_version_id": state.template.version_id,
                "line_id": validation.get("line_id"),
                "station_id": validation.get("station_id"),
                "part_name": validation.get("part_name"),
                "mp_check": validation.get("mp_check"),
                # data1/data2 mirror SQL contract: data1=part_ready confidence, data2=sticker confidence
                "data1": validation.get("data1"),
                "data2": validation.get("data2"),
                "decision": validation.get("decision"),
                "decision_code": validation.get("decision_code"),
                "reject_reason_code": validation.get("reject_reason_code"),
                "push_status": "pending",
                "actuation_status": "pending",  # Item 1: track PLC actuation ACK/NACK
                "retry_count": 0,
                "operator_user_id": validation.get("operator_user_id"),
                "part_ready_status": part_ready.get("status"),
                "part_ready_match_ratio": part_ready.get("match_ratio"),
                "part_ready_distance": part_ready.get("mean_distance"),
                "detected_class": validation.get("detected_class"),
                "expected_class": validation.get("expected_class"),
                "sticker_confidence": validation.get("sticker_confidence"),
                "sticker_bbox": validation.get("sticker_bbox"),
                "sticker_backend": validation.get("sticker_backend"),
                "text_bbox": validation.get("text_bbox"),
                "dot_bbox": validation.get("dot_bbox"),
                "dot_position": validation.get("dot_position"),
                "anchor_offset": validation.get("anchor_offset"),
                "pose_angle": validation.get("pose_angle"),
                "validation_details": validation.get("validation_details"),
                "part_ready_roi_meta": dict(part_ready_roi_meta),
                "sticker_roi_meta": dict(sticker_roi_meta),
                "targets": validation.get("targets") or [],
            }
        )
        state.last_persisted_at = datetime.now(UTC)
        state.last_persisted_key = persist_key
        return {"written": True, "result_id": record["id"]}


    # ---------------------------------------------------------------------------
    # Component Count Mode — helpers
    # ---------------------------------------------------------------------------

    def _point_in_which_roi(self, cx: float, cy: float, component_rois: list) -> int:
        """Return index of ROI containing normalized point (cx, cy), or -1.

        Supports axis-aligned ROIs only (rotation not yet supported in assign path).
        Point is in normalized [0,1] coordinates relative to full frame.
        """
        for idx, roi_rule in enumerate(component_rois):
            roi = roi_rule.roi
            rx, ry = float(roi.x), float(roi.y)
            rw, rh = float(roi.w), float(roi.h)
            if rw <= 0 or rh <= 0:
                continue
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                return idx
        return -1

    def _assign_detections_to_component_rois(
        self, detections: list[dict], component_rois: list, fw: int, fh: int
    ) -> None:
        """Assign tile_index to each detection based on center point falling inside ROI.

        Reads det["position"] = {x1, y1, x2, y2} (pixel coords from inference backend).
        Modifies detections in-place. Detections outside all ROIs get tile_index=-1.
        """
        if fw <= 0 or fh <= 0 or not component_rois:
            for det in detections:
                det["tile_index"] = -1
            return
        for det in detections:
            pos = det.get("position") or {}
            x1 = float(pos.get("x1", 0))
            y1 = float(pos.get("y1", 0))
            x2 = float(pos.get("x2", 0))
            y2 = float(pos.get("y2", 0))
            cx = ((x1 + x2) / 2.0) / float(fw)
            cy = ((y1 + y2) / 2.0) / float(fh)

