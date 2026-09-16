from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from shared.contracts.enums import SessionStatus
from shared.contracts.templates import InspectionTemplate


@dataclass(slots=True)
class SessionState:
    session_id: str
    client_id: str
    camera_index: int
    template: InspectionTemplate
    status: SessionStatus = SessionStatus.IDLE
    line_id: str | None = None
    station_id: str | None = None
    part_ready_roi_override: dict[str, Any] = field(default_factory=dict)
    sticker_roi_override: dict[str, Any] = field(default_factory=dict)
    latest_result: dict[str, Any] | None = None
    last_persisted_at: datetime | None = None
    last_persisted_key: str | None = None
    frame_index: int = 0
    current_presence: bool = False
    current_event_id: str | None = None
    current_event_key: str | None = None
    current_event_started_at: datetime | None = None
    current_event_stable_frames: int = 0
    current_event_committed: bool = False
    cooldown_until: datetime | None = None
    event_sequence: int = 0
    session_total: int = 0
    session_accept: int = 0
    session_reject: int = 0
    session_reject_breakdown: dict[str, int] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    last_committed_result: dict[str, Any] | None = None
    part_ready_ratio_history: list[float] = field(default_factory=list)
    part_ready_ema_ratio: float = -1.0
    # Adaptive HSV thresholds — updated at runtime when hsv_adaptive=True
    hsv_adaptive_lower: list[float] | None = None
    hsv_adaptive_upper: list[float] | None = None
    last_overlay_b64: str | None = None
    # Settle-time debounce: timestamp of the first frame where part_ready was True
    # in the current ready-run.  Reset to None whenever part_ready becomes False or
    # presence is lost.
    part_ready_settle_started_at: datetime | None = None
    # PLC constant-output mode: True once enqueue_part_ready() has been called for
    # the current ready-run, preventing duplicate triggers.  Reset when part leaves.
    plc_part_ready_triggered: bool = False
    plc_clamp_requested_at: float = 0.0
    plc_clamp_ready_at: float = 0.0
    plc_clamp_timeout: bool = False
    plc_clamp_event_id: str | None = None
    operator_sticker_delay_started_at: float = 0.0
    operator_sticker_ready_at: float = 0.0
    operator_state: str = "IDLE"
    inspection_has_run_for_current_part: bool = False
    inspection_result_cache: dict[str, Any] | None = None
    # Async inference state
    inference_result_cache: dict[str, Any] | None = None
    inference_result_ts: float = 0.0
    inference_frame_counter: int = 0
    inference_thread_busy: bool = False
    inference_submit_at: float = 0.0  # monotonic timestamp when last inference was submitted
    _inference_executor: concurrent.futures.ThreadPoolExecutor | None = None  # lazy-init per session
    # Inference generation counter — incremented each time a new inference result
    # is committed to the cache. Used by the accept gate to count distinct
    # inference runs (not just repeated reads of the same cached result).
    inference_result_generation: int = 0
    inference_accept_count: int = 0
    inference_accept_first_ts: float = 0.0
    inference_last_counted_generation: int = -1
    # Inference cache TTL (ms) — configurable per session from app config.
    inference_cache_ttl_ms: int = 10000
    gap_ref_cache: dict[str, Any] = field(default_factory=dict)  # cache for loaded gap reference patches
    part_removed_seen_at: datetime | None = None
    # Hysteresis counter: number of consecutive settled frames.
    # Reset to 0 when part_ready/presence is lost.
    settle_frame_count: int = 0
    # Timestamp when part first became settled (consecutive_part_ready_frames >= threshold).
    # Used for reject timeout: if no accept-commit within reject_timeout_ms, reject as COMMIT_TIMEOUT.
    # Reset to None when PLC returns to IDLE or when part leaves.
    part_ready_settled_at: datetime | None = None
    # Inference cooldown: timestamp (ms) of last inference run.
    # Prevents inference from running more than once per second.
    consecutive_part_ready_frames: int = 0
    last_inference_ms: int = 0
    # Inference interval (ms): minimum time between inference runs.
    # 0 = unlimited (every frame), 200 = max ~5 fps inference.
    inference_interval_ms: int = 0
    # Manual release COOLDOWN: timestamp (seconds since epoch) until which
    # re-clamp is blocked after IN1 (manual release). Prevents instant re-clamp.
    manual_release_cooldown_until: float = 0.0
    # Last activity timestamp (seconds since epoch) — updated on every frame process.
    # Used for idle timeout auto-end.
    last_activity_at: float = 0.0
    # Consecutive reject counter — incremented on each committed reject decision.
    # Reset to 0 on accept. Used to require N consecutive rejects before final reject.
    consecutive_reject_count: int = 0
    # Max consecutive rejects allowed before auto-commit reject (0 = immediate, no delay).
    max_consecutive_rejects: int = 0
    # ── Part-Ready Latch State ──
    # Once raw_part_ready is settled and clamp is requested, we latch so brief
    # drops in raw_part_ready (shadow, vibration) do not cancel the cycle.
    part_ready_latched: bool = False
    part_ready_latched_at: datetime | None = None
    # Timestamp when raw_part_ready last dropped while latched.
    # Used to debounce latch release via release_ms.
    part_ready_unsettled_at: datetime | None = None
    # ── Inspection Policy Stability Tracking ──
    # Policy key = hash of (decision, reject_reason_code, detected_class, expected_class)
    # Used to track consecutive stable frames before commit.
    # Cycle-level timestamp: when ACCEPT was first seen in this clamping cycle.
    # Persists across detection gaps (bridges holdover expiry) — reset only on commit
    # or when the cycle resets (part removed). Used for commit_grace_ms check.
    accept_cycle_started_at: datetime | None = None
    last_policy_key: str = ""
    policy_stable_started_at: datetime | None = None
    policy_stable_frames: int = 0
    # After ACCEPT commit, wait for part to actually leave before allowing next cycle.
    awaiting_part_removal_after_commit: bool = False
    policy_holdover_expires_at: datetime | None = None
    part_absent_started_at: datetime | None = None
    # ── Component Count Mode ──
    component_count_history: list = field(default_factory=list)
    consecutive_component_ok: int = 0
    # ── Logo Anti-Reclamp ──
    expected_logo_edge: object = None  # np.ndarray | None
    # ── Inference Result Cache for Hand-Obstruction Handling ──
    # When YOLO detects a valid class but part_ready drops temporarily (e.g., hand
    # obstructing during commit wait), we cache the last valid inference result and
    # timestamp. If part_ready returns within the grace window, we use the cached
    # result instead of re-running inference (which might fail due to obstruction).
    last_valid_inference: dict[str, Any] | None = None
    last_valid_inference_ts: float = 0.0  # monotonic timestamp
