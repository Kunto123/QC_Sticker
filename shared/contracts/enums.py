from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"


class DecisionCode(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ERROR = "ERROR"


class RejectReasonCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    WRONG_TYPE = "WRONG_TYPE"
    # deprecated: WRONG_TEXT kept for backward compat with old DB data
    WRONG_TEXT = "WRONG_TEXT"
    LOW_ROI_CONF = "LOW_ROI_CONF"
    LOW_CLASS_CONF = "LOW_CLASS_CONF"
    # deprecated: LOW_OCR_CONF kept for backward compat with old DB data
    LOW_OCR_CONF = "LOW_OCR_CONF"
    OUT_OF_POSITION = "OUT_OF_POSITION"
    OUT_OF_ANGLE = "OUT_OF_ANGLE"
    ANCHOR_NOT_FOUND = "ANCHOR_NOT_FOUND"
    ANCHOR_MISMATCH = "ANCHOR_MISMATCH"
    PART_NOT_READY = "PART_NOT_READY"
    COMMIT_TIMEOUT = "COMMIT_TIMEOUT"
    ERROR = "ERROR"
    PLC_FAULT = "PLC_FAULT"
    # New codes for counter/defect modes
    COMPONENT_COUNT_MISMATCH = "COMPONENT_COUNT_MISMATCH"
    UNEXPECTED_COMPONENT = "UNEXPECTED_COMPONENT"
    NO_COMPONENT_ROIS = "NO_COMPONENT_ROIS"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    NO_DEFECT_ROIS = "NO_DEFECT_ROIS"


class SessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class InspectionEventState(StrEnum):
    IDLE = "idle"
    PART_DETECTED = "part_detected"
    PART_READY = "part_ready"
    DECISION_PENDING = "decision_pending"
    DECISION_COMMITTED = "decision_committed"
    COOLDOWN = "cooldown"
