from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

from flask import Blueprint, Response, g, jsonify, request

from backend.app.core.container import audit_repo, inspection_results_repo, inspection_session_service, plc_worker, reject_log_repo
from backend.app.core.json_safety import safe_jsonify
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import DecisionCode, RejectReasonCode, UserRole


inspection_blueprint = Blueprint("inspection", __name__)


def _client_ip() -> str:
    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    return forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")


def _try_audit(event_type: str, **kwargs) -> None:
    try:
        audit_repo.log(event_type, **kwargs)
    except Exception:  # noqa: BLE001
        return


def _parse_optional_float(value: object, *, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


@inspection_blueprint.post("/inspection/sessions/start")
@require_auth
def start_session():
    payload = request.get_json(force=True) or {}
    try:
        session = inspection_session_service.start_session(
            client_id=str(payload.get("client_id") or "").strip() or str(g.current_user.id),
            camera_index=int(payload.get("camera_index") or 0),
            camera_rotation_degrees=float(payload.get("camera_rotation_degrees") or 0),
            template_version_id=int(payload.get("template_version_id") or 0),
            line_id=str(payload.get("line_id") or "").strip() or None,
            station_id=str(payload.get("station_id") or "").strip() or None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return safe_jsonify(session), 201


@inspection_blueprint.post("/inspection/sessions/<session_id>/frame")
@require_auth
def push_frame(session_id: str):
    payload = request.get_json(force=True) or {}
    try:
        result = inspection_session_service.process_frame(
            session_id,
            image_b64=str(payload.get("image_b64") or ""),
            response_mode=str(payload.get("response_mode") or "").strip() or None,
            username=g.current_user.username,
            user_id=g.current_user.id,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return safe_jsonify(result)


@inspection_blueprint.post("/inspection/sessions/<session_id>/roi")
@require_auth
def update_roi(session_id: str):
    payload = request.get_json(force=True) or {}
    try:
        result = inspection_session_service.update_roi(session_id, payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@inspection_blueprint.post("/inspection/sessions/<session_id>/stop")
@require_auth
def stop_session(session_id: str):
    try:
        result = inspection_session_service.stop_session(session_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@inspection_blueprint.post("/inspection/sessions/<session_id>/release")
@require_roles(UserRole.ADMIN, UserRole.OPERATOR)
def manual_release(session_id: str):
    """Operator NEUTRAL release: unclamp + reset cycle without committing a result."""
    try:
        result = inspection_session_service.manual_release(session_id, reason="manual_operator")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@inspection_blueprint.get("/inspection/reject-logs")
@require_roles(UserRole.ADMIN)
def list_reject_logs():
    try:
        limit = min(int(request.args.get("limit") or 100), 1000)
    except ValueError:
        limit = 100
    return jsonify(reject_log_repo.list_recent(limit=limit))


@inspection_blueprint.get("/inspection/latest-preview")
@require_auth
def latest_preview():
    result = inspection_session_service.get_latest_preview()
    if result is None:
        return jsonify({"error": "No active session with frames available."}), 404
    return safe_jsonify(result)


@inspection_blueprint.get("/inspections")
@require_auth
def list_inspections():
    try:
        template_version_id = (
            int(request.args["template_version_id"])
            if request.args.get("template_version_id") else None
        )
    except (ValueError, TypeError):
        return jsonify({"error": "template_version_id must be an integer"}), 400
    items = inspection_results_repo.list_results(
        line_id=request.args.get("line_id") or None,
        station_id=request.args.get("station_id") or None,
        part_name=request.args.get("part_name") or None,
        template_version_id=template_version_id,
        decision_code=request.args.get("decision_code") or None,
        push_status=request.args.get("push_status") or None,
        limit=min(int(request.args.get("limit") or 100), 1000),
        offset=int(request.args.get("offset") or 0),
    )
    return jsonify(items)


@inspection_blueprint.get("/inspections/<int:result_id>")
@require_auth
def get_inspection(result_id: int):
    item = inspection_results_repo.get_result(result_id)
    if item is None:
        return jsonify({"error": "Inspection result not found"}), 404
    return safe_jsonify(item)


@inspection_blueprint.patch("/inspections/<int:result_id>")
@require_roles(UserRole.ADMIN)
def patch_inspection(result_id: int):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    current = inspection_results_repo.get_result(result_id)
    if current is None:
        return jsonify({"error": "Inspection result not found"}), 404

    updates: dict = {}
    changed_fields: list[str] = []

    decision_raw = payload.get("decision_code", payload.get("decision"))
    decision_code: str | None = None
    if "decision_code" in payload or "decision" in payload:
        decision_candidate = str(decision_raw or "").strip().upper()
        try:
            decision_code = DecisionCode(decision_candidate).value
        except ValueError as exc:
            allowed = ", ".join(member.value for member in DecisionCode)
            return jsonify({"error": f"decision_code must be one of: {allowed}"}), 400
        updates["decision_code"] = decision_code
        updates["decision"] = decision_code
        changed_fields.extend(["decision_code", "decision"])

    if "reject_reason_code" in payload:
        reject_raw = payload.get("reject_reason_code")
        if reject_raw in (None, ""):
            updates["reject_reason_code"] = None
        else:
            reject_candidate = str(reject_raw or "").strip().upper()
            try:
                updates["reject_reason_code"] = RejectReasonCode(reject_candidate).value
            except ValueError:
                allowed = ", ".join(member.value for member in RejectReasonCode)
                return jsonify({"error": f"reject_reason_code must be one of: {allowed}"}), 400
        changed_fields.append("reject_reason_code")

    if decision_code == DecisionCode.ACCEPT.value and updates.get("reject_reason_code") not in (None, ""):
        return jsonify({"error": "reject_reason_code must be empty when decision_code is ACCEPT"}), 400

    if "part_ready_match_ratio" in payload:
        try:
            updates["part_ready_match_ratio"] = _parse_optional_float(
                payload.get("part_ready_match_ratio"),
                field_name="part_ready_match_ratio",
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        changed_fields.append("part_ready_match_ratio")

    if "sticker_confidence" in payload:
        try:
            updates["sticker_confidence"] = _parse_optional_float(
                payload.get("sticker_confidence"),
                field_name="sticker_confidence",
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        changed_fields.append("sticker_confidence")

    for field in ("detected_class", "expected_class"):
        if field in payload:
            updates[field] = str(payload.get(field) or "").strip() or None
            changed_fields.append(field)

    note = str(payload.get("note") or "").strip()
    if note:
        updates["correction_note"] = note

    if not updates:
        return jsonify({"error": "No supported fields to update"}), 400

    now = datetime.now(UTC).isoformat()
    corrections = list(current.get("corrections") or [])
    corrections.append(
        {
            "at": now,
            "by_user_id": g.current_user.id,
            "by_username": g.current_user.username,
            "fields": sorted(set(changed_fields)),
            "note": note or None,
        }
    )
    updates["corrections"] = corrections
    updates["corrected_at"] = now
    updates["corrected_by_user_id"] = g.current_user.id
    updates["corrected_by_username"] = g.current_user.username

    updated = inspection_results_repo.update_result(
        result_id,
        updates,
        requeue_mirror=True,
    )

    _try_audit(
        "inspection_corrected",
        user_id=updated.get("operator_user_id"),
        username=str(updated.get("mp_check") or "").strip() or None,
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=json.dumps(
            {
                "result_id": result_id,
                "fields": sorted(set(changed_fields)),
                "line_id": updated.get("line_id"),
                "station_id": updated.get("station_id"),
            },
            ensure_ascii=True,
        ),
    )
    return jsonify(updated)


@inspection_blueprint.delete("/inspections/<int:result_id>")
@require_roles(UserRole.ADMIN)
def delete_inspection(result_id: int):
    current = inspection_results_repo.get_result(result_id)
    if current is None:
        return jsonify({"error": "Inspection result not found"}), 404

    removed = inspection_results_repo.delete_result(result_id)
    _try_audit(
        "inspection_deleted",
        user_id=removed.get("operator_user_id"),
        username=str(removed.get("mp_check") or "").strip() or None,
        actor_id=g.current_user.id,
        actor_username=g.current_user.username,
        ip_address=_client_ip(),
        details=json.dumps(
            {
                "result_id": result_id,
                "line_id": removed.get("line_id"),
                "station_id": removed.get("station_id"),
                "decision_code": removed.get("decision_code"),
            },
            ensure_ascii=True,
        ),
    )
    return jsonify({"deleted": True, "id": result_id})


@inspection_blueprint.post("/inspections/<int:result_id>/retry-push")
@require_roles(UserRole.ADMIN)
def retry_inspection_push(result_id: int):
    try:
        item = inspection_results_repo.retry_result(result_id)
    except ValueError as exc:
        message = str(exc)
        status = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status
    return jsonify({"ok": item.get("push_status") == "sent", "result": item})


@inspection_blueprint.post("/inspections/retry-push")
@require_roles(UserRole.ADMIN)
def retry_failed_inspection_pushes():
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("result_ids") or []
    result_ids = []
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                result_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    limit = min(max(int(payload.get("limit") or 100), 1), 500)
    try:
        items = inspection_results_repo.retry_failed(result_ids=result_ids or None, limit=limit)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    succeeded = sum(1 for item in items if item.get("push_status") == "sent")
    return jsonify(
        {
            "attempted": len(items),
            "succeeded": succeeded,
            "failed": len(items) - succeeded,
            "items": items,
        }
    )


@inspection_blueprint.post("/inspections/push-worker/trigger")
@require_roles(UserRole.ADMIN)
def trigger_push_worker():
    """Manually trigger one push-worker batch. Returns batch statistics."""
    from backend.app.core.container import push_worker
    stats = push_worker.run_once()
    return jsonify({"ok": True, "stats": stats})


@inspection_blueprint.get("/inspections/push-worker/status")
@require_roles(UserRole.ADMIN)
def push_worker_status():
    """Return counts of results per push_status."""
    from backend.app.repositories.inspection_results_repository import InspectionResultsRepository
    local_repo = inspection_results_repo._local_repo  # noqa: SLF001
    if not isinstance(local_repo, InspectionResultsRepository):
        return jsonify({"error": "Status only available for local repo"}), 400
    all_items = local_repo.list_results(limit=100_000, offset=0)
    counts: dict[str, int] = {}
    for item in all_items:
        status = str(item.get("push_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return jsonify({"counts": counts, "total": len(all_items)})


@inspection_blueprint.get("/inspections/export")
@require_auth
def export_inspections():
    try:
        template_version_id = (
            int(request.args["template_version_id"])
            if request.args.get("template_version_id") else None
        )
    except (ValueError, TypeError):
        return jsonify({"error": "template_version_id must be an integer"}), 400
    items = inspection_results_repo.list_results(
        line_id=request.args.get("line_id") or None,
        station_id=request.args.get("station_id") or None,
        part_name=request.args.get("part_name") or None,
        template_version_id=template_version_id,
        decision_code=request.args.get("decision_code") or None,
        limit=10000,
        offset=0,
    )
    fields = [
        "id", "inspected_at", "line_id", "station_id", "part_name",
        "decision_code", "reject_reason_code",
        "part_ready_match_ratio", "sticker_confidence",
        "ocr_text", "ocr_confidence", "ocr_status", "anchor_offset", "pose_angle",
        "detected_class", "expected_class",
        "template_version_id", "push_status",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=inspections.csv"},
    )



@inspection_blueprint.get("/inspection/plc/status")
@require_roles(UserRole.ADMIN, UserRole.OPERATOR)
def plc_status():
    """Return the current PLC worker status (running, queue size, last command)."""
    if plc_worker is None:
        return jsonify({"enabled": False, "note": "PLC worker is disabled (QC_SUITE_PLC_ENABLED=0)"})
    return jsonify({"enabled": True, **plc_worker.status()})
