from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from backend.app.core.container import deployments_repo, templates_repo
from pathlib import Path as _Path
import json as _json
from backend.app.core.http import require_auth, require_roles
from shared.contracts.enums import UserRole


deployment_blueprint = Blueprint("deployments", __name__, url_prefix="/deployments")


@deployment_blueprint.post("")
@require_roles(UserRole.ADMIN)
def deploy_template():
    payload = request.get_json(force=True) or {}
    template_id = int(payload.get("template_id") or 0)
    template_version_id = int(payload.get("template_version_id") or 0)
    if not template_id or not template_version_id:
        return jsonify({"error": "template_id dan template_version_id wajib diisi"}), 400
    template = templates_repo.get_template(template_id)
    version = templates_repo.get_version(template_version_id)
    if template is None or version is None:
        return jsonify({"error": "Template or version not found"}), 404
    # Fail-fast: validate model has meta JSON with class_names
    _model_path = (version.get("vision") or {}).get("model_path", "")
    if _model_path:
        _p = _Path(_model_path)
        _meta_candidates = [
            _p.parent / (_p.stem + ".meta.json"),
            _p.with_suffix(".json"),
        ]
        _meta_path = None
        for _c in _meta_candidates:
            if _c.exists():
                _meta_path = str(_c)
                break
        if _meta_path:
            try:
                _meta = _json.loads(_Path(_meta_path).read_text(encoding="utf-8"))
                if not _meta.get("class_names"):
                    return jsonify({"error": f"Meta JSON tidak memiliki class_names: {_meta_path}"}), 400
            except Exception as _exc:
                return jsonify({"error": f"Meta JSON parse error: {_exc}"}), 400
    record = deployments_repo.deploy(
        template_id=template_id,
        template_version_id=template_version_id,
        deployed_by=None,
        template_name=template["name"],
        version_number=int(version["version_number"]),
    )
    return jsonify(record), 201


@deployment_blueprint.get("")
@require_roles(UserRole.ADMIN)
def list_deployments():
    return jsonify(deployments_repo.list_deployments())


@deployment_blueprint.get("/active")
@require_auth
def get_active_deployment():
    record = deployments_repo.get_active()
    return jsonify({"deployment": record})


@deployment_blueprint.delete("/<int:deployment_id>")
@require_roles(UserRole.ADMIN)
def deactivate_deployment(deployment_id: int):
    ok = deployments_repo.deactivate(deployment_id)
    if not ok:
        return jsonify({"error": "Deployment not found"}), 404
    return jsonify({"id": deployment_id, "is_active": False})


@deployment_blueprint.put("/<int:deployment_id>")
@require_roles(UserRole.ADMIN)
def update_deployment(deployment_id: int):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    current = deployments_repo.get_deployment(deployment_id)
    if current is None:
        return jsonify({"error": "Deployment not found"}), 404
    if not bool(current.get("is_active")):
        return jsonify({"error": "Inactive deployment cannot be updated."}), 409

    updates: dict = {}

    if "template_version_id" in payload:
        try:
            template_version_id = int(payload.get("template_version_id") or 0)
        except (TypeError, ValueError):
            return jsonify({"error": "template_version_id must be an integer"}), 400
        if template_version_id <= 0:
            return jsonify({"error": "template_version_id must be a positive integer"}), 400

        version = templates_repo.get_version(template_version_id)
        if version is None:
            return jsonify({"error": "Template version not found"}), 404

        version_template_id = int((version.get("template") or {}).get("id") or 0)
        current_template_id = int(current.get("template_id") or 0)
        if version_template_id != current_template_id:
            return jsonify({"error": "Template version does not belong to deployment template"}), 400

        updates["template_version_id"] = template_version_id
        updates["template_name"] = str((version.get("template") or {}).get("name") or current.get("template_name") or "")
        updates["version_number"] = int(version.get("version_number") or current.get("version_number") or 0)

    if "template_name" in payload:
        updates["template_name"] = str(payload.get("template_name") or "").strip()

    actor = getattr(g, "current_user", None)
    if actor is not None:
        updates["deployed_by"] = actor.id

    if not updates:
        return jsonify({"error": "At least one mutable field must be provided"}), 400

    try:
        record = deployments_repo.update_deployment(deployment_id, **updates)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            return jsonify({"error": message}), 404
        if "inactive" in message.lower():
            return jsonify({"error": message}), 409
        return jsonify({"error": message}), 400
    return jsonify(record)


@deployment_blueprint.post("/<int:deployment_id>/rollback")
@require_roles(UserRole.ADMIN)
def rollback_deployment(deployment_id: int):
    actor = getattr(g, "current_user", None)
    try:
        record = deployments_repo.rollback(
            deployment_id,
            rolled_back_by=actor.id if actor else None,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(record), 201
