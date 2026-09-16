from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.app.core.container import inspection_results_repo
from backend.app.core.http import require_auth


dashboard_blueprint = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard_blueprint.get("/summary")
@require_auth
def summary():
    return jsonify(
        inspection_results_repo.summary(
            line_id=request.args.get("line_id") or None,
            station_id=request.args.get("station_id") or None,
            part_name=request.args.get("part_name") or None,
            template_version_id=int(request.args["template_version_id"]) if request.args.get("template_version_id") else None,
            decision_code=request.args.get("decision_code") or None,
        )
    )


@dashboard_blueprint.get("/buckets")
@require_auth
def buckets():
    granularity = str(request.args.get("granularity") or "hour")
    if granularity not in {"minute", "hour", "day"}:
        granularity = "hour"
    return jsonify(
        inspection_results_repo.buckets(
            line_id=request.args.get("line_id") or None,
            station_id=request.args.get("station_id") or None,
            part_name=request.args.get("part_name") or None,
            template_version_id=int(request.args["template_version_id"]) if request.args.get("template_version_id") else None,
            decision_code=request.args.get("decision_code") or None,
            granularity=granularity,
            limit=min(int(request.args.get("limit") or 200), 1000),
        )
    )
