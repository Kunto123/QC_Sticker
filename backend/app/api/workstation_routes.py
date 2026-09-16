from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
import tempfile

from flask import Blueprint, g, jsonify, request, send_file

from backend.app.core.config import MODELS_DIR, PROJECT_ROOT
from backend.app.core.container import (
    app_config,
    augment_repo,
    dataset_versions_repo,
    datasets_repo,
    deployments_repo,
    models_repo,
    model_export_service,
    templates_repo,
    token_store,
    training_service,
    workstation_registry_repo,
)
from backend.app.core.model_catalog import list_base_models, resolve_base_model
from backend.app.core.http import require_auth, require_roles
from shared.contracts.augment import (
    GEOMETRIC_SAFE_TRANSFORMS,
    PHOTOMETRIC_TRANSFORMS,
    build_capabilities,
)
from shared.contracts.enums import UserRole


workstation_blueprint = Blueprint("workstation", __name__)


def _multipart_upload_files() -> list:
    files = request.files.getlist("files")
    if not files:
        files = request.files.getlist("files[]")
    if not files:
        files = request.files.getlist("file")
    return [item for item in files if getattr(item, "filename", "")]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _purge_model_files(model: dict) -> list[str]:
    removed: set[str] = set()
    allowed_root = MODELS_DIR.resolve()
    for field in ("path", "meta_path"):
        raw_value = str(model.get(field) or "").strip()
        if not raw_value:
            continue
        raw_path = Path(raw_value)
        candidate_paths = [raw_path] if raw_path.is_absolute() else [PROJECT_ROOT / raw_path, MODELS_DIR / raw_path, MODELS_DIR / raw_path.name]
        for candidate in candidate_paths:
            try:
                resolved = candidate.resolve()
            except Exception:  # noqa: BLE001
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            if resolved != allowed_root and allowed_root not in resolved.parents:
                continue
            try:
                resolved.unlink()
            except OSError:
                continue
            removed.add(str(resolved))
            break
    return sorted(removed)


@workstation_blueprint.get("/datasets")
@require_auth
def list_datasets():
    return jsonify(datasets_repo.list_datasets())


@workstation_blueprint.post("/datasets")
@require_roles(UserRole.ADMIN)
def create_dataset():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Dataset name is required"}), 400
    return jsonify(datasets_repo.create_dataset(name, str(payload.get("description") or ""))), 201


@workstation_blueprint.patch("/datasets/<dataset_id>")
@require_roles(UserRole.ADMIN)
def update_dataset(dataset_id: str):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    updates: dict = {}
    if "name" in payload:
        updates["name"] = payload.get("name")
    if "description" in payload:
        updates["description"] = payload.get("description")
    if not updates:
        return jsonify({"error": "At least one of name or description is required"}), 400

    try:
        record = datasets_repo.update_dataset(dataset_id, **updates)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(record)


@workstation_blueprint.delete("/datasets/<dataset_id>")
@require_roles(UserRole.ADMIN)
def delete_dataset(dataset_id: str):
    ok = datasets_repo.delete_dataset(dataset_id)
    if not ok:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({"deleted": True, "id": dataset_id})


@workstation_blueprint.get("/datasets/<dataset_id>/files")
@require_auth
def list_dataset_files(dataset_id: str):
    target = str(request.args.get("target") or "images")
    try:
        return jsonify(datasets_repo.list_files(dataset_id, target))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@workstation_blueprint.get("/datasets/<dataset_id>/files/<target>/<path:file_name>")
@require_auth
def download_dataset_file(dataset_id: str, target: str, file_name: str):
    allowed_targets = {"images", "labels", "exports"}
    if target not in allowed_targets:
        return jsonify({"error": f"Invalid dataset target '{target}'"}), 400
    file_path = datasets_repo.dataset_dir(dataset_id) / target / Path(file_name).name
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=False)


@workstation_blueprint.get("/datasets/<dataset_id>/versions")
@require_auth
def list_dataset_versions(dataset_id: str):
    if not datasets_repo.get_dataset(dataset_id):
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(dataset_versions_repo.list_versions(dataset_id))


def _validate_augment_eligibility(job: dict, dataset_id: str) -> str | None:
    """Return an actionable error string if the augment job is not eligible, else None.

    Photometric transforms are always allowed.
    Geometric-safe transforms are allowed only when ``geometric_augment_enabled=True``.
    Experimental transforms are never allowed.
    """
    if str(job.get("status") or "") != "completed":
        return (
            f"Augment job '{job.get('id')}' is not completed "
            f"(current status: '{job.get('status')}')."
        )
    if str(job.get("dataset_id") or "") != str(dataset_id):
        return (
            f"Augment job '{job.get('id')}' belongs to dataset '{job.get('dataset_id')}', "
            f"not '{dataset_id}'."
        )
    transforms = list(job.get("transforms") or [])
    geometric = [t for t in transforms if t in GEOMETRIC_SAFE_TRANSFORMS]
    experimental = [t for t in transforms if t not in PHOTOMETRIC_TRANSFORMS and t not in GEOMETRIC_SAFE_TRANSFORMS]

    if experimental:
        return (
            f"Augment job '{job.get('id')}' contains experimental transform(s) "
            f"({', '.join(sorted(experimental))}) which are not yet supported in version snapshots."
        )
    if geometric and not app_config.geometric_augment_enabled:
        return (
            f"Augment job '{job.get('id')}' contains geometric transform(s) "
            f"({', '.join(sorted(geometric))}) which require QC_SUITE_GEOMETRIC_AUGMENT_ENABLED=1. "
            "Use photometric-only transforms (brightness, contrast, blur, noise) "
            "or enable geometric augmentation in the server configuration."
        )
    return None


@workstation_blueprint.post("/datasets/<dataset_id>/versions")
@require_roles(UserRole.ADMIN)
def create_dataset_version(dataset_id: str):
    payload = request.get_json(force=True) or {}

    # Extract and validate optional augment job selection.
    raw_augment_ids = payload.pop("augment_job_ids", None)
    validated_augment_jobs: list[dict] = []

    if raw_augment_ids:
        if not isinstance(raw_augment_ids, list):
            return jsonify({"error": "augment_job_ids must be a list of job ID strings"}), 400
        for job_id_raw in raw_augment_ids:
            job_id = str(job_id_raw or "").strip()
            job = augment_repo.get_job(job_id)
            if job is None:
                return jsonify({"error": f"Augment job '{job_id}' not found."}), 404
            error = _validate_augment_eligibility(job, dataset_id)
            if error:
                return jsonify({"error": error}), 400
            validated_augment_jobs.append(job)

    try:
        return jsonify(
            dataset_versions_repo.create_version(
                dataset_id,
                payload,
                augment_jobs=validated_augment_jobs or None,
            )
        ), 201
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code


@workstation_blueprint.get("/datasets/<dataset_id>/versions/<version_id>")
@require_auth
def get_dataset_version(dataset_id: str, version_id: str):
    version = dataset_versions_repo.get_version(version_id)
    if version is None or str(version.get("dataset_id")) != str(dataset_id):
        return jsonify({"error": "Dataset version not found"}), 404
    return jsonify(version)


@workstation_blueprint.put("/datasets/<dataset_id>/versions/<version_id>")
@require_roles(UserRole.ADMIN)
def update_dataset_version(dataset_id: str, version_id: str):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    updates: dict = {}
    for field in ("name", "description", "status"):
        if field in payload:
            updates[field] = payload.get(field)
    if not updates:
        return jsonify({"error": "Only metadata fields can be updated: name, description, status"}), 400

    try:
        version = dataset_versions_repo.update_version(dataset_id, version_id, **updates)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(version)


@workstation_blueprint.post("/datasets/<dataset_id>/versions/<version_id>/export")
@require_roles(UserRole.ADMIN)
def export_dataset_version(dataset_id: str, version_id: str):
    try:
        return jsonify(dataset_versions_repo.export_version(dataset_id, version_id))
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code


@workstation_blueprint.post("/datasets/<dataset_id>/upload")
@require_roles(UserRole.ADMIN)
def upload_dataset_file(dataset_id: str):
    if request.files:
        target = str(request.form.get("target") or "images").strip()
        uploads = _multipart_upload_files()
        if not uploads:
            return jsonify({"error": "At least one file is required"}), 400
        batch: list[tuple[str, bytes]] = []
        for file_storage in uploads:
            file_name = str(file_storage.filename or "").strip()
            if not file_name:
                continue
            batch.append((file_name, file_storage.read()))
        if not batch:
            return jsonify({"error": "At least one file is required"}), 400
        try:
            saved = datasets_repo.save_files(dataset_id, target, batch)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"target": target, "count": len(saved), "items": saved}), 201

    payload = request.get_json(force=True) or {}
    file_name = str(payload.get("file_name") or "").strip()
    target = str(payload.get("target") or "images").strip()
    content_b64 = str(payload.get("content_b64") or "").strip()
    if not file_name or not content_b64:
        return jsonify({"error": "file_name and content_b64 are required"}), 400
    try:
        content = base64.b64decode(content_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid base64 content: {exc}"}), 400
    try:
        item = datasets_repo.save_file(dataset_id, target, file_name, content)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"target": target, "count": 1, "items": [item]}), 201


@workstation_blueprint.get("/datasets/<dataset_id>/annotations/<image_name>")
@require_auth
def get_annotation(dataset_id: str, image_name: str):
    return jsonify(datasets_repo.get_annotation(dataset_id, image_name))


@workstation_blueprint.post("/datasets/<dataset_id>/annotations/<image_name>")
@require_roles(UserRole.ADMIN)
def save_annotation(dataset_id: str, image_name: str):
    payload = request.get_json(force=True)
    if payload is None:
        payload = {}
    if isinstance(payload, list):
        labels = payload
    elif isinstance(payload, dict):
        labels = payload.get("labels")
        if labels is None:
            labels = payload.get("annotations")
    else:
        labels = None
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400
    return jsonify(datasets_repo.save_annotation(dataset_id, image_name, labels))


@workstation_blueprint.get("/augment/capabilities")
@require_auth
def get_augment_capabilities():
    """Return the transform catalog and current feature-flag state.

    The UI uses this to know which transforms are available, which need warnings,
    and whether geometric augmentation is enabled on this server.
    """
    return jsonify(build_capabilities(geometric_augment_enabled=app_config.geometric_augment_enabled))


@workstation_blueprint.get("/augment/jobs")
@require_auth
def list_augment_jobs():
    return jsonify(augment_repo.list_jobs())


@workstation_blueprint.get("/augment/jobs/<job_id>")
@require_auth
def get_augment_job(job_id: str):
    job = augment_repo.get_job(job_id)
    if job is None:
        return jsonify({"error": "Augment job not found"}), 404
    return jsonify(job)


@workstation_blueprint.post("/augment/jobs")
@require_roles(UserRole.ADMIN)
def create_augment_job():
    from shared.contracts.augment import TRANSFORM_CATALOG as _TC, EXPERIMENTAL_TRANSFORMS  # lazy import

    payload = request.get_json(force=True) or {}
    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not datasets_repo.get_dataset(dataset_id):
        return jsonify({"error": "Dataset not found"}), 404

    raw_transforms = payload.get("transforms")
    transforms: list[str] = list(raw_transforms) if isinstance(raw_transforms, list) else ["brightness", "blur"]
    multiplier = max(1, min(10, int(payload.get("multiplier") or 2)))

    # Reject unknown transform names.
    unknown = [t for t in transforms if t not in _TC]
    if unknown:
        known = ", ".join(sorted(_TC.keys()))
        return jsonify({
            "error": f"Unknown transform(s): {', '.join(sorted(unknown))}. Known: {known}."
        }), 400

    # Warn that experimental transforms will not produce correct labels.
    experimental_used = [t for t in transforms if t in EXPERIMENTAL_TRANSFORMS]
    if experimental_used:
        return jsonify({
            "error": (
                f"Experimental transform(s) ({', '.join(sorted(experimental_used))}) are not yet "
                "supported. Use photometric or geometric_safe transforms."
            )
        }), 400

    job = augment_repo.create_job(
        dataset_id,
        transforms=transforms,
        multiplier=multiplier,
        params=dict(payload),
    )
    return jsonify(job), 201


@workstation_blueprint.post("/augment/jobs/<job_id>/cancel")
@require_roles(UserRole.ADMIN)
def cancel_augment_job(job_id: str):
    try:
        result = augment_repo.cancel_job(job_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@workstation_blueprint.delete("/augment/jobs/<job_id>")
@require_roles(UserRole.ADMIN)
def delete_augment_job(job_id: str):
    try:
        removed = augment_repo.delete_job(job_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify({"deleted": True, "id": job_id, "job": removed})


@workstation_blueprint.get("/train/jobs")
@require_auth
def list_training_jobs():
    return jsonify(training_service.list_jobs())


@workstation_blueprint.get("/train/base-models")
@require_auth
def list_training_base_models():
    family = request.args.get("family")
    try:
        return jsonify(list_base_models(family))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@workstation_blueprint.post("/train/jobs")
@require_roles(UserRole.ADMIN)
def create_training_job():
    payload = request.get_json(force=True) or {}
    dataset_id = str(payload.get("dataset_id") or "").strip()
    base_model = str(payload.get("base_model") or "baseline").strip() or "baseline"
    base_model_family = str(payload.get("base_model_family") or "").strip() or None
    base_model_variant = str(payload.get("base_model_variant") or "").strip() or None
    device_mode = str(payload.get("device_mode") or payload.get("device") or "auto").strip().lower() or "auto"
    dataset_version_id = str(payload.get("dataset_version_id") or "").strip() or None
    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400
    if not datasets_repo.get_dataset(dataset_id):
        return jsonify({"error": "Dataset not found"}), 404
    if device_mode not in {"auto", "gpu", "cpu"}:
        return jsonify({"error": "device_mode must be one of: auto, gpu, cpu"}), 400

    for key, min_value, max_value in (("epochs", 1, 1000), ("imgsz", 64, 2048), ("batch", 1, 256), ("patience", 1, 500), ("workers", 0, 32)):
        if payload.get(key) in (None, ""):
            continue
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            return jsonify({"error": f"{key} must be an integer"}), 400
        if value < min_value or value > max_value:
            return jsonify({"error": f"{key} must be between {min_value} and {max_value}"}), 400
        payload[key] = value

    if "cache" in payload and not isinstance(payload.get("cache"), bool):
        payload["cache"] = str(payload.get("cache") or "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        base_model_spec = resolve_base_model(base_model, family=base_model_family, variant=base_model_variant)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if dataset_version_id:
        dataset_version = dataset_versions_repo.get_version(dataset_version_id)
        if dataset_version is None or str(dataset_version.get("dataset_id")) != str(dataset_id):
            return jsonify({"error": "Dataset version not found"}), 404
        export_root = Path(str(dataset_version.get("export_root") or "").strip())
        data_yaml_path = export_root / "data.yaml"
        if not export_root.exists() or not export_root.is_dir() or not data_yaml_path.exists():
            return jsonify({"error": "Dataset version export is not ready. Re-export the selected version first."}), 400
        payload["dataset_version_id"] = dataset_version["id"]
        payload["dataset_version_number"] = dataset_version.get("version_number")
        payload["dataset_version_name"] = dataset_version.get("name")
        payload["dataset_version_display_label"] = dataset_version.get("display_label")
        payload["dataset_version_status"] = dataset_version.get("status")
        payload["dataset_version_export_format"] = dataset_version.get("export_format")
        payload["dataset_version_export_root"] = dataset_version.get("export_root")
        payload["dataset_version_manifest_path"] = dataset_version.get("manifest_path")
        payload["dataset_version_split_ratios"] = dataset_version.get("split_ratios")
        payload["class_names"] = list(dataset_version.get("class_names") or [])
    if base_model_spec is not None:
        resolved_base_model = base_model_spec["id"]
        payload["base_model"] = base_model_spec["id"]
        payload["base_model_catalog_id"] = base_model_spec["id"]
        payload["base_model_family"] = base_model_spec["family"]
        payload["base_model_variant"] = base_model_spec["variant"]
        payload["base_model_display_name"] = base_model_spec["display_name"]
        payload["base_model_weights_name"] = base_model_spec["weights_name"]
        payload["base_model_runtime"] = base_model_spec["runtime"]
        payload["base_model_task"] = base_model_spec["task"]
        payload["base_model_source"] = base_model_spec["source"]
        payload["base_model_spec"] = base_model_spec
        # Offline-strict preflight: fail fast before queuing when download is disabled
        # and the weights file is not available locally.
        if not app_config.training_weights_download_allowed:
            weights_name = base_model_spec["weights_name"]
            local_weights = MODELS_DIR / weights_name
            if not local_weights.exists() or not local_weights.is_file():
                return jsonify({
                    "error": (
                        f"Weights file '{weights_name}' is not available locally "
                        f"({local_weights}) and download is disabled "
                        "(QC_SUITE_TRAINING_WEIGHTS_DOWNLOAD_ALLOWED=0). "
                        "Copy the weights file to MODELS_DIR or enable download."
                    )
                }), 400
    else:
        if base_model_family or base_model_variant:
            return jsonify({"error": "Unsupported base model family/variant"}), 400
        resolved_base_model = base_model
        payload["base_model"] = base_model
    payload["device_mode"] = device_mode
    try:
        job = training_service.create_job(dataset_id, resolved_base_model, dict(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(job), 201


@workstation_blueprint.get("/train/jobs/<job_id>")
@require_auth
def get_training_job(job_id: str):
    job = training_service.get_job(job_id)
    if job is None:
        return jsonify({"error": "Training job not found"}), 404
    return jsonify(job)


@workstation_blueprint.post("/train/jobs/<job_id>/cancel")
@require_roles(UserRole.ADMIN)
def cancel_training_job(job_id: str):
    try:
        result = training_service.cancel_job(job_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@workstation_blueprint.delete("/train/jobs/<job_id>")
@require_roles(UserRole.ADMIN)
def delete_training_job(job_id: str):
    try:
        removed = training_service.delete_job(job_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify({"deleted": True, "id": job_id, "job": removed})


@workstation_blueprint.get("/models")
@require_auth
def list_models():
    return jsonify(models_repo.list_models())


@workstation_blueprint.get("/models/<int:model_id>/export-manifest")
@require_roles(UserRole.ADMIN)
def get_model_export_manifest(model_id: int):
    try:
        manifest = model_export_service.build_export_manifest(model_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(manifest)


@workstation_blueprint.post("/models/<int:model_id>/export")
@require_roles(UserRole.ADMIN)
def export_model(model_id: int):
    try:
        bundle = model_export_service.create_export(model_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code

    archive_path = Path(bundle["archive_path"])

    response = send_file(
        archive_path,
        as_attachment=True,
        download_name=bundle["archive_name"],
        mimetype="application/zip",
    )

    def _cleanup_export() -> None:
        try:
            archive_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    response.call_on_close(_cleanup_export)
    return response


@workstation_blueprint.post("/models/upload")
@require_roles(UserRole.ADMIN)
def upload_model():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    file_name = str(payload.get("file_name") or "model.pt").strip()
    content_b64 = str(payload.get("content_b64") or "").strip()
    class_names_raw = payload.get("class_names") or []

    if not name or not content_b64:
        return jsonify({"error": "name and content_b64 are required"}), 400
    if not isinstance(class_names_raw, list):
        return jsonify({"error": "class_names must be a list"}), 400
    # P3-9: Sanitize filename — strip path components, prevent traversal
    file_name = Path(file_name).name  # strips any directory components
    if not file_name:
        return jsonify({"error": "file_name must not be empty"}), 400

    try:
        content = base64.b64decode(content_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Invalid base64 content: {exc}"}), 400

    # P3-9: Verify resolved path stays within MODELS_DIR (defense-in-depth)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = (MODELS_DIR / file_name).resolve()
    if not str(dest).startswith(str(MODELS_DIR.resolve())):
        return jsonify({"error": "file_name must be within models directory"}), 400
    dest.write_bytes(content)

    # Optional companion file (e.g., OpenVINO .bin paired with .xml)
    # Written to disk only — not registered in model DB.
    companion_file_name = str(payload.get("companion_file_name") or "").strip()
    companion_b64_str = str(payload.get("companion_b64") or "").strip()
    if companion_file_name and companion_b64_str:
        try:
            companion_content = base64.b64decode(companion_b64_str)
            # P3-9: Same sanitization for companion file
            companion_file_name = Path(companion_file_name).name
            companion_dest = (MODELS_DIR / companion_file_name).resolve()
            if not str(companion_dest).startswith(str(MODELS_DIR.resolve())):
                return jsonify({"error": "companion_file_name must be within models directory"}), 400
            companion_dest.write_bytes(companion_content)
        except Exception as exc:
            return jsonify({"error": f"Invalid companion file: {exc}"}), 400

    model = models_repo.add_model(
        name,
        str(dest),
        "upload",
        runtime=str(payload.get("runtime") or "ultralytics").strip() or "ultralytics",
        task=str(payload.get("task") or "detection").strip() or "detection",
        class_names=list(class_names_raw),
        architecture_family=str(payload.get("architecture_family") or "").strip() or None,
        architecture_variant=str(payload.get("architecture_variant") or "").strip() or None,
    )
    return jsonify({**model, "saved_to": str(dest)}), 201


@workstation_blueprint.post("/models/import")
@require_roles(UserRole.ADMIN)
def import_model():
    payload = request.get_json(silent=True) or {}
    form_data = request.form if request.form else None
    skip_validation = _truthy(form_data.get("skip_validation") if form_data is not None else payload.get("skip_validation"))
    force_rename = _truthy(form_data.get("force_rename") if form_data is not None else payload.get("force_rename"))
    requested_lifecycle = form_data.get("target_lifecycle") if form_data is not None else payload.get("target_lifecycle")
    target_lifecycle = str(requested_lifecycle or "draft").strip().lower() or "draft"

    temp_archive_path: Path | None = None
    original_filename: str | None = None
    try:
        if request.files:
            archive_file = request.files.get("zip_file") or request.files.get("file")
            if archive_file is None or not getattr(archive_file, "filename", ""):
                return jsonify({"error": "zip_file is required"}), 400
            original_filename = str(archive_file.filename or "").strip() or None
            temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-import-")
            temp_handle.close()
            temp_archive_path = Path(temp_handle.name)
            archive_file.save(str(temp_archive_path))
        else:
            content_b64 = str(payload.get("content_b64") or payload.get("zip_b64") or "").strip()
            if not content_b64:
                return jsonify({"error": "zip_file or content_b64 is required"}), 400
            try:
                content = base64.b64decode(content_b64)
            except Exception as exc:  # noqa: BLE001
                return jsonify({"error": f"Invalid base64 content: {exc}"}), 400
            temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-import-")
            temp_handle.write(content)
            temp_handle.close()
            temp_archive_path = Path(temp_handle.name)

        result = model_export_service.import_model_archive(
            temp_archive_path,
            original_filename=original_filename,
            skip_validation=skip_validation,
            force_rename=force_rename,
            target_lifecycle=target_lifecycle,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    finally:
        if temp_archive_path is not None:
            try:
                temp_archive_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    return jsonify(result), 201


@workstation_blueprint.post("/models")
@require_roles(UserRole.ADMIN)
def create_model():
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    path = str(payload.get("path") or "").strip()
    if not name or not path:
        return jsonify({"error": "name and path are required"}), 400
    class_names = payload.get("class_names")
    if class_names is not None and not isinstance(class_names, list):
        return jsonify({"error": "class_names must be a list"}), 400
    return jsonify(
        models_repo.add_model(
            name,
            path,
            str(payload.get("source") or "manual"),
            meta_path=str(payload.get("meta_path") or "").strip() or None,
            runtime=str(payload.get("runtime") or "ultralytics").strip() or "ultralytics",
            task=str(payload.get("task") or "detection").strip() or "detection",
            class_names=list(class_names or []),
            architecture_family=str(payload.get("architecture_family") or "").strip() or None,
            architecture_variant=str(payload.get("architecture_variant") or "").strip() or None,
        )
    ), 201


@workstation_blueprint.post("/models/<int:model_id>/transition")
@require_roles(UserRole.ADMIN)
def transition_model_lifecycle(model_id: int):
    payload = request.get_json(force=True) or {}
    new_status = str(payload.get("status") or "").strip().lower()
    if not new_status:
        return jsonify({"error": "status is required"}), 400
    note = str(payload.get("note") or "").strip() or None
    actor = getattr(g, "current_user", None)
    try:
        result = models_repo.transition_lifecycle(
            model_id,
            new_status,
            actor_id=actor.id if actor else None,
            note=note,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return jsonify({"error": message}), status_code
    return jsonify(result)


@workstation_blueprint.patch("/models/<int:model_id>")
@require_roles(UserRole.ADMIN)
def rename_model(model_id: int):
    payload = request.get_json(force=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be an object"}), 400

    allowed_fields = {"name"}
    extra_fields = set(payload.keys()) - allowed_fields
    if extra_fields:
        return jsonify({
            "error": f"Only 'name' can be updated. Unexpected field(s): {', '.join(sorted(extra_fields))}"
        }), 400

    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name must be a non-empty string"}), 400

    try:
        record = models_repo.update_model(model_id, name=name)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            return jsonify({"error": message}), 404
        if "seeded-default" in message.lower():
            return jsonify({"error": message}), 409
        return jsonify({"error": message}), 400
    return jsonify(record)


@workstation_blueprint.delete("/models/<int:model_id>")
@require_roles(UserRole.ADMIN)
def delete_model(model_id: int):
    model = models_repo.get_model(model_id)
    if model is None:
        return jsonify({"error": "Model not found"}), 404

    conflict = models_repo.find_active_model_conflict(
        str(model.get("path") or ""),
        templates_repo=templates_repo,
        deployments_repo=deployments_repo,
    )
    if conflict is not None:
        return jsonify({
            "error": "Model is referenced by an active deployment",
            "conflict": conflict,
        }), 409

    purge_files = _truthy(request.args.get("purge_files"))
    try:
        removed = models_repo.delete_model(model_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 409 if "cannot be deleted" in message.lower() else 400
        return jsonify({"error": message}), status_code

    purged_files = _purge_model_files(removed) if purge_files else []
    return jsonify({
        "deleted": True,
        "id": model_id,
        "purged_files": purged_files,
        "model": removed,
    })


# ---------------------------------------------------------------------------
# Workstation registry + heartbeat
# ---------------------------------------------------------------------------

@workstation_blueprint.get("/workstations")
@require_roles(UserRole.ADMIN)
def list_workstations():
    """List all registered workstations and their last-seen timestamps."""
    return jsonify(workstation_registry_repo.list_workstations())


@workstation_blueprint.delete("/workstations/<path:machine_id>")
@require_roles(UserRole.ADMIN)
def delete_workstation(machine_id: str):
    normalized = str(machine_id or "").strip()
    if not normalized:
        return jsonify({"error": "machine_id is required"}), 400
    ok = workstation_registry_repo.delete_workstation(normalized)
    if not ok:
        return jsonify({"error": "Workstation not found"}), 404
    return jsonify({"deleted": True, "machine_id": normalized})


@workstation_blueprint.post("/workstations/heartbeat")
@require_auth
def workstation_heartbeat():
    """Register or update a workstation's identity and trigger stale session cleanup.

    Body (all optional except machine_id):
        machine_id     — unique identifier for this client machine (required)
        client_version — client application version string
        line_id        — production line this workstation is assigned to
        station_id     — station within the line
    """
    payload = request.get_json(force=True) or {}
    machine_id = str(payload.get("machine_id") or "").strip()
    if not machine_id:
        return jsonify({"error": "machine_id is required"}), 400

    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    ip_address = forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")

    record = workstation_registry_repo.heartbeat(
        machine_id=machine_id,
        client_version=str(payload.get("client_version") or "").strip() or None,
        line_id=str(payload.get("line_id") or "").strip() or None,
        station_id=str(payload.get("station_id") or "").strip() or None,
        ip_address=ip_address or None,
    )

    # Best-effort stale session cleanup on each heartbeat (non-blocking)
    try:
        purged = token_store.purge_expired()
    except Exception:  # noqa: BLE001
        purged = 0

    return jsonify({
        "ok": True,
        "workstation": record,
        "sessions_purged": purged,
        "server_time": datetime.now(UTC).isoformat(),
    })
