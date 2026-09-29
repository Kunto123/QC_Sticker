"""Route registry model (+ heartbeat workstation).

Dataset / anotasi / augment / training sudah dihapus 2026-09-18 — training
dilakukan di software lain; app ini cuma import model yang sudah jadi.
"""
from __future__ import annotations

import base64
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file

from backend.app.core.config import MODELS_DIR
from backend.app.core.container import (
    deployments_repo,
    models_repo,
    model_export_service,
    sticker_inference_service,
    templates_repo,
    token_store,
    workstation_registry_repo,
)
from backend.app.core.http import require_auth, require_roles
from backend.app.services.model_export_service import (
    _safe_component,
    purge_model_files,
    runtime_for_path,
    write_meta_json,
)
from shared.contracts.enums import UserRole


workstation_blueprint = Blueprint("workstation", __name__)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
    """Registrasi satu file model yang dikirim sebagai base64 (.pt / .onnx / .tflite / .xml+.bin).

    File-nya masuk ke `data/models/<name>/`; kalau `class_names` diberikan,
    `<stem>.meta.json` ditulis di sebelahnya supaya inference backend bisa menemukannya.
    """
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    file_name = Path(str(payload.get("file_name") or "model.pt").strip()).name
    content_b64 = str(payload.get("content_b64") or "").strip()
    class_names_raw = payload.get("class_names") or []

    if not name or not content_b64:
        return jsonify({"error": "name dan content_b64 wajib diisi"}), 400
    if not isinstance(class_names_raw, list):
        return jsonify({"error": "class_names harus berupa list"}), 400
    if not file_name or Path(file_name).suffix.lower() not in {".pt", ".onnx", ".tflite", ".xml"}:
        return jsonify({"error": "file_name harus berakhiran .pt, .onnx, .tflite atau .xml"}), 400
    if models_repo.find_by_name(name) is not None:
        return jsonify({"error": f"Model dengan nama '{name}' sudah ada"}), 409

    try:
        content = base64.b64decode(content_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Konten base64 tidak valid: {exc}"}), 400

    runtime = str(payload.get("runtime") or "").strip().lower() or runtime_for_path(file_name)
    companion_file_name = Path(str(payload.get("companion_file_name") or "").strip()).name
    companion_b64 = str(payload.get("companion_b64") or "").strip()
    if runtime == "openvino" and not (companion_file_name and companion_b64):
        return jsonify({"error": "Upload OpenVINO butuh file pendamping .bin (companion_file_name + companion_b64)"}), 400

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_dir = MODELS_DIR / _safe_component(name)
    counter = 1
    while dest_dir.exists():
        dest_dir = MODELS_DIR / f"{_safe_component(name)}_{counter}"
        counter += 1
    dest_dir.mkdir(parents=True)
    dest = dest_dir / file_name
    dest.write_bytes(content)
    if companion_file_name and companion_b64:
        try:
            (dest_dir / companion_file_name).write_bytes(base64.b64decode(companion_b64))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"File pendamping tidak valid: {exc}"}), 400

    class_names = [str(c) for c in class_names_raw]
    meta_path = write_meta_json(dest, class_names, runtime=runtime, name=name) if class_names else None
    model = models_repo.add_model(
        name,
        str(dest),
        "upload",
        meta_path=str(meta_path) if meta_path else None,
        runtime=runtime,
        task=str(payload.get("task") or "detection").strip() or "detection",
        class_names=class_names,
    )
    warnings = []
    if runtime != "ultralytics" and not class_names:
        warnings.append("Class names tidak diberikan; hasil deteksi akan memakai label angka.")
    return jsonify({**model, "saved_to": str(dest), "warnings": warnings}), 201


@workstation_blueprint.post("/models/import")
@require_roles(UserRole.ADMIN)
def import_model():
    """Import paket model (.zip). Lihat ModelExportService.import_model_archive."""
    payload = request.get_json(silent=True) or {}
    form_data = request.form if request.form else None

    def _field(key: str):
        return form_data.get(key) if form_data is not None else payload.get(key)

    skip_validation = _truthy(_field("skip_validation"))
    force_rename = _truthy(_field("force_rename"))
    target_lifecycle = str(_field("target_lifecycle") or "draft").strip().lower() or "draft"
    name = str(_field("name") or "").strip() or None

    temp_archive_path: Path | None = None
    original_filename: str | None = None
    try:
        if request.files:
            archive_file = request.files.get("zip_file") or request.files.get("file")
            if archive_file is None or not getattr(archive_file, "filename", ""):
                return jsonify({"error": "zip_file wajib diisi"}), 400
            original_filename = str(archive_file.filename or "").strip() or None
            temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-import-")
            temp_handle.close()
            temp_archive_path = Path(temp_handle.name)
            archive_file.save(str(temp_archive_path))
        else:
            content_b64 = str(payload.get("content_b64") or payload.get("zip_b64") or "").strip()
            if not content_b64:
                return jsonify({"error": "zip_file atau content_b64 wajib diisi"}), 400
            original_filename = str(payload.get("file_name") or "").strip() or None
            try:
                content = base64.b64decode(content_b64)
            except Exception as exc:  # noqa: BLE001
                return jsonify({"error": f"Konten base64 tidak valid: {exc}"}), 400
            temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="qc-suite-import-")
            temp_handle.write(content)
            temp_handle.close()
            temp_archive_path = Path(temp_handle.name)

        result = model_export_service.import_model_archive(
            temp_archive_path,
            name=name,
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
    """Registrasi file model yang sudah ada di disk (bukan upload)."""
    payload = request.get_json(force=True) or {}
    name = str(payload.get("name") or "").strip()
    path = str(payload.get("path") or "").strip()
    if not name or not path:
        return jsonify({"error": "name dan path wajib diisi"}), 400
    class_names = payload.get("class_names")
    if class_names is not None and not isinstance(class_names, list):
        return jsonify({"error": "class_names harus berupa list"}), 400
    return jsonify(
        models_repo.add_model(
            name,
            path,
            str(payload.get("source") or "manual"),
            meta_path=str(payload.get("meta_path") or "").strip() or None,
            runtime=str(payload.get("runtime") or "").strip().lower() or runtime_for_path(path),
            task=str(payload.get("task") or "detection").strip() or "detection",
            class_names=[str(c) for c in (class_names or [])],
        )
    ), 201


@workstation_blueprint.post("/models/<int:model_id>/transition")
@require_roles(UserRole.ADMIN)
def transition_model_lifecycle(model_id: int):
    payload = request.get_json(force=True) or {}
    new_status = str(payload.get("status") or "").strip().lower()
    if not new_status:
        return jsonify({"error": "status wajib diisi"}), 400
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
        return jsonify({"error": "Request body harus berupa object"}), 400
    extra_fields = set(payload.keys()) - {"name"}
    if extra_fields:
        return jsonify({
            "error": f"Hanya 'name' yang bisa diupdate. Field tak dikenal: {', '.join(sorted(extra_fields))}"
        }), 400
    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name harus berupa teks yang tidak kosong"}), 400
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
        return jsonify({"error": "Model tidak ditemukan"}), 404

    conflict = models_repo.find_active_model_conflict(
        str(model.get("path") or ""),
        templates_repo=templates_repo,
        deployments_repo=deployments_repo,
    )
    if conflict is not None:
        return jsonify({
            "error": "Model masih dipakai oleh deployment yang aktif",
            "conflict": conflict,
        }), 409

    purge_files = _truthy(request.args.get("purge_files"))
    try:
        removed = models_repo.delete_model(model_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 409 if "cannot be deleted" in message.lower() else 400
        return jsonify({"error": message}), status_code

    purged_files: list[str] = []
    purge_warning: str | None = None
    if purge_files:
        # lepaskan weight yang di-mmap oleh inference cache sebelum menyentuh file
        sticker_inference_service.unload_model(str(removed.get("path") or ""))
        purged_files = purge_model_files(removed)
        leftover = Path(str(removed.get("path") or ""))
        if leftover.exists():
            purge_warning = f"Entri registry sudah dihapus tapi file masih dipakai sehingga tidak terhapus: {leftover.parent}"
    return jsonify({
        "deleted": True,
        "id": model_id,
        "purged_files": purged_files,
        "purge_warning": purge_warning,
        "model": removed,
    })


# ---------------------------------------------------------------------------
# Registry workstation + heartbeat
# ---------------------------------------------------------------------------

@workstation_blueprint.get("/workstations")
@require_roles(UserRole.ADMIN)
def list_workstations():
    """List semua workstation terdaftar beserta timestamp terakhir terlihat."""
    return jsonify(workstation_registry_repo.list_workstations())


@workstation_blueprint.delete("/workstations/<path:machine_id>")
@require_roles(UserRole.ADMIN)
def delete_workstation(machine_id: str):
    normalized = str(machine_id or "").strip()
    if not normalized:
        return jsonify({"error": "machine_id wajib diisi"}), 400
    ok = workstation_registry_repo.delete_workstation(normalized)
    if not ok:
        return jsonify({"error": "Workstation tidak ditemukan"}), 404
    return jsonify({"deleted": True, "machine_id": normalized})


@workstation_blueprint.post("/workstations/heartbeat")
@require_auth
def workstation_heartbeat():
    """Registrasi atau update identitas workstation, sekalian bersihkan sesi basi."""
    payload = request.get_json(force=True) or {}
    machine_id = str(payload.get("machine_id") or "").strip()
    if not machine_id:
        return jsonify({"error": "machine_id wajib diisi"}), 400

    forwarded_for = str(request.headers.get("X-Forwarded-For") or "").strip()
    ip_address = forwarded_for.split(",", 1)[0].strip() if forwarded_for else (request.remote_addr or "")

    record = workstation_registry_repo.heartbeat(
        machine_id=machine_id,
        client_version=str(payload.get("client_version") or "").strip() or None,
        line_id=str(payload.get("line_id") or "").strip() or None,
        station_id=str(payload.get("station_id") or "").strip() or None,
        ip_address=ip_address or None,
    )
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
