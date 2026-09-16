from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import AppConfig
from shared.contracts.enums import DecisionCode


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _new_aggregate() -> dict[str, Any]:
    return {
        "total_inspections": 0,
        "total_accept": 0,
        "total_reject": 0,
        "total_part_ready": 0,
        "total_part_not_ready": 0,
        "backend_ultralytics": 0,
        "backend_classic": 0,
        "backend_other": 0,
        "reject_not_found": 0,
        "reject_wrong_type": 0,
        "reject_out_of_position": 0,
        "reject_out_of_angle": 0,
        "reject_low_conf": 0,
        "reject_other": 0,
        "_sum_part_ready_match_ratio": 0.0,
        "_count_part_ready_match_ratio": 0,
        "_sum_sticker_confidence": 0.0,
        "_count_sticker_confidence": 0,
    }


def _finalize_aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    match_count = int(result.pop("_count_part_ready_match_ratio", 0) or 0)
    match_sum = float(result.pop("_sum_part_ready_match_ratio", 0.0) or 0.0)
    conf_count = int(result.pop("_count_sticker_confidence", 0) or 0)
    conf_sum = float(result.pop("_sum_sticker_confidence", 0.0) or 0.0)
    result["avg_part_ready_match_ratio"] = round(match_sum / match_count, 6) if match_count else None
    result["avg_sticker_confidence"] = round(conf_sum / conf_count, 6) if conf_count else None
    return result


class SqlServerInspectionResultsRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ensure_schema()

    def _connect(self):
        import pyodbc

        connection_string = (
            f"DRIVER={{{self._config.sql_driver}}};"
            f"SERVER={self._config.sql_server};"
            f"DATABASE={self._config.sql_database};"
            f"UID={self._config.sql_username};"
            f"PWD={self._config.sql_password};"
            "TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string, timeout=5)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                IF OBJECT_ID('dbo.qc_inspection_results', 'U') IS NULL
                BEGIN
                    CREATE TABLE dbo.qc_inspection_results (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        template_version_id INT NOT NULL,
                        line_id NVARCHAR(100) NULL,
                        station_id NVARCHAR(100) NULL,
                        part_name NVARCHAR(150) NULL,
                        mp_check NVARCHAR(100) NULL,
                        data1 FLOAT NULL,
                        data2 FLOAT NULL,
                        decision NVARCHAR(32) NOT NULL,
                        decision_code NVARCHAR(32) NOT NULL,
                        reject_reason_code NVARCHAR(64) NULL,
                        push_status NVARCHAR(32) NOT NULL,
                        retry_count INT NOT NULL,
                        operator_user_id INT NULL,
                        part_ready_status NVARCHAR(64) NULL,
                        part_ready_match_ratio FLOAT NULL,
                        part_ready_distance FLOAT NULL,
                        detected_class NVARCHAR(128) NULL,
                        expected_class NVARCHAR(128) NULL,
                        sticker_confidence FLOAT NULL,
                        sticker_backend NVARCHAR(64) NULL,
                        sticker_bbox_json NVARCHAR(MAX) NULL,
                        validation_details_json NVARCHAR(MAX) NULL,
                        part_ready_roi_meta_json NVARCHAR(MAX) NULL,
                        sticker_roi_meta_json NVARCHAR(MAX) NULL,
                        targets_json NVARCHAR(MAX) NOT NULL,
                        inspected_at DATETIMEOFFSET NOT NULL DEFAULT SYSUTCDATETIME()
                    )
                END
                """
            )
            for column_name, sql_type in (
                ("station_id", "NVARCHAR(100) NULL"),
                ("part_ready_status", "NVARCHAR(64) NULL"),
                ("part_ready_match_ratio", "FLOAT NULL"),
                ("part_ready_distance", "FLOAT NULL"),
                ("detected_class", "NVARCHAR(128) NULL"),
                ("expected_class", "NVARCHAR(128) NULL"),
                ("sticker_confidence", "FLOAT NULL"),
                ("sticker_backend", "NVARCHAR(64) NULL"),
                ("sticker_bbox_json", "NVARCHAR(MAX) NULL"),
                ("validation_details_json", "NVARCHAR(MAX) NULL"),
                ("part_ready_roi_meta_json", "NVARCHAR(MAX) NULL"),
                ("sticker_roi_meta_json", "NVARCHAR(MAX) NULL"),
            ):
                cursor.execute(
                    f"""
                    IF COL_LENGTH('dbo.qc_inspection_results', '{column_name}') IS NULL
                    BEGIN
                        ALTER TABLE dbo.qc_inspection_results ADD {column_name} {sql_type}
                    END
                    """
                )
            conn.commit()

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload)
        record.setdefault("push_status", "pending")
        record.setdefault("retry_count", 0)
        record.setdefault("inspected_at", datetime.now(UTC).isoformat())
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dbo.qc_inspection_results (
                    template_version_id,
                    line_id,
                    station_id,
                    part_name,
                    mp_check,
                    data1,
                    data2,
                    decision,
                    decision_code,
                    reject_reason_code,
                    push_status,
                    retry_count,
                    operator_user_id,
                    part_ready_status,
                    part_ready_match_ratio,
                    part_ready_distance,
                    detected_class,
                    expected_class,
                    sticker_confidence,
                    sticker_backend,
                    sticker_bbox_json,
                    validation_details_json,
                    part_ready_roi_meta_json,
                    sticker_roi_meta_json,
                    targets_json,
                    inspected_at
                )
                OUTPUT INSERTED.id
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                int(record["template_version_id"]),
                record.get("line_id"),
                record.get("station_id"),
                record.get("part_name"),
                record.get("mp_check"),
                record.get("data1"),
                record.get("data2"),
                record.get("decision"),
                record.get("decision_code"),
                record.get("reject_reason_code"),
                record.get("push_status"),
                int(record.get("retry_count") or 0),
                record.get("operator_user_id"),
                record.get("part_ready_status"),
                record.get("part_ready_match_ratio"),
                record.get("part_ready_distance"),
                record.get("detected_class"),
                record.get("expected_class"),
                record.get("sticker_confidence"),
                record.get("sticker_backend"),
                json.dumps(record.get("sticker_bbox"), ensure_ascii=True) if record.get("sticker_bbox") is not None else None,
                json.dumps(record.get("validation_details"), ensure_ascii=True) if record.get("validation_details") is not None else None,
                json.dumps(record.get("part_ready_roi_meta"), ensure_ascii=True) if record.get("part_ready_roi_meta") is not None else None,
                json.dumps(record.get("sticker_roi_meta"), ensure_ascii=True) if record.get("sticker_roi_meta") is not None else None,
                json.dumps(record.get("targets") or [], ensure_ascii=True),
                record["inspected_at"],
            )
            inserted_id = int(cursor.fetchone()[0])
            conn.commit()
        record["id"] = inserted_id
        return record

    def list_results(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
        push_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if line_id:
            where_parts.append("line_id = ?")
            params.append(line_id)
        if station_id:
            where_parts.append("station_id = ?")
            params.append(station_id)
        if part_name:
            where_parts.append("part_name = ?")
            params.append(part_name)
        if template_version_id is not None:
            where_parts.append("template_version_id = ?")
            params.append(int(template_version_id))
        if decision_code:
            where_parts.append("decision_code = ?")
            params.append(decision_code)
        if push_status:
            where_parts.append("push_status = ?")
            params.append(push_status)
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.extend([int(offset), int(limit)])
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT
                    id,
                    template_version_id,
                    line_id,
                    station_id,
                    part_name,
                    mp_check,
                    data1,
                    data2,
                    decision,
                    decision_code,
                    reject_reason_code,
                    push_status,
                    retry_count,
                    operator_user_id,
                    part_ready_status,
                    part_ready_match_ratio,
                    part_ready_distance,
                    detected_class,
                    expected_class,
                    sticker_confidence,
                    sticker_backend,
                    sticker_bbox_json,
                    validation_details_json,
                    part_ready_roi_meta_json,
                    sticker_roi_meta_json,
                    targets_json,
                    CONVERT(NVARCHAR(33), inspected_at, 127) AS inspected_at
                FROM dbo.qc_inspection_results
                {where_clause}
                ORDER BY id DESC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                *params,
            )
            rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_result(self, result_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    template_version_id,
                    line_id,
                    station_id,
                    part_name,
                    mp_check,
                    data1,
                    data2,
                    decision,
                    decision_code,
                    reject_reason_code,
                    push_status,
                    retry_count,
                    operator_user_id,
                    part_ready_status,
                    part_ready_match_ratio,
                    part_ready_distance,
                    detected_class,
                    expected_class,
                    sticker_confidence,
                    sticker_backend,
                    sticker_bbox_json,
                    validation_details_json,
                    part_ready_roi_meta_json,
                    sticker_roi_meta_json,
                    targets_json,
                    CONVERT(NVARCHAR(33), inspected_at, 127) AS inspected_at
                FROM dbo.qc_inspection_results
                WHERE id = ?
                """,
                int(result_id),
            )
            row = cursor.fetchone()
        return None if row is None else self._row_to_dict(row)

    def summary(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
    ) -> dict[str, Any]:
        return self._build_summary(
            self.list_results(
                line_id=line_id,
                station_id=station_id,
                part_name=part_name,
                template_version_id=template_version_id,
                decision_code=decision_code,
                limit=100_000,
            )
        )

    def buckets(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
        granularity: str = "hour",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        items = self.list_results(
            line_id=line_id,
            station_id=station_id,
            part_name=part_name,
            template_version_id=template_version_id,
            decision_code=decision_code,
            limit=100_000,
        )
        bucketed: dict[tuple[str, str | None, str | None, int | None, str | None], dict[str, Any]] = {}
        for item in items:
            inspected_at = _parse_dt(item.get("inspected_at"))
            if inspected_at is None:
                continue
            if granularity == "day":
                bucket_time = inspected_at.strftime("%Y-%m-%dT00:00:00+00:00")
            elif granularity == "minute":
                bucket_time = inspected_at.strftime("%Y-%m-%dT%H:%M:00+00:00")
            else:
                bucket_time = inspected_at.strftime("%Y-%m-%dT%H:00:00+00:00")
            key = (
                bucket_time,
                item.get("line_id"),
                item.get("station_id"),
                item.get("template_version_id"),
                item.get("part_name"),
            )
            bucket = bucketed.setdefault(
                key,
                {
                    "bucket_time": bucket_time,
                    "granularity": granularity,
                    "line_id": item.get("line_id"),
                    "station_id": item.get("station_id"),
                    "template_version_id": item.get("template_version_id"),
                    "part_name": item.get("part_name"),
                    **_new_aggregate(),
                },
            )
            self._accumulate(bucket, item)
        finalized = [_finalize_aggregate(item) for item in bucketed.values()]
        return sorted(finalized, key=lambda item: item["bucket_time"], reverse=True)[:limit]

    def _row_to_dict(self, row) -> dict[str, Any]:
        inspected_at = str(row.inspected_at) if row.inspected_at is not None else None
        return {
            "id": int(row.id),
            "template_version_id": int(row.template_version_id),
            "line_id": row.line_id,
            "station_id": row.station_id,
            "part_name": row.part_name,
            "mp_check": row.mp_check,
            "data1": row.data1,
            "data2": row.data2,
            "decision": row.decision,
            "decision_code": row.decision_code,
            "reject_reason_code": row.reject_reason_code,
            "push_status": row.push_status,
            "retry_count": int(row.retry_count),
            "operator_user_id": row.operator_user_id,
            "part_ready_status": row.part_ready_status,
            "part_ready_match_ratio": row.part_ready_match_ratio,
            "part_ready_distance": row.part_ready_distance,
            "detected_class": row.detected_class,
            "expected_class": row.expected_class,
            "sticker_confidence": row.sticker_confidence,
            "sticker_backend": row.sticker_backend,
            "sticker_bbox": json.loads(row.sticker_bbox_json or "null"),
            "validation_details": json.loads(row.validation_details_json or "null"),
            "part_ready_roi_meta": json.loads(row.part_ready_roi_meta_json or "null"),
            "sticker_roi_meta": json.loads(row.sticker_roi_meta_json or "null"),
            "targets": json.loads(row.targets_json or "[]"),
            "inspected_at": inspected_at,
        }

    def _build_summary(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        summary = _new_aggregate()
        for item in items:
            self._accumulate(summary, item)
        return _finalize_aggregate(summary)

    def _accumulate(self, target: dict[str, Any], item: dict[str, Any]) -> None:
        target["total_inspections"] += 1
        if item.get("decision") == DecisionCode.ACCEPT.value:
            target["total_accept"] += 1
        else:
            target["total_reject"] += 1

        part_ready_status = str(item.get("part_ready_status") or "")
        if part_ready_status in {"ready", "skipped", "disabled"}:
            target["total_part_ready"] += 1
        elif part_ready_status:
            target["total_part_not_ready"] += 1

        backend = str(item.get("sticker_backend") or "").strip().lower()
        if backend == "ultralytics":
            target["backend_ultralytics"] += 1
        elif backend == "classic":
            target["backend_classic"] += 1
        elif backend:
            target["backend_other"] += 1

        match_ratio = item.get("part_ready_match_ratio")
        if match_ratio is not None:
            target["_sum_part_ready_match_ratio"] += float(match_ratio)
            target["_count_part_ready_match_ratio"] += 1

        sticker_confidence = item.get("sticker_confidence")
        if sticker_confidence is not None:
            target["_sum_sticker_confidence"] += float(sticker_confidence)
            target["_count_sticker_confidence"] += 1

        reject = str(item.get("reject_reason_code") or "")
        if reject == "NOT_FOUND":
            target["reject_not_found"] += 1
        elif reject == "WRONG_TYPE":
            target["reject_wrong_type"] += 1
        elif reject == "OUT_OF_POSITION":
            target["reject_out_of_position"] += 1
        elif reject in {"LOW_ROI_CONF", "LOW_CLASS_CONF"}:
            target["reject_low_conf"] += 1
        elif reject:
            target["reject_other"] += 1
