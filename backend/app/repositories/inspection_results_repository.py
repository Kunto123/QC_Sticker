from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.repositories.base_json import JsonRepository
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


class InspectionResultsRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("inspection_results.json", {"results": []})

    def create_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = self.load()
        items = store["results"]
        record = dict(payload)
        record["id"] = self.next_id(items)
        record.setdefault("push_status", "pending")
        record.setdefault("retry_count", 0)
        record.setdefault("inspected_at", datetime.now(UTC).isoformat())
        items.append(record)
        self.save(store)
        return record

    def update_result(self, result_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        store = self.load()
        items = store["results"]
        for item in items:
            if int(item["id"]) != int(result_id):
                continue
            item.update(dict(patch))
            self.save(store)
            return item
        raise ValueError("Inspection result not found.")

    def delete_result(self, result_id: int) -> dict[str, Any]:
        store = self.load()
        items = store["results"]
        for index, item in enumerate(items):
            if int(item["id"]) != int(result_id):
                continue
            removed = dict(item)
            del items[index]
            self.save(store)
            return removed
        raise ValueError("Inspection result not found.")

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
        items = list(reversed(self.load()["results"]))
        filtered = []
        for item in items:
            if line_id and item.get("line_id") != line_id:
                continue
            if station_id and item.get("station_id") != station_id:
                continue
            if part_name and item.get("part_name") != part_name:
                continue
            if template_version_id is not None and int(item.get("template_version_id") or 0) != int(template_version_id):
                continue
            if decision_code and item.get("decision_code") != decision_code:
                continue
            if push_status and item.get("push_status") != push_status:
                continue
            filtered.append(item)
        return filtered[offset: offset + limit]

    def get_result(self, result_id: int) -> dict[str, Any] | None:
        return next((item for item in self.load()["results"] if int(item["id"]) == int(result_id)), None)

    def summary(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
        template_version_id: int | None = None,
        decision_code: str | None = None,
    ) -> dict[str, Any]:
        items = self.list_results(
            line_id=line_id,
            station_id=station_id,
            part_name=part_name,
            template_version_id=template_version_id,
            decision_code=decision_code,
            limit=100_000,
        )
        summary = _new_aggregate()
        for item in items:
            self._accumulate(summary, item)
        return _finalize_aggregate(summary)

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
            if key not in bucketed:
                bucketed[key] = {
                    "bucket_time": bucket_time,
                    "granularity": granularity,
                    "line_id": item.get("line_id"),
                    "station_id": item.get("station_id"),
                    "template_version_id": item.get("template_version_id"),
                    "part_name": item.get("part_name"),
                    **_new_aggregate(),
                }
            self._accumulate(bucketed[key], item)
        finalized = [_finalize_aggregate(item) for item in bucketed.values()]
        return sorted(finalized, key=lambda item: item["bucket_time"], reverse=True)[:limit]

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
