from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.repositories.base_json import JsonRepository


MIN_PROFILE_PIXELS = 64


def _is_expired(record: dict) -> bool:
    expires_at = record.get("expires_at")
    if not expires_at:
        return False
    try:
        exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= exp_dt
    except (ValueError, TypeError):
        return False


_UNSET = object()


class ProfilesRepository(JsonRepository):
    def __init__(self) -> None:
        super().__init__("profiles.json", {"profiles": []})

    def list_profiles(self) -> list[dict]:
        profiles = self.load()["profiles"]
        now = datetime.now(UTC)
        result = []
        for item in profiles:
            record = dict(item)
            record["is_expired"] = _is_expired(record)
            result.append(record)
        return result

    def get(self, profile_id: int) -> dict | None:
        item = next((item for item in self.load()["profiles"] if int(item["id"]) == int(profile_id)), None)
        if item is None:
            return None
        record = dict(item)
        record["is_expired"] = _is_expired(record)
        return record

    def get_active_for_scope(
        self,
        *,
        line_id: str | None = None,
        station_id: str | None = None,
        part_name: str | None = None,
    ) -> dict | None:
        """Return the most recent non-expired profile matching the given scope.

        Scope matching is hierarchical: an exact match beats a partial match.
        A None field on the stored profile means "any" (wildcard).
        """
        profiles = self.load()["profiles"]
        candidates = []
        for item in profiles:
            if _is_expired(item):
                continue
            scope_line = item.get("scope_line_id")
            scope_station = item.get("scope_station_id")
            scope_part = item.get("scope_part_name")
            # each scope field: None = wildcard, value = must match
            if scope_line is not None and scope_line != line_id:
                continue
            if scope_station is not None and scope_station != station_id:
                continue
            if scope_part is not None and scope_part != part_name:
                continue
            # specificity score: how many fields matched exactly
            score = (
                (1 if scope_line == line_id else 0)
                + (1 if scope_station == station_id else 0)
                + (1 if scope_part == part_name else 0)
            )
            candidates.append((score, item.get("version", 1), item))

        if not candidates:
            return None
        # prefer highest specificity, then latest version
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = candidates[0][2]
        result = dict(best)
        result["is_expired"] = False
        return result

    def _validate_profile_payload(self, profile: dict) -> None:
        if not isinstance(profile, dict):
            raise ValueError("profile must be an object")

        sampling_meta = dict(profile.get("sampling_meta") or {})
        raw_total_pixels = sampling_meta.get("total_pixels")
        if raw_total_pixels is None:
            raise ValueError("profile.sampling_meta.total_pixels is required")
        try:
            total_pixels = int(raw_total_pixels)
        except (TypeError, ValueError) as exc:
            raise ValueError("profile.sampling_meta.total_pixels must be an integer") from exc
        if total_pixels < MIN_PROFILE_PIXELS:
            raise ValueError(
                "profile.sampling_meta.total_pixels is too small for stable calibration. "
                f"Minimum {MIN_PROFILE_PIXELS} pixels required, got {total_pixels}."
            )

    def create(
        self,
        name: str,
        profile: dict,
        *,
        scope_line_id: str | None = None,
        scope_station_id: str | None = None,
        scope_part_name: str | None = None,
        expiry_interval_days: int | None = None,
    ) -> dict:
        payload = self.load()
        items = payload["profiles"]

        # Version numbering per scope
        scope_items = [
            item for item in items
            if item.get("scope_line_id") == scope_line_id
            and item.get("scope_station_id") == scope_station_id
            and item.get("scope_part_name") == scope_part_name
        ]
        version = max((int(item.get("version") or 1) for item in scope_items), default=0) + 1
        self._validate_profile_payload(profile)

        now = datetime.now(UTC)
        expires_at: str | None = None
        if expiry_interval_days and expiry_interval_days > 0:
            expires_at = (now + timedelta(days=expiry_interval_days)).isoformat()

        record: dict[str, Any] = {
            "id": self.next_id(items),
            "name": name,
            "profile": profile,
            "version": version,
            "scope_line_id": scope_line_id,
            "scope_station_id": scope_station_id,
            "scope_part_name": scope_part_name,
            "expires_at": expires_at,
            "created_at": now.isoformat(),
        }
        items.append(record)
        self.save(payload)
        result = dict(record)
        result["is_expired"] = False
        return result

    def update(
        self,
        profile_id: int,
        *,
        name: Any = _UNSET,
        profile: Any = _UNSET,
        scope_line_id: Any = _UNSET,
        scope_station_id: Any = _UNSET,
        scope_part_name: Any = _UNSET,
        expiry_interval_days: Any = _UNSET,
    ) -> dict:
        payload = self.load()
        now = datetime.now(UTC)
        for item in payload["profiles"]:
            if int(item["id"]) != int(profile_id):
                continue

            if name is not _UNSET:
                normalized_name = str(name or "").strip()
                if not normalized_name:
                    raise ValueError("name must not be empty")
                item["name"] = normalized_name

            if profile is not _UNSET:
                self._validate_profile_payload(profile)
                item["profile"] = profile

            if scope_line_id is not _UNSET:
                item["scope_line_id"] = str(scope_line_id or "").strip() or None
            if scope_station_id is not _UNSET:
                item["scope_station_id"] = str(scope_station_id or "").strip() or None
            if scope_part_name is not _UNSET:
                item["scope_part_name"] = str(scope_part_name or "").strip() or None

            if expiry_interval_days is not _UNSET:
                if expiry_interval_days in (None, ""):
                    item["expires_at"] = None
                else:
                    try:
                        expiry_days = int(expiry_interval_days)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("expiry_interval_days must be a positive integer") from exc
                    if expiry_days <= 0:
                        raise ValueError("expiry_interval_days must be a positive integer")
                    item["expires_at"] = (now + timedelta(days=expiry_days)).isoformat()

            item["updated_at"] = now.isoformat()
            self.save(payload)
            result = dict(item)
            result["is_expired"] = _is_expired(result)
            return result

        raise ValueError("Profile not found")

    def delete(self, profile_id: int) -> bool:
        payload = self.load()
        before = len(payload["profiles"])
        payload["profiles"] = [
            item for item in payload["profiles"] if int(item["id"]) != int(profile_id)
        ]
        if len(payload["profiles"]) == before:
            return False
        self.save(payload)
        return True

