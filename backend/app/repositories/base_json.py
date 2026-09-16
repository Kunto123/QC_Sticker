from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from backend.app.core.config import JSON_STORE_DIR, ensure_data_dirs


class JsonRepository:
    def __init__(self, filename: str, default_data: Any):
        ensure_data_dirs()
        self._path = JSON_STORE_DIR / filename
        self._default_data = default_data
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Any:
        with self._lock:
            if not self._path.exists():
                data = self._clone_default()
                self.save(data)
                return data
            return json.loads(self._path.read_text(encoding="utf-8-sig"))

    def save(self, data: Any) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to temp file in same dir, then os.replace.
            # Prevents corruption on crash/power-loss mid-write.
            content = json.dumps(data, ensure_ascii=True, indent=2)
            fd, tmp_path = tempfile.mkstemp(
                prefix=self._path.name + ".",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fd)
                os.replace(tmp_path, self._path)
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def _clone_default(self) -> Any:
        return json.loads(json.dumps(self._default_data))

    @staticmethod
    def next_id(items: list[dict[str, Any]], field_name: str = "id") -> int:
        if not items:
            return 1
        return max(int(item.get(field_name) or 0) for item in items) + 1

