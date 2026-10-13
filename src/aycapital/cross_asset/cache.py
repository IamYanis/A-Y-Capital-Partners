"""Immutable CSV snapshots plus a request pointer; refresh preserves old snapshots."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


class SnapshotCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _pointer(self, request: dict[str, Any]) -> Path:
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return self.directory / f"{key}.json"

    def read(self, request: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        pointer = self._pointer(request)
        if not pointer.exists():
            return None
        metadata = json.loads(pointer.read_text())
        path = self.directory / metadata["snapshot"]
        if path.parent != self.directory or path.name != metadata["snapshot"]:
            raise ValueError("Invalid cache snapshot path")
        if metadata["request"] != request:
            raise ValueError("Cache request mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Cache integrity check failed")
        return pd.read_csv(path, index_col=0), metadata

    def write(self, request: dict[str, Any], data: pd.DataFrame, **fields: Any) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        snapshot = f"snapshot-{uuid4().hex}.csv"
        path = self.directory / snapshot
        data.to_csv(path, float_format="%.17g")
        metadata = {
            "request": request,
            "snapshot": snapshot,
            "retrieved_at_utc": now,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **fields,
        }
        content = json.dumps(metadata, indent=2) + "\n"
        path.with_suffix(".json").write_text(content)
        pointer = self._pointer(request)
        temporary = pointer.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(content)
        temporary.replace(pointer)
        return metadata
