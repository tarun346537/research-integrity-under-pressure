"""Portable artifact storage and source identification; never stores API keys."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_hash() -> str:
    root = Path(__file__).parent
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))}
    return canonical_hash(sources)


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Artifact must contain a JSON object.")
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def envelope(environment: Any, *, kind: str, name: str, status: str) -> dict:
    return {
        "artifact_version": 1, "kind": kind, "name": name,
        "created_at": utc_now(), "source_sha256": source_hash(), "status": status,
        "environment": environment.export(), "grade": environment.grade(),
    }
