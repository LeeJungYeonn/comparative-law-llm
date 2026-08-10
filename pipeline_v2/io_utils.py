from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: object, length: int = 18) -> str:
    digest = sha256_text("\x1f".join(str(part) for part in parts))[:length]
    return f"{prefix}_{digest}"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def guard_outputs(paths: Iterable[Path], *, overwrite: bool, resume: bool = False) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite and not resume:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}; pass --overwrite or --resume")


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return canonical_json(value)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in materialized for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: csv_cell(row.get(key)) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def normalized_whitespace(value: object) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def normalize_for_grounding(value: str) -> str:
    return normalized_whitespace(value).casefold()

