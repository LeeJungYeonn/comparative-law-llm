from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl_recover(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete JSONL records and ignore only a malformed final partial line."""
    if not path.exists():
        return [], False
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows: list[dict[str, Any]] = []
    recovered = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                recovered = True
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object expected in {path} line {index + 1}")
        rows.append(value)
    return rows, recovered


def merge_records(path: Path, records: Iterable[dict[str, Any]], key: str = "case_id") -> None:
    existing, _ = read_jsonl_recover(path)
    merged = {str(row.get(key, "")): row for row in existing if row.get(key)}
    for row in records:
        record_key = str(row.get(key, ""))
        if not record_key:
            raise ValueError(f"Missing checkpoint key {key}")
        merged[record_key] = row
    atomic_write_jsonl(path, merged.values())

