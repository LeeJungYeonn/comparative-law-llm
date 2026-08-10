from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from .io_utils import sha256_text, write_json


DATASET_SERVER = "https://datasets-server.huggingface.co"


def _get_json(url: str, *, retries: int = 5, timeout: int = 120) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "comparative-law-llm-v2/2.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Hugging Face datasets-server request failed: {url}: {error}")


def dataset_info(dataset: str, config: str = "default") -> dict[str, Any]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config})
    return _get_json(f"{DATASET_SERVER}/info?{query}")


def iter_filtered_rows(
    dataset: str,
    *,
    where: str,
    config: str = "default",
    split: str = "train",
    page_size: int = 20,
    cache_dir: Path | None = None,
    limit: int = 0,
    allow_partial: bool = False,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    offset = 0
    yielded = 0
    while True:
        params = {"dataset": dataset, "config": config, "split": split, "where": where, "offset": offset, "length": page_size}
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        url = f"{DATASET_SERVER}/filter?{query}"
        cache_path = cache_dir / f"{sha256_text(url)}.json" if cache_dir else None
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _get_json(url)
            if cache_path:
                write_json(cache_path, payload)
        rows = payload.get("rows") or []
        meta = {
            "num_rows_total": int(payload.get("num_rows_total") or 0),
            "partial": bool(payload.get("partial")),
            "offset": offset,
            "page_size": page_size,
        }
        if meta["partial"] and not allow_partial:
            raise RuntimeError("datasets-server returned a partial filter result; refusing an incomplete audit")
        if not rows:
            break
        for wrapped in rows:
            row = wrapped.get("row") if isinstance(wrapped, dict) and "row" in wrapped else wrapped
            if not isinstance(row, dict):
                continue
            truncation = wrapped.get("truncated_cells") if isinstance(wrapped, dict) else None
            yield row, {**meta, "truncated_cells": truncation or []}
            yielded += 1
            if limit and yielded >= limit:
                return
        offset += len(rows)
        if offset >= meta["num_rows_total"]:
            break
