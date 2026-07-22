from __future__ import annotations

import json
import re
from typing import Any


class ResponseParseError(ValueError):
    pass


def _extract_balanced(text: str) -> str | None:
    start = next((i for i, char in enumerate(text) if char in "[{"), None)
    if start is None:
        return None
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted:
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
    return None


def parse_json_response(text: str, required_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    value = text.strip().lstrip("\ufeff")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", value, re.IGNORECASE)
    candidates = [value]
    if fence:
        candidates.insert(0, fence.group(1).strip())
    balanced = _extract_balanced(value)
    if balanced:
        candidates.append(balanced)
    errors: list[str] = []
    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if not isinstance(parsed, dict):
                errors.append("root is not an object")
                continue
            missing = [field for field in required_fields if field not in parsed]
            if missing:
                raise ResponseParseError(f"Missing schema-critical fields: {missing}")
            return parsed
    raise ResponseParseError("Unable to parse complete JSON response: " + (errors[-1] if errors else "no JSON object"))

