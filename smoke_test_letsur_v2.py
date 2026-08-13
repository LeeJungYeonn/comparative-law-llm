from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_v2.llm_runtime import DEFAULT_API_KEY_ENV, DEFAULT_LETSUR_BASE_URL, call_structured, configured_model

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make one tiny, secret-safe Letsur connectivity request.")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=DEFAULT_LETSUR_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--dotenv-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/letsur_connectivity_smoke"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    parsed, provenance = call_structured(
        case_id="CONNECTIVITY_SMOKE", stage="connectivity", prompt_version="letsur-connectivity-v1",
        model=configured_model(args.model), system_prompt="Return JSON with ok set to true.",
        user_payload={"task": "connectivity_check"}, schema_name="connectivity", schema=SCHEMA,
        raw_root=args.output_dir / "raw_api_responses", status_path=args.output_dir / "api_request_status.jsonl",
        max_retries=2, resume=args.resume, base_url=args.base_url, api_key_env=args.api_key_env,
        dotenv_path=args.dotenv_path,
    )
    passed = parsed.get("ok") is True
    print(json.dumps({"connectivity": "pass" if passed else "fail", "model": provenance.get("model_snapshot_or_returned_model_id"), "request_id": provenance.get("request_id"), "structured_output_mode": provenance.get("structured_output_mode")}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
