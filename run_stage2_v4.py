from __future__ import annotations

import sys

from run_stage2_v3 import main as run_profiled_stage2


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--generation-profile" not in values:
        values.extend(["--generation-profile", "english-v1"])
    if "--output-dir" not in values:
        values.extend([
            "--output-dir",
            "outputs/neutral/stage2-neutral-35x35-v4",
        ])
    return run_profiled_stage2(values)


if __name__ == "__main__":
    raise SystemExit(main())
