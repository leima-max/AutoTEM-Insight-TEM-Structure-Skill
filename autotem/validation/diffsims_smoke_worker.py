"""Subprocess entrypoint for timeout-guarded real diffsims smoke checks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from autotem.simulation.diffsims_adapter import simulate_diffraction_with_diffsims
from autotem.validation.diffsims_smoke import _material_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    try:
        kwargs = json.loads(Path(args.input).read_text(encoding="utf-8"))
        progress_path = kwargs.pop("_progress_path", None)

        def progress(stage: str) -> None:
            if not progress_path:
                return
            path = Path(str(progress_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": stage, "time": time.time()}) + "\n")
                handle.flush()

        result = {
            "ok": True,
            "result": _material_result(
                **dict(kwargs),
                simulate_func=simulate_diffraction_with_diffsims,
                progress_func=progress,
            ),
        }
    except BaseException as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
