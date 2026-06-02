"""Subprocess entrypoint for timeout-guarded diffsims simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotem.core import DiffractionResult
from autotem.simulation.basic import _simulate_diffsims_in_process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        diffraction = DiffractionResult.model_validate(payload["diffraction"])
        parameters = dict(payload.get("parameters") or {})
        simulation = _simulate_diffsims_in_process(diffraction, parameters)
        result = {
            "ok": True,
            "simulation": simulation.model_dump(mode="json"),
            "diffraction": diffraction.model_dump(mode="json"),
        }
    except BaseException as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
