"""Workflow orchestration facade for AutoTEM-Insight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autotem.atomic.atom_detection import detect_atoms
from autotem.core.models import AtomLattice


def _workflow_requests_step(workflow_path: Path | None, step_name: str) -> bool:
    if workflow_path is None or not workflow_path.exists():
        return False

    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
    workflow = data.get("workflow", data)
    steps = workflow.get("steps", [])
    for step in steps:
        if isinstance(step, str) and step == step_name:
            return True
        if isinstance(step, dict):
            name = step.get("name") or step.get("step")
            if name == step_name:
                return True
    return False


@dataclass(slots=True)
class TEMAnalysisSkill:
    """Facade that will coordinate load, calibration, analysis, and reporting steps."""

    def run(
        self,
        input_path: str | Path,
        workflow_path: str | Path | None = None,
        output_dir: str | Path = "outputs/run",
        detect_atoms_step: bool | None = None,
        atom_detection_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the configured analysis workflow.

        This placeholder keeps the package executable while scientific modules are added.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        workflow = Path(workflow_path) if workflow_path is not None else None

        result: dict[str, Any] = {
            "status": "AutoTEM workflow scaffold is ready.",
            "input_path": str(input_path),
            "workflow_path": str(workflow) if workflow else "",
            "output_dir": str(output_dir),
        }

        should_detect_atoms = (
            detect_atoms_step
            if detect_atoms_step is not None
            else _workflow_requests_step(workflow, "detect_atoms")
        )
        if not should_detect_atoms:
            return result

        options = dict(atom_detection_options or {})
        coordinate_unit = str(options.pop("coordinate_unit", "pixel"))
        detection_result = detect_atoms(input_path, **options)
        atom_lattice: AtomLattice = detection_result.to_atom_lattice(
            coordinate_unit=coordinate_unit
        )
        result["atom_lattice"] = atom_lattice
        result["atom_count"] = len(atom_lattice.atoms)
        result["status"] = (
            "AutoTEM workflow scaffold is ready. detect_atoms step executed."
        )
        return result
