"""Exercise the J15 field reader without manufacturing a host session."""

import json

import pytest

from tools.host_acceptance import verify


@pytest.mark.parametrize(
    ("variant", "operation", "code", "accepted"),
    [
        ("timeout", "quality.assess", "TIMEOUT", False),
        ("timeout", "simulation.diffraction", "TIMEOUT", False),
        ("timeout", "diffraction.match", "SIMULATION_TIMEOUT", False),
        ("timeout", "simulation.diffraction", "SIMULATION_TIMEOUT", True),
        ("broken_dependency", "simulation.diffraction", "DEPENDENCY_BROKEN", True),
        ("missing_dependency", "input.load", "DEPENDENCY_UNAVAILABLE", False),
    ],
)
def test_j15_binds_the_failure_to_the_backend_step(variant, operation, code, accepted):
    step = {"operation_id": operation, "status": "failed", "error_code": code}
    run = {"status": "failed", "error_code": code, "steps": [step]}
    args = ("J15", [], {"r": run}, [], {}, variant)
    if accepted:
        verify.check_case(*args)
    else:
        with pytest.raises(verify.EvidenceError, match="simulation_fault_not_observed"):
            verify.check_case(*args)
    if variant == "timeout" and code == "TIMEOUT":
        verify.check_case("J11", [], {"r": run}, [], {}, "timeout")


@pytest.mark.parametrize(
    ("variant", "operation", "source", "dependency", "accepted"),
    [
        (
            "missing_dependency",
            "simulation.diffraction",
            "simulation.diffraction",
            "diffsims",
            True,
        ),
        ("broken_dependency", "simulation.diffraction", "simulation.diffraction", "diffsims", True),
        ("missing_dependency", "quality.assess", "simulation.diffraction", "diffsims", False),
        ("missing_dependency", "simulation.diffraction", "input.load", "diffsims", False),
        (
            "broken_dependency",
            "simulation.diffraction",
            "simulation.diffraction",
            "unrelated",
            False,
        ),
    ],
)
def test_j15_accepts_a_bound_pre_submission_refusal(
    variant, operation, source, dependency, accepted
):
    code = "DEPENDENCY_BROKEN" if variant == "broken_dependency" else "DEPENDENCY_UNAVAILABLE"
    call = {
        "tool": "validate_plan",
        "arguments": {"plan": {"steps": [{"operation_id": operation}]}},
        "response": {
            "ok": False,
            "error": {"code": code, "message": source, "details": {"dependency": dependency}},
        },
    }
    if accepted:
        verify.check_case("J15", [call], {}, [], {}, variant)
    else:
        with pytest.raises(verify.EvidenceError, match="simulation_fault_not_observed"):
            verify.check_case("J15", [call], {}, [], {}, variant)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "no_match",
        "wrong_input",
        "inactive_backend",
        "fallback_sites",
        "no_array",
        "inactive_match",
    ],
)
def test_executed_j15_requires_the_actual_simulation_to_feed_the_match(tmp_path, change):
    sim = {
        "provenance": {"adapter": {"adapter": "diffsims", "active": change != "inactive_backend"}},
        "patterns": [
            {
                "simulation_parameters": {
                    "engine": "diffsims",
                    "atom_site_count": 1,
                    "structure_source": "single_atom_fallback"
                    if change == "fallback_sites"
                    else "candidate_atom_sites",
                }
            }
        ],
    }
    match = {
        "provenance": {
            "phase_matching": {
                "method": "diffsims_template_spot_match",
                "active": change != "inactive_match",
            }
        }
    }
    (tmp_path / "simulation.json").write_text(json.dumps(sim))
    (tmp_path / "match.json").write_text(json.dumps(match))
    artifacts = {
        i: {"role": role, "step_id": step, "file": file, "kind": kind, "execution_usable": True}
        for i, role, step, file, kind in (
            ("s", "simulation", "sim", "simulation.json", "json"),
            ("a", "simulated_pattern_0", "sim", "image.npy", "array"),
            ("d", "diffraction", "measured", "measured.json", "json"),
            ("m", "diffraction", "match", "match.json", "json"),
        )
    }
    if change == "no_array":
        del artifacts["a"]
    steps = [{"step_id": "sim", "operation_id": "simulation.diffraction", "status": "completed"}]
    if change != "no_match":
        steps.append(
            {
                "step_id": "match",
                "operation_id": "diffraction.match",
                "status": "completed",
                "consumed_artifacts": {
                    "diffraction": "d",
                    "simulation": "m" if change == "wrong_input" else "s",
                },
            }
        )
    run = {
        "status": "completed",
        "error_code": None,
        "steps": steps,
        "artifacts": artifacts,
        "_directory": tmp_path,
    }
    if change == "none":
        verify.check_case("J15", [], {"r": run}, [], {}, "executed")
    else:
        with pytest.raises(
            verify.EvidenceError,
            match="missing_linked_diffsims_match|requested_operations_or_results_missing",
        ):
            verify.check_case("J15", [], {"r": run}, [], {}, "executed")
