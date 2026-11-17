"""Detached release contract checks, including contradictory and incomplete evidence."""

import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from autotem.harness.cli import TOOL_SCHEMAS
from autotem.harness.registry import default_registry
from tools.release import contract_freeze as freeze


@pytest.fixture(scope="module")
def definitions():
    return freeze.validators()


def test_all_published_tool_arguments_match_the_frozen_contract():
    assert freeze.read(freeze.POLICY)["tools"] == TOOL_SCHEMAS


@pytest.mark.smoke
def test_all_registered_operation_contracts_match_the_release_freeze(monkeypatch):
    # Availability is environment-specific and is not part of the frozen contract.
    monkeypatch.setattr(
        "autotem.harness.registry.probe_dependency", lambda _: {"state": "available"}
    )
    rows = default_registry().capabilities()
    actual = {
        row["operation_id"]: {
            "version": row["version"],
            "status": row["implementation_status"],
            "profile": row["profile_id"],
            "input_roles": list(row["input_roles"]),
            "output_roles": list(row["output_roles"]),
            "contract_sha256": freeze.canonical(freeze.role_catalogue(row)),
            "role_schemas_sha256": freeze.canonical([row["input_schema"], row["output_schema"]]),
        }
        for row in rows
    }
    assert len(actual) == len(rows), "Duplicate published operation IDs"
    assert actual == freeze.read(freeze.POLICY)["operations"]


def test_core_reader_scope_and_tool_catalogue_are_explicit():
    policy = freeze.read(freeze.POLICY)
    assert set(policy["tools"]) == {
        "list_capabilities",
        "inspect_input",
        "validate_plan",
        "submit_analysis",
        "get_run_status",
        "cancel_run",
        "list_artifacts",
        "read_artifact",
        "export_report",
    }
    assert all(not s["additionalProperties"] for s in policy["tools"].values())
    assert policy["tools"]["read_artifact"]["properties"]["limit"]["maximum"] == 1000
    assert set(policy["formats"]["core_disk"]) == {".npy", ".tif", ".tiff"}
    assert ".dm4" in policy["formats"]["reader_branches_outside_release_claim"]
    assert policy["formats"]["metadata_only"] == [".emi"]
    assert sum(o["status"] == "implemented" for o in policy["operations"].values()) == 37
    assert sum(o["status"] == "reserved" for o in policy["operations"].values()) == 12


def test_scientific_claims_require_units_or_textual_limit(definitions):
    claim = {"claim": "Calibrated spacing", "artifact_id": "a" * 32, "value": [0.32]}
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(definitions["claim"], claim, "spacing")
    freeze.conform(definitions["claim"], {**claim, "unit": "nm"}, "spacing")
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(definitions["claim"], {**claim, "value": "cubic"}, "phase")
    freeze.conform(
        definitions["claim"], {**claim, "value": "cubic", "limitation": "candidate"}, "phase"
    )


def test_accepted_and_rejected_answers_have_distinct_lineage(definitions):
    row = {
        "response_id": "1" * 32,
        "request_id": "2" * 32,
        "schema_version": "1.0.0",
        "parent_run_id": "3" * 32,
        "child_run_id": None,
        "response": None,
        "validation": {"status": "rejected", "code": "INVALID_INPUT_RESPONSE"},
        "adopted_values": {},
        "created": 1.0,
    }
    freeze.conform(definitions["responseRecord"], row, "rejected")
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(definitions["responseRecord"], {**row, "child_run_id": "4" * 32}, "rejected")
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(
            definitions["responseRecord"], {**row, "validation": {"status": "accepted"}}, "accepted"
        )


def test_protocol_envelope_and_fake_host_success_are_rejected(definitions):
    envelope = {
        "schema_version": "1.0.0",
        "request_id": 1,
        "tool": "validate_plan",
        "ok": False,
        "result": None,
        "error": {"code": "INVALID_PLAN", "message": "No steps", "retryable": False},
    }
    freeze.conform(definitions["protocolEnvelope"], envelope, "envelope")
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(definitions["protocolEnvelope"], {**envelope, "ok": True}, "envelope")
    observation = {
        "schema_version": "1.0.0",
        "execution_layer": "real_host_capture",
        "host": "codex",
        "case_id": "J01",
        "case_verdict": "blocked",
        "capture_stage": "host_preflight",
        "runs": [],
        "tool_trace": [],
        "claims": [],
        "checks": {},
        "reason": "HOST_OFFLINE",
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
        "evidence": dict.fromkeys(
            ("blocker_sha256", "verifier_sha256", "matrix_sha256", "case_record_sha256"), "0" * 64
        ),
    }
    freeze.conform(definitions["hostCase"], observation, "blocked host")
    with pytest.raises(ValueError, match="schema mismatch"):
        freeze.conform(definitions["hostCase"], {**observation, "case_verdict": "passed"}, "host")


def tiny_bundle(path, required_dep=">=1", tag="py3-none-any", requires_python=""):
    root = path / "wheelhouse" / "traditional-core-mcp"
    root.mkdir(parents=True)
    m = {"target": {"python_version": "3.12.10"}, "artifacts": []}
    profile = {
        "wheelhouse": root.relative_to(path).as_posix(),
        "lock": "locks/traditional-core-mcp.txt",
    }
    hashes = {}
    for file_name, pkg, meta in [
        (
            "autotem_insight-1.0-py3-none-any.whl",
            "autotem-insight",
            f"Requires-Dist: support{required_dep}; extra == 'mcp'\n",
        ),
        (
            f"support-1.0-{tag}.whl",
            "support",
            f"Requires-Python: {requires_python}\n" if requires_python else "",
        ),
    ]:
        wheel = root / file_name
        with ZipFile(wheel, "w") as z:
            z.writestr(
                pkg.replace("-", "_") + "-1.0.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {pkg}\nVersion: 1.0\n{meta}\n",
            )
        hashes[pkg] = freeze.sha(wheel)
        m["artifacts"].append(
            {
                "path": wheel.relative_to(path).as_posix(),
                "size": wheel.stat().st_size,
                "sha256": hashes[pkg],
            }
        )
    index = root / "wheelhouse-manifest.json"
    index.write_text(
        json.dumps({"wheels": {Path(a["path"]).name: a["sha256"] for a in m["artifacts"]}})
    )
    m["artifacts"].append(
        {
            "path": index.relative_to(path).as_posix(),
            "size": index.stat().st_size,
            "sha256": freeze.sha(index),
        }
    )
    lock = path / profile["lock"]
    lock.parent.mkdir()
    lock.write_text(
        "".join(
            f"{name}{'[mcp]' if name == 'autotem-insight' else ''}==1.0 --hash=sha256:{h}\n"
            for name, h in hashes.items()
        ),
        encoding="ascii",
    )
    profile["lock_sha256"] = freeze.sha(lock)
    return m, profile, {"wheel_sha256": hashes["autotem-insight"], "lock_sha256": freeze.sha(lock)}


def test_complete_pinned_target_and_wheel_tamper(tmp_path):
    manifest, profile, identity = tiny_bundle(tmp_path)
    assert freeze.release_lock(tmp_path, manifest, profile, identity)["wheel_count"] == 2
    wheel = tmp_path / manifest["artifacts"][1]["path"]
    with wheel.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Wheel/lock mismatch"):
        freeze.release_lock(tmp_path, manifest, profile, identity)


@pytest.mark.parametrize(
    "req,tag,expected",
    [
        (">=2", "py3-none-any", "Unsatisfied target dependency"),
        (">=1", "cp312-cp312-linux_x86_64", "Wheel/lock mismatch"),
    ],
)
def test_matching_file_hash_does_not_certify_wrong_platform_or_dependency(
    tmp_path, req, tag, expected
):
    bundle = tiny_bundle(tmp_path, req, tag)
    with pytest.raises(ValueError, match=expected):
        freeze.release_lock(tmp_path, *bundle)


def test_generic_wheel_that_excludes_exact_python_patch_is_rejected(tmp_path):
    bundle = tiny_bundle(tmp_path, requires_python=">=3.12,!=3.12.10")
    with pytest.raises(ValueError, match="Incompatible Requires-Python"):
        freeze.release_lock(tmp_path, *bundle)


def test_recheck_requires_original_evidence_and_rejects_changed_certificate(tmp_path, monkeypatch):
    certificate = tmp_path / "contract-freeze.json"
    with pytest.raises(ValueError, match="Unavailable or malformed"):
        freeze.recheck(tmp_path, tmp_path, certificate)
    certificate.write_text(json.dumps({"probe_workdir": "../escape", "case_verdict": "passed"}))
    with pytest.raises(ValueError, match="Unsafe capture work directory"):
        freeze.recheck(tmp_path, tmp_path, certificate)
    certificate.write_text(json.dumps({"probe_workdir": "work", "case_verdict": "passed"}))
    monkeypatch.setattr(
        freeze, "assess", lambda *args: {"probe_workdir": "work", "case_verdict": "failed"}
    )
    with pytest.raises(ValueError, match="independently replayed"):
        freeze.recheck(tmp_path, tmp_path, certificate)


def test_artifact_reference_and_duplicate_evidence_keys(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    source = tmp_path / "outside.json"
    source.write_text('{"verdict":"failed","verdict":"passed"}')
    with pytest.raises(ValueError, match="Invalid evidence-relative path"):
        freeze.local(root, "../outside.json")
    with pytest.raises(ValueError, match="Duplicate key"):
        freeze.read(source)


def test_unobserved_host_matrix_can_be_built_without_a_named_evidence_directory(
    tmp_path, definitions
):
    from tools.host_acceptance import index

    src = tmp_path / "source.zip"
    with ZipFile(src, "w") as z:
        for name in ("index.py", "verify.py", "matrix.json"):
            z.write(index.MATRIX_FILE.with_name(name), "tools/host_acceptance/" + name)
    manifest = {
        "release_id": "fixture",
        "profiles": {"traditional-core-mcp": {}},
        "source": {"archive": src.name},
        "artifacts": [],
    }
    (tmp_path / "release-manifest.json").write_text(json.dumps(manifest))
    result = freeze.check_host_matrix(tmp_path, manifest, None, definitions)
    assert result["host_matrix_source"] == "empty_schema_baseline"
    assert result["host_required_variant_counts"] == {
        "passed": 0,
        "failed": 0,
        "blocked": 0,
        "not_evaluated": 81,
    }
    arbitrary = tmp_path / "observations.json"
    matrix = index.build(tmp_path)
    arbitrary.write_text(json.dumps(matrix))
    supplied = freeze.check_host_matrix(tmp_path, manifest, arbitrary, definitions)
    assert supplied["host_matrix_verdict"] == "not_evaluated"
    assert supplied["host_matrix_sha256"] == freeze.sha(arbitrary)
    matrix["hosts"]["codex"]["cases"]["J01"]["variants"][0]["verdict"] = "passed"
    arbitrary.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="differ from their release-bound"):
        freeze.check_host_matrix(tmp_path, manifest, arbitrary, definitions)
    matrix["hosts"]["codex"]["cases"]["J01"]["variants"][0]["summary_sha256"] = "a" * 64
    arbitrary.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="Missing referenced host"):
        freeze.check_host_matrix(tmp_path, manifest, arbitrary, definitions)


def test_frozen_operation_version_and_role_schema_cannot_drift(definitions):
    name = "fixture.measure"
    source = dict.fromkeys(freeze.CATALOGUE_FIELDS, "")
    source.update(
        operation_id=name,
        version="1.0.0",
        profile_id="fixture",
        implementation_status="implemented",
        implementation_source="fixture.measure",
        parameter_schema={"type": "object"},
        input_roles=[],
        output_roles=["result"],
        input_schema={},
        output_schema={"result": {"type": "number"}},
        scientific_accuracy="not_evaluated",
        availability={"state": "available"},
        negative_contract={"verdict": "passed", "observed_code": "INVALID_PARAMETERS"},
    )
    normative = {
        "version": "1.0.0",
        "status": "implemented",
        "profile": "fixture",
        "input_roles": [],
        "output_roles": ["result"],
        "contract_sha256": freeze.canonical(freeze.role_catalogue(source)),
        "role_schemas_sha256": freeze.canonical([source["input_schema"], source["output_schema"]]),
    }
    policy = {"tools": {}, "protocol_version": "1.0.0", "operations": {name: normative}}
    row = deepcopy(source)
    cap = {
        "schema_version": "1.0.0",
        "request_id": "cap",
        "tool": "list_capabilities",
        "ok": True,
        "error": None,
        "result": {"envelope_version": "1.0.0", "operations": [row]},
    }
    probe = {
        "mcp": {
            "server": "autotem-harness",
            "tools": {},
            "protocol": "fixture",
            "capabilities": cap,
        }
    }
    assert freeze.check_catalogue(policy, probe, {name: source}, definitions)["implemented"] == 1
    row["output_schema"]["result"] = {"type": "string"}
    with pytest.raises(ValueError, match="Role schema mismatch"):
        freeze.check_catalogue(policy, probe, {name: source}, definitions)
    row["version"] = "2.0.0"
    with pytest.raises(ValueError, match="Changed operation contract"):
        freeze.check_catalogue(policy, probe, {name: source}, definitions)
