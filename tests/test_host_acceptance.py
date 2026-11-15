"""Contract fixtures for the host evidence verifier. No real host journey is claimed."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.host_acceptance import verify as verifier


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def spacing_run(tmp_path):
    steps = [
        {"step_id": op, "operation_id": op, "status": "completed"}
        for op in verifier.OPERATIONS["J01"]
    ]
    run = {
        "run_id": "r",
        "modality": "HRTEM",
        "status": "completed",
        "_directory": tmp_path,
        "steps": steps,
        "artifacts": {},
        "jobs": [],
        "physical_measurements": [],
    }

    def artifact(role, content, step="diffraction.metrics", suffix="json"):
        aid = f"{len(run['artifacts']) + 1:032x}"
        path = tmp_path / f"{aid}.{suffix}"
        path.write_text(content, encoding="utf-8")
        run["artifacts"][aid] = {
            "role": role,
            "step_id": step,
            "file": path.name,
            "sha256": verifier.digest(path),
            "execution_usable": step != "_report",
        }
        return aid

    data = [
        {"value": val, "unit": "nm", "uncertainty": err}
        for val, err in [(0.15999999999999, None), (0.2448, 0.003)]
    ]
    metric = artifact(
        "diffraction", json.dumps({"d_spacings": data, "provenance": {"physical": True}})
    )
    run["physical_measurements"].append({"artifact_id": metric, "units": ["nm"]})
    table = artifact(
        "d_spacing_table",
        "value,unit,uncertainty\n0.15999999999999,nm,\n0.2448,nm,0.003\n",
        suffix="csv",
    )
    g_table = artifact(
        "g_vector_table", "magnitude\n\"{'value': 6.25, 'unit': '1/nm'}\"\n", suffix="csv"
    )
    manifest = artifact(
        "report_manifest",
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": table,
                        "sha256": run["artifacts"][table]["sha256"],
                        "execution_usable": True,
                    },
                ]
            }
        ),
        "_report",
    )
    html = artifact("report_html", f'<a href="{metric}.json">spacing</a>', "_report", "html")
    peak = artifact("peak_table", "x,y\n2,4\n", "diffraction.fft", "csv")
    return run, {
        "metric": metric,
        "table": table,
        "g": g_table,
        "manifest": manifest,
        "html": html,
        "peak": peak,
    }


@pytest.mark.parametrize(
    ("statement", "source"),
    [
        ("晶格间距约为 **160 pm**。", "metric"),
        ("The interplanar distances are 1.60 and 2.45 Å.", "table"),
        ("Espacement : 0,160 nm.", "g"),
        ("d = 2.448e-1 ± 0.003 nm", "manifest"),
        ("|peak|d-spacing (µm)|\n|---|---|\n|1|0.000160|", "html"),
    ],
)
def test_returned_spacing_from_registered_result_table_or_report(spacing_run, statement, source):
    run, ids = spacing_run
    ref = ids[source]
    answer = f"{statement}\nSource: {ref}"
    checked = verifier.check_spacing_answer(
        answer, [{"quote": statement, "artifact_ids": [ref]}], {"r": run}, {ref}
    )
    assert checked["quantities"] and checked["source_artifacts"][ref]["run_id"] == "r"
    assert checked["source_artifacts"][ref]["citation_sha256"] == run["artifacts"][ref]["sha256"]


@pytest.mark.parametrize(
    ("statement", "source", "reason"),
    [
        ("间距 9.876 nm", "metric", "spacing_value_or_unit"),
        ("d = 0.160 pm", "table", "spacing_value_or_unit"),
        (
            "The measured spacing is 0.160 nm. Its uncertainty is ±0.003 nm.",
            "metric",
            "spacing_uncertainty",
        ),
        ("Spacings 0.160 nm and 4.999 nm.", "manifest", "spacing_value_or_unit"),
        ("| d (pm) |\n|---|\n| 0.160 |", "html", "spacing_value_or_unit"),
        ("d = 0.16 nm", "peak", "spacing_source_not_cited"),
        (
            "Only the FFT and the report are available.",
            "table",
            "spacing_value_and_unit_not_observed",
        ),
    ],
)
def test_spacing_claim_rejects_mismatch_and_requires_quantitative_evidence(
    spacing_run, statement, source, reason
):
    run, ids = spacing_run
    ref = ids[source]
    with pytest.raises(verifier.EvidenceError, match=reason):
        verifier.check_spacing_answer(
            f"{statement}\n{ref}", [{"quote": statement, "artifact_ids": [ref]}], {"r": run}, {ref}
        )


def test_explicit_spacing_source_cannot_be_borrowed_from_another_claim_or_run(spacing_run):
    run, ids = spacing_run
    answer = f"The measured spacing is 0.160 nm. {ids['peak']}\nSource: {ids['metric']}"
    claims = [
        {"quote": answer.splitlines()[0], "artifact_ids": [ids["peak"]]},
        {"quote": answer.splitlines()[1], "artifact_ids": [ids["metric"]]},
    ]
    with pytest.raises(verifier.EvidenceError, match="spacing_claim_has_no_quantitative_source"):
        verifier.check_spacing_answer(answer, claims, {"r": run}, set(ids.values()))
    other = "f" * 32
    with pytest.raises(verifier.EvidenceError, match="unreturned_artifact"):
        verifier.check_spacing_answer(answer + " " + other, claims, {"r": run}, set(ids.values()))


@pytest.fixture
def coverage_capture(tmp_path):
    import numpy as np

    def make_run(name, counts):
        directory = tmp_path / name
        directory.mkdir()
        run = {
            "run_id": name,
            "status": "completed",
            "_directory": directory,
            "steps": [],
            "artifacts": {},
            "strain_validity": [],
        }
        rows = []

        def register(role, path, step):
            aid = path.stem
            run["artifacts"][aid] = {
                "role": role,
                "file": path.name,
                "step_id": step,
                "sha256": verifier.digest(path),
                "execution_usable": step != "_report",
            }
            return aid

        for sequence, valid in enumerate(counts):
            step = f"g{sequence}"
            run["steps"].append(
                {"step_id": step, "operation_id": "strain.gpa", "status": "completed"}
            )
            mask = (np.arange(2000) < valid).reshape(40, 50)
            path = directory / f"{name[:28]}{sequence:02x}01.npy"
            np.save(path, mask)
            mask_id = register("gpa.quality_mask", path, step)
            run["strain_validity"].append(
                {"step_id": step, "valid_count": int(mask.sum()), "total_count": mask.size}
            )
            gpa = directory / f"{name[:28]}{sequence:02x}02.json"
            write_json(gpa, {"provenance": {"quality_mask": {"uri": str(path)}}})
            rows.append(
                {
                    "artifact_id": register("gpa", gpa, step),
                    "operation_id": "strain.gpa",
                    "step_id": step,
                    "validity_mask": {"artifact_id": mask_id},
                }
            )
        manifest = directory / f"{name[:28]}ffff.json"
        write_json(
            manifest,
            {
                "run": {"run_id": name},
                "strain_results": {"all": rows},
                "artifacts": [dict(artifact_id=i, **a) for i, a in run["artifacts"].items()],
            },
        )
        register("report_manifest", manifest, "_report")
        html = directory / f"{name[:28]}fffe.html"
        html.write_text(
            " ".join(f'<a href="{r["validity_mask"]["artifact_id"]}.npy">mask</a>' for r in rows)
        )
        register("report_html", html, "_report")
        return run

    return make_run


def check_coverage(run, text, roles=("gpa.quality_mask",)):
    refs = [i for i, a in run["artifacts"].items() if a["role"] in roles]
    answer = text + " " + " ".join(refs)
    compared = {(run["run_id"], s["step_id"]) for s in run["steps"]}
    return verifier.check_gpa_coverage_answer(
        answer, [{"quote": answer, "artifact_ids": refs}], {run["run_id"]: run}, set(refs), compared
    )


@pytest.mark.parametrize(
    ("text", "role", "quantities"),
    [
        (
            "全图有效像素占比约为 73.55%，参考框内有效覆盖率 100%，εxx 为 2%。",
            "gpa.quality_mask",
            1,
        ),
        ("有效像素占全图的 74%（1,471 / 2,000 个像素）", "gpa", 2),
        ("参考框内的有效覆盖率 100%，整幅图像有效覆盖率 73.6％", "report_manifest", 1),
        ("Whole-image valid pixel coverage is 73.55% (1,471 of 2,000 pixels).", "report_html", 2),
        ("73.55% of all pixels are valid.", "gpa.quality_mask", 1),
        ("全图共有 1471/2000 个像素有效", "report_manifest", 1),
        ("全图有效像素占比0.7355", "gpa.quality_mask", 1),
        ("全图有 73.55% 的像素有效", "gpa", 1),
        ("## 全图\n有效覆盖率 74%\n## 参考框\n有效覆盖率 100%", "gpa.quality_mask", 1),
    ],
)
def test_j03_answer_matches_the_registered_full_image_mask(
    coverage_capture, text, role, quantities
):
    run = coverage_capture("a" * 32, [1471])
    result = check_coverage(run, text, (role,))
    mask = next(iter(result["masks"]))
    assert result["masks"][mask] == {
        "run_id": run["run_id"],
        "step_id": "g0",
        "mask_sha256": run["artifacts"][mask]["sha256"],
        "valid_pixels": 1471,
        "total_pixels": 2000,
    }
    assert len(result["quantities"]) == quantities


@pytest.mark.parametrize(
    "text",
    [
        "有效覆盖率 100%",
        "GPA 掩膜显示全图覆盖率为 100%",
        "全图所有像素均有效（100%）",
        "全图有效覆盖率 73.54%",
        "有效像素 1471/1999",
        "Whole-image valid pixel fraction is 1.00",
        "全图有效像素占比 1471/2000 (100%)",
        "全图有效覆盖率 74%。另一项全图有效覆盖率 100%。",
    ],
)
def test_j03_answer_rejects_explicit_wrong_numeric_coverage(coverage_capture, text):
    with pytest.raises(verifier.EvidenceError, match="answer_gpa_coverage_differs"):
        check_coverage(coverage_capture("a" * 32, [1471]), text)


@pytest.mark.parametrize(
    "text",
    [
        "参考区有效覆盖率 100%，全图应变 73.55%",
        "Valid pixel coverage is 100% within the ROI.",
        "有效覆盖率 100%（参考框内）。",
        "参考框：\n有效像素占比 100%",
        "invalid pixel coverage is 26.45%.",
        "全图 εxx 73.55%，不确定度 100%，掩膜已登记。",
        "有效覆盖率在 60% 至 90% 之间",
    ],
)
def test_j03_roi_or_unlabeled_percentage_is_not_a_whole_image_measurement(coverage_capture, text):
    with pytest.raises(verifier.NotEvaluated, match="whole_image_gpa_coverage"):
        check_coverage(coverage_capture("a" * 32, [1471]), text)


def test_j03_coverage_binds_one_compared_step_and_actual_report(coverage_capture):
    first = coverage_capture("a" * 32, [1471, 500])
    second = coverage_capture("b" * 32, [2000])
    runs = {r["run_id"]: r for r in (first, second)}
    m1, m2 = (i for i, a in first["artifacts"].items() if a["role"] == "gpa.quality_mask")
    report = next(i for i, a in first["artifacts"].items() if a["role"] == "report_manifest")
    other = next(i for i, a in second["artifacts"].items() if a["role"] == "gpa.quality_mask")
    compared = {(first["run_id"], "g0"), (second["run_id"], "g0")}

    def claim(rows):
        claims = [{"quote": f"{s} {' '.join(ids)}", "artifact_ids": ids} for s, ids in rows]
        return verifier.check_gpa_coverage_answer(
            "\n".join(c["quote"] for c in claims),
            claims,
            runs,
            {i for c in claims for i in c["artifact_ids"]},
            compared,
        )

    assert (
        len(claim([("有效覆盖率 74%", [m1, report]), ("有效覆盖率 100%", [other])])["masks"]) == 2
    )
    with pytest.raises(verifier.NotEvaluated, match="ambiguous_mask"):
        claim([("全图有效覆盖率 74%", [report])])
    with pytest.raises(verifier.EvidenceError, match="citations_disagree"):
        claim([("有效覆盖率 100%", [other, report])])
    with pytest.raises(verifier.EvidenceError, match="step_was_not_compared"):
        claim([("有效覆盖率 25%", [m2, report])])
    with pytest.raises(verifier.EvidenceError, match="differs_from_registered_mask"):
        claim([("有效覆盖率 100%", [m1]), ("有效覆盖率 100%", [other])])
    answer = f"全图有效覆盖率 100%。两次运行的掩膜编号为 {m1} 和 {other}。"
    with pytest.raises(verifier.NotEvaluated, match="no_local_mask_reference"):
        verifier.check_gpa_coverage_answer(
            answer,
            [{"quote": "全图有效覆盖率 100%", "artifact_ids": [other]}],
            runs,
            {m1, other},
            compared,
        )
    path = first["_directory"] / first["artifacts"][report]["file"]
    data = json.loads(path.read_text("utf-8"))
    next(a for a in data["artifacts"] if a["artifact_id"] == m1)["sha256"] = "0" * 64
    write_json(path, data)
    with pytest.raises(verifier.EvidenceError, match="report_mask_registration_mismatch"):
        claim([("有效覆盖率 74%", [m1, report])])
    data["run"]["run_id"] = second["run_id"]
    write_json(path, data)
    with pytest.raises(verifier.EvidenceError, match="gpa_report_run_mismatch"):
        claim([("有效覆盖率 74%", [report])])


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("none", None),
        ("wrong_count", "differs_from_registered_mask"),
        ("wrong_percent", "differs_from_registered_mask"),
        ("different_pixels", "identical_gpa_masks_with_different_content"),
        ("different_run", "citation_mixes_runs"),
        ("foreign_artifact", "citation_mixes_runs"),
        ("reversed_steps", "not_first_and_repeat"),
        ("different_settings", "not_first_and_repeat"),
        ("other_path", "mask_link_path_mismatch"),
        ("detached", "no_linked_citation"),
        ("other_sample", "no_linked_citation"),
        ("next_section", "no_linked_citation"),
        ("one_link", "no_linked_citation"),
        ("unreturned", "no_linked_citation"),
        ("missing_identity_claim", "no_linked_citation"),
    ],
)
def test_adjacent_citations_for_a_repeated_gpa_result(coverage_capture, change, error):
    import numpy as np

    def pair(run_id):
        run = coverage_capture(run_id, [1471, 1471])
        raw = run["_directory"] / "image.npy"
        np.save(raw, np.arange(2000, dtype="f4").reshape(40, 50))
        run["artifacts"]["raw-id"] = {
            "role": "raw",
            "step_id": "load",
            "file": raw.name,
            "sha256": verifier.digest(raw),
            "execution_usable": True,
        }
        run["_request"] = {
            "steps": [
                {"id": s["step_id"], "parameters": {"image_role": "raw"}} for s in run["steps"]
            ]
        }
        for measured in run["strain_validity"]:
            measured["image_artifact_id"] = "raw-id"
        masks = [i for i, a in run["artifacts"].items() if a["role"] == "gpa.quality_mask"]
        return run, masks

    run, masks = pair("a" * 32)
    runs = {run["run_id"]: run}
    paths = [run["_directory"] / run["artifacts"][i]["file"] for i in masks]
    if change == "different_pixels":
        data = np.load(paths[1], allow_pickle=False)
        shifted = np.roll(data, 1, axis=0)
        assert int(data.sum()) == int(shifted.sum()) and not np.array_equal(data, shifted)
        np.save(paths[1], shifted)
        run["artifacts"][masks[1]]["sha256"] = verifier.digest(paths[1])
    if change == "different_settings":
        run["_request"]["steps"][1]["parameters"]["quality_percentile"] = 50
    if change in {"different_run", "foreign_artifact"}:
        other, others = pair("b" * 32)
        runs[other["run_id"]] = other
        if change == "different_run":
            paths[1] = other["_directory"] / other["artifacts"][others[1]]["file"]
    if change == "reversed_steps":
        paths.reverse()
    if change == "other_path":
        paths[0] = paths[0].parent / "unregistered" / paths[0].name
    count = "1,470" if change == "wrong_count" else "1,471"
    percent = "100" if change == "wrong_percent" else "73.55"
    result = f"两次诊断性 GPA 均保留 **{count} / 2,000 个有效像素（{percent}%）**。"
    if change != "missing_identity_claim":
        result += "两次掩膜完全相同。"
    cite = f"下表统计首次 GPA 掩膜内像素。[首次掩膜]({paths[0]})"
    if change != "one_link":
        cite += f" · [复算掩膜]({paths[1]})"
    if change in {"other_sample", "next_section"}:
        cite = ("另一个样本：" if change == "other_sample" else "## Other section\n") + cite
    if change == "foreign_artifact":
        cite += " " + next(i for i, a in other["artifacts"].items() if a["role"] == "report_html")
    gap = "\n\n中间另有说明。\n\n" if change == "detached" else "\n\n"
    final = (
        result
        + gap
        + cite
        + "\n\n框内 **1,024 / 1,024** 个像素有效，图像最外圈 **0 / 284** 个像素有效。"
    )
    ids = {p.stem for p in paths if p.stem in cite}
    if change == "unreturned":
        ids.discard(paths[1].stem)
    compared = {(r["run_id"], s["step_id"]) for r in runs.values() for s in r["steps"]}
    args = (final, [{"quote": cite, "artifact_ids": sorted(ids)}], runs, ids, compared)
    if error:
        exc = verifier.NotEvaluated if error == "no_linked_citation" else verifier.EvidenceError
        with pytest.raises(exc, match=error):
            verifier.check_gpa_coverage_answer(*args)
    else:
        proof = verifier.check_gpa_coverage_answer(*args)
        assert set(proof["masks"]) == set(masks)
        assert [q["as_displayed"] for q in proof["quantities"]] == ["1,471 / 2,000", "73.55%"]
        assert all(q["first_and_repeat_mask_ids"] == masks for q in proof["quantities"])


def test_j03_report_and_result_must_link_to_the_registered_mask(coverage_capture):
    run = coverage_capture("a" * 32, [1471])
    arts = run["artifacts"]
    mask = next(i for i, a in arts.items() if a["role"] == "gpa.quality_mask")
    gpa = next(a for a in arts.values() if a["role"] == "gpa")
    write_json(
        run["_directory"] / gpa["file"], {"provenance": {"quality_mask": {"uri": "other.npy"}}}
    )
    html = next(a for a in arts.values() if a["role"] == "report_html")
    (run["_directory"] / html["file"]).write_text(f'<!-- <a href="{mask}.npy">mask</a> -->')
    for role in ("gpa", "report_html"):
        with pytest.raises(verifier.NotEvaluated, match="claim_has_no_linked_mask"):
            check_coverage(run, "有效覆盖率 74%", (role,))


def test_j02_requires_same_raw_refinement_and_physical_distances(tmp_path):
    products = [
        ("input.load", "raw", {}),
        ("atomic.detect", "atoms", {"raw": "raw"}),
        ("atomic.refine", "refined_atoms", {"raw": "raw", "atoms": "atoms"}),
        ("atomic.distances", "atom_distances", {"refined_atoms": "refined_atoms"}),
        ("atomic.intensity", "atomic_intensity", {"raw": "raw", "refined_atoms": "refined_atoms"}),
    ]
    steps = [
        {"step_id": op, "operation_id": op, "status": "completed", "consumed_artifacts": used}
        for op, _, used in products
    ]
    artifacts = {
        role: {"step_id": op, "role": role, "file": role, "execution_usable": True}
        for op, role, _ in products
    }
    lattice = {
        "atoms": [{"y": 1, "x": 1}],
        "provenance": {"raw_refinement": {"point_status": ["accepted"]}},
    }
    write_json(tmp_path / "refined_atoms", lattice)
    measurement = {"artifact_id": "atom_distances", "units": ["nm"]}
    run = {
        "status": "completed",
        "error_code": None,
        "modality": "HAADF-STEM",
        "_directory": tmp_path,
        "steps": steps,
        "artifacts": artifacts,
        "physical_measurements": [measurement],
    }

    def check():
        verifier.check_case("J02", [], {"r": run}, [], {})

    check()
    for step, role in [
        (steps[2], "atoms"),
        (steps[3], "refined_atoms"),
        (steps[4], "raw"),
        (steps[4], "refined_atoms"),
    ]:
        step["consumed_artifacts"][role] = "unrelated"
        with pytest.raises(verifier.EvidenceError, match="raw_refined_physical"):
            check()
        step["consumed_artifacts"][role] = role
    run["physical_measurements"] = []
    with pytest.raises(verifier.EvidenceError, match="raw_refined_physical"):
        check()
    run["physical_measurements"] = [measurement]
    lattice["provenance"]["raw_refinement"]["point_status"] = []
    write_json(tmp_path / "refined_atoms", lattice)
    with pytest.raises(verifier.EvidenceError, match="raw_refined_physical"):
        check()


def test_gpa_geometry_binds_full_image_and_requested_reference(tmp_path):
    import numpy as np

    mask = np.ones((6, 8), dtype=bool)
    mask[0] = False
    values = np.where(mask, 0.25, np.nan)
    files = {}
    for name in (
        "quality_mask",
        "displacement_x",
        "displacement_y",
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "rotation_xy",
        "uncertainty",
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, mask if name == "quality_mask" else values)
        files["g", f"gpa.{name}"] = path
    files["g", "gpa"] = tmp_path / "gpa.json"
    step = {
        "step_id": "g",
        "operation_id": "strain.gpa",
        "result": json.dumps({"consumed_artifacts": {"processed": "filtered"}}),
    }
    image = {"role": "processed", "kind": "array", "execution_usable": True, "shape": [6, 8]}
    config = {"image_role": "processed", "reference_region": [0, 4, 0, 3]}
    source = {
        "reference_region": config["reference_region"],
        "provenance": {"complete_arrays": True},
    }

    def check():
        write_json(files["g", "gpa"], source)
        return verifier.check_strain_fields(files, step, config, {"filtered": image})

    result = check()
    assert result["image_artifact_id"] == "filtered"
    assert (result["valid_count"], result["total_count"]) == (40, 48)
    assert result["reference_coverage"]["g1"] == {
        "box": [0, 4, 0, 3],
        "valid_count": 9,
        "total_count": 12,
    }
    image["shape"] = [12, 16]
    with pytest.raises(verifier.EvidenceError, match="do_not_cover_consumed_image"):
        check()
    image["shape"] = [6, 8]
    config["image_role"] = "raw"
    with pytest.raises(verifier.EvidenceError, match="do_not_cover_consumed_image"):
        check()
    config["image_role"] = "processed"
    source["reference_region"] = [0, 4, 1, 4]
    with pytest.raises(verifier.EvidenceError, match="reference_differs_from_request"):
        check()
    for box in ([0, 60, 0, 80], [-3, 6, 0, 8], [0, True, 0, 8], [4, 3, 1, 5]):
        source["reference_region"] = config["reference_region"] = box
        with pytest.raises(verifier.EvidenceError, match="invalid_gpa_reference_bounds"):
            check()

    source["reference_region"] = config["reference_region"] = None
    config["absolute_reference"] = source["provenance"]["absolute_reference_accepted"] = True
    assert check()["reference_coverage"] == {}
    box1, box2 = [0, 5, 0, 3], [1, 4, 2, 5]
    config["reference_regions"] = [box1, box2]
    refs = source["provenance"]["reference_regions"] = {"g1": box1, "g2": box2}
    assert check()["reference_coverage"]["g2"]["valid_count"] == 9
    refs["g1"], refs["g2"] = box2, box1
    with pytest.raises(verifier.EvidenceError, match="reference_differs_from_request"):
        check()
    refs["g1"], refs["g2"] = box1, box2
    box2[1] = 3
    with pytest.raises(verifier.EvidenceError, match="reference_without_valid_pixels"):
        check()


@pytest.mark.parametrize("separate_run", [False, True])
def test_j03_requires_repeat_of_measured_fields(tmp_path, separate_run):
    config = {"g_vectors": [[0.125, 0], [0, 0.125]], "reference_region": [16, 48, 16, 48]}
    measurement = {
        "status": "completed",
        "error_code": None,
        "_directory": tmp_path,
        "steps": [
            {
                "step_id": "g",
                "operation_id": "strain.gpa",
                "status": "completed",
                "consumed_artifacts": {"raw": "image"},
            }
        ],
        "_request": {"steps": [{"id": "g", "parameters": config}]},
        "artifacts": {
            k: {"role": k, "step_id": "g", "execution_usable": True} for k in verifier.ROLES["J03"]
        },
        "source_sha256": "synthetic-source",
        "run_id": "gpa",
    }
    measurement["artifacts"]["image"] = {
        "role": "raw",
        "sha256": "raw-hash",
        "execution_usable": True,
    }
    runs = {"gpa": measurement}
    with pytest.raises(verifier.EvidenceError, match="same_parameter_gpa_comparison_missing"):
        verifier.check_case("J03", [], runs, [], {})

    if separate_run:
        repeat = {
            **measurement,
            "run_id": "repeat",
            "steps": [],
            "_request": {"steps": []},
            "artifacts": {"other_id": dict(measurement["artifacts"]["image"])},
        }
        runs["repeat"] = repeat
    else:
        repeat = measurement
    raw = "other_id" if separate_run else "image"
    repeat["steps"].append(
        {
            "step_id": "compare",
            "operation_id": "strain.gpa_compare",
            "status": "completed",
            "consumed_artifacts": {"raw": raw},
        }
    )
    parameters = {"reference_config": dict(config), "candidate_config": dict(config)}
    repeat["_request"]["steps"].append({"id": "compare", "parameters": parameters})
    artifact = {
        "role": "gpa_comparison",
        "step_id": "compare",
        "execution_usable": True,
        "file": "compare.json",
    }
    repeat["artifacts"]["comparison"] = artifact
    result = {
        "config_a": config,
        "config_b": dict(config),
        "quality": {"intersection_fraction": 0.62},
        "diagnosis": {"configs_aligned": True, "max_strain_mae_percent": 0.0},
    }

    def store():
        write_json(tmp_path / "compare.json", result)
        artifact["sha256"] = verifier.digest(tmp_path / "compare.json")

    store()
    verifier.check_case("J03", [], runs, [], {})
    result["quality"]["intersection_fraction"] = 0.0
    store()
    with pytest.raises(verifier.EvidenceError, match="same_parameter_gpa_comparison_missing"):
        verifier.check_case("J03", [], runs, [], {})
    result["quality"]["intersection_fraction"] = 0.62
    result["config_a"] = result["config_b"] = {**config, "g_vectors": [[0, 0.1], [0.1, 0]]}
    store()
    with pytest.raises(verifier.EvidenceError, match="same_parameter_gpa_comparison_missing"):
        verifier.check_case("J03", [], runs, [], {})
    result["config_a"] = result["config_b"] = config
    store()
    parameters["candidate_config"]["reference_region"] = [20, 40, 20, 40]
    with pytest.raises(verifier.EvidenceError, match="same_parameter_gpa_comparison_missing"):
        verifier.check_case("J03", [], runs, [], {})
    if separate_run:
        parameters["candidate_config"] = dict(config)
        repeat["artifacts"][raw]["sha256"] = "different-bytes"
        with pytest.raises(verifier.EvidenceError, match="same_parameter_gpa_comparison_missing"):
            verifier.check_case("J03", [], runs, [], {})


def sample(tmp_path):
    record = tmp_path / "case.json"
    verifier.main(["init", str(tmp_path), "--host", "codex", "--case", "J10"])
    spec = json.loads(record.read_text("utf-8"))
    now = datetime.now(timezone.utc)
    spec.update(
        observation="recorded",
        version_file="version.txt",
        host_version="1.2.3",
        started_at=now.isoformat(),
        ended_at=(now + timedelta(minutes=1)).isoformat(),
        turns=[{"trace": "host.jsonl", "prompt": "user.txt"}],
    )
    write_json(record, spec)
    (tmp_path / "version.txt").write_text("codex 1.2.3")
    (tmp_path / "user.txt").write_text("Use the reserved analysis capability.", encoding="utf-8")
    answer = {
        "schema_version": "1.0.0",
        "tool": "validate_plan",
        "ok": False,
        "result": None,
        "error": {"code": "DEFERRED_THIS_PHASE", "message": "reserved"},
    }
    events = [
        {"type": "thread.started", "thread_id": "test-only"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "tool": "validate_plan",
                "server": "autotem",
                "arguments": {"plan": {"steps": [{"id": "ai", "operation_id": "ai.denoise"}]}},
                "result": {"content": [{"type": "text", "text": json.dumps(answer)}]},
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "The requested capability is reserved. FFT is available.",
            },
        },
        {"type": "turn.completed"},
    ]
    trace = tmp_path / "host.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return record, spec, trace, events


@pytest.mark.parametrize(
    "order,reason",
    [
        ((0, 1, 4, 2, 3), "host_item_outside_turn"),
        ((0, 2, 1, 3, 4), "host_item_outside_turn"),
        ((1, 0, 2, 3, 4), "host_turn_event_out_of_order"),
        ((0, 1, 2, 2, 3, 4), "duplicate_host_item_completion"),
    ],
)
def test_codex_discards_out_of_turn_or_repeated_events(tmp_path, order, reason):
    record, _, trace, events = sample(tmp_path)
    trace.write_text("\n".join(json.dumps(events[i]) for i in order), encoding="utf-8")
    report = verifier.verify(record, root=tmp_path)
    assert (report["case_verdict"], report["reason"]) == ("failed", reason)


@pytest.mark.parametrize("key", ["structuredContent", "structured_content"])
def test_visible_harness_response_copies_agree(key):
    response = {
        "tool": "read_artifact",
        "schema_version": "1.0.0",
        "ok": True,
        "result": {"rows": [[True, 2]]},
        "error": None,
    }
    payload = {key: response}
    assert verifier.envelope("read_artifact", payload) == response
    displayed = {**response, "result": {"rows": [[True, 2.0]]}}
    payload["content"] = [{"type": "text", "text": json.dumps(displayed, sort_keys=True)}]
    assert verifier.envelope("read_artifact", payload) == response
    displayed["result"]["rows"][0][0] = 1
    payload["content"][0]["text"] = json.dumps(displayed)
    with pytest.raises(verifier.EvidenceError, match="conflicting_harness_response_copies"):
        verifier.envelope("read_artifact", payload)


def test_distinct_codex_polls_can_repeat_the_same_call(tmp_path):
    record, _, trace, events = sample(tmp_path)
    repeated = {**events[2], "item": {**events[2]["item"], "id": "item_2"}}
    events.insert(3, repeated)
    events.insert(1, {"type": "item.completed", "item": {"id": "startup", "type": "error"}})
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    report = verifier.verify(record, root=tmp_path)
    assert report["checks"]["case_execution"] == "passed"


@pytest.mark.parametrize(
    "probe,method",
    [
        ("list_mcp_resources", "resources/list"),
        ("list_mcp_resource_templates", "resources/templates/list"),
    ],
)
def test_codex_resource_discovery_is_visible_but_never_counts_as_a_domain_call(
    tmp_path, probe, method
):
    _, _, trace, events = sample(tmp_path)
    start = {
        "id": "protocol",
        "type": "mcp_tool_call",
        "server": "autotem",
        "tool": probe,
        "arguments": {"server": "autotem"},
        "status": "in_progress",
        "result": None,
        "error": None,
    }
    end = {
        **start,
        "status": "failed",
        "error": {
            "message": f"{method} failed: {method} failed for `autotem`: Mcp error: -32601: Method not found"
        },
    }
    events[2:2] = [{"type": "item.started", "item": start}, {"type": "item.completed", "item": end}]

    def read():
        trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
        return verifier.read_turn(
            tmp_path,
            {"prompt": "user.txt", "trace": trace.name},
            "codex",
            "codex-exec-jsonl",
            "autotem",
        )

    turn = read()
    assert [c["tool"] for c in turn["calls"]] == ["validate_plan"]
    assert turn["trace_sha256"] == verifier.digest(trace)
    del events[4]  # Without the actual Harness call this capture contains only discovery.
    assert read()["calls"] == []
    for change in (
        {"tool": "run_analysis"},
        {"arguments": {"server": "other"}},
        {"error": {"message": "connection closed"}},
        {"result": {"content": []}},
        {"id": "not-started"},
    ):
        events[3]["item"] = {**end, **change}
        with pytest.raises(verifier.EvidenceError, match="unexpected_tool_in_autotem_namespace"):
            read()


@pytest.mark.parametrize("tool", ["list_capabilities", "list_mcp_resources"])
@pytest.mark.parametrize("foreign_completion", [False, True])
def test_codex_turn_cannot_end_with_an_unanswered_autotem_call(tmp_path, tool, foreign_completion):
    record, _, trace, events = sample(tmp_path)
    pending = {
        "id": "pending",
        "type": "mcp_tool_call",
        "server": "autotem",
        "tool": tool,
        "arguments": {},
        "status": "in_progress",
        "error": None,
        "result": None,
    }
    events.insert(2, {"type": "item.started", "item": pending})
    if foreign_completion:
        events.insert(3, {"type": "item.completed", "item": {**pending, "server": "other"}})
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    report = verifier.verify(record, root=tmp_path)
    assert (report["case_verdict"], report["reason"]) == (
        "not_evaluated",
        "unfinished_host_mcp_call",
    )


def test_unattempted_blocked_and_missing_proof_distinct(tmp_path):
    record, spec, _, _ = sample(tmp_path)
    partial = verifier.verify(record, root=tmp_path)
    assert (
        partial["case_verdict"] == "not_evaluated"
        and partial["checks"]["case_execution"] == "passed"
    )
    assert partial["reason"] == "no_verified_release_binding" and partial["release_ready"] is False
    assert "capability is reserved" not in json.dumps(partial)
    spec.update(
        observation="blocked",
        blocker_code="GATEWAY_UNAVAILABLE",
        blocker_file="gateway.log",
        host="openclaw",
    )
    write_json(record, spec)
    assert verifier.verify(record)["case_verdict"] == "not_evaluated"
    (tmp_path / "gateway.log").write_text("1006 closed", encoding="utf-8")
    blocked = verifier.verify(record)
    assert blocked["case_verdict"] == "blocked" and blocked["reason"] == "GATEWAY_UNAVAILABLE"
    spec.update(observation="not_evaluated")
    write_json(record, spec)
    assert verifier.verify(record)["case_verdict"] == "not_evaluated"


@pytest.mark.parametrize(
    "alter,reason",
    [
        ("fake-cli", "no_real_autotem_host_call"),
        ("different-server", "no_real_autotem_host_call"),
        ("different-tool", "tool_envelope_mismatch"),
        ("unfinished", "incomplete_host_turn"),
    ],
)
def test_discards_nonhost_cli_and_unmatched_results(tmp_path, alter, reason):
    record, _, trace, events = sample(tmp_path)
    call = events[2]["item"]
    if alter == "fake-cli":
        call["type"] = "command_execution"
    if alter == "different-server":
        call["server"] = "some_other_server"
    if alter == "different-tool":
        call["tool"] = "submit_analysis"
    if alter == "unfinished":
        events.pop()
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    report = verifier.verify(record, root=tmp_path)
    assert report["case_verdict"] != "passed" and report["reason"] == reason


@pytest.fixture
def failed_codex_result(tmp_path):
    from copy import deepcopy

    _, spec, trace, events = sample(tmp_path)
    completed = events[2]["item"]
    failure = {
        "schema_version": "1.0.0",
        "request_id": None,
        "tool": "validate_plan",
        "ok": False,
        "result": None,
        "error": {
            "code": "DEPENDENCY_BROKEN",
            "message": "strain.gpa",
            "retryable": False,
            "details": {"reason": "import_timeout"},
        },
    }
    completed.update(
        status="failed",
        error=None,
        result={
            "content": [{"type": "text", "text": json.dumps(failure)}],
            "structured_content": deepcopy(failure),
        },
    )
    started = dict(deepcopy(completed), status="in_progress", result=None)
    retry = dict(
        deepcopy(completed),
        id="item_2",
        status="completed",
        result={"structured_content": dict(failure, ok=True, error=None, result={"valid": True})},
    )
    events[2:3] = [
        {"type": "item.started", "item": started},
        events[2],
        {"type": "item.completed", "item": retry},
    ]

    def read():
        trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
        return verifier.read_turn(
            tmp_path, spec["turns"][0], "codex", "codex-exec-jsonl", "autotem"
        )

    return started, completed, read


def test_codex_retains_received_business_error_and_later_success(failed_codex_result):
    _, completed, read = failed_codex_result
    for copy_key in ("structured_content", "content"):
        # Both native encodings carry a complete envelope, including text-only hosts.
        original = completed["result"].pop(copy_key)
        calls = read()["calls"]
        assert [c["response"]["ok"] for c in calls] == [False, True]
        assert calls[0]["response"]["error"]["details"] == {"reason": "import_timeout"}
        assert calls[0]["arguments"] == completed["arguments"] and calls[0]["line"] == 4
        completed["result"][copy_key] = original


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("completed", "error"), {"message": "server disconnected"}),
        (("completed", "result"), None),
        (("started", "id"), "unrelated"),
        (("started", "tool"), "submit_analysis"),
        (("started", "arguments"), {"plan": {"steps": []}}),
        (("started", "server"), "other"),
        (("started", "status"), "completed"),
        (("completed", "result", "content", 0, "text"), '{"schema_version":'),
        (("completed", "result", "content", 0, "text"), '{"ok": true}'),
        (("completed", "result", "isError"), False),
        (("response", "tool"), "submit_analysis"),
        (("response", "request_id"), "other-request"),
        (("response", "result"), {}),
        (("response", "error", "retryable"), 0),
        (("response", "error", "code"), ""),
    ],
)
def test_codex_failed_item_rejects_unbound_or_incomplete_results(failed_codex_result, path, value):
    started, completed, read = failed_codex_result
    payload = completed["result"]
    target = {"started": started, "completed": completed, "response": payload["structured_content"]}
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "response":
        payload["content"][0]["text"] = json.dumps(payload["structured_content"])
    with pytest.raises((verifier.EvidenceError, verifier.NotEvaluated, ValueError)):
        read()


def test_codex_failed_item_rejects_an_otherwise_valid_positive_envelope(failed_codex_result):
    _, completed, read = failed_codex_result
    response = completed["result"]["structured_content"]
    response.update(ok=True, result={"valid": True}, error=None)
    completed["result"]["content"][0]["text"] = json.dumps(response)
    with pytest.raises(verifier.EvidenceError, match="invalid_failed_mcp_error_envelope"):
        read()


def test_real_disk_hash_and_readonly_database_verification(tmp_path):
    import time
    import uuid

    from autotem.harness.store import Store, encode

    store = Store(tmp_path / "h")
    run_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    directory = store.run_dir(run_id)
    raw = directory / "raw" / "input.npy"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"verifier unit fixture")
    now = time.time()
    plan = {
        "input_mode": "snapshot",
        "steps": [{"id": "load", "operation_version": "1.0.0"}],
        "budgets": {},
    }
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated,input_path,"
            "source_hash,job_id) VALUES(?,?,?,'completed',?,?,?,?,?)",
            (run_id, "test", encode(plan), now, now, str(raw), verifier.digest(raw), job_id),
        )
        db.execute(
            "INSERT INTO jobs(job_id,run_id,status,created) VALUES(?,?,'completed',?)",
            (job_id, run_id, now),
        )
    a = store.write_json(run_id, "_request", "example", {"status": "diagnostic"})
    result = verifier.run_evidence(store.root, run_id, (now - 1, now + 1), None)
    assert result["artifacts"][a["artifact_id"]]["sha256"] == a["sha256"]
    from pathlib import Path

    Path(a["path"]).write_text("changed")
    with pytest.raises(verifier.EvidenceError, match="artifact_hash_mismatch"):
        verifier.run_evidence(store.root, run_id, (now - 1, now + 1), None)
    with pytest.raises(verifier.EvidenceError, match="run_outside_host_capture"):
        verifier.run_evidence(store.root, run_id, (now + 600, now + 800), None)


def test_manual_export_needs_original_observations_and_no_relabelled_native(tmp_path):
    record, spec, trace, events = sample(tmp_path)
    spec.update(host="workbuddy", capture_format="observed-host-export")
    write_json(record, spec)
    manual = {
        "host": "workbuddy",
        "session_id": "observed-test",
        "calls": [
            {
                "server": "autotem",
                "tool": "validate_plan",
                "arguments": events[2]["item"]["arguments"],
                "response": events[2]["item"]["result"],
                "original_evidence": "evidence.txt",
            }
        ],
        "answer": "Reserved capability.",
        "answer_evidence": "evidence.txt",
    }
    write_json(trace, manual)
    assert verifier.verify(record, root=tmp_path)["reason"] == "capture_file_missing_or_empty"
    (tmp_path / "evidence.txt").write_text("Synthetic test attachment")
    spec.update(delegation_mode="native")
    write_json(record, spec)
    assert verifier.verify(record, root=tmp_path)["reason"] == "incomplete_native_profile_traces"
    manual["calls"][0]["original_evidence"] = "../outside.txt"
    write_json(trace, manual)
    assert verifier.verify(record, root=tmp_path)["reason"] == "capture_path_outside_directory"


def test_automatic_host_proofs_are_scoped_without_a_reviewer(tmp_path):
    record, spec, trace, events = sample(tmp_path)
    events.insert(
        2,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "Get-Content .agents/skills/autotem/SKILL.md",
                "exit_code": 0,
                "aggregated_output": "---\nname: autotem\n---",
            },
        },
    )
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    (tmp_path / "discovery.txt").write_text("\n".join(sorted(verifier.TOOLS)), encoding="utf-8")
    (tmp_path / "consent.txt").write_text("AutoTEM request was not approved", encoding="utf-8")
    spec["host_observations"] = {"discovery": "discovery.txt", "authorization": "consent.txt"}
    write_json(record, spec)
    result = verifier.verify(record, root=tmp_path)
    checks = result["checks"]
    assert result["observed_tools"] == ["validate_plan"]
    assert checks["effective_tool_permission"] == checks["full_tool_inventory"] == "passed"
    assert (
        checks["session_skill_discovery"] == "passed"
        and checks["first_authorization"] == "not_evaluated"
    )
    assert result["case_verdict"] == "not_evaluated"
    spec["host_version"] = "1.2"
    write_json(record, spec)
    assert verifier.verify(record, root=tmp_path)["reason"] == "host_version_evidence_mismatch"


@pytest.mark.parametrize(
    ("original", "passed"),
    [
        ("User approved access to AutoTEM tools", True),
        ("AutoTEM request was approved", True),
        ("AutoTEM\nPermission granted", True),
        ("AutoTEM accepted", False),
        ("AutoTEM access granted. AutoTEM access denied", False),
        ("AutoTEM 权限：已授权", True),
        ("已允许调用 AutoTEM 工具", True),
        ("AutoTEM request was not approved", False),
        ("AutoTEM accepted 42 refined columns", False),
        ("AutoTEM instructions mention the approved CLI fallback", False),
        ("Other server permission granted. AutoTEM", False),
        ("Allow AutoTEM tools?", False),
        ('{"type":"approval_configuration","server":"autotem","status":"granted"}', False),
        ('{"type":"permission_decision","server":"autotem","decision":"granted"}', False),
        (
            '{"type":"permission_decision","session_id":"session",'
            '"server":"autotem","decision":"granted"}',
            False,
        ),
        ('{"type":"permission_decision","server":"other","decision":"granted"}', False),
        (
            '{"type":"permission_decision","server":"autotem","session_id":"elsewhere","decision":"granted"}',
            False,
        ),
        (
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"AutoTEM access granted"}}',
            False,
        ),
    ],
)
def test_first_consent_requires_an_observed_decision_for_this_service(original, passed):
    if not original.startswith("{"):
        original = "Codex tool permissions\n" + original
    assert verifier.has_grant(original, {"session"}, "codex") is passed


@pytest.mark.parametrize("method", ["mcpServer/elicitation/request", "tool/requestUserInput"])
@pytest.mark.parametrize(
    "question", ["Allow AutoTEM tool to rescale this frame?", "Allow this MCP tool to run?"]
)
def test_service_defined_forms_never_establish_host_permission(method, question):
    request = {
        "id": 17,
        "method": method,
        "params": {
            "serverName": "autotem",
            "threadId": "session",
            "mode": "form",
            "message": question,
            "requestedSchema": {"type": "object", "properties": {"allow": {"type": "boolean"}}},
        },
    }
    response = {"id": 17, "result": {"action": "accept", "content": {"allow": True}}}
    if method == "tool/requestUserInput":
        request["params"] = {
            "threadId": "session",
            "turnId": "turn",
            "itemId": "item",
            "isBlocking": True,
            "questions": [{"id": "allow", "header": "AutoTEM", "question": question}],
        }
        response["result"] = {"answers": {"allow": {"answers": ["yes"]}}}
    text = "\n".join(json.dumps(v) for v in (request, response))
    assert not verifier.has_grant(text, {"session"}, "codex")
    assert not verifier.has_grant("AutoTEM permission granted", {"session"}, "codex")
    assert not verifier.has_grant(
        "MCP server form\nAutoTEM permission granted", {"session"}, "codex"
    )
    assert verifier.has_grant("Codex: AutoTEM access granted", {"session"}, "codex")


@pytest.mark.parametrize(
    ("change", "passed"),
    [
        ({}, True),
        ({"origin": "autotem"}, False),
        ({"origin": "workbuddy"}, False),
        ({"actor": "server"}, False),
        ({"server": "other"}, False),
        ({"session_id": "elsewhere"}, False),
        ({"decision": "denied"}, False),
    ],
)
def test_separate_host_permission_record_requires_user_host_server_and_session(change, passed):
    event = {
        "type": "tool_permission_decision",
        "origin": "codex",
        "actor": "user",
        "server": "autotem",
        "session_id": "session",
        "decision": "granted",
        **change,
    }
    response = {"id": 17, "result": {"action": "accept", "content": event}}
    assert not verifier.has_grant(json.dumps(response), {"session"}, "codex")
    assert verifier.has_grant(json.dumps([response, event]), {"session"}, "codex") is passed


def test_only_the_original_catalogue_counts(tmp_path):
    names = sorted(verifier.TOOLS)
    catalog = [{"name": "mcp__autotem__" + name} for name in names]
    wrapper = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "output": [
                {"type": "input_text", "text": json.dumps(catalog)},
            ],
        },
    }
    assert verifier.has_inventory(json.dumps(wrapper))
    assert not verifier.has_inventory(json.dumps({"serverName": "other", "tools": catalog}))
    server = {
        "name": "autotem",
        "authStatus": "notLoggedIn",
        "resources": [],
        "resourceTemplates": [],
        "tools": {n: {"name": n, "inputSchema": {}} for n in names},
    }
    status = {"id": 4, "result": {"data": [server], "nextCursor": None}}
    assert verifier.has_inventory(json.dumps(status))
    assert not verifier.has_grant(json.dumps(status), {"session"})
    server["name"] = "other"
    assert not verifier.has_inventory(json.dumps(status))
    assert verifier.has_inventory(json.dumps({"result": {"tools": [{"name": n} for n in names]}}))
    assert verifier.has_inventory("\n".join(names))
    instructions = "All tools mentioned in the manual include " + ", ".join(names) + "."
    assert not verifier.has_inventory(instructions)
    assert not verifier.has_inventory(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "aggregated_output": instructions},
            }
        )
    )
    path = tmp_path / "request.txt"
    path.write_text("Analyze this image", encoding="utf-8")
    spec = {
        "host_observations": {
            "authorization": {
                "session_id": "session",
                "original_evidence": path.name,
                "observed_text": "AutoTEM access granted",
            }
        }
    }
    result = {"checks": {}, "evidence": {}}
    verifier.host_observations(
        spec, tmp_path, [{"session": "session", "skill_opened": False}], [], result
    )
    assert result["checks"]["first_authorization"] == "not_evaluated"


def test_ui_transcription_needs_a_decodable_image_and_actual_readable_grant(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "screen.png"
    spec = {
        "host": "codex",
        "host_observations": {
            "authorization": {
                "session_id": "session",
                "original_evidence": path.name,
                "observed_text": "AutoTEM access granted",
            }
        },
    }
    turns = [{"session": "session", "skill_opened": False}]

    def check():
        result = {"checks": {}, "evidence": {}}
        verifier.host_observations(spec, tmp_path, turns, [], result)
        return result

    path.write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
    monkeypatch.setattr(verifier, "windows_ocr", lambda _: pytest.fail("Invalid image reached OCR"))
    assert check()["checks"]["first_authorization"] == "not_evaluated"
    path.write_text("AutoTEM access granted", encoding="utf-8")
    assert check()["checks"]["first_authorization"] == "not_evaluated"
    Image.new("RGB", (16, 16), "white").save(path)
    # Isolate recognition outcomes here; the Windows integration probe uses real screen pixels.
    for original, expected in (
        ("", False),
        ("Accepted refinement (42)", False),
        ("AutoTEM access granted", False),
        ("Codex tool permissions\nAutoTEM access granted", True),
    ):
        monkeypatch.setattr(verifier, "windows_ocr", lambda _, text=original: text)
        result = check()
        assert result["evidence"]["authorization_image_decoded"]
        assert (result["checks"]["first_authorization"] == "passed") is expected


def test_windows_capture_time_uses_all_available_microseconds():
    assert verifier.stamp("2026-11-12T13:17:38.1920650+08:00") == verifier.stamp(
        "2026-11-12T05:17:38.192065Z"
    )


def test_physical_units_and_case_negative_variants():
    assert (
        verifier.positive_distance_units(
            "atom_distances", {"nearest_distances": [1, 2], "unit": "pixel"}
        )
        == []
    )
    assert (
        verifier.positive_distance_units(
            "atom_distances", {"nearest_distances": [float("nan"), None], "unit": "nm"}
        )
        == []
    )
    result = {"d_spacings": [{"value": 0.29, "unit": "nm"}], "provenance": {"physical": True}}
    assert verifier.positive_distance_units("diffraction", result) == ["nm"]
    tool = {
        "tool": "validate_plan",
        "response": {"ok": False, "error": {"code": "DEPENDENCY_BROKEN"}},
    }
    verifier.check_case("J11", [tool], {}, [{}], {}, "broken_dependency")
    with pytest.raises(verifier.EvidenceError, match="fault_variant_not_observed"):
        verifier.check_case("J11", [tool], {}, [{}], {}, "missing_dependency")
    run = {
        "status": "needs_input",
        "requested_fields": {"q": ["reference_region"]},
        "artifacts": {"a": {"kind": "image"}},
        "question_evidence": {"q": ["a"]},
        "_children": [],
        "error_code": "NEEDS_INPUT",
    }
    retrieved = {
        "tool": "list_artifacts",
        "response": {"ok": True, "result": [{"artifact_id": "a"}]},
    }
    verifier.check_case("J08", [retrieved], {"r": run}, [{}], {}, "unanswered")
    run["_children"].append("unexpected")
    with pytest.raises(verifier.EvidenceError, match="reply_started_unexpected_child"):
        verifier.check_case("J08", [retrieved], {"r": run}, [{}], {}, "unanswered")


def test_matrix_missing_blocked_baselines_and_conflicts(tmp_path):
    from tools.host_acceptance import index

    release = tmp_path / "release"
    release.mkdir()
    manifest = {
        "release_id": "contract-test",
        "profiles": {"traditional-core-mcp": {"lock_sha256": "lock"}},
        "source": {"archive_sha256": "source"},
        "package_version": "1.0",
        "artifacts": [],
    }
    write_json(release / "release-manifest.json", manifest)
    record, spec, _, _ = sample(tmp_path)
    spec.update(
        observation="blocked", blocker_code="GATEWAY_UNAVAILABLE", blocker_file="gateway.txt"
    )
    write_json(record, spec)
    (tmp_path / "gateway.txt").write_text("closed with 1006", encoding="utf-8")
    summary = verifier.verify(record, bundle=release)
    public = tmp_path / "public.json"
    write_json(public, summary)
    matrix = index.build(release, [public])
    assert len(matrix["hosts"]) == 3
    assert all(len(h["cases"]) == 15 for h in matrix["hosts"].values())
    assert matrix["hosts"]["codex"]["cases"]["J10"]["verdict"] == "blocked"
    assert matrix["hosts"]["openclaw"]["cases"]["J01"]["verdict"] == "not_evaluated"
    assert matrix["release_ready"] is False and matrix["host_journey_matrix"] == "blocked"
    simulation = matrix["hosts"]["codex"]["cases"]["J15"]["variants"]
    assert [r["variant"] for r in simulation if r["required"]] == ["missing_dependency"]
    native = matrix["hosts"]["codex"]["cases"]["J14"]["variants"]
    assert [r["variant"] for r in native if r["required"]] == ["sequential"]
    with pytest.raises(ValueError, match="Mixed host versions"):
        index.build(release, [public], versions={"codex": "9.8.7"})
    summary["case_verdict"] = "passed"
    write_json(public, summary)
    with pytest.raises(ValueError, match="lacks verified release"):
        index.build(release, [public])
    summary["attempted_release_manifest_sha256"] = "0" * 64
    write_json(public, summary)
    with pytest.raises(ValueError, match="Mixed release identities"):
        index.build(release, [public])


def test_passed_verifier_requires_active_project_and_observed_skill(tmp_path, monkeypatch):
    import shutil
    import sys
    from pathlib import Path

    from autotem.acceptance import driver
    from autotem.installation import host_entry

    record, spec, trace, events = sample(tmp_path)
    events.insert(
        2,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "exit_code": 0,
                "command": "Get-Content .agents/skills/autotem/SKILL.md",
                "aggregated_output": "---\nname: autotem\n---",
            },
        },
    )
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    project = tmp_path / "project"
    root = project / ".autotem" / "runs"
    root.mkdir(parents=True)
    deployed = root.parent / "host-entry.py"
    shutil.copy(host_entry.__file__, deployed)
    adapter = project / "autotem-host" / "codex" / "adapter.json"
    adapter.parent.mkdir(parents=True)
    entry = {"command": sys.executable, "args": ["-I", str(deployed), "mcp"]}
    write_json(adapter, {"host": "codex", "mcp": {"mcpServers": {"autotem": entry}}})
    config = project / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "[mcp_servers.autotem]\n"
        + "\n".join(k + " = " + json.dumps(v) for k, v in {**entry, "cwd": str(project)}.items()),
        encoding="utf-8",
    )
    deployed_skill = project / ".agents" / "skills" / "autotem" / "SKILL.md"
    deployed_skill.parent.mkdir(parents=True)
    deployed_skill.write_text("synthetic verifier fixture", encoding="utf-8")
    bundle = tmp_path / "release"
    bundle.mkdir()
    write_json(bundle / "release-manifest.json", {"release_id": "test"})
    # These two controlled substitutes exercise the verifier contract only.
    monkeypatch.setattr(
        driver,
        "runtime_record",
        lambda *a: {
            "skill_resource_sha256": {"SKILL.md": verifier.digest(deployed_skill)},
            "release_identity": {"profile": "traditional-core-mcp"},
        },
    )
    monkeypatch.setattr(host_entry, "active_python", lambda c: Path(sys.executable))
    result = verifier.verify(record, bundle=bundle, root=root, project=project)
    assert result["case_verdict"] == "passed" and not result["release_ready"]
    assert result["checks"]["first_authorization"] == "not_evaluated"
    config.write_text('[mcp_servers.autotem]\ncommand="wrong"\n', encoding="utf-8")
    assert (
        verifier.verify(record, bundle=bundle, root=root, project=project)["reason"]
        == "active_host_mcp_entry_mismatch"
    )
