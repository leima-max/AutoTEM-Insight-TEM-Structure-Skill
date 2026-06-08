from __future__ import annotations

from apps.b2_diffraction_ui import (
    _build_phase_lookup_pool_rows,
    _lookup_active_phase_candidate_params,
    _merge_candidate_phase_params,
)


def test_active_phase_lookup_feeds_local_candidate_params_to_manual_path() -> None:
    rows, warnings, provenance = _lookup_active_phase_candidate_params(
        {
            "indexing_mode": "lattice_fit",
            "phase_candidate_materials": ["Silicon"],
            "candidate_phases": ["Silicon"],
            "phase_lookup_sources": ["local_builtin"],
            "phase_lookup_limit_per_source": 3,
            "phase_lookup_timeout_s": 1,
        }
    )

    assert warnings == []
    assert provenance["candidate_count"] == 1
    assert rows[0]["formula"] == "Si"
    assert rows[0]["identifier"] == "builtin:Si:diamond"

    merged = _merge_candidate_phase_params(
        [
            {
                "name": "Graphite",
                "formula": "C",
                "a": 0.246,
                "b": 0.246,
                "c": 0.6708,
                "alpha": 90,
                "beta": 90,
                "gamma": 120,
                "space_group": "P63/mmc",
            }
        ],
        [],
        extra_rows=rows,
    )

    assert any(row.get("formula") == "Si" for row in merged)


def test_phase_lookup_pool_rows_show_source_and_formula_metadata() -> None:
    rows = _build_phase_lookup_pool_rows(
        [
            {
                "name": "COD_1533063",
                "formula": "- Br3 Cs Pb -",
                "formula_original": "- Br3 Cs Pb -",
                "formula_canonical": "Br3CsPb",
                "source": "crystallography_open_database",
                "lookup_sources": ["crystallography_open_database"],
                "identifier": "1533063",
                "space_group": "Pm-3m",
                "space_group_number": 221,
                "a": 0.5874,
                "b": 0.5874,
                "c": 0.5874,
                "structure_source": "cod_cif",
                "lookup_source_count": 1,
            }
        ]
    )

    assert rows == [
        {
            "name": "COD_1533063",
            "formula": "- Br3 Cs Pb -",
            "formula_original": "- Br3 Cs Pb -",
            "formula_canonical": "Br3CsPb",
            "source": "crystallography_open_database",
            "lookup_sources": "crystallography_open_database",
            "identifier": "1533063",
            "space_group": "Pm-3m",
            "space_group_number": 221,
            "a_nm": 0.5874,
            "b_nm": 0.5874,
            "c_nm": 0.5874,
            "structure_source": "cod_cif",
            "equivalent_source_count": 1,
        }
    ]
