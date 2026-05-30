from __future__ import annotations

import sys
import types

from autotem.diffraction.phase_lookup import (
    lookup_phase_candidates,
    resolve_materials_project_api_key,
    write_materials_project_api_config,
)


def test_phase_lookup_resolves_material_name_from_local_builtin_library() -> None:
    result = lookup_phase_candidates(
        formula=None,
        elements=None,
        material_names=["Silicon"],
        sources=["local_builtin"],
        limit_per_source=3,
        timeout_s=1,
    )

    assert not result.warnings
    assert result.candidates
    top = result.candidates[0]
    assert top["name"] == "Silicon"
    assert top["formula"] == "Si"
    assert top["source"] == "local_builtin_crystal_library"
    assert top["a"] == 0.5431
    assert result.provenance["material_names"] == ["Silicon"]


def test_materials_project_api_source_guides_first_use_when_key_missing(
    monkeypatch,
    tmp_path,
) -> None:
    mp_api_pkg = types.ModuleType("mp_api")
    mp_api_client = types.ModuleType("mp_api.client")
    mp_api_client.MPRester = object
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_pkg)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client)
    monkeypatch.delenv("MP_API_KEY", raising=False)

    result = lookup_phase_candidates(
        formula="Si",
        elements=None,
        sources=["materials_project_api"],
        limit_per_source=1,
        timeout_s=1,
        materials_project_config_path=str(tmp_path / "missing.json"),
    )

    assert not result.candidates
    assert any("Materials Project API key is not configured" in text for text in result.warnings)
    assert result.provenance["materials_project_api"][0]["reason"] == "missing_api_key"


def test_materials_project_api_source_returns_candidates_with_fake_mp_api(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeLattice:
        a = 5.431
        b = 5.431
        c = 5.431
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    class FakeStructure:
        lattice = FakeLattice()
        sites = [
            types.SimpleNamespace(
                specie=types.SimpleNamespace(symbol="Si"),
                frac_coords=[0.0, 0.0, 0.0],
                species=None,
            )
        ]

        def __iter__(self):
            return iter(self.sites)

    class FakeSymmetry:
        symbol = "Fd-3m"
        number = 227

    class FakeDoc:
        material_id = "mp-149"
        formula_pretty = "Si"
        structure = FakeStructure()
        symmetry = FakeSymmetry()

    class FakeSummary:
        def search(self, **kwargs):
            assert kwargs["formula"] == "Si"
            assert "structure" in kwargs["fields"]
            return [FakeDoc()]

    class FakeMaterials:
        summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "secret"

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, *args):
            return False

    mp_api_pkg = types.ModuleType("mp_api")
    mp_api_client = types.ModuleType("mp_api.client")
    mp_api_client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_pkg)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client)
    monkeypatch.delenv("MP_API_KEY", raising=False)

    config_path = write_materials_project_api_config(
        "secret",
        config_path=tmp_path / "mp.json",
    )
    resolved, info = resolve_materials_project_api_key(config_path=config_path)
    assert resolved == "secret"
    assert info["source"] == "config_file"

    result = lookup_phase_candidates(
        formula="Si",
        elements=None,
        sources=["materials_project_api"],
        limit_per_source=1,
        timeout_s=1,
        materials_project_config_path=str(config_path),
    )

    assert not result.warnings
    assert result.candidates
    top = result.candidates[0]
    assert top["source"] == "materials_project_api"
    assert top["identifier"] == "mp-149"
    assert top["formula"] == "Si"
    assert top["a"] == 0.5431
    assert top["space_group"] == "Fd-3m"
    assert top["space_group_number"] == 227
    assert top["atom_sites"][0]["element"] == "Si"


def test_materials_project_api_ranks_and_filters_candidate_pool(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeLattice:
        a = 5.431
        b = 5.431
        c = 5.431
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    class FakeStructure:
        lattice = FakeLattice()
        sites = [
            types.SimpleNamespace(
                specie=types.SimpleNamespace(symbol="Si"),
                frac_coords=[0.0, 0.0, 0.0],
                species=None,
            )
        ]

        def __iter__(self):
            return iter(self.sites)

        def __len__(self):
            return len(self.sites)

    class FakeDoc:
        def __init__(
            self,
            material_id: str,
            *,
            energy_above_hull: float,
            is_stable: bool,
            deprecated: bool = False,
            theoretical: bool = False,
        ) -> None:
            self.material_id = material_id
            self.formula_pretty = "Si"
            self.structure = FakeStructure()
            self.symmetry = {"symbol": "Fd-3m", "number": 227}
            self.energy_above_hull = energy_above_hull
            self.is_stable = is_stable
            self.deprecated = deprecated
            self.theoretical = theoretical

    class FakeSummary:
        def search(self, **kwargs):
            calls.append(dict(kwargs))
            assert kwargs["formula"] == "Si"
            assert kwargs["chunk_size"] == 4
            assert "energy_above_hull" in kwargs["fields"]
            assert "is_stable" in kwargs["fields"]
            return [
                FakeDoc("mp-metastable", energy_above_hull=0.20, is_stable=False),
                FakeDoc("mp-deprecated", energy_above_hull=0.0, is_stable=True, deprecated=True),
                FakeDoc("mp-149", energy_above_hull=0.0, is_stable=True),
            ]

    class FakeMaterials:
        summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "secret"

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, *args):
            return False

    mp_api_pkg = types.ModuleType("mp_api")
    mp_api_client = types.ModuleType("mp_api.client")
    mp_api_client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_pkg)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client)
    monkeypatch.delenv("MP_API_KEY", raising=False)

    config_path = write_materials_project_api_config(
        "secret",
        config_path=tmp_path / "mp.json",
    )
    result = lookup_phase_candidates(
        formula="Si",
        elements=None,
        sources=["materials_project_api"],
        limit_per_source=1,
        timeout_s=1,
        materials_project_config_path=str(config_path),
        materials_project_candidate_pool_size=4,
        materials_project_max_energy_above_hull=0.05,
    )

    assert calls
    assert not result.warnings
    assert result.candidates[0]["identifier"] == "mp-149"
    assert result.candidates[0]["materials_project"]["is_stable"] is True
    assert result.candidates[0]["materials_project"]["energy_above_hull"] == 0.0
    mp_provenance = result.provenance["materials_project_api"][0]
    assert mp_provenance["candidate_count_before_filter"] == 3
    assert mp_provenance["filter"]["deprecated_removed_count"] == 1
    assert mp_provenance["filter"]["energy_above_hull_removed_count"] == 1
    assert mp_provenance["ranking"]["sort_by"] == "stability"


def test_material_name_alias_resolves_to_formula_for_materials_project_api(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeLattice:
        a = 4.913
        b = 4.913
        c = 5.405
        alpha = 90.0
        beta = 90.0
        gamma = 120.0

    class FakeStructure:
        lattice = FakeLattice()
        sites = [
            types.SimpleNamespace(
                specie=types.SimpleNamespace(symbol="Si"),
                frac_coords=[0.0, 0.0, 0.0],
                species=None,
            ),
            types.SimpleNamespace(
                specie=types.SimpleNamespace(symbol="O"),
                frac_coords=[0.3, 0.3, 0.0],
                species=None,
            ),
        ]

        def __iter__(self):
            return iter(self.sites)

    class FakeDoc:
        material_id = "mp-quartz"
        formula_pretty = "SiO2"
        structure = FakeStructure()
        symmetry = {"symbol": "P3121", "number": 152}

    class FakeSummary:
        def search(self, **kwargs):
            calls.append(dict(kwargs))
            assert kwargs["formula"] == "SiO2"
            return [FakeDoc()]

    class FakeMaterials:
        summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "secret"

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, *args):
            return False

    mp_api_pkg = types.ModuleType("mp_api")
    mp_api_client = types.ModuleType("mp_api.client")
    mp_api_client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_pkg)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client)
    monkeypatch.delenv("MP_API_KEY", raising=False)

    config_path = write_materials_project_api_config(
        "secret",
        config_path=tmp_path / "mp.json",
    )
    result = lookup_phase_candidates(
        formula=None,
        elements=None,
        material_names=["quartz"],
        sources=["materials_project_api"],
        limit_per_source=1,
        timeout_s=1,
        materials_project_config_path=str(config_path),
    )

    assert calls
    assert not result.warnings
    assert result.provenance["resolved_formulas"] == ["SiO2"]
    assert result.provenance["resolved_material_queries"] == [
        {"material": "quartz", "query_type": "formula", "value": "SiO2"}
    ]
    assert result.candidates[0]["identifier"] == "mp-quartz"
    assert result.candidates[0]["formula"] == "SiO2"
    assert result.candidates[0]["atom_sites"][1]["element"] == "O"


def test_materials_project_id_can_be_entered_as_material_name(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeLattice:
        a = 5.431
        b = 5.431
        c = 5.431
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    class FakeStructure:
        lattice = FakeLattice()
        sites = []

        def __iter__(self):
            return iter(self.sites)

    class FakeDoc:
        material_id = "mp-149"
        formula_pretty = "Si"
        structure = FakeStructure()
        symmetry = {"symbol": "Fd-3m", "number": 227}

    class FakeSummary:
        def search(self, **kwargs):
            calls.append(dict(kwargs))
            assert kwargs["material_ids"] == ["mp-149"]
            assert "formula" not in kwargs
            return [FakeDoc()]

    class FakeMaterials:
        summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "secret"

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, *args):
            return False

    mp_api_pkg = types.ModuleType("mp_api")
    mp_api_client = types.ModuleType("mp_api.client")
    mp_api_client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", mp_api_pkg)
    monkeypatch.setitem(sys.modules, "mp_api.client", mp_api_client)
    monkeypatch.delenv("MP_API_KEY", raising=False)

    config_path = write_materials_project_api_config(
        "secret",
        config_path=tmp_path / "mp.json",
    )
    result = lookup_phase_candidates(
        formula=None,
        elements=None,
        material_names=["mp-149"],
        sources=["materials_project_api"],
        limit_per_source=1,
        timeout_s=1,
        materials_project_config_path=str(config_path),
    )

    assert calls
    assert result.provenance["materials_project_ids"] == ["mp-149"]
    assert result.provenance["resolved_material_queries"] == [
        {"material": "mp-149", "query_type": "materials_project_id", "value": "mp-149"}
    ]
    assert result.candidates[0]["identifier"] == "mp-149"
