from __future__ import annotations

from pathlib import Path

import numpy as np

from autotem.diffraction.phase_mvp import (
    load_cif_phase_entries,
    match_candidates,
    plot_annotated_fft,
    plot_atom_projection,
    plot_experimental_vs_simulated_overlay,
    plot_simulated_diffraction,
    simulate_phase_diffraction,
)


def _write_test_cif(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 0.352",
                "_cell_length_b 0.352",
                "_cell_length_c 0.352",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "_space_group_name_H-M_alt 'Fm-3m'",
                "_chemical_formula_sum 'Ni'",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "Ni1 Ni 0.0 0.0 0.0",
                "Ni2 Ni 0.5 0.5 0.5",
            ]
        ),
        encoding="utf-8",
    )


def test_phase_mvp_load_simulate_match_and_plot(tmp_path: Path) -> None:
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    cif_path = cif_dir / "ni_test.cif"
    _write_test_cif(cif_path)

    entries, warnings = load_cif_phase_entries(str(cif_dir))
    assert entries
    assert not any("No valid CIF" in item for item in warnings)
    entry = entries[0]
    assert entry.name == "ni_test"
    assert entry.atoms

    sim_rows = simulate_phase_diffraction(entry, max_hkl=5, zone_axis=(0, 0, 1))
    assert sim_rows

    rng = np.random.default_rng(1234)
    angle = np.deg2rad(7.5)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    peak_rows: list[dict[str, float | int]] = []
    for idx, row in enumerate(sim_rows[:18]):
        gx = float(row["gx"])
        gy = float(row["gy"])
        gx_r = c * gx - s * gy + float(rng.normal(0.0, 0.01))
        gy_r = s * gx + c * gy + float(rng.normal(0.0, 0.01))
        peak_rows.append(
            {
                "peak_id": idx,
                "refined_x": 100.0 + 15.0 * gx_r,
                "refined_y": 100.0 + 15.0 * gy_r,
                "gx_1_per_nm": gx_r,
                "gy_1_per_nm": gy_r,
                "intensity": 1000.0 - 10.0 * idx,
                "snr": 8.0,
            }
        )

    ranking, details, match_warnings = match_candidates(
        peak_rows,
        entries,
        max_hkl=5,
        zone_axis=(0, 0, 1),
        d_tol_rel=0.12,
        angle_tol_deg=15.0,
    )
    assert ranking
    assert ranking[0]["name"] == "ni_test"
    assert ranking[0]["matched_peaks"] >= 6
    assert isinstance(match_warnings, list)

    top = details["ni_test"]
    assignments = list(top["assignments"])
    assert assignments

    fft = np.zeros((220, 220), dtype=np.float32)
    fig1 = plot_annotated_fft(fft, peak_rows, assignments)
    fig2 = plot_simulated_diffraction(sim_rows, title="sim")
    fig3 = plot_experimental_vs_simulated_overlay(peak_rows, assignments, sim_rows)
    fig4 = plot_atom_projection(entry, zone_axis=(0, 0, 1))
    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
