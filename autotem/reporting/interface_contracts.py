"""Interface completeness checks for Member B/C integration contracts."""

from __future__ import annotations

from typing import TypedDict

from autotem.core import AnalysisReport


class InterfaceCheck(TypedDict):
    owner: str
    item: str
    status: str
    detail: str


def evaluate_bc_interface_completeness(report: AnalysisReport) -> list[InterfaceCheck]:
    """Evaluate baseline completeness for B/C output contracts."""
    checks: list[InterfaceCheck] = []

    has_diffraction_overlay = any(
        figure.figure_type == "diffraction_overlay" for figure in report.figures
    )
    has_atom_overlay = any("atom_overlay" in figure.figure_type for figure in report.figures) or (
        report.atom_lattice is not None
        and report.atom_lattice.overlay is not None
        and bool(report.atom_lattice.overlay.uri)
    )

    def add(owner: str, item: str, ok: bool, detail: str) -> None:
        checks.append(
            InterfaceCheck(
                owner=owner,
                item=item,
                status="pass" if ok else "missing",
                detail=detail,
            )
        )

    add(
        "B",
        "TEMDataset",
        report.dataset is not None,
        "Input dataset object for downstream calibration/diffraction.",
    )
    add(
        "B",
        "DiffractionResult",
        report.diffraction is not None,
        "Reciprocal-space container for peaks, d-spacing, and phase candidates.",
    )
    add(
        "B",
        "Diffraction peaks",
        report.diffraction is not None and len(report.diffraction.peaks) > 0,
        "At least one detected peak is expected.",
    )
    add(
        "B",
        "d-spacing table",
        report.diffraction is not None and len(report.diffraction.d_spacings) > 0,
        "d-spacing entries should be exported for phase matching.",
    )
    add(
        "B",
        "Phase candidates",
        report.diffraction is not None and len(report.diffraction.phase_candidates) > 0,
        "Top-N phase candidates for downstream review.",
    )
    add(
        "B",
        "Diffraction overlay figure",
        has_diffraction_overlay,
        "FFT peak overlay in figures.",
    )

    add(
        "C",
        "AtomLattice",
        report.atom_lattice is not None,
        "Atom lattice object for atom/strain pipelines.",
    )
    add(
        "C",
        "Atom coordinates",
        report.atom_lattice is not None and len(report.atom_lattice.atoms) > 0,
        "At least one atom coordinate expected when atom step is active.",
    )
    add(
        "C",
        "Atom overlay figure",
        bool(has_atom_overlay),
        "Atom overlay visual for reportability.",
    )
    add(
        "C",
        "StrainResult",
        report.strain is not None,
        "Structured strain result object.",
    )
    add(
        "C",
        "Strain map availability",
        report.strain is not None
        and any(
            field is not None
            for field in [
                report.strain.strain_xx,
                report.strain.strain_yy,
                report.strain.strain_xy,
                report.strain.rotation_xy,
            ]
        ),
        "At least one strain/rotation map should be provided.",
    )

    return checks
