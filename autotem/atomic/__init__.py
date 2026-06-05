"""Atom-column detection and atomic-structure modules."""

from autotem.atomic.atom_detection import AtomDetectionResult, detect_atoms
from autotem.atomic.displacement import (
    AtomicDisplacementResult,
    compute_atomic_displacement_from_basis,
)
from autotem.atomic.distance_statistics import (
    DistanceStatisticsResult,
    compute_distance_statistics,
    compute_distance_statistics_from_lattice,
)
from autotem.atomic.gaussian_refinement import RefinementResult, refine_atom_coordinates
from autotem.atomic.intensity_quantification import (
    AtomicColumnIntensityResult,
    AtomicIntensityGroupResult,
    cluster_atomic_column_intensities,
    group_atomic_column_intensities_by_lattice,
    quantify_atomic_column_intensity,
)
from autotem.atomic.lattice_constraint import (
    LatticeConstraintResult,
    apply_lattice_constraint,
)
from autotem.atomic.plane_spacing import (
    AtomicPlaneSpacingResult,
    compute_atomic_plane_spacing_segments,
    compute_atomic_plane_spacing_segments_from_basis,
)

__all__ = [
    "AtomDetectionResult",
    "AtomicDisplacementResult",
    "DistanceStatisticsResult",
    "RefinementResult",
    "LatticeConstraintResult",
    "AtomicColumnIntensityResult",
    "AtomicIntensityGroupResult",
    "AtomicPlaneSpacingResult",
    "detect_atoms",
    "compute_atomic_displacement_from_basis",
    "compute_distance_statistics",
    "compute_distance_statistics_from_lattice",
    "refine_atom_coordinates",
    "apply_lattice_constraint",
    "quantify_atomic_column_intensity",
    "cluster_atomic_column_intensities",
    "group_atomic_column_intensities_by_lattice",
    "compute_atomic_plane_spacing_segments",
    "compute_atomic_plane_spacing_segments_from_basis",
]
