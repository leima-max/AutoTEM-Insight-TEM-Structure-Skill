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
    compute_atomic_plane_spacing_segments_by_angle,
    compute_atomic_plane_spacing_segments_from_angle_constrained_basis,
    compute_atomic_plane_spacing_segments_from_basis,
    compute_atomic_plane_spacing_segments_from_lattice_angle,
    compute_atomic_plane_spacing_segments_from_picked_direction,
)
from autotem.atomic.sublattice import (
    SublatticeClassificationResult,
    apply_manual_sublattice_labels,
    classify_sublattices_auto,
    classify_sublattices_from_model,
    parse_projected_model_csv,
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
    "SublatticeClassificationResult",
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
    "compute_atomic_plane_spacing_segments_from_angle_constrained_basis",
    "compute_atomic_plane_spacing_segments_from_basis",
    "compute_atomic_plane_spacing_segments_from_lattice_angle",
    "compute_atomic_plane_spacing_segments_by_angle",
    "compute_atomic_plane_spacing_segments_from_picked_direction",
    "classify_sublattices_auto",
    "classify_sublattices_from_model",
    "apply_manual_sublattice_labels",
    "parse_projected_model_csv",
]
