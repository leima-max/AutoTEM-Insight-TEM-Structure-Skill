"""FFT, diffraction, indexing, and phase matching modules."""

from autotem.diffraction.basic import run_diffraction_analysis
from autotem.diffraction.fft import compute_fft_analysis, locate_direct_beam
from autotem.diffraction.gvector import compute_gvectors_and_dspacing
from autotem.diffraction.indexing import B4IndexingResult, run_b4_indexing
from autotem.diffraction.open_source_adapters import (
    SUPPORTED_PEAK_ADAPTER_ENGINES,
    resolve_peak_adapter_engine,
)
from autotem.diffraction.peak_detection import detect_bragg_peaks
from autotem.diffraction.phase_lookup import (
    PhaseLookupResult,
    lookup_phase_candidates,
    materials_project_api_config_path,
    materials_project_api_setup_guidance,
    resolve_materials_project_api_key,
    supported_phase_lookup_sources,
    write_materials_project_api_config,
)
from autotem.diffraction.phase_matching import (
    PhaseMatchingResult,
    match_diffsims_patterns_to_diffraction,
)

__all__ = [
    "SUPPORTED_PEAK_ADAPTER_ENGINES",
    "B4IndexingResult",
    "PhaseLookupResult",
    "PhaseMatchingResult",
    "compute_fft_analysis",
    "compute_gvectors_and_dspacing",
    "detect_bragg_peaks",
    "lookup_phase_candidates",
    "locate_direct_beam",
    "materials_project_api_config_path",
    "materials_project_api_setup_guidance",
    "match_diffsims_patterns_to_diffraction",
    "resolve_peak_adapter_engine",
    "resolve_materials_project_api_key",
    "run_b4_indexing",
    "run_diffraction_analysis",
    "supported_phase_lookup_sources",
    "write_materials_project_api_config",
]
