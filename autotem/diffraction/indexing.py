"""Public hkl indexing entrypoints.

This module keeps the B4 task-board contract stable while the implementation
continues to live in the MVP helper module.
"""

from autotem.diffraction.b4_indexing import B4IndexingResult, run_b4_indexing

__all__ = ["B4IndexingResult", "run_b4_indexing"]
