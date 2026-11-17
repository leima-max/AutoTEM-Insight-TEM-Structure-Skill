"""Exercise the same standalone numerical checks used by installed release maintainers."""

import os

import pytest

from autotem.acceptance.driver import runtime_record
from tools.acceptance.verify_variants import atomic_grid, cubic_simulation, scale_bar


def test_atomic_operation_variants_from_numbered_refined_coordinates(tmp_path):
    atomic_grid(tmp_path, runtime_record(False))


def test_scalebar_lookup_and_delivery_operations(tmp_path):
    scale_bar(tmp_path, runtime_record(False))


def test_simulation_allows_explicit_cold_budget_and_keeps_default():
    from autotem.harness.contracts import Budgets
    from autotem.harness.registry import default_registry

    operation = default_registry().get("simulation.diffraction")
    timeout = operation.parameter_model.model_json_schema()["properties"]["diffsims_timeout_s"]
    assert timeout["default"] == Budgets().simulation_seconds == 30
    assert timeout["maximum"] == operation.timeout_seconds == 120


@pytest.mark.real_diffsims
@pytest.mark.skipif(
    os.environ.get("AUTOTEM_RUN_REAL_DIFFSIMS_TESTS") != "1",
    reason="Real diffsims suite requires explicit opt-in and an installed backend",
)
def test_optional_real_simulation_and_matching_from_disk(tmp_path):
    cubic_simulation(tmp_path, runtime_record(False))
