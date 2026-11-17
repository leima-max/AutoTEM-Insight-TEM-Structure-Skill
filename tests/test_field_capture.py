"""A field operator may name a specific file when replaying a single-input scenario."""

import pytest

from tools.host_acceptance.field_capture import check_expected_input


@pytest.mark.parametrize(
    ("hashes", "verdict"),
    [(["a" * 64, "a" * 64], "passed"), (["a" * 64, "b" * 64], "failed"), ([], "not_evaluated")],
)
def test_expected_input_covers_every_registered_run(hashes, verdict):
    summary = {
        "case_verdict": "passed",
        "runs": [{"source_sha256": value} for value in hashes],
        "checks": {},
        "evidence": {},
    }
    check_expected_input(summary, "a" * 64)
    assert summary["case_verdict"] == summary["checks"]["expected_input"] == verdict
    assert summary["evidence"]["expected_input_sha256"] == "a" * 64


def test_expected_input_preserves_existing_host_failure():
    summary = {"case_verdict": "blocked", "reason": "HOST_ROUTE_UNAVAILABLE"}
    check_expected_input(summary, "a" * 64)
    assert summary == {"case_verdict": "blocked", "reason": "HOST_ROUTE_UNAVAILABLE"}
