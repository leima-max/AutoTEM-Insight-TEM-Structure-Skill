"""Pydantic schema models for reporting contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from autotem.core.models import CURRENT_SCHEMA_VERSION


class ReportingContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReportBundleSummaryModel(ReportingContractModel):
    schema_version: Literal["0.1.0"] = CURRENT_SCHEMA_VERSION
    summary: str = ""
    module_snapshots: list[str] = Field(default_factory=list)
    figure_count: int = 0
    table_count: int = 0
    warning_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class OutputManifestModel(ReportingContractModel):
    schema_version: Literal["0.1.0"] = CURRENT_SCHEMA_VERSION
    report_summary: str = ""
    requested_formats: list[str] = Field(default_factory=list)
    output_root: str
    directories: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    artifact_kind_counts: dict[str, int] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class InterfaceCompletenessSummaryModel(ReportingContractModel):
    total: int = 0
    passed: int = 0
    missing: int = 0
    pass_rate: float = 0.0


class InterfaceCompletenessCheckModel(ReportingContractModel):
    owner: str
    item: str
    status: Literal["pass", "missing"]
    detail: str


class InterfaceCompletenessContractModel(ReportingContractModel):
    schema_version: Literal["0.1.0"] = CURRENT_SCHEMA_VERSION
    contract: Literal["A3_BC_interface_completeness"] = "A3_BC_interface_completeness"
    summary: InterfaceCompletenessSummaryModel
    checks: list[InterfaceCompletenessCheckModel] = Field(default_factory=list)


def reporting_contract_schemas() -> dict[str, dict[str, Any]]:
    return {
        "ReportBundleSummary": ReportBundleSummaryModel.model_json_schema(),
        "OutputManifest": OutputManifestModel.model_json_schema(),
        "InterfaceCompletenessContract": InterfaceCompletenessContractModel.model_json_schema(),
    }
