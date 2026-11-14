"""Interpret registered GPA fields using the mask recorded by the same analysis step."""

from pathlib import PureWindowsPath

GPA_MASK_INTERPRETATION = (
    "Use only pixels where the GPA quality mask for this step is true. Outside the mask, "
    "read_artifact returns null. Downloaded arrays can contain zeros at those pixels. "
    "Exclude them from measurements and statistics."
)


def gpa_result_mask(reference, same_step_artifacts: list[dict]) -> dict:
    # Registry filenames retain the artifact ID when a capture moves between computers.
    uri = reference.get("uri") if isinstance(reference, dict) else None
    filename = PureWindowsPath(uri).name if isinstance(uri, str) else None
    for mask in same_step_artifacts:
        if (
            mask["role"] == "gpa.quality_mask"
            and mask["kind"] == "array"
            and filename == f"{mask['artifact_id']}.npy"
        ):
            return {"artifact_id": mask["artifact_id"], "interpretation": GPA_MASK_INTERPRETATION}
    return {
        "artifact_id": None,
        "interpretation": "A verified quality mask reference is unavailable for this GPA result. "
        "Downloaded arrays can contain zeros in invalid pixels. "
        "Exclude these fields from measurements and statistics.",
    }
