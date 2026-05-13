# B0 I/O Contract Notes

This document summarizes Member B B0 conventions for image readers, metadata mapping,
warning codes, and `TEMDataset` conversion.

## Metadata Mapping

`autotem.io.metadata.FORMAT_METADATA_MAPPINGS` defines prioritized key lookup for each
format family:

- `image`: TIFF/PNG/JPG/JPEG
- `dm`: DM3/DM4
- `mrc`: MRC
- `emd`: HDF5/EMD
- `ser`: SER/EMI

Each mapping entry includes:

- `keys`: metadata key candidates (supports nested path syntax, e.g. `pixelSize.0`)
- `unit`: default unit when raw value has no explicit unit
- `priority`: larger value means preferred candidate

## Warning Schema

I/O warnings are structured dictionaries with:

- `code`: stable warning code
- `message`: human-readable description
- `severity`: `info`, `warning`, or `error`
- `source`: origin module (e.g. `metadata`, `reader`)
- `field`: optional field locator

`autotem.io.metadata.METADATA_WARNING_CATALOG` is the canonical warning-code catalog.

## Metadata Confidence

`normalize_metadata` now emits:

- `metadata_completeness`: ratio of core fields present
- `metadata_confidence`: completeness adjusted by invalid-field penalties

Both values are in `[0, 1]` and are intended for workflow-level quality gating.

## Reader Registry

`ReaderRegistry` now supports:

- optional reader registration with dependency checks
- unavailable-format placeholders with install hints
- `list_supported_formats(only_available=...)`
- `describe_formats()` for per-format capability introspection
- `read_series(...)` for directory/sequence batch read
- `read_series_as_temdataset(...)` for direct `TEMDataset` batch conversion

## TEMDataset Conversion

`to_temdataset` now:

- preserves normalized/raw metadata in `TEMDataset.metadata`
- infers `modality` from source format + metadata hints
- keeps small arrays inline and stores large arrays as URI references
- serializes structured warning records into stable warning strings
