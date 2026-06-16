"""Run the AutoTEM denoise operator on one image or NumPy array.

This lightweight example intentionally loads ``autotem/preprocessing/denoise.py``
directly so the denoise skill can be used before the full workflow environment
is installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_apply_denoise() -> Any:
    denoise_path = _repo_root() / "autotem" / "preprocessing" / "denoise.py"
    spec = importlib.util.spec_from_file_location("autotem_preprocessing_denoise", denoise_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load denoise module from {denoise_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_denoise


def _load_array(input_path: Path) -> np.ndarray:
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(input_path)
    else:
        array = np.asarray(Image.open(input_path).convert("L"), dtype=np.float32)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = np.mean(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got shape {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, copy=False)
    return array


def _normalize_unit_interval(array: np.ndarray) -> np.ndarray:
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return np.asarray((array - min_value) / (max_value - min_value), dtype=np.float32)


def _save_png(array: np.ndarray, path: Path) -> None:
    scaled = np.clip(array, 0.0, 1.0)
    Image.fromarray((scaled * 255.0).astype(np.uint8)).save(path)


def _save_comparison(raw: np.ndarray, filtered: np.ndarray, path: Path) -> None:
    raw_img = Image.fromarray((np.clip(raw, 0.0, 1.0) * 255.0).astype(np.uint8))
    filtered_img = Image.fromarray((np.clip(filtered, 0.0, 1.0) * 255.0).astype(np.uint8))
    canvas = Image.new("L", (raw_img.width + filtered_img.width, raw_img.height))
    canvas.paste(raw_img, (0, 0))
    canvas.paste(filtered_img, (raw_img.width, 0))
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply AutoTEM baseline image denoise.")
    parser.add_argument("input", type=Path, help="Input .jpg/.png/.tif/.tiff/.npy file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/filter_basic"),
        help="Directory for filtered outputs.",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "none", "gaussian", "median", "wiener", "nlm"),
        default="wiener",
        help="Denoise method.",
    )
    parser.add_argument("--sigma", type=float, default=0.8, help="Gaussian sigma.")
    parser.add_argument("--median-size", type=int, default=3, help="Median filter window size.")
    parser.add_argument("--wiener-window", type=int, default=5, help="Wiener filter window size.")
    parser.add_argument("--wiener-noise", type=float, default=None, help="Optional Wiener noise.")
    parser.add_argument("--nlm-h", type=float, default=None, help="Optional NLM h parameter.")
    parser.add_argument("--nlm-patch-size", type=int, default=5, help="NLM patch size.")
    parser.add_argument("--nlm-patch-distance", type=int, default=6, help="NLM patch distance.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    apply_denoise = _load_apply_denoise()
    raw = _normalize_unit_interval(_load_array(args.input))
    filtered, effective_method, diagnostics, warnings = apply_denoise(
        raw,
        method=args.method,
        sigma=args.sigma,
        median_size=args.median_size,
        wiener_window=args.wiener_window,
        wiener_noise=args.wiener_noise,
        nlm_h=args.nlm_h,
        nlm_patch_size=args.nlm_patch_size,
        nlm_patch_distance=args.nlm_patch_distance,
    )
    filtered = np.asarray(filtered, dtype=np.float32)
    if filtered.shape != raw.shape:
        raise RuntimeError(f"Filtered shape mismatch: {filtered.shape} != {raw.shape}")
    if not np.isfinite(filtered).all():
        raise RuntimeError("Filtered image contains non-finite values.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_method = str(effective_method).replace(":", "_").replace("/", "_")
    prefix = f"{args.input.stem}_filtered_{safe_method}"
    array_path = args.output_dir / f"{prefix}.npy"
    png_path = args.output_dir / f"{prefix}.png"
    comparison_path = args.output_dir / f"{prefix}_comparison.png"
    metrics_path = args.output_dir / f"{prefix}_metrics.json"

    np.save(array_path, filtered)
    _save_png(filtered, png_path)
    _save_comparison(raw, filtered, comparison_path)

    std_before = float(np.std(raw))
    std_after = float(np.std(filtered))
    metrics = {
        "input": str(args.input),
        "method_requested": args.method,
        "method_effective": effective_method,
        "shape": [int(v) for v in filtered.shape],
        "dtype": str(filtered.dtype),
        "finite": True,
        "std_before": std_before,
        "std_after": std_after,
        "std_reduction_fraction": ((std_before - std_after) / std_before) if std_before else 0.0,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "outputs": {
            "array": str(array_path),
            "png": str(png_path),
            "comparison": str(comparison_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
