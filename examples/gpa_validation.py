from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from autotem.strain import run_gpa


def _synthetic_lattice_exact_bin(
    *,
    shape: tuple[int, int] = (256, 256),
    g1: tuple[float, float] = (20 / 256, 0.0),
    g2: tuple[float, float] = (0.0, 18 / 256),
    exx: float = 0.0,
    eyy: float = 0.0,
    exy: float = 0.0,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0

    ux = exx * (xx - cx) + exy * (yy - cy)
    uy = exy * (xx - cx) + eyy * (yy - cy)

    phase1 = 2.0 * np.pi * (g1[0] * (xx - ux) + g1[1] * (yy - uy))
    phase2 = 2.0 * np.pi * (g2[0] * (xx - ux) + g2[1] * (yy - uy))
    return (np.cos(phase1) + 0.8 * np.cos(phase2)).astype(np.float32)


def _summarize_result(name: str, result: Any) -> dict[str, float]:
    stats = dict(result.provenance.get("map_stats", {}))
    print(f"\n[{name}]")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if result.warnings:
        print("warnings:")
        for item in result.warnings:
            print(f"- {item}")
    return {
        "mean_exx": float(stats.get("mean_exx", np.nan)),
        "mean_eyy": float(stats.get("mean_eyy", np.nan)),
        "mean_exy": float(stats.get("mean_exy", np.nan)),
    }


def run_synthetic_validation() -> None:
    print("=== Synthetic GPA validation ===")
    shape = (256, 256)
    g1 = (20 / 256, 0.0)
    g2 = (0.0, 18 / 256)
    ref = (64, 192, 64, 192)

    zero_image = _synthetic_lattice_exact_bin(shape=shape, g1=g1, g2=g2)
    zero_out = run_gpa(
        image=zero_image,
        method="gpa_b3",
        g_vectors=[g1, g2],
        g_vector_unit="1/pixel",
        reference_region=ref,
        gpa_mask_sigma_px=3.0,
        map_inline_mode="full",
    )
    zero_stats = _summarize_result("zero_strain", zero_out)
    zero_ok = (
        abs(zero_stats["mean_exx"]) < 1e-3
        and abs(zero_stats["mean_eyy"]) < 1e-3
        and abs(zero_stats["mean_exy"]) < 1e-3
    )
    print(f"zero_strain_check={'PASS' if zero_ok else 'FAIL'}")

    exx_target = 0.01
    exx_image = _synthetic_lattice_exact_bin(shape=shape, g1=g1, g2=g2, exx=exx_target)
    exx_out = run_gpa(
        image=exx_image,
        method="gpa_b3",
        g_vectors=[g1, g2],
        g_vector_unit="1/pixel",
        reference_region=ref,
        gpa_mask_sigma_px=3.0,
        map_inline_mode="full",
    )
    exx_stats = _summarize_result("exx_only", exx_out)
    exx_ok = (
        abs(exx_stats["mean_exx"] - exx_target) < 0.02
        and abs(exx_stats["mean_eyy"]) < 0.02
    )
    print(f"exx_only_check={'PASS' if exx_ok else 'FAIL'}")


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _peak_xy_from_row(row: dict[str, Any]) -> tuple[float, float]:
    x = float(row.get("refined_x", row.get("x", 0.0)) or 0.0)
    y = float(row.get("refined_y", row.get("y", 0.0)) or 0.0)
    return x, y


def _gxy_from_row(row: dict[str, Any]) -> tuple[float, float]:
    gx = float(row.get("gx_1_per_nm", row.get("g_x", 0.0)) or 0.0)
    gy = float(row.get("gy_1_per_nm", row.get("g_y", 0.0)) or 0.0)
    return gx, gy


def _infer_pixel_size_nm_from_rows(
    rows: list[dict[str, Any]],
    image_shape: tuple[int, int],
) -> float | None:
    h, w = image_shape
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    values: list[float] = []
    for row in rows:
        x, y = _peak_xy_from_row(row)
        gx, gy = _gxy_from_row(row)
        dx = x - cx
        dy = y - cy
        if abs(gx) > 1e-12 and abs(dx) > 1e-6:
            values.append(dx / (gx * float(w)))
        if abs(gy) > 1e-12 and abs(dy) > 1e-6:
            values.append(dy / (gy * float(h)))
    positive = [float(v) for v in values if float(v) > 0]
    if not positive:
        return None
    return float(np.median(np.asarray(positive, dtype=np.float64)))


def _target_xy_from_g(
    gx: float,
    gy: float,
    *,
    pixel_size_nm: float,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    h, w = image_shape
    x = (float(w) / 2.0) + gx * float(w) * pixel_size_nm
    y = (float(h) / 2.0) + gy * float(h) * pixel_size_nm
    return x, y


def run_peak_table_validation(
    *,
    csv_path: Path,
    image_shape: tuple[int, int],
    peak_indices: list[int],
    pixel_size_nm: float | None,
) -> None:
    print("\n=== Peak-table GPA scale validation ===")
    rows = _load_csv_rows(csv_path)
    selected = [rows[idx] for idx in peak_indices]
    implied = _infer_pixel_size_nm_from_rows(selected, image_shape=image_shape)
    print(f"csv_path={csv_path}")
    print(f"selected_peaks={peak_indices}")
    print(f"image_shape={image_shape}")
    print(f"input_pixel_size_nm={pixel_size_nm}")
    print(f"implied_pixel_size_nm={implied}")
    if implied is None:
        print("Could not infer pixel size from selected peaks.")
        return

    for idx, row in zip(peak_indices, selected, strict=False):
        actual_x, actual_y = _peak_xy_from_row(row)
        gx, gy = _gxy_from_row(row)
        forced_x, forced_y = _target_xy_from_g(
            gx,
            gy,
            pixel_size_nm=implied,
            image_shape=image_shape,
        )
        print(
            f"peak[{idx}] actual=({actual_x:.4f}, {actual_y:.4f}) "
            f"forced_target=({forced_x:.4f}, {forced_y:.4f}) "
            f"forced_delta_r={np.hypot(forced_x - actual_x, forced_y - actual_y):.4f}px"
        )
        if pixel_size_nm is not None:
            test_x, test_y = _target_xy_from_g(
                gx,
                gy,
                pixel_size_nm=float(pixel_size_nm),
                image_shape=image_shape,
            )
            print(
                f"          input_target=({test_x:.4f}, {test_y:.4f}) "
                f"input_delta_r={np.hypot(test_x - actual_x, test_y - actual_y):.4f}px"
            )


def run_real_image_validation(
    *,
    image_path: Path,
    g1: tuple[float, float],
    g2: tuple[float, float],
    g_vector_unit: str,
    pixel_size_nm: float | None,
    roi: tuple[int, int, int, int] | None,
    sigma_px: float,
) -> None:
    print("\n=== Real-image GPA validation ===")
    result = run_gpa(
        image=image_path,
        method="gpa_b3",
        g_vectors=[g1, g2],
        g_vector_unit=g_vector_unit,
        pixel_size_nm=pixel_size_nm,
        reference_region=roi,
        gpa_mask_sigma_px=sigma_px,
        include_debug_maps=True,
        map_inline_mode="full",
    )
    _summarize_result("real_image", result)
    print("reference_region:", result.reference_region)
    print("provenance_parameters:")
    print(json.dumps(result.provenance.get("parameters", {}), indent=2, ensure_ascii=False))


def _parse_roi(values: list[int] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--roi requires exactly 4 integers: y0 y1 x0 x1")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone GPA validation helper for AutoTEM-Insight.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip the exact-bin synthetic GPA validation cases.",
    )
    parser.add_argument("--csv", type=Path, help="Peak-table CSV exported from B2.")
    parser.add_argument(
        "--image-shape",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="Image shape used when interpreting peak-table coordinates.",
    )
    parser.add_argument(
        "--peak-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Peak indices to use from the CSV for scale/mask validation.",
    )
    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        help="Pixel size to test against the peak-table implied value or real-image GPA.",
    )
    parser.add_argument("--image", type=Path, help="Real image path for direct GPA validation.")
    parser.add_argument(
        "--g1",
        nargs=2,
        type=float,
        metavar=("GX", "GY"),
        help="Manual g1 vector.",
    )
    parser.add_argument(
        "--g2",
        nargs=2,
        type=float,
        metavar=("GX", "GY"),
        help="Manual g2 vector.",
    )
    parser.add_argument(
        "--g-unit",
        default="1/nm",
        choices=["1/nm", "1/pixel"],
        help="Unit for --g1/--g2 when validating a real image.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Reference ROI for real-image GPA validation.",
    )
    parser.add_argument(
        "--sigma-px",
        type=float,
        default=8.0,
        help="Bragg mask sigma in pixels for real-image GPA validation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.skip_synthetic:
        run_synthetic_validation()

    if args.csv is not None:
        if args.image_shape is None:
            raise SystemExit("--image-shape H W is required when using --csv")
        run_peak_table_validation(
            csv_path=args.csv,
            image_shape=(int(args.image_shape[0]), int(args.image_shape[1])),
            peak_indices=[int(v) for v in args.peak_indices],
            pixel_size_nm=args.pixel_size_nm,
        )

    if args.image is not None:
        if args.g1 is None or args.g2 is None:
            raise SystemExit("--image requires both --g1 and --g2")
        run_real_image_validation(
            image_path=args.image,
            g1=(float(args.g1[0]), float(args.g1[1])),
            g2=(float(args.g2[0]), float(args.g2[1])),
            g_vector_unit=str(args.g_unit),
            pixel_size_nm=args.pixel_size_nm,
            roi=_parse_roi(args.roi),
            sigma_px=float(args.sigma_px),
        )


if __name__ == "__main__":
    main()
