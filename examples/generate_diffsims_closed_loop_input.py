"""Generate a tiny synthetic input for the diffsims closed-loop example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def synthetic_cubic_fft_image(size: int = 80) -> np.ndarray:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 2200.0
    for dy, dx, amp in [
        (0, 10, 320.0),
        (0, -10, 320.0),
        (10, 0, 300.0),
        (-10, 0, 300.0),
        (10, 10, 200.0),
        (-10, -10, 200.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    image = image - float(image.min())
    image = image / max(float(image.max()), 1.0)
    return image.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic input for examples/diffsims_closed_loop.yaml."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/diffsims_closed_loop_input.npy"),
        help="Output .npy path.",
    )
    parser.add_argument("--size", type=int, default=80, help="Square image size.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, synthetic_cubic_fft_image(max(32, int(args.size))))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
