"""Display the stored raw-refinement result in its registered PNG."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _mark(draw, x: float, y: float, accepted: bool) -> None:
    if accepted:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=3)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#56b4e9", width=2)
    else:
        diagonals = ((x - 3, y - 3, x + 3, y + 3), (x - 3, y + 3, x + 3, y - 3))
        for width, color in ((3, "black"), (1, "#ffbf47")):
            for line in diagonals:
                draw.line(line, fill=color, width=width)


def save_refinement_overlay(
    raw: np.ndarray, coords: np.ndarray, path: Path, point_status: list[str] | None = None
) -> dict:
    low, high = np.percentile(raw, [1, 99])
    pixels = np.uint8(np.clip((raw - low) / max(float(high - low), 1e-12), 0, 1) * 255)
    scale = max(1, min(4, 800 // max(raw.shape)))
    canvas = Image.fromarray(pixels).convert("RGB")
    canvas = canvas.resize((raw.shape[1] * scale, raw.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    points = coords * scale
    metadata = {"display_scale": scale, "labels_shown": min(len(points), 200)}
    known = ("accepted", "fit_failed", "max_shift_rejected", "out_of_bounds")
    if not (
        point_status
        and len(point_status) == len(points)
        and all(status in known for status in point_status)
    ):
        # Preserve the unclassified overlay for older results without aligned statuses.
        for i, (y, x) in enumerate(points):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(40, 255, 100))
            if i < 200:
                draw.text((x + 3, y + 3), str(i), fill=(255, 120, 40))
        canvas.save(path)
        return metadata

    # Draw indices first so a nearby label cannot paint over a state marker.
    for i, (y, x) in enumerate(points[:200]):
        draw.text((x + 5, y + 3), str(i), fill="white", stroke_width=1, stroke_fill="black")
    for (y, x), status in zip(points, point_status, strict=True):
        _mark(draw, x, y, status == "accepted")

    accepted = point_status.count("accepted")
    retained = len(points) - accepted
    labels = (
        f"Raw-image refinement ({len(points)} total)",
        f"Accepted refinement ({accepted})",
        f"Input position retained ({retained})",
    )
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # Pillow 10.0 has a fixed-size default font.
        font = ImageFont.load_default()
    width = max(canvas.width, 34 + int(max(font.getlength(label) for label in labels)))
    output = Image.new("RGB", (width, canvas.height + 68), "#202428")
    output.paste(canvas)
    legend = ImageDraw.Draw(output)
    legend.text((8, canvas.height + 3), labels[0], font=font, fill="white")
    for i, label in enumerate(labels[1:]):
        top = canvas.height + 24 + i * 20
        _mark(legend, 13, top + 8, i == 0)
        legend.text((26, top), label, font=font, fill="white")
    output.save(path)
    metadata.update(
        refinement_status_source="atomic_refinement.provenance.point_status",
        refinement_counts={"total": len(points), "accepted": accepted, "retained": retained},
    )
    return metadata
