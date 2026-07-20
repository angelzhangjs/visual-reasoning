#!/usr/bin/env python3
"""Download silhouette (Option A) and render intermediate tangram snapshots."""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw

OUT = Path("/data/scratch/angelz/tangram_256/letter_a_snapshots")
SIZE = 256

# Target silhouette. The gstatic thumbnail from the sheet is expired (404);
# use the matching letter-A tangram silhouette for this SevenShapeInfo row.
URL = (
    "https://image.jimcdn.com/app/cms/image/transf/none/"
    "path/sb0abad0b84d20c80/image/iba25398b4cc89773/version/1407242734/"
    "tangram-letter-a.png"
)
# Original (dead) sheet URL kept for reference:
# https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQF5HYTWLp1gSInGfunkwus6iT7jIv3jZUBB458uNG2Tncfo2LT

DATA = {
    "MoveShapeSequence": [2, 7, 4, 6, 4, 6, 4, 3, 7, 2, 7, 1, 3, 1, 4, 5, 4, 2, 3, 4, 3],
    "SevenShapeInfo": [
        {
            "shapeId": 1,
            "shapeType": 0,
            "shapePosition": {"x": 1.2000000476837159, "y": 1.0},
            "shapeRotationId": 15,
        },
        {
            "shapeId": 2,
            "shapeType": 0,
            "shapePosition": {"x": 0.8999999761581421, "y": -1.2999999523162842},
            "shapeRotationId": 0,
        },
        {
            "shapeId": 3,
            "shapeType": 3,
            "shapePosition": {"x": -1.7000000476837159, "y": -2.0},
            "shapeRotationId": 3,
        },
        {
            "shapeId": 4,
            "shapeType": 4,
            "shapePosition": {"x": -1.0, "y": -0.5},
            "shapeRotationId": 3,
        },
        {
            "shapeId": 5,
            "shapeType": 1,
            "shapePosition": {"x": 1.899999976158142, "y": 0.19999998807907105},
            "shapeRotationId": 12,
        },
        {
            "shapeId": 6,
            "shapeType": 2,
            "shapePosition": {"x": -0.30000001192092898, "y": 1.7999999523162842},
            "shapeRotationId": 18,
        },
        {
            "shapeId": 7,
            "shapeType": 2,
            "shapePosition": {"x": -1.0, "y": 0.30000001192092898},
            "shapeRotationId": 15,
        },
    ],
}

# Classic tangram polygons in local coords (unit: half of large-triangle leg = 1)
# shapeType: 0 large △ (x2), 1 square, 2 small △ (x2), 3 medium △, 4 parallelogram
S = math.sqrt(2)


def local_polygon(shape_type: int) -> list[tuple[float, float]]:
    if shape_type == 0:  # large isosceles right triangle, legs=2
        return [(0, 0), (2, 0), (0, 2)]
    if shape_type == 1:  # square side=1
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    if shape_type == 2:  # small isosceles right triangle, legs=1
        return [(0, 0), (1, 0), (0, 1)]
    if shape_type == 3:  # medium isosceles right triangle, legs=√2
        return [(0, 0), (S, 0), (0, S)]
    if shape_type == 4:  # parallelogram
        return [(0, 0), (1, 0), (1 + 0.5, 0.5), (0.5, 0.5)]
    raise ValueError(shape_type)


COLORS = {
    1: (220, 70, 70),
    2: (70, 140, 220),
    3: (70, 180, 100),
    4: (240, 180, 50),
    5: (180, 90, 200),
    6: (50, 180, 180),
    7: (230, 120, 60),
}


def rotate(points, deg, flip=False):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    out = []
    for x, y in points:
        if flip:
            x = -x
        out.append((x * c - y * s, x * s + y * c))
    return out


def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def transform_piece(shape: dict) -> list[tuple[float, float]]:
    poly = local_polygon(shape["shapeType"])
    # Center local poly at origin so position is piece center-ish
    cx, cy = centroid(poly)
    poly = [(x - cx, y - cy) for x, y in poly]

    rid = int(shape["shapeRotationId"])
    # Discrete 45° steps; ids >= 8 often encode parallelogram flip in Unity apps
    angle = (rid % 8) * 45.0
    flip = rid >= 8 and shape["shapeType"] == 4
    poly = rotate(poly, angle, flip=flip)

    px = float(shape["shapePosition"]["x"])
    py = float(shape["shapePosition"]["y"])
    # Unity-like: often y-up; image y-down → flip y for drawing
    return [(px + x, -(py + y)) for x, y in poly]


def world_to_pixel(points, size=SIZE, margin=0.12):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Keep a fixed world box so frames align across snapshots
    # Expand with margin around all pieces of the full board
    return min_x, max_x, min_y, max_y


def make_mapper(all_points, size=SIZE, pad=0.15):
    min_x, max_x, min_y, max_y = world_to_pixel(all_points)
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)
    side = max(w, h) * (1 + pad)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    def map_pt(x, y):
        u = (x - cx) / side + 0.5
        v = (y - cy) / side + 0.5
        return u * (size - 1), v * (size - 1)

    return map_pt


def render_board(shapes_by_id, visible_ids, mapper, size=SIZE, bg=(255, 255, 255)):
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    # Draw in shapeId order for stability
    for sid in sorted(visible_ids):
        shape = shapes_by_id[sid]
        poly = transform_piece(shape)
        pix = [mapper(x, y) for x, y in poly]
        draw.polygon(pix, fill=COLORS[sid], outline=(30, 30, 30))
    return img


def download_silhouette(url: str, out: Path) -> Image.Image:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img_orig = img.copy()
    img256 = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    img_orig.save(out / "00_target_silhouette_orig.png")
    img256.save(out / "00_target_silhouette_256.png")
    return img256


def first_appearance_order(seq: list[int]) -> list[int]:
    seen = []
    for s in seq:
        if s not in seen:
            seen.append(s)
    return seen


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sil = download_silhouette(URL, OUT)
    print("saved silhouette", OUT / "00_target_silhouette_256.png")

    shapes_by_id = {s["shapeId"]: s for s in DATA["SevenShapeInfo"]}
    all_pts = []
    for s in DATA["SevenShapeInfo"]:
        all_pts.extend(transform_piece(s))
    mapper = make_mapper(all_pts)

    # Frame 0: empty boar
    empty = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    empty.save(OUT / "snap_00_empty.png")

    # Intermediate snapshots: add pieces in first-appearance order from MoveShapeSequence
    order = first_appearance_order(DATA["MoveShapeSequence"])
    visible = []
    frames = [empty]
    for i, sid in enumerate(order, start=1):
        visible.append(sid)
        frame = render_board(shapes_by_id, visible, mapper)
        path = OUT / f"snap_{i:02d}_after_piece_{sid}.png"
        frame.save(path)
        frames.append(frame)
        print("wrote", path)

    # Final full board (all 7)
    final = render_board(shapes_by_id, list(shapes_by_id.keys()), mapper)
    final.save(OUT / "snap_final_all_pieces.png")
    frames.append(final)

    # Also: one frame per move index (piece highlighted / cumulative unique set grown only on first see)
    # Contact strip for quick viewing
    strip = Image.new("RGB", (SIZE * (1 + len(order) + 1), SIZE), (240, 240, 240))
    strip.paste(sil, (0, 0))
    for i, sid in enumerate(order, start=1):
        im = Image.open(OUT / f"snap_{i:02d}_after_piece_{sid}.png")
        strip.paste(im, (SIZE * i, 0))
    strip.paste(final, (SIZE * (1 + len(order)), 0))
    strip.save(OUT / "contact_sheet_silhouette_to_final.png")
    print("wrote contact sheet")

    # Side-by-side silhouette | final for LVM-style pair
    pair = Image.new("RGB", (SIZE * 2, SIZE), (255, 255, 255))
    pair.paste(sil, (0, 0))
    pair.paste(final, (SIZE, 0))
    pair.save(OUT / "pair_silhouette_final_256.png")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
