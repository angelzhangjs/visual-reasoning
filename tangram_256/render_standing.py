#!/usr/bin/env python3
"""Render standing-tangram silhouette + intermediate snapshots from sheet JSON."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw

OUT = Path("/data/scratch/angelz/tangram_256/standing_snapshots")
SIZE = 256

URL = (
    "https://image.jimcdn.com/app/cms/image/transf/none/"
    "path/sb0abad0b84d20c80/image/ic31e29e3b81ea22f/version/1417439562/"
    "tangram-standing.png"
)

DATA = {
    "MoveShapeSequence": [
        3, 5, 1, 3, 4, 7, 4, 7, 3, 7, 6, 1, 5, 4, 3, 7, 3, 7, 1, 6, 1, 5, 4, 2
    ],
    "SevenShapeInfo": [
        {
            "shapeId": 1,
            "shapeType": 0,
            "shapePosition": {"x": -2.799999952316284, "y": 0.8000000715255737},
            "shapeRotation": {"x": 315.0, "y": 90.00000762939453, "z": 89.9999771118164},
            "shapeRotationId": 0,
        },
        {
            "shapeId": 2,
            "shapeType": 0,
            "shapePosition": {"x": 3.5, "y": -2.9000000953674318},
            "shapeRotation": {"x": 0.000010245284101983998, "y": 270.0, "z": 270.0},
            "shapeRotationId": 15,
        },
        {
            "shapeId": 3,
            "shapeType": 3,
            "shapePosition": {"x": -3.700000286102295, "y": 3.700000286102295},
            "shapeRotation": {"x": 270.0, "y": 180.00001525878907, "z": 0.0},
            "shapeRotationId": 3,
        },
        {
            "shapeId": 4,
            "shapeType": 4,
            "shapePosition": {"x": 2.0, "y": -2.200000047683716},
            "shapeRotation": {"x": 270.0, "y": 180.00001525878907, "z": 0.0},
            "shapeRotationId": 9,
        },
        {
            "shapeId": 5,
            "shapeType": 1,
            "shapePosition": {"x": 0.19999998807907105, "y": -1.4000000953674317},
            "shapeRotation": {"x": 0.000023905662601464428, "y": 90.0, "z": 270.0},
            "shapeRotationId": 21,
        },
        {
            "shapeId": 6,
            "shapeType": 2,
            "shapePosition": {"x": -0.8000000715255737, "y": -0.30000001192092898},
            "shapeRotation": {"x": 315.0, "y": 90.0, "z": 270.0},
            "shapeRotationId": 18,
        },
        {
            "shapeId": 7,
            "shapeType": 2,
            "shapePosition": {"x": -1.7999999523162842, "y": 2.9000000953674318},
            "shapeRotation": {"x": 45.000003814697269, "y": 270.0, "z": 90.0},
            "shapeRotationId": 6,
        },
    ],
}

# Match the colored standing reference roughly by shape role
COLORS = {
    1: (255, 140, 40),   # large - orange
    2: (60, 150, 230),   # large - blue
    3: (255, 105, 180),  # medium - pink
    4: (255, 210, 40),   # parallelogram - yellow
    5: (80, 200, 70),    # square - green
    6: (230, 70, 70),    # small - red
    7: (160, 90, 200),   # small - purple
}

S = math.sqrt(2)


def local_polygon(shape_type: int):
    """Canonical tangram polygons; origin at mesh center."""
    if shape_type == 0:  # large right triangle, legs=2
        pts = [(0, 0), (2, 0), (0, 2)]
    elif shape_type == 1:  # square side=1
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    elif shape_type == 2:  # small right triangle, legs=1
        pts = [(0, 0), (1, 0), (0, 1)]
    elif shape_type == 3:  # medium right triangle, legs=√2
        pts = [(0, 0), (S, 0), (0, S)]
    elif shape_type == 4:  # parallelogram
        pts = [(0, 0), (1, 0), (1.5, 0.5), (0.5, 0.5)]
    else:
        raise ValueError(shape_type)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(x - cx, y - cy) for x, y in pts]


def rot2d(points, deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def angle_from_shape(shape: dict) -> tuple[float, bool]:
    """
    Map Unity discrete rotation id → 2D angle.
    This dataset uses 24 bins (15°). ids >= 12 often include parallelogram flip.
    """
    rid = int(shape["shapeRotationId"])
    angle = (rid % 24) * 15.0
    flip = shape["shapeType"] == 4 and rid >= 12
    return angle, flip


def transform_piece(shape: dict):
    poly = local_polygon(shape["shapeType"])
    angle, flip = angle_from_shape(shape)
    if flip:
        poly = [(-x, y) for x, y in poly]
    poly = rot2d(poly, angle)
    px = float(shape["shapePosition"]["x"])
    py = float(shape["shapePosition"]["y"])
    # Unity y-up → image y-down
    return [(px + x, -(py + y)) for x, y in poly]


def make_mapper(all_points, size=SIZE, pad=0.18):
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    side = max(max_x - min_x, max_y - min_y, 1e-6) * (1 + pad)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    def map_pt(x, y):
        u = (x - cx) / side + 0.5
        v = (y - cy) / side + 0.5
        return u * (size - 1), v * (size - 1)

    return map_pt


def render_board(shapes_by_id, visible_ids, mapper, bg=(255, 255, 255), size=SIZE):
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for sid in sorted(visible_ids):
        pix = [mapper(*p) for p in transform_piece(shapes_by_id[sid])]
        draw.polygon(pix, fill=COLORS[sid], outline=(20, 20, 20))
    return img


def first_appearance(seq):
    out = []
    for s in seq:
        if s not in out:
            out.append(s)
    return out


def download(url, out: Path):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(out / "00_target_silhouette_orig.png")
    img256 = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    img256.save(out / "00_target_silhouette_256.png")
    return img256


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sil = download(URL, OUT)

    shapes_by_id = {s["shapeId"]: s for s in DATA["SevenShapeInfo"]}
    all_pts = []
    for s in DATA["SevenShapeInfo"]:
        all_pts.extend(transform_piece(s))
    mapper = make_mapper(all_pts)

    # empty
    Image.new("RGB", (SIZE, SIZE), (255, 255, 255)).save(OUT / "snap_00_empty.png")

    order = first_appearance(DATA["MoveShapeSequence"])
    visible = []
    for i, sid in enumerate(order, 1):
        visible.append(sid)
        frame = render_board(shapes_by_id, visible, mapper)
        path = OUT / f"snap_{i:02d}_after_piece_{sid}.png"
        frame.save(path)
        print("wrote", path.name)

    final = render_board(shapes_by_id, list(shapes_by_id), mapper)
    final.save(OUT / "snap_final_all_pieces.png")

    # black-bg final to compare with silhouette style
    final_black = render_board(
        shapes_by_id, list(shapes_by_id), mapper, bg=(0, 0, 0)
    )
    final_black.save(OUT / "snap_final_black_bg.png")

    # contact sheet: silhouette | intermediates | final
    n = 1 + len(order) + 1
    strip = Image.new("RGB", (SIZE * n, SIZE), (30, 30, 30))
    strip.paste(sil, (0, 0))
    for i, sid in enumerate(order, 1):
        strip.paste(Image.open(OUT / f"snap_{i:02d}_after_piece_{sid}.png"), (SIZE * i, 0))
    strip.paste(final, (SIZE * (n - 1), 0))
    strip.save(OUT / "contact_sheet_silhouette_to_final.png")

    pair = Image.new("RGB", (SIZE * 2, SIZE), (0, 0, 0))
    pair.paste(sil, (0, 0))
    pair.paste(final_black, (SIZE, 0))
    pair.save(OUT / "pair_silhouette_final_256.png")

    print("order", order)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
