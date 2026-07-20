#!/usr/bin/env python3
"""
Option A intermediates for tangram-standing:
1) download target image → 256x256
2) segment the 7 colored pieces from that image
3) reveal them in MoveShapeSequence first-appearance order
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

OUT = Path("/data/scratch/angelz/tangram_256/standing_snapshots")
SIZE = 256

URL = (
    "https://image.jimcdn.com/app/cms/image/transf/none/"
    "path/sb0abad0b84d20c80/image/ic31e29e3b81ea22f/version/1417439562/"
    "tangram-standing.png"
)

# From sheet cell D2
MOVE_SEQ = [3, 5, 1, 3, 4, 7, 4, 7, 3, 7, 6, 1, 5, 4, 3, 7, 3, 7, 1, 6, 1, 5, 4, 2]

# shapeId → approximate RGB in standing.png (from measured means)
SHAPE_COLOR = {
    3: (252, 154, 206),  # medium / pink
    5: (11, 190, 8),     # square / green
    1: (253, 155, 0),    # large / orange
    4: (253, 253, 0),    # parallelogram / yellow
    7: (148, 79, 184),   # small / purple
    6: (208, 33, 20),    # small / red
    2: (0, 153, 250),    # large / blue
}


def download(url: str) -> Image.Image:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def first_appearance(seq):
    out = []
    for s in seq:
        if s not in out:
            out.append(s)
    return out


def color_masks(img: np.ndarray, color_map: dict[int, tuple[int, int, int]], thresh2: float = 70**2):
    """Nearest-color assignment on non-black pixels → per-shapeId boolean masks."""
    h, w, _ = img.shape
    fg = img.sum(axis=2) > 40
    ids = sorted(color_map)
    cols = np.array([color_map[i] for i in ids], dtype=np.float32)
    masks = {i: np.zeros((h, w), dtype=bool) for i in ids}

    ys, xs = np.where(fg)
    if len(xs) == 0:
        return masks
    pix = img[ys, xs].astype(np.float32)
    d = ((pix[:, None, :] - cols[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    mind = d.min(1)
    for j, sid in enumerate(ids):
        sel = (lab == j) & (mind < thresh2)
        masks[sid][ys[sel], xs[sel]] = True
    return masks


def compose(base_rgb: np.ndarray, masks: dict[int, np.ndarray], visible: list[int], bg=(0, 0, 0)):
    out = np.zeros_like(base_rgb)
    out[:] = bg
    for sid in visible:
        m = masks[sid]
        out[m] = base_rgb[m]
    return Image.fromarray(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    orig = download(URL)
    orig.save(OUT / "00_target_silhouette_orig.png")
    img256 = orig.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    img256.save(OUT / "00_target_silhouette_256.png")
    arr = np.array(img256)

    masks = color_masks(arr, SHAPE_COLOR)
    for sid, m in masks.items():
        print(f"shape {sid}: {m.sum()} px")

    order = first_appearance(MOVE_SEQ)
    print("reveal order", order)

    # snap_00 empty
    Image.new("RGB", (SIZE, SIZE), (0, 0, 0)).save(OUT / "snap_00_empty.png")

    visible = []
    frames = []
    for i, sid in enumerate(order, 1):
        visible.append(sid)
        frame = compose(arr, masks, visible)
        path = OUT / f"snap_{i:02d}_after_piece_{sid}.png"
        frame.save(path)
        frames.append(frame)
        print("wrote", path.name)

    final = compose(arr, masks, list(SHAPE_COLOR))
    final.save(OUT / "snap_final_all_pieces.png")

    # contact sheet
    n = 1 + len(order) + 1
    strip = Image.new("RGB", (SIZE * n, SIZE), (20, 20, 20))
    strip.paste(img256, (0, 0))
    for i, sid in enumerate(order, 1):
        strip.paste(Image.open(OUT / f"snap_{i:02d}_after_piece_{sid}.png"), (SIZE * i, 0))
    strip.paste(final, (SIZE * (n - 1), 0))
    strip.save(OUT / "contact_sheet_silhouette_to_final.png")

    pair = Image.new("RGB", (SIZE * 2, SIZE), (0, 0, 0))
    pair.paste(img256, (0, 0))
    pair.paste(final, (SIZE, 0))
    pair.save(OUT / "pair_silhouette_final_256.png")

    # Also save per-piece isolated sprites (useful for prompting)
    piece_dir = OUT / "pieces"
    piece_dir.mkdir(exist_ok=True)
    for sid in SHAPE_COLOR:
        compose(arr, masks, [sid]).save(piece_dir / f"piece_{sid}.png")

    print("done ->", OUT)


if __name__ == "__main__":
    main()
