#!/usr/bin/env python3
"""
Iterate tangram_response.csv → sequence/0001, sequence/0002, ...

Each folder:
  00_silhouette.png
  01_step.png ... N_step.png
  (N+1)_final.png
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFilter

ROOT = Path("/data/scratch/angelz/vision_reasoning/tangram_256")
CSV_PATH = ROOT / "tangram_response.csv"
SEQ_ROOT = ROOT / "sequence"
SIZE = 256
TIMEOUT = 25


def parse_solution(raw: str) -> dict | None:
    if not raw or not raw.strip().startswith("{"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def first_appearance(seq: list[int]) -> list[int]:
    out = []
    for s in seq:
        if s not in out:
            out.append(s)
    return out


def download(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _largest_component(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    best = None
    best_n = 0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or vis[y, x]:
                continue
            q = deque([(y, x)])
            vis[y, x] = True
            cells = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        q.append((ny, nx))
                        cells.append((ny, nx))
            if len(cells) > best_n:
                best_n = len(cells)
                best = cells
    out = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        out[ys, xs] = True
    return out


def to_silhouette(arr: np.ndarray) -> Image.Image:
    """Black figure on white background."""
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    sat = mx - mn
    dark_frac = float((mx <= 40).mean())
    bright_frac = float((mn >= 220).mean())
    sat_frac = float((sat > 25).mean())

    # Prefer saturation so white margins / letterbox are not treated as figure.
    if sat_frac > 0.03:
        fg = (sat > 25) & (mn < 235)
    elif dark_frac >= 0.25 and dark_frac >= bright_frac:
        fg = mx > 40
    else:
        fg = (mn < 230) & ((sat > 15) | (mx < 200))

    pil = Image.fromarray((fg.astype(np.uint8) * 255))
    pil = pil.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    fg = np.array(pil) > 127
    fg = _largest_component(fg)

    out = np.ones_like(arr) * 255
    out[fg] = 0
    return Image.fromarray(out)


def cluster_colors(arr: np.ndarray, max_colors: int = 7):
    fg = arr.max(axis=2) > 25
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return []
    pix = arr[ys, xs].astype(np.float32)
    q = (pix // 24).astype(np.int32)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    clusters = []
    H, W = fg.shape
    used = np.zeros(len(xs), dtype=bool)
    for idx in order:
        if len(clusters) >= max_colors:
            break
        if counts[idx] < 80:
            continue
        sel = inv == idx
        if used[sel].mean() > 0.5:
            continue
        mask = np.zeros((H, W), dtype=bool)
        mask[ys[sel], xs[sel]] = True
        clusters.append((mask, float(ys[sel].mean()), float(xs[sel].mean())))
        used[sel] = True
    clusters.sort(key=lambda c: (c[1], c[2]))
    return [c[0] for c in clusters]


def to_white_background(arr: np.ndarray) -> np.ndarray:
    """Replace dark/black background with white; keep colored pieces."""
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    sat = mx - mn
    dark_frac = float((mx <= 40).mean())
    sat_frac = float((sat > 25).mean())
    out = arr.copy()
    if sat_frac > 0.03:
        is_piece = (sat > 20) & (mn < 240)
        if dark_frac >= 0.2:
            bg = ~is_piece
        else:
            bg = ((mx <= 40) | ((sat <= 15) & (mx >= 240))) & ~is_piece
    elif dark_frac >= 0.25:
        bg = mx <= 40
    else:
        bg = mx <= 40
    out[bg] = 255
    return out


def compose(arr: np.ndarray, masks: dict[int, np.ndarray], visible: list[int]) -> Image.Image:
    out = np.ones_like(arr) * 255  # white background
    for sid in visible:
        m = masks.get(sid)
        if m is not None:
            out[m] = arr[m]
    return Image.fromarray(out)


def process_row(row_idx: int, row: dict) -> tuple[bool, str]:
    url = (row.get("Tangram URL") or "").strip()
    sol = parse_solution(row.get("Solution") or "")
    folder = SEQ_ROOT / f"{row_idx:04d}"
    folder.mkdir(parents=True, exist_ok=True)

    # clear old pngs in folder
    for p in folder.glob("*.png"):
        p.unlink()

    if not url:
        return False, "empty url"

    img = download(url)
    if img is None:
        return False, f"download fail"

    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img)
    arr_white = to_white_background(arr)
    sil = to_silhouette(arr)
    sil.save(folder / "00_silhouette.png")

    if not sol or "MoveShapeSequence" not in sol:
        Image.fromarray(arr_white).save(folder / "01_final.png")
        return True, "sil+final only"

    order = first_appearance(sol["MoveShapeSequence"])
    clusters = cluster_colors(arr, max_colors=max(7, len(order)))
    masks = {sid: clusters[i] for i, sid in enumerate(order) if i < len(clusters)}

    visible = []
    for i, sid in enumerate(order, start=1):
        visible.append(sid)
        compose(arr_white, masks, visible).save(folder / f"{i:02d}_step.png")

    Image.fromarray(arr_white).save(folder / f"{len(order) + 1:02d}_final.png")
    return True, f"ok steps={len(order)} clusters={len(clusters)}"


def main():
    SEQ_ROOT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = fail = 0
    log_path = ROOT / "build_sequences_log.txt"
    with log_path.open("w") as log:
        for i, row in enumerate(rows, start=1):
            t0 = time.time()
            success, msg = process_row(i, row)
            line = f"{i:04d} {'OK' if success else 'FAIL'} {msg} ({time.time() - t0:.1f}s)"
            print(line, flush=True)
            log.write(line + "\n")
            ok += int(success)
            fail += int(not success)

    print(f"done ok={ok} fail={fail} -> {SEQ_ROOT}")


if __name__ == "__main__":
    main()
