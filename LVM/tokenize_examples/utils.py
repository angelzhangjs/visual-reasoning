"""Helpers for LVM tokenize_examples (missing from upstream repo)."""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def is_image(path: str) -> bool:
    return path.endswith(IMAGE_EXTS)


def is_video(path: str) -> bool:
    return path.endswith(VIDEO_EXTS)


def list_dir_with_full_path(directory: str):
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if not f.startswith(".")
    ]


def read_image_to_tensor(image_path, crop: bool = False):
    """Load image as float32 HxWxC in [0, 1], resized to 256x256."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if crop:
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr


def randomly_subsample_frame_indices(n_total, n_frames, max_stride=4, random_start=True):
    if n_total < n_frames:
        raise ValueError(f"n_total={n_total} < n_frames={n_frames}")
    max_possible_stride = max(1, (n_total - 1) // max(1, n_frames - 1))
    stride = random.randint(1, min(max_stride, max_possible_stride))
    span = stride * (n_frames - 1)
    if span >= n_total:
        stride = 1
        span = n_frames - 1
    start_max = n_total - span - 1
    start = random.randint(0, max(0, start_max)) if random_start else 0
    return [start + i * stride for i in range(n_frames)]


def match_mulitple_path(root_dir, regex):
    import re

    videos = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.startswith("."):
                videos.append(os.path.join(root, file))

    grouped_path = {}
    compiled = [re.compile(r) for r in regex]
    for r in compiled:
        for v in videos:
            matched = r.findall(v)
            if matched:
                groups = matched[0]
                grouped_path.setdefault(groups, []).append(v)

    return [
        tuple(v)
        for k, v in grouped_path.items()
        if len(v) == len(regex)
    ]


def read_frames_from_video(video_path):
    """Optional helper; requires opencv if used."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames
