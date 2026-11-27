"""Skeleton drawing helpers (server-side, for debug sheets and thumbnails)."""
from __future__ import annotations

import cv2
import numpy as np

from .config import KPT_THR

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]
LEFT = {5, 7, 9, 11, 13, 15, 1, 3}


HIGHLIGHT = (0, 230, 255)   # yellow (BGR)


def _crop_box(k: np.ndarray, s: np.ndarray, shape, thr=KPT_THR, margin=0.3):
    ok = (s > thr) & np.isfinite(k).all(1)
    if ok.sum() < 4:
        return 0, 0, shape[1], shape[0]
    x0, y0 = k[ok].min(0)
    x1, y1 = k[ok].max(0)
    h = (y1 - y0) * (1 + 2 * margin)
    w = max((x1 - x0) * (1 + 2 * margin), 0.75 * h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    return a, b, int(min(shape[1], cx + w / 2)), int(min(shape[0], cy + h / 2))


def keyframe_panel(img: np.ndarray, k: np.ndarray, s: np.ndarray, highlight: list[int], flip: bool = False,
                   out_h: int = 420, measure: str | None = None, hand: str = "right") -> np.ndarray:
    """Crop around the person, draw the skeleton and emphasise the joints a checkpoint is about."""
    out = draw_pose(img, k, s)
    lw = max(3, int(round(img.shape[0] / 160)))
    hs = set(highlight)
    for a, b in EDGES:
        if a in hs and b in hs and s[a] > KPT_THR and s[b] > KPT_THR:
            cv2.line(out, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), HIGHLIGHT, lw, cv2.LINE_AA)
    for j in highlight:
        if s[j] > KPT_THR and np.isfinite(k[j]).all():
            cv2.circle(out, tuple(int(v) for v in k[j]), lw + 3, HIGHLIGHT, 2, cv2.LINE_AA)
    if measure == "horizontal" and np.isfinite(k[[11, 12]]).all():
        # vertical reference line through the hip centre: horizontal distances are measured from it
        hx = int((k[11, 0] + k[12, 0]) / 2)
        cv2.line(out, (hx, 0), (hx, out.shape[0]), (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
    x0, y0, x1, y1 = _crop_box(k, s, img.shape)
    crop = out[y0:y1, x0:x1]
    if crop.size == 0:
        crop = out
    if flip:
        crop = crop[:, ::-1]
    scale = out_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), out_h), interpolation=cv2.INTER_AREA)


def side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    canvas = np.full((h, a.shape[1] + gap + b.shape[1], 3), 24, np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1] + gap:] = b
    return canvas


def draw_pose(img: np.ndarray, k: np.ndarray, s: np.ndarray, thr: float = KPT_THR) -> np.ndarray:
    out = img.copy()
    lw = max(2, int(round(img.shape[0] / 300)))
    for a, b in EDGES:
        if s[a] > thr and s[b] > thr and not np.isnan(k[[a, b]]).any():
            col = (255, 160, 0) if (a in LEFT and b in LEFT) else (0, 140, 255) if (a not in LEFT and b not in LEFT) else (200, 200, 200)
            cv2.line(out, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), col, lw, cv2.LINE_AA)
    for j in range(17):
        if s[j] > thr and not np.isnan(k[j]).any():
            col = (255, 160, 0) if j in LEFT else (0, 140, 255)
            cv2.circle(out, tuple(int(v) for v in k[j]), lw + 2, col, -1, cv2.LINE_AA)
    return out
