"""Run pose extraction on a video and write a skeleton-overlay contact sheet for visual checking.

uv run python scripts/debug_pose.py <video> <out.jpg> [--every 0.5] [--start 0] [--end 999]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tennis_coach.pose import extract_cached  # noqa: E402
from tennis_coach.render import draw_pose  # noqa: E402
from tennis_coach.video import iter_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--crop", action="store_true", help="crop around the tracked person")
    a = ap.parse_args()
    seq, hit = extract_cached(a.video, progress=lambda p: print(f"\r{p:.0%}", end=""))
    print(f"\ncache_hit={hit} frames={len(seq.t)} meta={seq.meta}")
    tiles, next_t = [], a.start
    for i, f in iter_frames(a.video):
        t = seq.t[i]
        if t < next_t or t > a.end:
            continue
        next_t = t + a.every
        img = draw_pose(f, seq.kpts[i], seq.scores[i])
        cv2.putText(img, f"{t:.2f}s #{i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if a.crop and not np.isnan(seq.bbox[i]).any():
            x0, y0, x1, y1 = seq.bbox[i]
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8
            xa, ya = int(max(0, cx - s)), int(max(0, cy - s))
            img = img[ya:int(cy + s), xa:int(cx + s)]
        tiles.append(cv2.resize(img, (320, int(320 * img.shape[0] / max(img.shape[1], 1)))))
    if not tiles:
        print("no frames in range"); return
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = []
    for r in range(0, len(tiles), a.cols):
        row = tiles[r:r + a.cols]
        row += [np.zeros_like(tiles[0])] * (a.cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, np.vstack(rows))
    print("wrote", a.out, len(tiles), "tiles")


if __name__ == "__main__":
    main()
