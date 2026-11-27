"""Render one forehand frame per THETIS player (near the max wrist speed) for manual hand annotation.

uv run python scripts/thetis_hand_sheet.py out.jpg
The manual result is stored in data/labels/thetis_apparent_hand.csv (hand as it APPEARS in the video;
THETIS RGB is a Kinect stream and is mirrored for most subjects).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from tennis_coach.pose import extract_cached
from tennis_coach.video import read_frame_at

out = sys.argv[1]
tiles = []
players = sorted({p.name.split("_")[0] for p in Path("data/raw/thetis/forehand_flat").glob("*.avi")}, key=lambda s: int(s[1:]))
for pl in players:
    f = sorted(Path("data/raw/thetis/forehand_flat").glob(f"{pl}_*.avi"))[0]
    pose, _ = extract_cached(f)
    k = pose.kpts
    sp = np.nan_to_num(np.linalg.norm(np.diff(k[:, [9, 10]], axis=0), axis=-1)).max(1)
    i = int(np.argmax(sp)) + 1
    img = read_frame_at(f, max(0, i - 2))
    if img is None:
        continue
    bb = pose.bbox[i]
    if np.isfinite(bb).all():
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        s = max(bb[2] - bb[0], bb[3] - bb[1]) * 0.75
        img = img[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)]
    img = cv2.resize(img, (220, 220))
    cv2.putText(img, pl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    tiles.append(img)
cols = 6
rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols]))) for r in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows))
print(len(tiles), "players ->", out)
