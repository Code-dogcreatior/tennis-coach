"""Render frames around the estimated contact (and take-back end / trophy) for manual phase-error checking.

uv run python scripts/phase_check_sheet.py out.jpg clip1.avi[:hand] clip2.avi[:hand] ...
Each row: frames ic-3 .. ic+3 (estimated contact in the middle, marked), then the take-back end / trophy frame.
"""
import sys

import cv2
import numpy as np

from tennis_coach.analysis import analyse_video
from tennis_coach.pose import extract_cached
from tennis_coach.video import read_frame_at

out = sys.argv[1]
rows = []
for arg in sys.argv[2:]:
    path, hand = (arg.rsplit(":", 1) if arg.endswith((":left", ":right")) else (arg, ""))
    pose, _ = extract_cached(path)
    r = analyse_video(path, time_scale=1.0, hand_override=hand or None, pose=pose)
    sws = [s for s in r["swings"] if s["phases"]]
    if not sws:
        print(path, "no analysable swing"); continue
    sw = max(sws, key=lambda s: s["window"]["peak_speed"])
    ev = sw["phases"]["events"]
    fc = int(np.argmin(np.abs(pose.t - ev["contact"]["tv"])))
    tiles = []
    for d in range(-3, 4):
        f = read_frame_at(path, fc + d)
        if f is None:
            f = np.zeros((480, 640, 3), np.uint8)
        bb = pose.bbox[min(len(pose.bbox) - 1, max(0, fc))]
        if np.isfinite(bb).all():
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            s = max(bb[2] - bb[0], bb[3] - bb[1]) * 0.7
            f = f[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)]
        f = cv2.resize(f, (170, 170))
        cv2.putText(f, f"{d:+d}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if d == 0 else (0, 255, 255), 2)
        tiles.append(f)
    label = f"{path.split('/')[-1][:22]} {sw['action']}"
    bar = np.zeros((20, 170 * 7, 3), np.uint8)
    cv2.putText(bar, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.vstack([bar, np.hstack(tiles)]))
    print(path, sw["action"], "contact frame", fc, "events", {k: round(v["tv"], 2) for k, v in ev.items()})
cv2.imwrite(out, np.vstack(rows))
