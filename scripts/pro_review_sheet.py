"""Contact sheets for reviewing the swings of reference videos (tier=pro): for every swing, frames at
backswing_end / contact / end (or trophy / contact for serves) with the automatic label, so a human can write
data/labels/pro_review.csv.

uv run python scripts/pro_review_sheet.py OUT_DIR [--yid ID ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tennis_coach import db, service  # noqa: E402
from tennis_coach.render import keyframe_panel  # noqa: E402
from tennis_coach.video import read_frame_at  # noqa: E402


def frame_at(v, pose, tv, h=200):
    fi = int(np.argmin(np.abs(pose.t - tv)))
    img = read_frame_at(v["path"], pose.video_frame(fi))
    return None if img is None else keyframe_panel(img, pose.kpts[fi], pose.scores[fi], [], out_h=h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--yid", nargs="*")
    ap.add_argument("--swings", nargs="*", type=int)
    ap.add_argument("--size", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for vrow in db.list_videos("reference"):
        meta = vrow["meta"]
        if meta.get("tier") != "pro" or (a.yid and meta.get("youtube_id") not in a.yid):
            continue
        v = db.get_video(vrow["id"])
        an = v.get("analysis")
        if not an:
            continue
        pose = service.load_pose(v["id"])
        t0 = an.get("video_t0", 0.0)
        rows = []
        for sw in an["swings"]:
            if a.swings and sw["index"] not in a.swings:
                continue
            ev = (sw.get("phases") or {}).get("events") or {}
            names = ["trophy", "contact", "end"] if "trophy" in ev else ["backswing_end", "contact", "end"]
            tiles = []
            for n in names:
                tv = ev[n]["tv"] if n in ev else (sw["window"]["tv_peak"] if n == "contact" else None)
                p = frame_at(v, pose, tv + t0, a.size) if tv is not None else None
                tiles.append(p if p is not None else np.zeros((a.size, a.size * 3 // 4, 3), np.uint8))
            row = np.concatenate([cv2.resize(x, (a.size * 3 // 4, a.size)) for x in tiles], axis=1)
            lab = f"#{sw['index']} {sw['action']} ({sw['classification'].get('rule_action')}) {sw['view_key']} " \
                  f"t={sw['window']['tv_peak']:.1f}s pk={sw['window']['peak_speed']:.0f} q={sw['quality']['level']}"
            bar = np.full((22, row.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, lab, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(np.concatenate([bar, row], axis=0))
        if not rows:
            continue
        per = 4   # swings per line
        lines = []
        for i in range(0, len(rows), per):
            chunk = rows[i:i + per]
            while len(chunk) < per:
                chunk.append(np.full_like(rows[0], 40))
            lines.append(np.concatenate(chunk, axis=1))
        for j in range(0, len(lines), 5):
            cv2.imwrite(str(out / f"{meta['youtube_id']}_{j // 5}.jpg"), np.concatenate(lines[j:j + 5], axis=0),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(meta["youtube_id"], meta.get("player"), len(an["swings"]), "swings")


if __name__ == "__main__":
    main()

