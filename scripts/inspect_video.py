"""Analyse a video and print per-swing results; optionally render event frames for visual checking.

uv run python scripts/inspect_video.py <video> [--hand right|left] [--sheet out.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tennis_coach.analysis import analyse_video  # noqa: E402
from tennis_coach.pose import extract_cached  # noqa: E402
from tennis_coach.render import draw_pose  # noqa: E402
from tennis_coach.video import read_frame_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--hand")
    ap.add_argument("--sheet")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--json")
    ap.add_argument("--ts", default="1.0", help="time scale (real s per video s) or 'unknown'")
    a = ap.parse_args()
    pose, _ = extract_cached(a.video)
    ts = None if a.ts == "unknown" else float(a.ts)
    r = analyse_video(a.video, hand_override=a.hand, pose=pose, time_scale=ts)
    print(f"time_scale={r['time_scale']:.3f} known={r['time_known']}")
    print(f"duration={r['duration']:.1f}s fps_grid={r['fps_grid']:.1f} swings={r['n_swings']} "
          f"lr_swaps={r['lr_swaps_fixed']} torso_px={r['torso_px']:.0f} elapsed={r['elapsed_s']}s")
    print("hand:", r["hand"]["hand"], r["hand"]["source"], r["hand"]["confidence"], r["hand"]["evidence"])
    v = r["view_overall"]
    print("view overall:", v["view"], v["facing"], round(v["confidence"], 2), "| hipw", round(v["hip_width"], 2),
          "lr", round(v["lr_order"], 2), "face", round(v["face_score"], 2))
    print("quality:", r["quality"]["level"], [i["code"] for i in r["quality"]["issues"]])
    rows = []
    for sw in r["swings"]:
        w, c, vv = sw["window"], sw["classification"], sw["view"]
        ph = sw["phases"]
        evs = {k: round(e["tv"], 2) for k, e in ph["events"].items()} if ph else {}
        print(f"#{sw['index']:2d} tv={w['tv_peak']:6.2f}s peak={w['peak_speed']:5.1f} wrist={w['wrist']:5s} "
              f"view={vv['view']}/{vv['facing']}({vv['confidence']:.2f}, hipw={vv['hip_width']:.2f}) key={sw['view_key']:15s} "
              f"-> {c['action']:12s} conf={c['confidence']:.2f} | {c['evidence'][-1] if c['evidence'] else ''}")
        if ph:
            print("      events:", {k: v for k, v in evs.items() if not k.startswith("window")},
                  "warn:", ph["warnings"])
            ms = sw["metrics"]
            print("      metrics:", {k: round(m["value"], 2) for k, m in ms.items()})
        if a.sheet and len(rows) < a.max:
            names = ["window_start", "start", "backswing_end", "trophy", "drop", "contact", "end"] if ph else []
            tiles = []
            for n in names:
                if n not in (ph or {}).get("events", {}):
                    continue
                t = ph["events"][n]["tv"]
                fi = int(np.argmin(np.abs(pose.t - t)))
                f = read_frame_at(a.video, fi)
                if f is None:
                    continue
                img = draw_pose(f, pose.kpts[fi], pose.scores[fi])
                bb = pose.bbox[fi]
                if np.isfinite(bb).all():
                    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    s = max(bb[2] - bb[0], bb[3] - bb[1]) * 0.75
                    img = img[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)]
                img = cv2.resize(img, (240, 240))
                cv2.putText(img, f"#{sw['index']} {n[:9]} {t:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(img)
            if tiles:
                tiles += [np.zeros_like(tiles[0])] * (6 - len(tiles))
                rows.append(np.hstack(tiles[:6]))
    if a.sheet and rows:
        cv2.imwrite(a.sheet, np.vstack(rows)); print("sheet ->", a.sheet)
    if a.json:
        from tennis_coach.analysis import strip_arrays
        Path(a.json).write_text(json.dumps(strip_arrays(r), ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
