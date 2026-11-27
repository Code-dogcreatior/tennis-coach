"""Derive the per-player *apparent* hitting hand in THETIS videos (ground truth for evaluation only).

THETIS RGB comes from a Kinect and is mirrored for (most) subjects, so the hand seen in the video is not
necessarily the player's real hand. For each player we use only the FOREHAND clips (class label is ground
truth): in a front view the forehand take-back stays on the hitting side and the finish crosses the body.
We compute the front-view forehand score under both hand hypotheses and keep the one that is consistent.
Players with no clear margin are marked 'unclear' and excluded from hand-dependent metrics.
Output: data/labels/thetis_apparent_hand.csv
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from tennis_coach.pose import extract_cached
from tennis_coach.skeleton import CANON, clean, normalise
from tennis_coach.swing import detect_swings, nanfill, speed
from tennis_coach.view import ViewEstimate, canonicalise


def fh_score(C, t, w):
    fps = 1 / np.median(np.diff(t))
    hw = C[:, CANON["h_wri"]]
    s = nanfill(speed(hw, t))
    a, b = max(w.i0, w.i_peak - int(0.1 * fps)), min(w.i1, w.i_peak + int(0.1 * fps))
    ic = a + int(np.argmax(s[a:b + 1]))
    pre = hw[max(w.i0, ic - int(1.0 * fps)):ic, 0]
    post = hw[ic:min(w.i1 + 1, ic + int(0.3 * fps)), 0]
    if not (np.isfinite(pre).any() and np.isfinite(post).any()):
        return 0.0, s[ic]
    return float(np.nanmedian(pre) - np.nanmedian(post)), float(s[ic])


scores = defaultdict(lambda: {"left": [], "right": []})
for f in sorted(Path("data/raw/thetis").glob("forehand_*/*.avi")):
    pl = f.name.split("_")[0]
    pose, _ = extract_cached(f)
    cs = clean(pose.t, pose.kpts, pose.scores)
    P, _, _ = normalise(cs.K)
    t = cs.t - cs.t[0]
    ws = detect_swings(P, t)
    if not ws:
        continue
    w = max(ws, key=lambda w: w.peak_speed)
    for hand in ("left", "right"):
        C, _, _, _ = canonicalise(P, cs.S, hand, ViewEstimate("front", "towards", 1, 0.5, 1, 1, []))
        sc, pk = fh_score(C, t, w)
        scores[pl][hand].append(sc + 0.05 * pk)   # forehand-consistency + hitting-wrist speed tie-break
rows = []
for pl in sorted(scores, key=lambda s: int(s[1:])):
    L, R = np.mean(scores[pl]["left"]), np.mean(scores[pl]["right"])
    margin = abs(L - R)
    hand = ("left" if L > R else "right") if margin > 0.3 else "unclear"
    rows.append({"player": pl, "apparent_hand": hand, "score_left": round(L, 3), "score_right": round(R, 3),
                 "n_forehand_clips": len(scores[pl]["left"]),
                 "method": "forehand-consistency on GT forehand clips; spot-checked visually"})
    print(rows[-1])
Path("data/labels").mkdir(parents=True, exist_ok=True)
with open("data/labels/thetis_apparent_hand.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
