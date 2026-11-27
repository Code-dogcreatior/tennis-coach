"""Write data/labels/pro_review.csv from the automatic (rule) labels of the pro reference videos, constrained by
what each clip is known to contain (manifest 'content' column = the player's stroke repertoire in that clip):

* a swing whose rule label is not in the clip's repertoire is excluded (e.g. 'backhand_1h' for a player with a
  two-handed backhand: either a misclassification or a slice - neither is a valid reference);
* unknown / non-stroke swings are excluded;
* clips flagged with slow-motion segments: swings with peak wrist speed < 0.6 x the clip median are excluded
  (unknown slow-motion factor would corrupt timing checkpoints).

This is a conservative filter, not a human review; spot checks are reported in docs/REPORT.md.
uv run python scripts/pro_autolabel.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tennis_coach import db  # noqa: E402

man = {r["youtube_id"]: r for r in csv.DictReader(open("data/labels/pro_manifest.csv", encoding="utf-8-sig"))}
rows, stats = [], {}
for vrow in db.list_videos("reference"):
    meta = vrow["meta"]
    yid = meta.get("youtube_id")
    if meta.get("tier") != "pro" or yid not in man:
        continue
    an = db.get_video(vrow["id"]).get("analysis")
    if not an:
        print("not analysed yet:", yid)
        continue
    allowed = set(man[yid]["content"].split("+"))
    slowmo = "slow-motion" in man[yid]["note"]
    peaks = np.array([s["window"]["peak_speed"] for s in an["swings"]]) if an["swings"] else np.array([1.0])
    med = float(np.median(peaks))
    for sw in an["swings"]:
        a = sw["classification"].get("rule_action") or "unknown"
        if sw["classification"].get("scores", {}).get("non_stroke"):
            a = "unknown"
        lab = a if a in allowed else "exclude"
        if slowmo and sw["window"]["peak_speed"] < 0.6 * med:
            lab = "exclude"
        rows.append({"youtube_id": yid, "swing_index": sw["index"], "action": lab, "rule_action": a})
        stats.setdefault(yid, {}).setdefault(lab, 0)
        stats[yid][lab] += 1
# manual spot-check corrections win over the automatic filter
manual_p = Path("data/labels/pro_review_manual.csv")
if manual_p.exists():
    manual = {(r["youtube_id"], int(r["swing_index"])): r["action"] for r in csv.DictReader(open(manual_p, encoding="utf-8-sig"))}
    for r in rows:
        if (r["youtube_id"], r["swing_index"]) in manual:
            r["action"] = manual[(r["youtube_id"], r["swing_index"])]
with open("data/labels/pro_review.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["youtube_id", "swing_index", "action", "rule_action"])
    w.writeheader()
    w.writerows(rows)
for k, v in stats.items():
    print(k, man[k]["player"], v)
