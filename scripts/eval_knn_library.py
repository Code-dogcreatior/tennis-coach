"""Does the size of the live reference library bias action recognition?

The app's kNN votes over ALL enabled references (pros + demo), so adding 175 pro serves can pull any
overhead motion towards 'serve'. This compares vote strategies on the SAME library the app uses:

  plain     current behaviour: k nearest swings
  source    each player/video contributes at most its closest swing per action (k nearest *sources*)
  balanced  per-action score = mean cost of that action's m closest sources; label = lowest score

Queries:
  THETIS clips (strongest swing per clip, gt hand); refs of the same THETIS player are removed.
  Smash / volley clips are non-target (should become 'unknown').
  Pro swings, leave-one-player-out (kNN label only; they have no rule result stored).

uv run python scripts/eval_knn_library.py [--out reports/knn_library_eval.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_thetis import GT, LABELS, split_of  # noqa: E402

from tennis_coach import service  # noqa: E402
from tennis_coach.analysis import analyse_video, swing_record  # noqa: E402
from tennis_coach.classify import Classification, combine_with_knn  # noqa: E402
from tennis_coach.match import STRATEGIES, knn_costs, knn_vote  # noqa: E402

THR = 1.6   # the app's cost threshold


def summarise(rows, key):
    tgt = [r for r in rows if r["gt"] != "unknown"]
    non = [r for r in rows if r["gt"] == "unknown"]
    acc = [r[key] == r["gt"] for r in tgt if r[key] != "unknown"]
    out = {"n": len(rows),
           "accuracy_overall": round(float(np.mean([r[key] == r["gt"] for r in rows])), 3),
           "target_rejected": round(float(np.mean([r[key] == "unknown" for r in tgt])), 3),
           "target_acc_when_accepted": round(float(np.mean(acc)), 3) if acc else None}
    for c in sorted({r["thetis"] for r in non}):
        rr = [r for r in non if r["thetis"] == c]
        out[f"{c}_rejected"] = round(float(np.mean([r[key] == "unknown" for r in rr])), 3)
        out[f"{c}_as"] = dict(Counter(r[key] for r in rr if r[key] != "unknown"))
    m = {g: Counter() for g in LABELS}
    for r in rows:
        m[r["gt"]][r[key]] += 1
    out["confusion"] = {g: dict(c) for g, c in m.items() if c}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/thetis")
    ap.add_argument("--out", default="reports/knn_library_eval.json")
    a = ap.parse_args()
    refs = service.load_refs()
    print("library:", Counter(r["action"] for r in refs))
    gt_hand = {r["player"]: r["apparent_hand"] for r in csv.DictReader(open("data/labels/thetis_apparent_hand.csv", encoding="utf-8"))}
    rows = []
    files = sorted(Path(a.root).glob("*/*.avi"))
    for n, f in enumerate(files):
        player = f.name.split("_")[0]
        ho = gt_hand.get(player) if gt_hand.get(player) in ("left", "right") else None
        res = analyse_video(str(f), time_scale=1.0, hand_override=ho)
        row = {"file": f.as_posix(), "player": player, "split": split_of(player), "thetis": f.parent.name,
               "gt": GT[f.parent.name]}
        if not res["swings"]:
            row.update({f"pred_{s}": "unknown" for s in STRATEGIES}, rule="unknown"); rows.append(row); continue
        sw = max(res["swings"], key=lambda s: s["window"]["peak_speed"])
        c = sw["classification"]
        pool = [r for r in refs if r.get("player") != f"THETIS {player}"]
        q = swing_record(sw, None)
        costs = knn_costs(q, pool)
        for s in STRATEGIES:
            base = Classification(c["rule_action"], c["rule_confidence"], [], c["rule_action"], c["rule_confidence"],
                                  scores=c.get("scores", {}))
            if c.get("scores", {}).get("non_stroke"):
                row[f"pred_{s}"] = "unknown"; continue
            kv = knn_vote(q, pool, strategy=s, costs=costs)
            if kv:
                kv["cost_threshold"] = THR
            row[f"pred_{s}"] = combine_with_knn(base, kv).action
            row[f"knn_{s}"] = kv["label"] if kv else None
        row["rule"] = c["rule_action"]
        row["smash_like"] = bool((c.get("scores") or {}).get("smash_like"))
        rows.append(row)
        if n % 50 == 0:
            print(f"thetis {n}/{len(files)}", flush=True)
    rep = {"threshold": THR, "library": dict(Counter(r["action"] for r in refs)), "thetis": {}}
    for s in ("rule",) + STRATEGIES:
        key = "rule" if s == "rule" else f"pred_{s}"
        rep["thetis"][s] = summarise(rows, key)
    # pros, leave-one-player-out, kNN label only
    pros = [r for r in refs if r.get("tier") == "pro"]
    pro_rows = []
    for r in pros:
        pool = [x for x in refs if x.get("player") != r.get("player") and x["video_id"] != r["video_id"]]
        costs = knn_costs(r, pool)
        pr = {"action": r["action"], "player": r.get("player")}
        for s in STRATEGIES:
            kv = knn_vote(r, pool, strategy=s, costs=costs)
            pr[s] = kv["label"] if kv else None
        pro_rows.append(pr)
    # serve calls flagged "smash-like" (toss hand never below the hips before the toss), app strategy
    rep["smash_like_flag"] = {}
    for c in sorted({r["thetis"] for r in rows}):
        sv = [r for r in rows if r["thetis"] == c and r.get("pred_balanced") == "serve"]
        if sv:
            rep["smash_like_flag"][c] = {"called_serve": len(sv), "flagged": sum(r.get("smash_like", False) for r in sv)}
    print("smash-like flag among serve calls:", rep["smash_like_flag"])
    rep["pro_lopo_knn_label"] = {}
    for s in STRATEGIES:
        per = {}
        for act in sorted({p["action"] for p in pro_rows}):
            pp = [p for p in pro_rows if p["action"] == act]
            per[act] = {"n": len(pp), "acc": round(float(np.mean([p[s] == act for p in pp])), 3),
                        "as": dict(Counter(p[s] for p in pp if p[s] != act))}
        per["macro_acc"] = round(float(np.mean([v["acc"] for v in per.values()])), 3)
        rep["pro_lopo_knn_label"][s] = per
    Path(a.out).write_text(json.dumps({**rep, "rows": rows}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for s, v in rep["thetis"].items():
        print("THETIS", s, {k: x for k, x in v.items() if k != "confusion"})
    for s, v in rep["pro_lopo_knn_label"].items():
        print("PRO", s, v)


if __name__ == "__main__":
    main()
