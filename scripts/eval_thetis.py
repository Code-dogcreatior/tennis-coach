"""Player-isolated evaluation on the THETIS RGB subset.

1. extract (cached) poses for every clip
2. analyse each clip (time_scale=1, real time), keep the strongest swing per clip
3. rule-only recognition on the TEST players
4. rule + kNN with a reference library built ONLY from the TRAIN players
5. report confusion matrices, rejection rates, hand/view statistics, timing

uv run python scripts/eval_thetis.py [--root data/raw/thetis] [--pose-only] [--out reports/thetis_eval.json]
Split: players are assigned to train/test by index parity within each level (beginner/expert); no clip of a
player ever appears in both sets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tennis_coach.analysis import analyse_video, swing_record
from tennis_coach.match import knn_vote
from tennis_coach.classify import combine_with_knn, Classification
from tennis_coach.pose import extract_cached

GT = {"forehand_flat": "forehand", "forehand_openstands": "forehand", "backhand": "backhand_1h",
      "backhand2hands": "backhand_2h", "flat_service": "serve", "kick_service": "serve", "slice_service": "serve",
      "smash": "unknown", "forehand_volley": "unknown", "backhand_volley": "unknown"}
LABELS = ["forehand", "backhand_1h", "backhand_2h", "serve", "unknown"]


def split_of(player: str) -> str:
    i = int(player[1:])
    # beginners p1..p31, experts p32..p55; the downloaded subset uses odd indices -> alternate them
    order = (i - 1) // 2 if i <= 31 else (i - 33) // 2
    return "train" if order % 2 == 0 else "test"


def confusion(rows, key="pred"):
    m = {g: Counter() for g in LABELS}
    for r in rows:
        m[r["gt"]][r[key]] += 1
    return {g: dict(c) for g, c in m.items()}


def summarise(rows, key="pred"):
    tgt = [r for r in rows if r["gt"] != "unknown"]
    non = [r for r in rows if r["gt"] == "unknown"]
    acc_all = np.mean([r[key] == r["gt"] for r in rows]) if rows else float("nan")
    acc_tgt_accepted = [r[key] == r["gt"] for r in tgt if r[key] != "unknown"]
    return {
        "n": len(rows), "n_target": len(tgt), "n_nontarget": len(non),
        "accuracy_overall(unknown counts as class)": round(float(acc_all), 3),
        "target_rejected_rate": round(float(np.mean([r[key] == "unknown" for r in tgt])), 3) if tgt else None,
        "target_accuracy_when_accepted": round(float(np.mean(acc_tgt_accepted)), 3) if acc_tgt_accepted else None,
        "nontarget_rejected_rate": round(float(np.mean([r[key] == "unknown" for r in non])), 3) if non else None,
        "nontarget_false_accept": dict(Counter(r[key] for r in non if r[key] != "unknown")),
        "confusion(gt->pred)": confusion(rows, key),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/thetis")
    ap.add_argument("--pose-only", action="store_true")
    ap.add_argument("--out", default="reports/thetis_eval.json")
    ap.add_argument("--hand", default=None, help="force hand (e.g. right) to isolate recognition from hand errors")
    ap.add_argument("--cached-only", action="store_true", help="skip clips whose pose is not cached yet")
    a = ap.parse_args()
    files = sorted(Path(a.root).glob("*/*.avi"))
    if a.cached_only:
        import hashlib
        from tennis_coach import config
        from tennis_coach.pose import file_sha1

        def cached(f):
            key = hashlib.sha1((file_sha1(f) + config.POSE_MODEL_ID + str(config.DET_EVERY)).encode()).hexdigest()[:20]
            return (config.DERIVED_DIR / "pose_cache" / f"{key}.npz").exists()
        files = [f for f in files if cached(f)]
    print(f"{len(files)} clips")
    t_pose = []
    for k, f in enumerate(files):
        t0 = time.perf_counter()
        _, hit = extract_cached(f)
        if not hit:
            t_pose.append(time.perf_counter() - t0)
        if k % 25 == 0:
            print(f"pose {k}/{len(files)}", flush=True)
    if a.pose_only:
        return
    import csv
    gt_hand = {}
    hp = Path("data/labels/thetis_apparent_hand.csv")
    if hp.exists():
        gt_hand = {r["player"]: r["apparent_hand"] for r in csv.DictReader(open(hp, encoding="utf-8"))}
    rows, recs = [], {}
    t_an = []
    for f in files:
        cls_dir = f.parent.name
        player = f.name.split("_")[0]
        t0 = time.perf_counter()
        if a.hand == "gt":
            ho = gt_hand.get(player) if gt_hand.get(player) in ("left", "right") else None
        else:
            ho = a.hand
        res = analyse_video(str(f), time_scale=1.0, hand_override=ho)
        t_an.append(time.perf_counter() - t0)
        sws = res["swings"]
        row = {"file": f.as_posix(), "player": player, "split": split_of(player), "thetis": cls_dir, "gt": GT[cls_dir],
               "n_swings": len(sws), "hand": res["hand"]["hand"], "hand_src": res["hand"]["source"],
               "hand_est": res["hand"]["estimated"], "hand_gt": gt_hand.get(player),
               "hand_conf": res["hand"]["confidence"], "view": res["view_overall"]["view"]}
        if not sws:
            row.update(pred="unknown", rule="unknown", conf=0, reason="no swing detected", view_key=None)
        else:
            sw = max(sws, key=lambda s: s["window"]["peak_speed"])
            c = sw["classification"]
            row.update(pred=c["action"], rule=c["rule_action"], conf=c["confidence"], view_key=sw["view_key"],
                       reason=c["evidence"][-1] if c["evidence"] else "", swing_view=sw["view"]["view"],
                       peak=sw["window"]["peak_speed"], cues=c.get("scores"))
            recs[f.as_posix()] = (sw, res)
        rows.append(row)
    # ---- kNN with a TRAIN-player reference library (labels = ground truth of train clips)
    lib = []
    for r in rows:
        if r["split"] == "train" and r["gt"] != "unknown" and r["file"] in recs:
            sw, _ = recs[r["file"]]
            sw2 = dict(sw); sw2["action"] = r["gt"]
            lib.append(swing_record(sw2, r["file"]))
    # kNN votes: the pool never contains clips of the same player (leave-one-player-out inside train)
    kvs = {}
    for r in rows:
        if r["file"] not in recs:
            continue
        sw, _ = recs[r["file"]]
        if sw["classification"].get("scores", {}).get("non_stroke"):
            continue
        pool = [x for x in lib if x["ref_id"].split("/")[-1].split("_")[0] != r["player"]]
        kvs[r["file"]] = knn_vote(swing_record(sw, None), pool, k=5)

    def predict(r, thr):
        if r["file"] not in recs or r["file"] not in kvs:
            return "unknown"
        sw, _ = recs[r["file"]]
        c = sw["classification"]
        base = Classification(c["rule_action"], c["rule_confidence"], list(c["evidence"]), c["rule_action"],
                              c["rule_confidence"], scores=c.get("scores", {}))
        kv = dict(kvs[r["file"]]) if kvs[r["file"]] else None
        if kv:
            kv["cost_threshold"] = thr
        return combine_with_knn(base, kv).action

    # choose the rejection threshold on TRAIN players only (balanced accuracy of accept/reject + correctness)
    cands = np.round(np.arange(0.6, 3.01, 0.1), 2)

    def objective(thr):
        tr = [r for r in rows if r["split"] == "train"]
        tg = [r for r in tr if r["gt"] != "unknown"]
        nt = [r for r in tr if r["gt"] == "unknown"]
        a1 = np.mean([predict(r, thr) == r["gt"] for r in tg]) if tg else 0
        a2 = np.mean([predict(r, thr) == "unknown" for r in nt]) if nt else 0
        return (a1 + a2) / 2
    scores_thr = {float(t): round(float(objective(t)), 4) for t in cands}
    best_thr = max(scores_thr, key=scores_thr.get)
    print("kNN threshold chosen on train:", best_thr, scores_thr[best_thr])
    for r in rows:
        r["pred_knn"] = predict(r, best_thr)
        kv = kvs.get(r["file"])
        r["knn_label"] = kv["label"] if kv else None
        r["knn_cost"] = kv["nearest_cost"] if kv else None
    test = [r for r in rows if r["split"] == "test"]
    train = [r for r in rows if r["split"] == "train"]
    rep = {
        "n_clips": len(rows), "players": {"train": sorted({r["player"] for r in train}), "test": sorted({r["player"] for r in test})},
        "composition": {s: dict(Counter((r["thetis"]) for r in rows if r["split"] == s)) for s in ("train", "test")},
        "rule_only": {"train": summarise(train, "pred"), "test": summarise(test, "pred")},
        "rule_plus_knn(train-player library)": {"train(LOPO)": summarise(train, "pred_knn"), "test": summarise(test, "pred_knn")},
        "knn_threshold": {"chosen_on_train": best_thr, "train_objective_curve": scores_thr},
        "hand_estimates": dict(Counter((r["hand"], r["hand_src"]) for r in rows).most_common()) if True else None,
        "view_estimates": dict(Counter(r["view"] for r in rows)),
        "n_swings_per_clip": dict(Counter(r["n_swings"] for r in rows)),
        "timing_s": {"pose_per_clip_mean": round(float(np.mean(t_pose)), 2) if t_pose else "cached",
                     "analysis_per_clip_mean": round(float(np.mean(t_an)), 3)},
        "library_size": len(lib),
        "rows": rows,
    }
    rep["hand_estimates"] = {f"{k[0]}|{k[1]}": v for k, v in Counter((r["hand"], r["hand_src"]) for r in rows).items()}
    hv = [r for r in rows if r.get("hand_gt") in ("left", "right")]
    rep["hand_estimation_vs_gt(per clip)"] = {
        "n": len(hv),
        "correct": sum(r["hand_est"] == r["hand_gt"] for r in hv),
        "wrong": sum(r["hand_est"] not in (None, r["hand_gt"]) for r in hv),
        "undetermined": sum(r["hand_est"] is None for r in hv),
        "by_class": {c: {"correct": sum(r["hand_est"] == r["hand_gt"] for r in hv if r["thetis"] == c),
                         "wrong": sum(r["hand_est"] not in (None, r["hand_gt"]) for r in hv if r["thetis"] == c),
                         "undetermined": sum(r["hand_est"] is None for r in hv if r["thetis"] == c)}
                     for c in sorted({r["thetis"] for r in hv})},
    }
    print("hand estimation vs GT:", rep["hand_estimation_vs_gt(per clip)"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for k in ("rule_only", "rule_plus_knn(train-player library)"):
        for s, v in rep[k].items():
            print(f"== {k} [{s}]", {kk: vv for kk, vv in v.items() if kk != "confusion(gt->pred)"})
            for g, c in v["confusion(gt->pred)"].items():
                print(f"   {g:12s} -> {c}")
    print("hand:", rep["hand_estimates"]); print("view:", rep["view_estimates"]); print("swings/clip:", rep["n_swings_per_clip"])
    print("timing:", rep["timing_s"])


if __name__ == "__main__":
    sys.exit(main())
