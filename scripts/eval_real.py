"""Evaluate the system against the hand-labelled real-world clips (data/labels/annotations/, made with the
annotation page /annotate.html).

For every video whose annotation is marked 'done' the app pipeline is run (auto hand / view, the local
reference library for kNN), system swings are matched one-to-one to labelled strokes by contact time, and:
  * detection   - recall (labelled strokes found) / precision (system swings that are real strokes)
  * recognition - confusion labelled stroke -> system action (smash / volley / other should become 'unknown')
  * contact     - system contact estimate minus labelled contact frame, in ms and frames
  * hand / view - per video
With two annotators on the same video, their agreement (stroke type, contact frame) is reported as well.

  uv run python scripts/eval_real.py [--annotator user] [--include-unfinished] [--out reports/eval_real]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tennis_coach import service
from tennis_coach.analysis import analyse_video
from tennis_coach.annotate import LABEL_DIR, RAW, labelled_videos, load_labels

MATCH_S = 0.4                 # max |system contact - labelled contact| for a match
TARGET = {"forehand", "backhand_1h", "backhand_2h", "serve"}
VIEW_GROUP = {"side_hitting": "side", "side_nonhitting": "side", "side": "side", "front": "front", "back": "back",
              "oblique_front": "oblique_front", "oblique_back": "oblique_back"}


def system_swings(path: Path, refs) -> tuple[dict, list[dict]]:
    an = analyse_video(str(path), refs=refs, time_scale=1.0)
    t0 = an.get("video_t0", 0.0)
    out = []
    for sw in an["swings"]:
        ev = (sw.get("phases") or {}).get("events", {}).get("contact")
        t = (ev["tv"] if ev else sw["window"]["tv_peak"]) + t0
        out.append({"t": t, "action": sw["action"], "view": sw.get("view_key"), "has_phases": bool(sw.get("phases")),
                    "non_stroke": bool(sw["classification"].get("scores", {}).get("non_stroke")),
                    "smash_like": bool(sw["classification"].get("scores", {}).get("smash_like")),
                    "conf": sw["classification"]["confidence"]})
    return an, out


def match(gt: list[dict], sys_: list[dict], fps: float):
    """Greedy one-to-one matching by time difference."""
    pairs = sorted(((abs(g["frame"] / fps - s["t"]), i, j) for i, g in enumerate(gt) for j, s in enumerate(sys_)
                    if abs(g["frame"] / fps - s["t"]) <= MATCH_S))
    gi, sj, out = set(), set(), []
    for d, i, j in pairs:
        if i in gi or j in sj:
            continue
        gi.add(i); sj.add(j); out.append((i, j))
    return out


def agreement(a: list[dict], b: list[dict]) -> dict:
    """Two annotators on one video: same strokes found (within 3 frames), same type, contact-frame difference."""
    used, same, dfr = set(), 0, []
    for e in a:
        cand = [(abs(e["frame"] - f["frame"]), k) for k, f in enumerate(b) if k not in used and abs(e["frame"] - f["frame"]) <= 6]
        if not cand:
            continue
        _, k = min(cand); used.add(k)
        dfr.append(b[k]["frame"] - e["frame"]); same += e["stroke"] == b[k]["stroke"]
    return {"n_a": len(a), "n_b": len(b), "matched": len(dfr), "same_stroke": same,
            "frame_diff_abs_median": float(np.median(np.abs(dfr))) if dfr else None,
            "frame_diff_abs_max": int(np.max(np.abs(dfr))) if dfr else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default=None, help="whose labels are ground truth (default: the first 'done' one)")
    ap.add_argument("--include-unfinished", action="store_true")
    ap.add_argument("--out", default="reports/eval_real")
    a = ap.parse_args()
    refs = service.load_refs()
    rows, per_video, agree = [], [], []
    for video in labelled_videos():
        lab = load_labels(video)
        if "probe" not in lab:
            continue
        anns = lab.get("annotations", {})
        usable = {k: v for k, v in anns.items() if a.include_unfinished or v.get("status") == "done"}
        if not usable:
            continue
        who = a.annotator if a.annotator in usable else sorted(usable)[0]
        g = usable[who]
        if len(usable) >= 2:
            others = [k for k in sorted(usable) if k != who]
            for o in others:
                agree.append({"video": lab["video"], "a": who, "b": o, **agreement(g["events"], usable[o]["events"])})
        path = RAW / lab["video"]
        if not path.exists():
            print("missing video (run fetch_eval_real.py):", path); continue
        fps = lab["probe"]["fps"]
        an, ss = system_swings(path, refs)
        gt = g["events"]
        m = match(gt, ss, fps)
        mg = {i: j for i, j in m}
        for i, e in enumerate(gt):
            s = ss[mg[i]] if i in mg else None
            rows.append({"video": lab["video"], "frame": e["frame"], "gt": e["stroke"], "unsure": e.get("unsure", False),
                         "pred": (s["action"] if s else "missed"), "dt_ms": (round((s["t"] - e["frame"] / fps) * 1000, 1) if s and s["has_phases"] else None),
                         "fps": fps, "smash_like": s["smash_like"] if s else None})
        fp = [s for j, s in enumerate(ss) if j not in {jj for _, jj in m}]
        per_video.append({"video": lab["video"], "annotator": who, "n_gt": len(gt), "n_sys": len(ss), "matched": len(m),
                          "false_swings": len(fp), "false_swings_target": sum(s["action"] in TARGET for s in fp),
                          "hand_gt": g.get("hand"), "hand_sys": an["hand"]["hand"], "hand_src": an["hand"]["source"],
                          "view_gt": g.get("view"), "view_sys": dict(Counter(VIEW_GROUP.get(s["view"], s["view"]) for s in ss))})
        print(f"{lab['video']}: gt {len(gt)} sys {len(ss)} matched {len(m)}", flush=True)
    if not rows:
        print("no finished annotations found in", LABEL_DIR); return
    n_gt = len(rows)
    found = [r for r in rows if r["pred"] != "missed"]
    conf = defaultdict(Counter)
    for r in rows:
        conf[r["gt"]][r["pred"]] += 1
    tgt = [r for r in found if r["gt"] in TARGET]
    non = [r for r in found if r["gt"] not in TARGET]
    dts = np.array([r["dt_ms"] for r in found if r["dt_ms"] is not None])
    dtf = np.array([r["dt_ms"] / 1000 * r["fps"] for r in found if r["dt_ms"] is not None])
    rep = {
        "n_videos": len(per_video), "n_labelled_strokes": n_gt,
        "detection": {"recall": round(len(found) / n_gt, 3),
                      "precision": round(sum(v["matched"] for v in per_video) / max(1, sum(v["n_sys"] for v in per_video)), 3),
                      "false_swings_called_target": sum(v["false_swings_target"] for v in per_video)},
        "recognition": {"target_found": len(tgt),
                        "target_correct": sum(r["pred"] == r["gt"] for r in tgt),
                        "target_rejected_unknown": sum(r["pred"] == "unknown" for r in tgt),
                        "target_wrong_class": sum(r["pred"] not in (r["gt"], "unknown") for r in tgt),
                        "nontarget_found": len(non), "nontarget_rejected": sum(r["pred"] == "unknown" for r in non),
                        "confusion(gt->pred)": {k: dict(v) for k, v in conf.items()}},
        "contact_error": {"n": int(len(dts)), "median_ms": float(np.median(dts)) if len(dts) else None,
                          "mae_ms": float(np.mean(np.abs(dts))) if len(dts) else None,
                          "p90_abs_ms": float(np.percentile(np.abs(dts), 90)) if len(dts) else None,
                          "within_1_frame": float(np.mean(np.abs(dtf) <= 1)) if len(dtf) else None,
                          "within_3_frames": float(np.mean(np.abs(dtf) <= 3)) if len(dtf) else None,
                          "by_stroke_median_ms": {k: float(np.median([r["dt_ms"] for r in found if r["gt"] == k and r["dt_ms"] is not None]))
                                                  for k in sorted({r["gt"] for r in found if r["dt_ms"] is not None})}},
        "hand": {"videos_with_gt": sum(1 for v in per_video if v["hand_gt"]),
                 "correct": sum(1 for v in per_video if v["hand_gt"] and v["hand_gt"] == v["hand_sys"])},
        "per_video": per_video, "annotator_agreement": agree, "rows": rows,
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    d, r_, c = rep["detection"], rep["recognition"], rep["contact_error"]
    md = [f"# 真实场景评测（{rep['n_videos']} 段视频，{n_gt} 次标注挥拍）", "",
          f"* 挥拍检出率 {d['recall']:.0%}，检出的挥拍里是真实击球的比例 {d['precision']:.0%}；"
          f"没有对应标注却被判成目标动作的挥拍 {d['false_swings_called_target']} 次",
          f"* 目标动作（正手/单反/双反/发球）{r_['target_found']} 次：判对 {r_['target_correct']}，拒识 {r_['target_rejected_unknown']}，"
          f"判错类别 {r_['target_wrong_class']}",
          f"* 非目标动作（高压/截击/其他）{r_['nontarget_found']} 次：拒识 {r_['nontarget_rejected']}",
          (f"* 击球时刻误差（系统 − 标注）：中位数 {c['median_ms']:+.0f} ms，平均绝对误差 {c['mae_ms']:.0f} ms，"
           f"90% 在 {c['p90_abs_ms']:.0f} ms 以内；±1 帧内 {c['within_1_frame']:.0%}，±3 帧内 {c['within_3_frames']:.0%}") if c["n"] else "* 击球时刻：无可比数据",
          f"* 持拍手：{rep['hand']['correct']}/{rep['hand']['videos_with_gt']} 段视频正确", "",
          "| 标注 → 系统 | " + " | ".join(["forehand", "backhand_1h", "backhand_2h", "serve", "unknown", "missed"]) + " |",
          "|---|" + "---|" * 6]
    for k in ["forehand", "backhand_1h", "backhand_2h", "serve", "smash", "volley", "other"]:
        if k in conf:
            md.append(f"| {k} | " + " | ".join(str(conf[k].get(p, 0)) for p in ["forehand", "backhand_1h", "backhand_2h", "serve", "unknown", "missed"]) + " |")
    if agree:
        md += ["", "## 标注一致性", ""] + [f"* {x['video']}：{x['a']} {x['n_a']} 条 / {x['b']} {x['n_b']} 条，对上 {x['matched']} 条，"
                                          f"动作一致 {x['same_stroke']}，击球帧差中位数 {x['frame_diff_abs_median']} 帧（最大 {x['frame_diff_abs_max']}）"
                                          for x in agree]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
