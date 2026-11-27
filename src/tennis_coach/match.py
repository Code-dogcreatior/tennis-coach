"""Reference selection, phase-anchored DTW matching and metric comparison against a reference group."""
from __future__ import annotations

import numpy as np

from .classify import ACTION_LABELS
from .dtw import dtw, phase_dtw
from .features import FEATURE_WEIGHTS, METRIC_INFO
from .view import COMPATIBLE, VIEW_KEY_LABELS

# minimal differences considered meaningful (avoid flagging noise when the reference group is tight)
TOLERANCE = {
    "dur_backswing": 0.08, "dur_forward": 0.05, "dur_follow": 0.1, "tempo_ratio": 0.35, "dur_toss": 0.12,
    "dur_drop_to_contact": 0.08, "peak_speed": 2.0, "elbow_contact": 15.0, "trophy_elbow": 15.0,
    "knee_min": 12.0, "trophy_knee": 12.0, "backswing_amp": 0.25, "backswing_behind": 0.2,
    "backswing_height": 0.2, "contact_front": 0.2, "contact_height": 0.2, "follow_height": 0.25,
    "follow_path": 0.4, "follow_cross": 0.2, "elbow_before_wrist": 25.0, "hip_shift": 0.2, "toss_height": 0.25,
    "drop_depth": 0.2, "contact_above_head": 0.2, "leg_drive": 0.1,
}

KNN_WINDOW = (-1.0, 0.6)    # seconds around estimated contact used for action-agnostic kNN


def phase_segments(sw: dict) -> list[tuple[int, int]]:
    """Index ranges (relative to the swing's feature array) used for anchored DTW."""
    e = sw["phases"]["events"]
    off = int(sw["feat_offset"])
    if e.get("trophy") is not None:
        names = ["start", "trophy", "drop", "contact", "end"]
    else:
        names = ["start", "backswing_end", "contact", "end"]
    idx = [int(e[n]["i"]) - off for n in names]
    return [(idx[k], idx[k + 1] + 1) for k in range(len(idx) - 1)]


def knn_slice(sw: dict) -> np.ndarray:
    F = sw["feat"]
    fps = float(sw["fps"])
    c = int(sw["contact_i"]) - int(sw["feat_offset"])
    a, b = max(0, c + int(KNN_WINDOW[0] * fps)), min(len(F), c + int(KNN_WINDOW[1] * fps) + 1)
    return F[a:b]


def resample(F: np.ndarray, n: int = 48) -> np.ndarray:
    if len(F) < 2:
        return np.repeat(F, n, axis=0) if len(F) else np.zeros((n, F.shape[1]))
    x = np.linspace(0, 1, len(F))
    xi = np.linspace(0, 1, n)
    return np.column_stack([np.interp(xi, x, F[:, c]) for c in range(F.shape[1])])


STRATEGIES = ("plain", "source", "balanced")
BALANCED_M = 2   # sources per action averaged in the balanced score


def _source(r: dict):
    """Who a reference swing comes from: one player (or one video) counts once per action."""
    return r.get("player") or r.get("video_id") or r["ref_id"]


def knn_costs(q: dict, refs: list[dict]) -> list[tuple[float, str, object, object]]:
    """(cost, action, ref_id, source) for every view-compatible reference, cheapest first."""
    cands = [r for r in refs if r["view_key"] in COMPATIBLE.get(q["view_key"], {q["view_key"]})]
    qf = resample(knn_slice(q))
    out = [(float(dtw(qf, resample(knn_slice(r)), FEATURE_WEIGHTS, band=0.2)[0]), r["action"], r["ref_id"], _source(r))
           for r in cands]
    out.sort(key=lambda x: x[0])
    return out


def knn_vote(q: dict, refs: list[dict], k: int = 3, strategy: str = "plain",
             costs: list | None = None) -> dict | None:
    """Nearest-reference vote. strategy (see STRATEGIES):
    plain    - k nearest swings (a player with many swings in the library can fill all k slots)
    source   - k nearest *sources*: each player/video contributes only its closest swing per action
    balanced - per action, mean cost of its BALANCED_M closest sources; lowest wins (library size per action
               no longer decides). The rejection cost is the overall nearest swing in every strategy."""
    scored = costs if costs is not None else knn_costs(q, refs)
    if not scored:
        return None
    if strategy != "plain":
        seen, dedup = set(), []
        for s in scored:
            if (s[1], s[3]) not in seen:
                seen.add((s[1], s[3])); dedup.append(s)
        pool = dedup
    else:
        pool = scored
    top = [(c, a, rid) for c, a, rid, *_ in pool[:k]]
    votes: dict = {}
    for _, a, _ in top:
        votes[a] = votes.get(a, 0) + 1
    label = max(votes, key=votes.get)
    class_score = None
    if strategy == "balanced":
        per: dict = {}
        for c, a, *_ in pool:
            if len(per.setdefault(a, [])) < BALANCED_M:
                per[a].append(c)
        class_score = {a: float(np.mean(v)) for a, v in per.items()}
        label = min(class_score, key=class_score.get)
    nearest = float(scored[0][0])
    # threshold from within-library nearest-neighbour costs (fallback default)
    thr = q.get("knn_cost_threshold") or 1.6
    out = {"label": label, "votes": votes, "k": len(top), "nearest_cost": nearest, "strategy": strategy,
           "cost_threshold": float(thr), "neighbours": [{"cost": round(float(c), 3), "action": a, "ref_id": rid}
                                                     for c, a, rid in top]}
    if class_score is not None:
        out["class_score"] = {a: round(v, 3) for a, v in class_score.items()}
    return out


def match(q: dict, refs: list[dict], top_k: int = 3) -> dict:
    """Match one analysed query swing against reference swings.

    Returns {"status", "message", "candidates": [...], "group": {...}, "comparison": {...}}.
    status: ok | no_refs | view_mismatch | action_unknown
    """
    if q["action"] == "unknown":
        return {"status": "action_unknown", "message": "动作未识别为四类目标动作，不做参考匹配。可人工指定动作类型后重新匹配。",
                "candidates": [], "comparison": {}}
    same_action = [r for r in refs if r["action"] == q["action"]]
    if not same_action:
        return {"status": "no_refs", "message": f"参考库中还没有「{ACTION_LABELS[q['action']]}」的参考，请先上传示范视频。",
                "candidates": [], "comparison": {}}
    compat = COMPATIBLE.get(q["view_key"], {q["view_key"]})
    pool = [r for r in same_action if r["view_key"] in compat]
    if not pool:
        have = sorted({VIEW_KEY_LABELS.get(r["view_key"], r["view_key"]) for r in same_action})
        return {"status": "view_mismatch",
                "message": (f"当前视频机位为「{VIEW_KEY_LABELS.get(q['view_key'], q['view_key'])}」，参考库中该动作只有："
                            f"{'、'.join(have)}。不同机位的 2D 骨架无法可靠对比，请换用同机位参考或按参考机位重拍。"),
                "candidates": [], "comparison": {}}
    segq = phase_segments(q)
    cands = []
    for r in pool:
        cost, path, seg_costs = phase_dtw(q["feat"], r["feat"], segq, phase_segments(r), FEATURE_WEIGHTS)
        cands.append({"ref_id": r["ref_id"], "cost": round(cost, 4), "segment_costs": [round(c, 4) for c in seg_costs],
                      "path": [(int(i), int(j)) for i, j in path[:: max(1, len(path) // 200)]],
                      "review_status": r.get("review_status", "demo"), "player": r.get("player"),
                "video_id": r.get("video_id"), "swing_index": r.get("swing_index"), "tier": r.get("tier"),
                      "style": r.get("style")})
    cands.sort(key=lambda c: c["cost"])
    comparison = compare_metrics(q["metrics"], [r["metrics"] for r in pool])
    seg_names = ["引拍", "加速", "随挥"] if q["action"] != "serve" else ["抛球/蓄力", "下落", "加速", "随挥"]
    best = cands[0]
    worst_seg = int(np.argmax(best["segment_costs"])) if best["segment_costs"] else None
    basis = (f"按动作类型「{ACTION_LABELS[q['action']]}」+ 机位「{VIEW_KEY_LABELS.get(q['view_key'])}」"
             f"筛选出 {len(pool)} 条参考；以检测到的阶段事件为锚点逐阶段做 DTW（特征：持拍腕/肘、非持拍腕、双肩相对髋部位置，"
             f"肘角、膝角、躯干倾角）。最接近的参考 DTW 平均距离 {best['cost']:.2f}"
             + (f"，差异最大的阶段：{seg_names[worst_seg]}" if worst_seg is not None and worst_seg < len(seg_names) else ""))
    return {"status": "ok", "message": basis, "candidates": cands[:top_k], "n_pool": len(pool),
            "segment_names": seg_names, "comparison": comparison}


def compare_metrics(qm: dict, ref_metrics: list[dict]) -> dict:
    out = {}
    for key, m in qm.items():
        vals = [rm[key]["value"] for rm in ref_metrics if key in rm]
        if not vals:
            continue
        vals = np.asarray(vals, float)
        mean, std = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        lo, hi = float(vals.min()), float(vals.max())
        tol = TOLERANCE.get(key, 0.1)
        if key == "peak_speed":
            tol = max(tol, 0.15 * abs(mean))
        band_lo, band_hi = min(lo, mean - 2 * std) - tol, max(hi, mean + 2 * std) + tol
        v = m["value"]
        status = "within" if band_lo <= v <= band_hi else ("above" if v > band_hi else "below")
        out[key] = {"value": v, "ref_mean": round(mean, 4), "ref_std": round(std, 4), "ref_min": round(lo, 4),
                    "ref_max": round(hi, 4), "n_ref": int(len(vals)), "band": [round(band_lo, 4), round(band_hi, 4)],
                    "status": status, "reliable": m["reliable"], "label": m["label"], "unit": m["unit"],
                    "t0": m["t0"], "t1": m["t1"], "desc": METRIC_INFO[key][3]}
    return out
