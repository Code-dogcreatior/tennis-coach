"""End-to-end analysis of one video: pose -> clean -> swings -> hand/view -> classify -> phases -> metrics."""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from . import config
from .classify import Classification, combine_with_knn, rule_classify
from .features import compute_metrics, curves, frame_features
from .pose import PoseSeq, cuts_cached, extract_cached
from .quality import check as quality_check
from .skeleton import clean, normalise, torso_track
from . import technique
from .skeleton import L_WRI, R_WRI
from .swing import (combine_hand, detect_swings, estimate_hand, nanfill, segment_phases, shot_boundary_mask,
                    speed)
from .view import VIEW_KEY_LABELS, canonicalise, estimate_view

log = logging.getLogger(__name__)

TYPICAL_PEAK = 10.0   # torso/s, only used to window slow-motion clips of unknown factor
TIME_METRICS = {"dur_backswing", "dur_forward", "dur_follow", "dur_toss", "dur_drop_to_contact", "peak_speed",
                "elbow_before_wrist", "tempo_ratio"}


def _add_video_time(obj, scale: float):
    """Recursively add 'tv'/'tv0'/'tv1' (video-time seconds) next to analysis-time keys t/t0/t1."""
    if isinstance(obj, dict):
        for k in ("t", "t0", "t1"):
            if k in obj and isinstance(obj[k], (int, float)):
                obj["tv" + k[1:]] = obj[k] / scale
        for v in obj.values():
            _add_video_time(v, scale)
    elif isinstance(obj, list):
        for v in obj:
            _add_video_time(v, scale)


def merge_serve_preparation(P, t, windows, fps_p, shot):
    """A serve's preparation (racket swing-down and lift into the trophy position) can form its own wrist
    speed peak. If a swing shows the overhead serve pattern (either hand assignment), swallow earlier peaks
    within 2.5 s as its preparation and extend its window backwards."""
    from dataclasses import replace

    from .swing import serve_pattern
    if len(windows) < 2:
        return windows
    keep = list(windows)
    for k in range(len(windows) - 1, 0, -1):
        w = windows[k]
        if w not in keep:
            continue
        if not (serve_pattern(P, t, w, R_WRI, L_WRI)[0] or serve_pattern(P, t, w, L_WRI, R_WRI)[0]):
            continue
        lo = max(0, w.i_peak - int(2.5 * fps_p))
        if shot is not None:
            lo = max(lo, int(np.where(shot == shot[w.i_peak])[0].min()))
        prev = [p for p in keep if p is not w and p.i_peak < w.i_peak and p.i_peak >= lo]
        if prev:
            for p in prev:
                keep.remove(p)
            new_i0 = max(lo, min(p.i0 for p in prev))
            keep[keep.index(w)] = replace(w, i0=new_i0, t0=float(t[new_i0]),
                                          truncated_start=(w.i_peak - new_i0) < 0.6 * fps_p)
    return keep


def analyse_video(video_path: str, hand_override: str | None = None, view_override: dict | None = None,
                  action_overrides: dict | None = None, refs: list[dict] | None = None,
                  progress: Callable[[str, float], None] | None = None, pose: PoseSeq | None = None,
                  time_scale: float | None = 1.0) -> dict:
    """time_scale: real seconds per video second (1.0 = real time, 0.25 = 4x slow motion, None = unknown).
    With an unknown time scale, durations/speeds are not comparable and are marked unreliable."""
    """Analyse every swing found in a video.

    hand_override: "right"/"left" (user correction or metadata) - takes precedence over estimation.
    view_override: {"view": "side|front|back|oblique_front|oblique_back", "facing": "..."}
    action_overrides: {swing_index: action}
    refs: reference swings (see ``swing_record``) used for kNN-assisted recognition.
    """
    prog = progress or (lambda stage, p: None)
    t_start = time.perf_counter()
    cache_hit = True
    if pose is None:
        prog("pose", 0.0)
        pose, cache_hit = extract_cached(video_path, progress=lambda p: prog("pose", p))
    prog("analyse", 0.0)
    cs = clean(pose.t, pose.kpts, pose.scores)
    tt, torso_adaptive = torso_track(cs.K, cs.fps)
    P, Hip, torso = normalise(cs.K, tt if torso_adaptive else None)
    # --- shots (edits / camera cuts): never compute motion across a cut ---
    try:
        cuts = cuts_cached(video_path)
        fi = pose.frame_index   # pose may be strided (high-fps input): cuts are per video frame
        if len(cuts) and len(fi) == len(pose.t) and int(fi[-1]) < len(cuts):
            shot_src = np.cumsum(cuts)[fi]
        else:
            shot_src = np.zeros(len(pose.t), int)
    except Exception:  # noqa: BLE001
        shot_src = np.zeros(len(pose.t), int)
    shot = shot_src[np.clip(cs.src_index, 0, len(shot_src) - 1)]
    n_shots = int(shot.max()) + 1 if len(shot) else 1
    bmask = shot_boundary_mask(shot, len(cs.t))
    # --- time scale: real seconds per video second (slow motion < 1) ---
    if time_scale is not None:
        ts_used, time_known = float(time_scale), True
    else:
        sp = np.maximum(nanfill(speed(P[:, L_WRI], cs.t)), nanfill(speed(P[:, R_WRI], cs.t)))
        sp[bmask] = 0
        s995 = float(np.percentile(sp, 99.5))
        # provisional scale for *windowing only*; assumes a full swing reaches ~10 torso/s in real time
        ts_used, time_known = float(np.clip(s995 / TYPICAL_PEAK, 0.05, 1.0)), False
    t = (cs.t - cs.t[0]) * ts_used
    windows = detect_swings(P, t, shot=shot if n_shots > 1 else None)
    fps_p = 1.0 / float(np.median(np.diff(t)))
    windows = merge_serve_preparation(P, t, windows, fps_p, shot)

    def ready_frames(w):
        """Frames in the ready position before the take-back and after the finish: the body is roughly
        square to the net there, so they reflect the camera position rather than trunk rotation."""
        pre = np.arange(max(0, w.i_peak - int(1.6 * fps_p)), max(0, w.i_peak - int(0.7 * fps_p)))
        post = np.arange(min(len(t), w.i_peak + int(0.6 * fps_p)), min(len(t), w.i_peak + int(1.2 * fps_p)))
        idx = np.concatenate([pre, post]).astype(int)
        return idx if len(idx) >= 5 else np.arange(w.i0, w.i1 + 1)

    views = []
    for w in windows:
        v = estimate_view(P, cs.S, ready_frames(w))
        if view_override and view_override.get("view"):
            v.view = view_override["view"]
            v.facing = view_override.get("facing", v.facing)
            v.confidence = 1.0
            v.evidence = v.evidence + ["机位已由用户指定"]
        views.append(v)
    hand_est = [estimate_hand(P, t, w, v.view) for w, v in zip(windows, views)]
    hand_all = combine_hand(hand_est)
    hand = hand_override or hand_all.hand or "right"
    hand_info = {"hand": hand, "estimated": hand_all.hand, "confidence": round(hand_all.confidence, 3),
                 "evidence": hand_all.evidence, "source": "user" if hand_override else
                 ("estimated" if hand_all.hand else "default_right_unverified"),
                 "per_swing": [{"hand": e.hand, "confidence": round(e.confidence, 3), "evidence": e.evidence,
                                "two_handed_like": e.two_handed_like} for e in hand_est]}
    all_ready = np.unique(np.concatenate([ready_frames(w) for w in windows])) if windows else None
    view_all = estimate_view(P, cs.S, all_ready)
    swings = []
    for k, w in enumerate(windows):
        v = views[k]
        C, Sc, key, sign = canonicalise(P, cs.S, hand, v)
        hsign = (-1.0 if hand == "left" else 1.0) * sign
        Hc = Hip.copy(); Hc[:, 0] *= hsign
        cls = rule_classify(C, t, w, key)
        sw = {"index": k, "window": w.to_dict(), "view": v.to_dict(), "view_key": key,
              "view_key_label": VIEW_KEY_LABELS.get(key, key), "hand": hand, "x_sign": hsign,
              "fps": fps_p, "fps_video": cs.fps, "torso_px": torso}
        F = frame_features(C)
        if refs:
            from .match import knn_vote  # local import to avoid cycle
            tmp = {"feat": F[w.i0:w.i1 + 1], "feat_offset": w.i0, "contact_i": w.i_peak, "fps": fps_p,
                   "view_key": key}
            cls = combine_with_knn(cls, knn_vote(tmp, refs, strategy="balanced"))
        else:
            cls = combine_with_knn(cls, None)
        action = cls.action
        if action_overrides and k in action_overrides:
            action = action_overrides[k]
            cls = Classification(action, 1.0, cls.evidence + ["动作类型已由用户指定"], cls.rule_action,
                                 cls.rule_confidence, cls.knn)
        sw["classification"] = cls.to_dict()
        sw["action"] = action
        q = quality_check(pose, None, swing_ok=not (w.truncated_start and w.truncated_end),
                          i0=int(cs.src_index[w.i0]), i1=int(cs.src_index[w.i1]) + 1)
        sw["quality"] = q
        if action != "unknown":
            ph = segment_phases(C, t, w, action)
            sw["phases"] = ph.to_dict()
            sw["contact_i"] = ph.events["contact"]["i"]
            sw["metrics"] = compute_metrics(C, Hc, t, ph, key)
            sw["technique"] = technique.compute(C, t, ph, action, key)
            if torso_adaptive:
                for mk in {"hip_shift", "leg_drive"} & set(sw["metrics"]):
                    sw["metrics"][mk]["reliable"] = False
                    sw["metrics"][mk]["why_unreliable"] = "画面中人体大小变化明显（镜头推拉/跟拍），髋部全局位移不可比"
            if not time_known:
                for mk in TIME_METRICS & set(sw["metrics"]):
                    sw["metrics"][mk]["reliable"] = False
                    sw["metrics"][mk]["why_unreliable"] = "视频时间尺度未知（可能为倍率未知的慢动作），时长/速度不可比"
            i0, i1 = ph.events["start"]["i"], ph.events["end"]["i"]
        else:
            sw["phases"] = None
            sw["contact_i"] = w.i_peak
            sw["metrics"] = {}
            sw["technique"] = {}
            i0, i1 = w.i0, w.i1
        sw["feat"] = F[w.i0:w.i1 + 1]
        sw["feat_offset"] = w.i0
        sw["curves"] = curves(C[w.i0:w.i1 + 1], t[w.i0:w.i1 + 1])
        sw["analysed_range"] = [float(t[i0]), float(t[i1])]
        swings.append(sw)
        prog("analyse", (k + 1) / max(1, len(windows)))
    overall_q = quality_check(pose, video_path, swing_ok=bool(windows))
    if n_shots > 1:
        overall_q["issues"].append({"code": "edited", "level": "warn", "value": n_shots,
                                    "message": f"检测到 {n_shots} 个镜头（剪辑/转场）。首版面向连续单机位拍摄；"
                                               "每个镜头已单独分析，跨镜头的动作会被截断。"})
        if overall_q["level"] == "good":
            overall_q["level"] = "warn"
    for sw in swings:
        sw["window"]["tv_peak"] = sw["window"]["t_peak"] / ts_used
        sw["curves"]["tv"] = [round(x / ts_used, 4) for x in sw["curves"]["t"]]
        sw["analysed_range_v"] = [x / ts_used for x in sw["analysed_range"]]
    _add_video_time(swings, ts_used)
    return {
        "time_scale": ts_used, "time_known": time_known, "video_t0": float(cs.t[0]), "n_shots": n_shots,
        "analysis_version": config.ANALYSIS_VERSION, "pose_model": pose.model_id,
        "pose_meta": pose.meta, "pose_cache_hit": cache_hit, "fps_grid": cs.fps, "duration": float(t[-1]),
        "lr_swaps_fixed": cs.lr_swaps, "torso_px": torso, "torso_adaptive": torso_adaptive,
        "hand": hand_info, "view_overall": view_all.to_dict(), "quality": overall_q,
        "swings": swings, "n_swings": len(swings),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }


def swing_record(sw: dict, ref_id, extra: dict | None = None) -> dict:
    """Compact record used as a reference in matching / kNN."""
    r = {"ref_id": ref_id, "action": sw["action"], "view_key": sw["view_key"], "feat": np.asarray(sw["feat"]),
         "feat_offset": sw["feat_offset"], "contact_i": sw["contact_i"], "fps": sw["fps"], "phases": sw["phases"],
         "metrics": sw["metrics"], "technique": sw.get("technique") or {}, "hand": sw.get("hand"), "view": sw.get("view"),
         "window": sw.get("window") or {}, "quality": sw.get("quality") or {}}
    if extra:
        r.update(extra)
    return r


def strip_arrays(result: dict) -> dict:
    """JSON-safe copy (drops raw feature arrays)."""
    out = dict(result)
    out["swings"] = [{k: v for k, v in sw.items() if k != "feat"} for sw in result["swings"]]
    return out
