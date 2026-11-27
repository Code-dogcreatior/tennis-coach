"""Video / pose quality checks with concrete re-shoot reasons."""
from __future__ import annotations

import cv2
import numpy as np

from .config import KPT_THR
from .pose import PoseSeq
from .video import iter_frames

FACE = [0, 1, 2]
ANKLES = [15, 16]
ARM = [5, 6, 7, 8, 9, 10]


def check(seq: PoseSeq, video_path: str | None = None, swing_ok: bool | None = None,
          i0: int = 0, i1: int | None = None) -> dict:
    """Return {"ok": bool, "level": "good|warn|bad", "issues": [{code, level, message, value}]}.

    i0/i1 restrict the check to a frame range (e.g. the analysed swing).
    """
    i1 = len(seq.t) if i1 is None else i1
    K, S, B = seq.kpts[i0:i1], seq.scores[i0:i1], seq.bbox[i0:i1]
    T = len(K)
    issues = []

    def add(code, level, msg, value=None):
        issues.append({"code": code, "level": level, "message": msg, "value": value})

    has = np.isfinite(K[:, :, 0]).all(1) & (S.mean(1) > 0.25)
    det_rate = float(has.mean()) if T else 0.0
    if det_rate < 0.6:
        add("no_person", "bad", f"只有 {det_rate:.0%} 的帧检测到完整人体。请确保球员全程在画面内、光线充足。", det_rate)
    elif det_rate < 0.9:
        add("person_dropout", "warn", f"{1 - det_rate:.0%} 的帧人体检测丢失，部分时间段的结果会插值或缺失。", det_rate)
    if has.any():
        Kh, Sh, Bh = K[has], S[has], B[has]
        w, h = seq.width, seq.height
        margin = 0.01
        inside = ((Kh[..., 0] > w * margin) & (Kh[..., 0] < w * (1 - margin)) &
                  (Kh[..., 1] > h * margin) & (Kh[..., 1] < h * (1 - margin)))
        feet_vis = float(((Sh[:, ANKLES] > KPT_THR) & inside[:, ANKLES]).all(1).mean())
        head_vis = float(((Sh[:, FACE] > KPT_THR).any(1) & inside[:, FACE].any(1)).mean())
        if feet_vis < 0.7:
            add("feet_cut", "bad" if feet_vis < 0.4 else "warn",
                f"仅 {feet_vis:.0%} 的帧双脚清晰可见。请拉远机位或调整角度，保证全身（含脚）入镜。", feet_vis)
        if head_vis < 0.7:
            add("head_cut", "warn", f"仅 {head_vis:.0%} 的帧头部可见（背面机位时属正常）。", head_vis)
        arm_conf = float(np.median(Sh[:, ARM]))
        if arm_conf < 0.45:
            add("arm_occluded", "warn", f"手臂关键点置信度偏低（中位 {arm_conf:.2f}），可能被身体遮挡或模糊。", arm_conf)
        hfrac = float(np.nanmedian((Bh[:, 3] - Bh[:, 1]) / h))
        if hfrac < 0.18:
            add("person_small", "warn" if hfrac >= 0.1 else "bad",
                f"球员在画面中只占约 {hfrac:.0%} 高度，关节定位误差会变大。建议靠近或使用更高分辨率（球员高度最好 ≥ 1/3 画面）。", hfrac)
    det_frames = seq.n_persons[i0:i1] >= 0
    if det_frames.any():
        others = seq.other_area[i0:i1][det_frames]
        multi = float((others > 0.35).mean())
        if multi > 0.2:
            add("multi_person", "warn",
                f"{multi:.0%} 的检测帧中存在另一个较大的人。首版只分析单人，可能跟踪错人；请尽量单人入镜。", multi)
    if seq.fps < 24:
        add("low_fps", "warn", f"帧率 {seq.fps:.0f} fps 偏低，击球附近时刻误差约 ±{1000 / seq.fps:.0f} ms，建议 60 fps 以上。", seq.fps)
    if video_path:
        sharp = _sharpness(video_path, seq, i0, i1)
        if sharp is not None and sharp < 30:
            add("blurry", "warn", f"人体区域清晰度偏低（拉普拉斯方差 {sharp:.0f}），可能失焦或运动模糊。", sharp)
    if swing_ok is False:
        add("no_swing", "bad", "未检测到完整挥拍（手腕速度没有明显峰值或挥拍被截断）。请录制一次从准备到收拍的完整动作。")
    level = "bad" if any(i["level"] == "bad" for i in issues) else "warn" if issues else "good"
    return {"ok": level != "bad", "level": level, "issues": issues}


def _sharpness(video_path: str, seq: PoseSeq, i0: int, i1: int, n: int = 6) -> float | None:
    want = sorted(set(np.linspace(i0, max(i0, i1 - 1), n).astype(int).tolist()))
    vals = []
    cap = cv2.VideoCapture(str(video_path))
    for i in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq.video_frame(i))
        ok, f = cap.read()
        if ok and np.isfinite(seq.bbox[i]).all():
            x0, y0, x1, y1 = seq.bbox[i].astype(int)
            crop = f[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if crop.size > 100:
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                vals.append(cv2.Laplacian(g, cv2.CV_64F).var())
    cap.release()
    return float(np.median(vals)) if vals else None
