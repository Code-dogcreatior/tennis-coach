"""Camera-view estimation and canonicalisation.

The camera view is estimated from statistics over the whole clip (not a single frame), so that
a player turning the body during the swing is not mistaken for a camera change:

* hip width / torso length (hips rotate much less than shoulders during a stroke),
* left/right ordering of shoulders + hips on screen (person facing towards vs away from camera),
* visibility of face keypoints,
* nose offset from the ear/shoulder centre (facing direction in side views).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .skeleton import (L_EAR, L_EYE, L_HIP, L_SHO, NOSE, R_EAR, R_EYE, R_HIP, R_SHO, mirror_lr)

VIEW_LABELS = {"side": "侧面", "front": "正面", "back": "背面", "oblique_front": "斜前方", "oblique_back": "斜后方",
               "unknown": "未知"}

SIDE_MAX_HIP = 0.20      # median hip width / torso below which the camera is treated as side-on
FRONT_MIN_HIP = 0.34     # above which the camera is treated as front/back


@dataclass
class ViewEstimate:
    view: str                  # side / front / back / oblique_front / oblique_back / unknown
    facing: str                # screen_left / screen_right / towards / away / unknown
    confidence: float
    hip_width: float
    lr_order: float            # fraction of frames with person's right on screen-left (=facing camera)
    face_score: float
    evidence: list

    def to_dict(self):
        d = asdict(self); d["label"] = VIEW_LABELS.get(self.view, self.view); return d


def estimate_view(P: np.ndarray, S: np.ndarray, frames=None) -> ViewEstimate:
    """P: body-space (T,17,2) BEFORE any mirroring; S: scores. frames: slice or index array."""
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    if frames is not None:
        P, S = P[frames], S[frames]
    hipw = np.abs(P[:, L_HIP, 0] - P[:, R_HIP, 0])
    shw = np.abs(P[:, L_SHO, 0] - P[:, R_SHO, 0])
    ok = np.isfinite(hipw)
    if ok.sum() < 5:
        return ViewEstimate("unknown", "unknown", 0.0, float("nan"), float("nan"), float("nan"), ["有效骨架帧太少"])
    hw = float(np.nanmedian(hipw))
    sw = float(np.nanmedian(shw))
    # person's right joints appear on screen-left when facing the camera
    order = np.concatenate([(P[:, R_SHO, 0] < P[:, L_SHO, 0])[ok], (P[:, R_HIP, 0] < P[:, L_HIP, 0])[ok]])
    lr = float(order.mean())
    face = float(np.nanmean(S[:, [NOSE, L_EYE, R_EYE]]))
    ear_c = np.nanmean(P[:, [L_EAR, R_EAR], 0], axis=1)
    sho_c = np.nanmean(P[:, [L_SHO, R_SHO], 0], axis=1)
    nose_off = float(np.nanmedian(P[:, NOSE, 0] - np.where(np.isfinite(ear_c), ear_c, sho_c)))
    ev = [f"髋宽/躯干长 中位数 {hw:.2f}（侧面 < {SIDE_MAX_HIP}，正/背面 > {FRONT_MIN_HIP}）",
          f"肩宽/躯干长 中位数 {sw:.2f}", f"人体右侧在画面左边的帧占比 {lr:.0%}", f"面部关键点平均置信度 {face:.2f}",
          f"鼻子相对耳朵中心水平偏移 {nose_off:+.2f}"]
    if hw < SIDE_MAX_HIP:
        view = "side"
        facing = "screen_right" if nose_off > 0 else "screen_left"
        conf = float(np.clip((SIDE_MAX_HIP - hw) / 0.1, 0, 1)) * float(np.clip(abs(nose_off) / 0.08, 0.3, 1))
    else:
        towards = lr > 0.5
        # face visibility must agree with left/right ordering for a confident front/back call
        agree = (towards and face > 0.45) or ((not towards) and face < 0.55)
        if hw > FRONT_MIN_HIP:
            view = "front" if towards else "back"
            conf = float(np.clip((hw - FRONT_MIN_HIP) / 0.1 + 0.3, 0, 1))
        else:
            view = "oblique_front" if towards else "oblique_back"
            conf = 0.5
        facing = "towards" if towards else "away"
        conf *= float(np.clip(abs(lr - 0.5) * 2, 0, 1)) * (1.0 if agree else 0.6)
        if not agree:
            ev.append("左右顺序与面部可见性不一致，正/背面判断置信度下调")
    return ViewEstimate(view, facing, float(conf), hw, lr, face, ev)


def mirror_view(v: ViewEstimate) -> ViewEstimate:
    """View of the mirrored (handedness-canonicalised) sequence: side views swap facing."""
    f = {"screen_left": "screen_right", "screen_right": "screen_left"}.get(v.facing, v.facing)
    return ViewEstimate(v.view, f, v.confidence, v.hip_width, v.lr_order, v.face_score, list(v.evidence))


def view_key(view: str, facing: str) -> str:
    """Comparable view class *after* handedness canonicalisation (right-handed).

    For side views, facing direction + handedness determine which side the camera is on:
    a right-hander facing screen-right is filmed from his hitting-arm side.
    """
    if view == "side":
        return "side_hitting" if facing == "screen_right" else "side_nonhitting"
    return view


VIEW_KEY_LABELS = {"side_hitting": "侧面（持拍手一侧）", "side_nonhitting": "侧面（非持拍手一侧）", "front": "正面",
                   "back": "背面", "oblique_front": "斜前方", "oblique_back": "斜后方", "unknown": "未知"}

# views considered close enough to compare directly
COMPATIBLE = {
    "side_hitting": {"side_hitting"}, "side_nonhitting": {"side_nonhitting"},
    "front": {"front", "oblique_front"}, "oblique_front": {"oblique_front", "front"},
    "back": {"back", "oblique_back"}, "oblique_back": {"oblique_back", "back"},
}


def canonicalise(P: np.ndarray, S: np.ndarray, hand: str, view: ViewEstimate):
    """Return canonical-space sequence (right-handed, x sign convention per view) + canonical view key.

    1. left-hander -> mirror_lr (x flip + label swap) => equivalent right-hander.
    2. x sign convention: side views +x = facing direction (towards the net);
       front/back views +x = hitting-arm side. Pure sign convention within one view class.
    """
    C, Sc = P.copy(), S.copy()
    v = view
    if hand == "left":
        C, Sc = mirror_lr(C, Sc)
        v = mirror_view(view)
    key = view_key(v.view, v.facing)
    sign = 1.0
    if v.view == "side":
        sign = 1.0 if v.facing == "screen_right" else -1.0
    elif v.view in ("front", "oblique_front"):
        sign = -1.0   # right-hander facing camera: hitting side is on screen-left
    C[..., 0] *= sign
    return C, Sc, key, sign
