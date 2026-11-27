"""Per-frame feature vectors (for DTW) and interpretable 2D swing metrics.

All metrics are 2D *projections* in the canonical body frame (torso-length units, y up). They are
not 3D joint angles, forces or energies. Each metric carries a per-view reliability flag; unreliable
metrics are displayed for transparency but excluded from feedback.
"""
from __future__ import annotations

import numpy as np

from .skeleton import CANON, NOSE, L_HIP, R_HIP, L_SHO, R_SHO, angle
from .swing import Phases, nanfill, speed

H = CANON

FEATURE_NAMES = ["h_wri_x", "h_wri_y", "h_elb_x", "h_elb_y", "n_wri_x", "n_wri_y", "h_sho_x", "h_sho_y",
                 "n_sho_x", "n_sho_y", "elbow_h", "knee_h", "knee_n", "trunk"]
FEATURE_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.5, 0.7])


def frame_features(C: np.ndarray) -> np.ndarray:
    hw, he, nw, hs, ns = C[:, H["h_wri"]], C[:, H["h_elb"]], C[:, H["n_wri"]], C[:, H["h_sho"]], C[:, H["n_sho"]]
    elb = angle(hs, he, hw) / 90.0
    kh = angle(C[:, H["h_hip"]], C[:, H["h_knee"]], C[:, H["h_ank"]]) / 90.0
    kn = angle(C[:, H["n_hip"]], C[:, H["n_knee"]], C[:, H["n_ank"]]) / 90.0
    sc = (hs + ns) / 2
    trunk = np.degrees(np.arctan2(sc[:, 0], sc[:, 1])) / 45.0   # lean of shoulder centre from vertical
    F = np.column_stack([hw, he, nw, hs, ns, elb, kh, kn, trunk])
    for c in range(F.shape[1]):
        F[:, c] = nanfill(F[:, c])
    return F


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

METRIC_INFO = {
    # key: (label, unit, higher_is, description)
    "dur_backswing": ("引拍时长", "s", None, "引拍开始→引拍终点"),
    "dur_forward": ("加速时长", "s", None, "引拍终点→估计击球"),
    "dur_follow": ("随挥时长", "s", None, "估计击球→随挥结束"),
    "tempo_ratio": ("引拍/加速 时长比", "", None, "节奏：引拍时长 ÷ 加速时长"),
    "peak_speed": ("手腕相对峰值速度", "躯干长/s", None, "手腕相对髋部的 2D 投影速度峰值（相对强度，不是球拍速度或真实物理速度）"),
    "elbow_contact": ("击球附近肘角（投影）", "°", None, "持拍侧 肩-肘-腕 的 2D 投影夹角"),
    "backswing_amp": ("引拍幅度", "躯干长", None, "引拍终点到估计击球点的手腕位移"),
    "backswing_behind": ("引拍手腕在身后距离", "躯干长", None, "引拍终点时手腕在髋部中心后方的水平距离（侧面）"),
    "backswing_height": ("引拍终点手腕高度", "躯干长", None, "相对髋部中心，肩部约为 1.0"),
    "contact_front": ("击球点前后位置", "躯干长", None, "估计击球时手腕在髋部中心前方的水平距离（侧面）"),
    "contact_height": ("击球点高度", "躯干长", None, "估计击球时手腕相对髋部中心的高度"),
    "follow_height": ("随挥最高手腕高度", "躯干长", None, "击球后手腕到达的最高位置"),
    "follow_path": ("随挥手腕路径长度", "躯干长", None, "击球后至随挥结束的手腕轨迹长度"),
    "follow_cross": ("随挥越过身体中线", "躯干长", None, "击球后手腕向非持拍侧的最远水平位置（正/背面，负值=越过中线）"),
    "knee_min": ("最小膝角（投影）", "°", None, "引拍到击球之间两膝 2D 投影角度的最小值，越小表示屈膝越多"),
    "elbow_before_wrist": ("肘先于腕的速度峰值时差", "ms", None, "肘部速度峰值早于手腕速度峰值的时间（2D 投影观察）"),
    "hip_shift": ("髋部向前移动", "躯干长", None, "引拍终点→随挥结束 髋部中心沿击球方向的位移（侧面）"),
    # serve
    "dur_toss": ("抛球→奖杯时长", "s", None, "抛球手开始上举→最高点"),
    "dur_drop_to_contact": ("奖杯→击球时长", "s", None, "抛球手最高点→估计击球"),
    "toss_height": ("抛球手最高高度", "躯干长", None, "非持拍手腕最高点相对髋部中心"),
    "trophy_knee": ("奖杯姿势膝角（投影）", "°", None, "抛球手最高点附近两膝 2D 投影最小角"),
    "trophy_elbow": ("奖杯姿势持拍肘角（投影）", "°", None, "抛球手最高点时持拍侧肘角"),
    "drop_depth": ("持拍手腕下落深度", "躯干长", None, "持拍手腕相对持拍肩的最低位置（负值=低于肩）"),
    "contact_above_head": ("击球点高于头部", "躯干长", None, "估计击球时手腕高出鼻子的距离"),
    "leg_drive": ("髋部上升", "躯干长", None, "奖杯→击球 髋部中心最大上升量（2D 投影，蹬地的间接观察）"),
}

SIDE_ONLY = {"backswing_behind", "contact_front", "elbow_before_wrist", "hip_shift", "leg_drive"}
FRONTBACK_ONLY = {"follow_cross"}
LOW_RELIABILITY_FRONTBACK = {"knee_min", "trophy_knee", "elbow_contact", "trophy_elbow"}


def _at(x, i):
    v = x[i]
    return float(v) if np.isfinite(v) else None


def compute_metrics(C: np.ndarray, Hip: np.ndarray, t: np.ndarray, ph: Phases, view_key: str) -> dict:
    """Return {key: {value, unit, label, reliable, note, t0, t1}} for one swing."""
    ev = ph.events
    hw, he = C[:, H["h_wri"]], C[:, H["h_elb"]]
    nw, hs = C[:, H["n_wri"]], C[:, H["h_sho"]]
    sw = nanfill(speed(hw, t))
    se = nanfill(speed(he, t))
    ic, i0, i1 = ev["contact"]["i"], ev["start"]["i"], ev["end"]["i"]
    side = view_key.startswith("side")
    out: dict = {}

    def put(key, val, t0, t1, note=""):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return
        label, unit, _, desc = METRIC_INFO[key]
        reliable, why = True, ""
        if key in SIDE_ONLY and not side:
            reliable, why = False, "该指标需要侧面机位，当前机位下为深度方向，不可靠"
        if key in FRONTBACK_ONLY and side:
            reliable, why = False, "该指标需要正面/背面机位"
        if key in LOW_RELIABILITY_FRONTBACK and not side:
            reliable, why = False, "正/背面机位下该关节角的 2D 投影受深度影响大，暂停用"
        out[key] = {"value": round(float(val), 4), "label": label, "unit": unit, "desc": desc,
                    "reliable": reliable, "why_unreliable": why, "t0": float(t0), "t1": float(t1), "note": note}

    kn = np.fmin(angle(C[:, H["h_hip"]], C[:, H["h_knee"]], C[:, H["h_ank"]]),
                 angle(C[:, H["n_hip"]], C[:, H["n_knee"]], C[:, H["n_ank"]]))
    elb = angle(hs, he, hw)
    tc = t[ic]
    put("peak_speed", sw[ic], tc, tc)
    put("elbow_contact", _at(elb, ic), tc, tc)
    put("contact_height", _at(hw[:, 1], ic), tc, tc)
    if ph.action == "serve":
        it, idp = ev["trophy"]["i"], ev["drop"]["i"]
        put("dur_toss", t[it] - t[i0], t[i0], t[it])
        put("dur_drop_to_contact", t[ic] - t[it], t[it], t[ic])
        put("dur_follow", t[i1] - t[ic], t[ic], t[i1])
        put("toss_height", _at(nw[:, 1], it), t[it], t[it])
        a, b = max(0, it - int(0.3 / np.median(np.diff(t)))), it + 1
        put("trophy_knee", float(np.nanmin(kn[a:min(b + 5, ic)])) if np.isfinite(kn[a:min(b + 5, ic)]).any() else None,
            t[a], t[min(b + 5, ic)])
        put("trophy_elbow", _at(elb, it), t[it], t[it])
        put("drop_depth", _at(hw[:, 1] - hs[:, 1], idp), t[idp], t[idp])
        put("contact_above_head", _at(hw[:, 1] - C[:, NOSE, 1], ic), tc, tc)
        if np.isfinite(Hip[it:ic + 1, 1]).any():
            put("leg_drive", float(np.nanmax(Hip[it:ic + 1, 1]) - Hip[it, 1]) if np.isfinite(Hip[it, 1]) else None,
                t[it], t[ic])
    else:
        ib = ev["backswing_end"]["i"]
        put("dur_backswing", t[ib] - t[i0], t[i0], t[ib])
        put("dur_forward", t[ic] - t[ib], t[ib], t[ic])
        put("dur_follow", t[i1] - t[ic], t[ic], t[i1])
        if t[ic] - t[ib] > 0:
            put("tempo_ratio", (t[ib] - t[i0]) / (t[ic] - t[ib]), t[i0], t[ic])
        put("backswing_amp", float(np.linalg.norm(hw[ib] - hw[ic])), t[ib], t[ic])
        put("backswing_behind", _at(-hw[:, 0], ib), t[ib], t[ib])
        put("backswing_height", _at(hw[:, 1], ib), t[ib], t[ib])
        put("contact_front", _at(hw[:, 0], ic), tc, tc)
        if i1 > ic:
            seg = hw[ic:i1 + 1]
            put("follow_height", float(np.nanmax(seg[:, 1])), tc, t[i1])
            put("follow_path", float(np.nansum(np.linalg.norm(np.diff(seg, axis=0), axis=-1))), tc, t[i1])
            put("follow_cross", float(np.nanmin(seg[:, 0])), tc, t[i1])
        if ic > i0 and np.isfinite(kn[i0:ic + 1]).any():
            put("knee_min", float(np.nanmin(kn[i0:ic + 1])), t[i0], tc)
        a = max(ib, ic - int(0.4 / np.median(np.diff(t))))
        ie = a + int(np.argmax(se[a:ic + 3]))
        put("elbow_before_wrist", (t[ic] - t[ie]) * 1000, t[ie], tc, "正值=肘部速度峰值先出现")
        if np.isfinite(Hip[ib, 0]) and np.isfinite(Hip[i1, 0]):
            # Hip is in body/screen units; canonical sign applied by caller via Hip x already
            put("hip_shift", float(Hip[i1, 0] - Hip[ib, 0]), t[ib], t[i1])
    return out


def curves(C: np.ndarray, t: np.ndarray) -> dict:
    """Time series for plotting (canonical space)."""
    hw, he = C[:, H["h_wri"]], C[:, H["h_elb"]]
    return {
        "t": np.round(t, 4).tolist(),
        "wrist_speed": np.round(nanfill(speed(hw, t)), 3).tolist(),
        "elbow_speed": np.round(nanfill(speed(he, t)), 3).tolist(),
        "elbow_angle": np.round(nanfill(angle(C[:, H["h_sho"]], he, hw)), 1).tolist(),
        "knee_angle": np.round(nanfill(np.fmin(angle(C[:, H["h_hip"]], C[:, H["h_knee"]], C[:, H["h_ank"]]),
                                               angle(C[:, H["n_hip"]], C[:, H["n_knee"]], C[:, H["n_ank"]]))), 1).tolist(),
        "wrist_x": np.round(nanfill(hw[:, 0]), 3).tolist(),
        "wrist_y": np.round(nanfill(hw[:, 1]), 3).tolist(),
    }
