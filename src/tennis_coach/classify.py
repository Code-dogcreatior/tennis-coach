"""Stroke recognition: interpretable rule baseline + optional nearest-reference (DTW) vote.

Classes: forehand, backhand_1h, backhand_2h, serve, plus ``unknown`` (non-target or ambiguous).
No classifier is trained by default; see docs for when a learned model would be justified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .skeleton import CANON, NOSE
from .swing import MIN_PEAK_SPEED, SwingWindow, nanfill, serve_pattern, speed

ACTIONS = ["forehand", "backhand_1h", "backhand_2h", "serve"]
ACTION_LABELS = {"forehand": "正手", "backhand_1h": "单手反手", "backhand_2h": "双手反手", "serve": "发球",
                 "unknown": "未知/非目标动作"}
H = CANON
TWO_HAND_THR = 0.40


@dataclass
class Classification:
    action: str
    confidence: float
    evidence: list
    rule_action: str
    rule_confidence: float
    knn: dict | None = None
    scores: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self); d["label"] = ACTION_LABELS.get(self.action, self.action); return d


SMASH_LIKE_THR = 0.0   # toss_hand_low >= this -> smash-like (chosen on THETIS train players, REPORT 8e)


def toss_hand_low(C: np.ndarray, t: np.ndarray, i_contact: int, i_toss: int) -> float | None:
    """Lowest height of the non-hitting wrist relative to the hip centre (torso lengths, up = +) in the up to
    2.5 s before the toss peak. A serve starts with the ball held at or below the waist; before a smash the
    free arm is usually already up. None when less than 0.3 s before the toss is visible."""
    dt = float(np.median(np.diff(t)))
    a = max(0, i_contact - int(2.5 / dt))
    if (i_toss - a) * dt < 0.3:
        return None
    hip_y = (C[a:i_toss, H["h_hip"], 1] + C[a:i_toss, H["n_hip"], 1]) / 2
    y = C[a:i_toss, H["n_wri"], 1] - hip_y
    return float(np.nanmin(y)) if np.isfinite(y).any() else None


GROUND_REACH_THR = -0.6   # lowest wrist - knee height (torso lengths); below this -> bending down, not a stroke


def ground_reach(C: np.ndarray, t: np.ndarray, ic: int, half_s: float = 0.4) -> float | None:
    """Lowest wrist (either hand) relative to the knee height in +-half_s around the wrist-speed peak
    (torso lengths, up = +). Strongly negative when the player bends down to the court."""
    fps = 1 / float(np.median(np.diff(t)))
    lo, hi = max(0, ic - int(half_s * fps)), min(len(t), ic + int(half_s * fps) + 1)
    knee = (C[lo:hi, H["h_knee"], 1] + C[lo:hi, H["n_knee"], 1]) / 2
    wr = np.fmin(C[lo:hi, H["h_wri"], 1], C[lo:hi, H["n_wri"], 1])
    if not (np.isfinite(knee).any() and np.isfinite(wr).any()):
        return None
    return float(np.nanmin(wr) - np.nanmedian(knee))


def rule_classify(C: np.ndarray, t: np.ndarray, w: SwingWindow, view_key: str,
                  min_speed: float = MIN_PEAK_SPEED) -> Classification:
    fps = 1 / np.median(np.diff(t))
    hw, nw = C[:, H["h_wri"]], C[:, H["n_wri"]]
    s = nanfill(speed(hw, t))
    a, b = max(w.i0, w.i_peak - int(0.1 * fps)), min(w.i1, w.i_peak + int(0.1 * fps))
    ic = a + int(np.argmax(s[a:b + 1]))
    ev = []
    peak = float(s[ic])
    # the hitting wrist itself must reach a substantial share of the swing speed
    if peak < min_speed or peak < 0.5 * w.peak_speed:
        return Classification("unknown", 0.8, [f"持拍手腕相对速度峰值 {peak:.1f} 低于挥拍阈值 {min_speed:.1f}，判定为非挥拍动作"],
                              "unknown", 0.8, scores={"non_stroke": 1.0})
    reach = ground_reach(C, t, ic)
    if reach is not None and reach < GROUND_REACH_THR:
        # bending down (picking up a ball, touching the court): fast wrist motion but no stroke. Real strokes
        # never bring a wrist this far below the knees (THETIS + pro library, 792 swings incl. volleys: >= -0.15).
        return Classification("unknown", 0.8, [f"峰值前后 0.4 s 内手腕降到膝盖以下 {-reach:.2f} 躯干长，"
                                               "判定为弯腰（捡球等）而非挥拍"], "unknown", 0.8,
                              scores={"non_stroke": 1.0, "ground_reach": reach})
    ok_serve, i_con, hw_above, toss, i_toss = serve_pattern(C, t, w, H["h_wri"], H["n_wri"])
    ev.append(f"峰值前持拍手腕最高比鼻子高 {hw_above:+.2f} 躯干长；此前非持拍手腕最高比鼻子高 {toss:+.2f}")
    if hw_above > 0.25:
        if ok_serve:
            # THETIS evaluation: flat serve and smash have overlapping distributions of these 2D features,
            # so the serve call is capped and the ambiguity is stated instead of being hidden.
            conf = float(np.clip(0.3 + hw_above, 0, 0.6))
            ev.append("高点击球且击球前非持拍手上举（抛球）→ 发球")
            low = toss_hand_low(C, t, i_con, i_toss)
            scores = {"toss_hand_low": low}
            if low is None:
                ev.append("注意：高压球在 2D 姿态上与发球很难区分，结果也可能是高压")
            elif low >= SMASH_LIKE_THR:
                # no ball-holding phase below the hips before the toss: smash-like. Not rejected - on THETIS test
                # players this would also drop ~1 in 5 real serves (see REPORT 8e) - but stated and down-weighted.
                conf = min(conf, 0.35)
                scores["smash_like"] = 1.0
                ev.append(f"抛球前非持拍手始终没有降到髋部以下（最低 {low:+.2f} 躯干长；发球通常先在腰部以下托球）"
                          "→ 更像高压，按发球分析但请自行确认")
            else:
                ev.append(f"抛球前非持拍手降到髋部以下（{low:+.2f} 躯干长，托球准备）→ 符合发球；高压仍不能完全排除")
            return Classification("serve", conf, ev, "serve", conf, scores=scores)
        return Classification("unknown", 0.6, ev + ["高点击球但未见抛球动作（可能为高压球），不属于首版四类"], "unknown", 0.6)
    # two-handed: both hands stay on the racket into the follow-through. Tuned on THETIS *train* players:
    # median wrist distance 0.1-0.3 s after the contact estimate: 2HBH p90 = 0.33, 1HBH p10 = 0.48 torso.
    # (Before contact the free hand of a one-handed backhand still holds the throat, so that window is ambiguous.)
    fa, fb = ic + int(0.1 * fps), min(w.i1, ic + int(0.3 * fps))
    if fb - fa < 2:
        fa, fb = max(w.i0, ic - int(0.1 * fps)), min(w.i1, ic + int(0.05 * fps))
    dist = np.linalg.norm(hw[fa:fb + 1] - nw[fa:fb + 1], axis=-1)
    dmed = float(np.nanmedian(dist)) if np.isfinite(dist).any() else 9
    ev.append(f"击球后 0.1–0.3 s 两手腕距离中位数 {dmed:.2f} 躯干长（双手 < {TWO_HAND_THR}）")
    two_conf = float(np.clip((TWO_HAND_THR - dmed) / 0.2 + 0.5, 0, 0.95))
    # hands also together just before contact? (free hand of a forehand is away from the racket there; many
    # professionals catch the racket throat with it *after* a forehand, so the post-contact window alone
    # confuses forehands with two-handed backhands in side views)
    pa, pb = max(w.i0, ic - int(0.2 * fps)), max(w.i0 + 1, ic - int(0.04 * fps))
    dpre = np.linalg.norm(hw[pa:pb] - nw[pa:pb], axis=-1)
    dpre_med = float(np.nanmedian(dpre)) if np.isfinite(dpre).any() else 9
    ev.append(f"击球前 0.2–0.04 s 两手腕距离中位数 {dpre_med:.2f} 躯干长")
    # forehand vs backhand
    ba, bb = max(w.i0, ic - int(1.2 * fps)), max(w.i0 + 1, ic - int(0.08 * fps))
    d = np.linalg.norm(hw[ba:bb] - hw[ic], axis=-1)
    ib = ba + int(np.nanargmax(np.nan_to_num(d, nan=-1)))
    # view-independent cue: position of the hitting wrist along the shoulder line at the end of the take-back
    # (0 = shoulder centre, +0.5 = hitting shoulder, -0.5 = non-hitting shoulder). Affine ratios along a line
    # survive 2D projection, so this does not depend on the x-sign / facing convention of the view.
    hs_, ns_ = C[ib, H["h_sho"]], C[ib, H["n_sho"]]
    sl = hs_ - ns_
    line_pos = float(np.dot(hw[ib] - (hs_ + ns_) / 2, sl) / max(float(np.dot(sl, sl)), 1e-3))
    scores_extra = {"line_pos": line_pos, "dpre": dpre_med, "dpost": dmed}
    if view_key.startswith("side"):
        # +x = facing / hitting direction. Forehand: hitting shoulder behind non-hitting shoulder at end of
        # backswing (body turned with the non-hitting shoulder forward); one-handed backhand: the opposite.
        dsh = float(C[ib, H["h_sho"], 0] - C[ib, H["n_sho"], 0])
        ev.append(f"引拍终点时持拍肩相对非持拍肩的前后位置 {dsh:+.2f} 躯干长（负=持拍肩在后）")
        margin = abs(dsh)
        lab = "forehand" if dsh < 0 else "backhand_1h"
        scores_extra["side_dsh"] = dsh
    elif view_key in ("front", "back", "oblique_front", "oblique_back"):
        # +x = hitting-arm side. Forehand: the wrist stays on the hitting side before contact and finishes
        # across the body; backhand: crosses to the non-hitting side in the take-back and finishes on the
        # hitting side. Score tuned on THETIS train players (min pre-contact x: 1HBH p90 0.00, FH p10 0.10).
        pre = hw[max(w.i0, ic - int(1.0 * fps)):ic, 0]
        post = hw[ic:min(w.i1 + 1, ic + int(0.3 * fps)), 0]
        x_min = float(np.nanmin(pre)) if np.isfinite(pre).any() else 0.0
        x_post = float(np.nanmedian(post)) if np.isfinite(post).any() else 0.0
        score = x_min - 0.5 * x_post
        ev.append(f"击球前持拍手腕最靠非持拍侧的横向位置 {x_min:+.2f}、击球后横向位置 {x_post:+.2f} 躯干长"
                  f"（正=持拍侧）→ 正手得分 {score:+.2f}")
        margin = abs(score)
        lab = "forehand" if score > 0 else "backhand_1h"
        scores_extra["front_score"] = score
    else:
        return Classification("unknown", 0.3, ev + ["机位未知，无法区分正反手"], "unknown", 0.3)
    conf = float(np.clip(margin / 0.3, 0, 0.95))
    # combine with the view-independent cue (thresholds from THETIS train players: line>=0.8 & free hand
    # >=0.7 torso away before contact -> 55/68 forehands, 2/33 one-handed backhands; line<0.4 -> 3/68 forehands)
    cue = "forehand" if (line_pos >= 0.8 and dpre_med >= 0.7) else ("backhand_1h" if line_pos < 0.4 else None)
    if cue is not None:
        ev.append(f"引拍终点持拍手腕在肩线上的位置 {line_pos:+.2f}（+0.5=持拍肩，-0.5=非持拍肩）→ {'正手侧' if cue == 'forehand' else '反手侧'}")
        if cue == lab:
            conf = float(min(0.95, conf + 0.2))
        elif view_key in ("front", "back") and conf >= 0.6:
            ev.append("与机位规则不一致；正/背面机位下以机位规则为准，降低置信度")
            conf = 0.3
        else:
            ev.append("与机位规则不一致；斜向/侧面机位下横向位置受透视影响，以肩线位置为准")
            lab, conf = cue, 0.55
    two = dmed < TWO_HAND_THR
    if two and (lab == "backhand_1h" or conf < 0.25 or dpre_med < TWO_HAND_THR):
        c2 = two_conf if lab == "backhand_1h" else two_conf * 0.8
        return Classification("backhand_2h", c2, ev + ["反手侧挥拍且两手在击球前后都握在一起 → 双手反手"], "backhand_2h", c2,
                              scores=scores_extra)
    if two and lab == "forehand":
        ev.append("击球后两手靠近，但击球前非持拍手离开球拍 → 视为正手（随挥时非持拍手接拍）")
    if conf < 0.25:
        return Classification("unknown", 1 - conf, ev + ["正/反手判据接近边界，不强行判断"], "unknown", 1 - conf,
                              scores=scores_extra)
    return Classification(lab, conf, ev, lab, conf, scores=scores_extra)


def combine_with_knn(rule: Classification, knn: dict | None) -> Classification:
    """knn: {"label", "votes": {label: n}, "k", "nearest_cost", "cost_threshold"}."""
    if not knn:
        rule.evidence.append("参考库中没有可比对的同机位参考，仅使用规则判断")
        return rule
    rule.knn = knn
    if rule.scores.get("ground_reach") is not None:
        return rule     # bending down: posture rules this out, the reference vote must not relabel it
    if knn["nearest_cost"] > knn["cost_threshold"]:
        rule.evidence.append(f"与最近参考的 DTW 距离 {knn['nearest_cost']:.2f} 超过阈值 {knn['cost_threshold']:.2f}，与库内动作都不相似")
        if rule.rule_confidence < 0.6:
            return Classification("unknown", 0.5, rule.evidence, rule.rule_action, rule.rule_confidence, knn, scores=rule.scores)
        rule.confidence *= 0.7
        return rule
    agree = knn["label"] == rule.rule_action
    frac = knn["votes"].get(knn["label"], 0) / max(knn["k"], 1)
    if agree:
        cap = 0.98
        if rule.rule_action == "serve":       # serve vs smash ambiguity (see rule_classify)
            cap = 0.35 if rule.scores.get("smash_like") else 0.6
        rule.confidence = float(min(cap, 0.5 * rule.rule_confidence + 0.5 * frac + 0.1))
        rule.evidence.append(f"最近参考投票 {knn['votes']}，与规则一致")
        return rule
    if rule.rule_action == "unknown" and frac >= 0.67:
        return Classification(knn["label"], 0.5 * frac, rule.evidence + [f"规则无法判断，最近参考投票 {knn['votes']}"],
                              rule.rule_action, rule.rule_confidence, knn, scores=rule.scores)
    rule.evidence.append(f"最近参考投票 {knn['votes']} 与规则判断（{ACTION_LABELS.get(rule.rule_action)}）不一致")
    return Classification("unknown", 0.4, rule.evidence, rule.rule_action, rule.rule_confidence, knn, scores=rule.scores)
