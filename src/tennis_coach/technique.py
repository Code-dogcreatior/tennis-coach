"""Technique checkpoints: coaching-style 2D observations at phase events, judged against a *professional*
reference distribution.

Each checkpoint is a single number measured on the canonical skeleton (right-handed, hip-centred,
torso-length units, y up; side views: +x = towards the net). What counts as "normal" is not invented here:
it is the 10th-90th percentile range of the same number over the professional reference swings of the same
stroke (and the same view group for checkpoints that depend on the camera). A checkpoint only produces a
diagnosis when the value lies outside that range by more than a measurement tolerance AND the direction is
one coaches consider a fault. Messages are templates; nothing is generated.

Limits (stated in every report): 2D projection, no racket or ball, so racket face, spin, and timing relative
to the ball are not observable. The "contact" is the estimated moment of peak wrist speed (groundstrokes) or
maximum wrist height (serve).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import CANON, NOSE, angle, side_joints

H = CANON
GROUND = ("forehand", "backhand_1h", "backhand_2h")
MIN_PRO = 5            # pro values needed before a checkpoint can be judged
SIDE, FRONT, BACK = "side", "front", "back"
ANY = (SIDE, FRONT, BACK)


def view_group(view_key: str) -> str:
    if view_key.startswith("side"):
        return SIDE
    if view_key in ("front", "oblique_front"):
        return FRONT
    return BACK


@dataclass
class Checkpoint:
    key: str
    label: str
    actions: tuple
    views: tuple            # view groups in which the measurement is meaningful
    event: str              # phase event used as visual evidence
    unit: str
    tol: float              # measurement tolerance (noise) added outside the pro range
    importance: int         # 1..3, used for ranking
    desc: str               # what is measured
    low: tuple | None = None    # (title, why/what it means, practice) when below the pro range
    high: tuple | None = None   # same, above the pro range
    joints: tuple = ()          # semantic joints to highlight in the comparison image
    pool_views: bool = False    # True: pool pro values over all view groups (vertical measures)
    extra: dict = field(default_factory=dict)


T = "躯干长"
CHECKPOINTS: list[Checkpoint] = [
    # ---------------------------------------------------------------- groundstrokes
    Checkpoint("gs_shoulder_turn", "引拍转肩", GROUND, (SIDE,), "backswing_end", T, 0.15, 3,
               "引拍阶段双肩连线在击球方向上的最大投影长度（侧面看：越大表示侧身越充分）",
               low=("引拍时转肩不足",
                    "引拍到位时肩膀转得比职业参考少，身体更接近正对球网，挥拍更多依赖手臂，力量和稳定性都受影响。",
                    "分腿垫步后先转肩再引拍（unit turn）：持拍手和非持拍手一起把球拍带到身体侧面，让对侧肩膀指向来球方向；可做无球影子挥拍，停在引拍终点检查肩线。"),
               joints=("h_sho", "n_sho")),
    Checkpoint("gs_turn_fb", "引拍转肩（正/背面）", GROUND, (FRONT, BACK), "backswing_end", "", 0.12, 3,
               "正/背面看：引拍阶段肩宽投影相对本次挥拍最大肩宽缩小的比例（0=没有转，接近 1=完全侧身）",
               low=("引拍时转肩不足",
                    "从正面/背面看，引拍时肩宽几乎没有变窄，说明身体基本没有侧转，挥拍主要靠手臂。",
                    "分腿垫步后先转肩再引拍（unit turn），让对侧肩膀转向来球方向；对镜子做影子挥拍，检查引拍终点时是否能看到自己的背部。"),
               joints=("h_sho", "n_sho")),
    Checkpoint("gs_rotation_fb", "击球前转体（正/背面）", GROUND, (FRONT, BACK), "contact", "", 0.12, 3,
               "正/背面看：从最大转肩到估计击球时，肩宽投影恢复的比例（越大表示身体转回越充分）",
               low=("击球时身体没有转过来，像是手臂单独挥拍",
                    "从引拍到击球，肩宽恢复量明显小于职业参考，躯干没有带动挥拍。",
                    "练习“先转髋、再转肩、最后手臂”的顺序：慢速挥拍，击球时让胸口转向球网。"),
               joints=("h_sho", "n_sho")),
    Checkpoint("gs_spacing_fb", "击球点与身体的横向距离（正/背面）", ("forehand",), (FRONT, BACK), "contact", T, 0.15, 2,
               "正/背面看：估计击球时持拍手腕在髋部中心持拍侧的横向距离",
               low=("击球时离球太近，手臂被挤住",
                    "击球时手腕离身体的横向距离明显小于职业参考，通常导致夹肘、击球受限。",
                    "调整站位，让球在身体侧前方约一臂距离处；多做移动后调整步（小碎步）再击球。"),
               joints=("h_wri", "h_hip", "n_hip")),
    Checkpoint("gs_cross_fb", "随挥越过身体（正/背面）", ("forehand",), (FRONT, BACK), "end", T, 0.2, 2,
               "正/背面看：击球后持拍手腕到达的最靠非持拍侧的横向位置（负值=越过身体中线）",
               high=("随挥没有越过身体",
                     "击球后手腕停在持拍侧，没有像职业参考那样收到身体另一侧，挥拍提前减速。",
                     "击球后让球拍继续向前并收到对侧肩膀附近，完整完成随挥。"),
               joints=("h_wri", "h_sho", "n_sho")),
    Checkpoint("gs_sink", "重心下降（屈膝）", GROUND, ANY, "backswing_end", "", 0.05, 2,
               "引拍到击球之间髋部离地高度（髋到较低脚踝）的最小值 ÷ 本次挥拍中的站立高度；越小表示屈膝下沉越多（垂直方向，与机位基本无关）",
               high=("屈膝下沉不足",
                     "挥拍过程中髋部高度几乎没有下降，腿部没有蓄力，和职业参考差别明显。",
                     "引拍时屈膝降低重心，像要坐下一样，再在击球时蹬地向上向前。"),
               joints=("h_hip", "h_knee", "h_ank", "n_hip", "n_knee", "n_ank"), pool_views=True),
    Checkpoint("gs_hip_turn", "引拍转髋", GROUND, (SIDE,), "backswing_end", T, 0.12, 1,
               "引拍阶段髋部连线在击球方向上的最大投影长度",
               low=("引拍时髋部几乎没有转动",
                    "下肢和髋部没有参与蓄力，转体主要来自上身。",
                    "引拍时外侧脚（正手为右脚）踩稳并让髋部随肩一起转，重心落在外侧脚上。"),
               joints=("h_hip", "n_hip")),
    Checkpoint("gs_rotation", "击球前转体（肩部打开量）", GROUND, (SIDE,), "contact", T, 0.15, 3,
               "从引拍最大转肩到估计击球时刻，双肩投影的变化量：表示击球时身体转回来多少",
               low=("击球时身体没有转过来，像是手臂单独挥拍",
                    "从引拍到击球，肩膀的转回量明显小于职业参考，说明发力主要来自手臂而不是躯干转动。",
                    "练习“先转髋、再转肩、最后手臂”的顺序：慢速挥拍，击球时让胸口转向球网，持拍侧肩膀向前。"),
               joints=("h_sho", "n_sho")),
    Checkpoint("gs_contact_front", "击球点前后", GROUND, (SIDE,), "contact", T, 0.15, 3,
               "估计击球时刻持拍手腕在髋部中心前方的水平距离（侧面）",
               low=("击球点偏后（离身体太近或在身体侧后方）",
                    "估计击球时手腕位置明显比职业参考靠后。常见原因是准备偏晚或者站得离球太近，击球时手臂被挤住。",
                    "提前完成引拍，迎前在身体前方击球；喂球练习时在前脚前方约一臂距离处放标志物作为击球点提示。"),
               joints=("h_wri", "h_hip", "n_hip")),
    Checkpoint("gs_arm_ext", "击球时持拍臂伸展（肘角）", ("forehand", "backhand_1h"), (SIDE,), "contact", "°", 12, 2,
               "估计击球时刻 肩-肘-腕 的投影夹角，180° 为完全伸直",
               low=("击球时持拍手臂弯曲过多（夹肘）",
                    "击球附近肘部明显比职业参考更弯，通常伴随击球点偏近身体。",
                    "让球在身体前方、离身体更远的位置击打；影子挥拍时感受击球瞬间手臂向前伸展。"),
               joints=("h_sho", "h_elb", "h_wri")),
    Checkpoint("gs_knee_load", "屈膝蓄力", GROUND, (SIDE,), "backswing_end", "°", 12, 2,
               "引拍到击球之间两膝投影角的最小值（越小屈膝越多）",
               high=("屈膝不足，腿部没有蓄力",
                     "整个挥拍过程中膝盖弯曲明显少于职业参考，站得偏直。",
                     "引拍时降低重心、膝盖弯曲，击球时向上向前蹬伸；可做“坐-蹬”分解练习。"),
               joints=("h_hip", "h_knee", "h_ank", "n_hip", "n_knee", "n_ank")),
    Checkpoint("gs_leg_drive", "下肢蹬伸", GROUND, ANY, "contact", T, 0.06, 2,
               "髋部高出较低脚踝的距离，从引拍最低点到击球时的增加量（垂直方向，与机位基本无关）",
               low=("击球时没有向上蹬伸",
                    "从引拍到击球，髋部相对脚的高度几乎没有升高，下肢力量没有传到挥拍中。",
                    "引拍时屈膝下沉，击球时脚蹬地、髋部向上向前，感觉“从地面开始挥拍”。"),
               joints=("h_hip", "n_hip", "h_ank", "n_ank"), pool_views=True),
    Checkpoint("gs_low_to_high", "挥拍路径由低到高", GROUND, ANY, "contact", T, 0.12, 2,
               "击球后 0.25 秒内手腕最高点与击球前（引拍终点之后）手腕最低点的高度差：挥拍穿过击球区时的上升幅度",
               low=("挥拍路径偏平或由上往下",
                    "击球前手腕没有先降到击球点下方再向上挥，和职业参考的“低到高”路径差别明显，不利于制造上旋和过网高度。",
                    "引拍后让拍头先落到球的下方，再向前向上挥拍；可对着网带上方的目标做慢速挥拍。"),
               joints=("h_wri",), pool_views=True),
    Checkpoint("gs_follow_height", "随挥收拍高度", GROUND, ANY, "end", T, 0.2, 2,
               "击球后持拍手腕到达的最高位置（相对髋部，肩部约为 1.0）",
               low=("随挥不完整，收拍过低",
                    "击球后手腕没有随挥到职业参考那样的高度，挥拍在击球后过早停止。",
                    "击球后让球拍继续向上向前，正手收到对侧肩膀上方、反手收到持拍侧肩膀上方，保持到动作结束。"),
               joints=("h_wri", "h_sho"), pool_views=True),
    Checkpoint("gs_nonhit_arm", "非持拍手前伸", ("forehand",), (SIDE,), "backswing_end", T, 0.15, 1,
               "引拍转肩时非持拍手腕在髋部中心前方的水平距离",
               low=("非持拍手没有前伸辅助转体",
                    "引拍时左手（右手持拍者）没有伸向前方，转肩和平衡通常会受影响。",
                    "引拍时让非持拍手扶拍转体后向前伸，指向来球，击球时再收回到身体前方。"),
               joints=("n_sho", "n_elb", "n_wri")),
    Checkpoint("gs_trunk_lean", "击球时躯干前倾", GROUND, (SIDE,), "contact", "°", 8, 2,
               "估计击球时肩部中心相对髋部中心偏离竖直方向的角度，正值为向前（球网方向）倾",
               low=("击球时身体后仰，重心在后",
                    "击球时上身向后倾，和职业参考的直立或略前倾不同，常见于被动击球或重心没有向前转移。",
                    "击球时保持上身直立，重心随挥拍从后脚转到前脚；可做击球后前脚站稳停 2 秒的平衡练习。"),
               joints=("h_sho", "n_sho", "h_hip", "n_hip")),
    Checkpoint("gs_backswing_depth", "引拍深度", GROUND, (SIDE,), "backswing_end", T, 0.2, 1,
               "引拍终点时持拍手腕在髋部中心后方的水平距离",
               low=("引拍过短",
                    "引拍终点时手腕离身体后方的距离明显小于职业参考，挥拍加速距离短。",
                    "结合转肩完成引拍，让球拍自然到达身体侧后方，而不是只用手臂往后拉。"),
               high=("引拍过大（拉得过远）",
                     "引拍终点时手腕在身体后方的距离明显大于职业参考，容易导致击球迟到。",
                     "引拍以转肩为主，手臂不要单独往后拉过身体太多；尝试更紧凑的引拍。"),
               joints=("h_wri", "h_hip", "n_hip")),
    Checkpoint("gs_head_still", "击球时头部稳定", GROUND, ANY, "contact", T, 0.06, 1,
               "估计击球前后 0.1 秒内鼻子相对髋部的最大位移",
               high=("击球瞬间头部晃动或过早抬头",
                     "击球前后头部位移明显大于职业参考，常见于过早看出球去向。",
                     "击球时眼睛盯住击球点，击球后头部保持不动片刻再转头看球。"),
               joints=("nose",), pool_views=True),
    # ---------------------------------------------------------------- serve
    Checkpoint("sv_toss_arm", "抛球手上举", ("serve",), ANY, "trophy", T, 0.15, 2,
               "奖杯姿势时非持拍手腕高出非持拍肩的距离",
               low=("抛球手没有充分上举",
                    "奖杯姿势时抛球手的高度明显低于职业参考，抛球和身体的上下拉伸都不充分。",
                    "抛球时手臂伸直向上送球，球出手后手臂继续指向球，保持到持拍手开始下落。"),
               joints=("n_sho", "n_elb", "n_wri"), pool_views=True),
    Checkpoint("sv_knee_load", "奖杯姿势屈膝", ("serve",), (SIDE,), "trophy", "°", 12, 2,
               "奖杯姿势前后两膝投影角的最小值",
               high=("发球蓄力时屈膝不足",
                     "奖杯姿势附近膝盖弯曲明显少于职业参考，腿部几乎没有参与发力。",
                     "抛球的同时屈膝下沉，在奖杯姿势时膝盖弯曲，随后向上蹬伸击球。"),
               joints=("h_hip", "h_knee", "h_ank", "n_hip", "n_knee", "n_ank")),
    Checkpoint("sv_leg_drive", "发球蹬伸", ("serve",), ANY, "contact", T, 0.08, 2,
               "髋部高出较低脚踝的距离，从奖杯姿势前后的最低点到击球时的增加量",
               low=("发球时向上蹬伸不足",
                    "从奖杯姿势到击球，髋部相对脚的高度增加明显少于职业参考。",
                    "奖杯姿势屈膝后，用蹬地把身体向上送向击球点，而不是只靠手臂向上挥。"),
               joints=("h_hip", "n_hip", "h_ank", "n_ank"), pool_views=True),
    Checkpoint("sv_shoulder_tilt", "奖杯姿势肩部倾斜", ("serve",), (FRONT, BACK), "trophy", T, 0.12, 2,
               "奖杯姿势时抛球侧肩膀高出持拍侧肩膀的距离（正面/背面）",
               low=("奖杯姿势时肩膀没有倾斜",
                    "抛球侧肩膀没有明显高于持拍侧肩膀，身体没有形成向上击球所需的倾斜。",
                    "抛球时让抛球手带动同侧肩膀向上，持拍侧肩膀自然下沉，形成“肩线指向球”的姿势。"),
               joints=("h_sho", "n_sho")),
    Checkpoint("sv_drop", "持拍手下落", ("serve",), ANY, "drop", T, 0.15, 1,
               "持拍手腕相对持拍肩的最低位置（越低表示下落越充分；注意是手腕，不是拍头）",
               high=("持拍手下落不充分",
                     "奖杯姿势之后持拍手腕没有像职业参考那样落到较低位置，加速距离较短。",
                     "奖杯姿势后放松持拍手臂，让球拍在背后自然下落，再向上加速。"),
               joints=("h_sho", "h_elb", "h_wri"), pool_views=True),
    Checkpoint("sv_contact_reach", "击球点高度", ("serve",), ANY, "contact", T, 0.15, 3,
               "估计击球时持拍手腕高出鼻子的距离",
               low=("发球击球点偏低",
                    "击球时手腕高出头部的距离明显小于职业参考，可能是抛球偏低、等球下落或手臂没有伸直。",
                    "把球抛到手臂和球拍完全伸展时能够到的高度，在最高点附近击球；可练习只抛球不击球，检查抛球高度和位置。"),
               joints=("h_sho", "h_elb", "h_wri", "nose"), pool_views=True),
    Checkpoint("sv_arm_ext", "击球时手臂伸展", ("serve",), ANY, "contact", "°", 12, 2,
               "估计击球时 肩-肘-腕 的投影夹角",
               low=("发球击球时手臂没有伸直",
                    "击球时肘部弯曲明显多于职业参考。",
                    "向上击球，击球瞬间手臂伸展到最高点；可先做无球的向上挥拍练习。"),
               joints=("h_sho", "h_elb", "h_wri"), pool_views=True),
    Checkpoint("sv_tuck", "非持拍手回收", ("serve",), ANY, "contact", T, 0.2, 1,
               "估计击球时非持拍手腕相对非持拍肩的高度（负值=已收到肩部以下）",
               high=("击球时抛球手没有回收",
                     "击球时抛球手仍停在高处，没有像职业参考那样收回身体前方，会影响转体和平衡。",
                     "持拍手向上挥拍时，抛球手顺势收到腹部前方。"),
               joints=("n_sho", "n_elb", "n_wri"), pool_views=True),
]
BY_KEY = {c.key: c for c in CHECKPOINTS}


# ----------------------------------------------------------------------------------------------
def _i(t: np.ndarray, i: int, ds: float) -> int:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    return int(np.clip(i + round(ds / dt), 0, len(t) - 1))


def _fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


def _nanmax(a):
    a = np.asarray(a, float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else None


def _nanmin(a):
    a = np.asarray(a, float)
    return float(np.nanmin(a)) if np.isfinite(a).any() else None


def compute(C: np.ndarray, t: np.ndarray, phases, action: str, view_key: str) -> dict:
    """Return {key: {"value", "event", "i", "t"}} for the checkpoints valid for this action/view."""
    ev = phases.events if hasattr(phases, "events") else phases["events"]
    g = view_group(view_key)
    hs, ns, hh, nh = C[:, H["h_sho"]], C[:, H["n_sho"]], C[:, H["h_hip"]], C[:, H["n_hip"]]
    hw, he, nw = C[:, H["h_wri"]], C[:, H["h_elb"]], C[:, H["n_wri"]]
    knee = np.fmin(angle(hh, C[:, H["h_knee"]], C[:, H["h_ank"]]), angle(nh, C[:, H["n_knee"]], C[:, H["n_ank"]]))
    hip_above_ankle = -np.fmin(C[:, H["h_ank"], 1], C[:, H["n_ank"], 1])   # hip centre is the origin
    elb = angle(hs, he, hw)
    i0, ic, i1 = ev["start"]["i"], ev["contact"]["i"], ev["end"]["i"]
    out: dict = {}

    def put(key, val, i):
        cp = BY_KEY[key]
        if action not in cp.actions or g not in cp.views:
            return
        val = _fin(val)
        if val is None:
            return
        out[key] = {"value": round(val, 4), "event": cp.event, "i": int(i), "t": float(t[i])}

    if action in GROUND:
        ib = ev["backswing_end"]["i"]
        s = 1.0 if action == "forehand" else -1.0       # turn direction for this stroke
        turn_sh = s * (ns[:, 0] - hs[:, 0])
        turn_hip = s * (nh[:, 0] - hh[:, 0])
        a, b = i0, max(i0 + 1, min(ic, _i(t, ib, 0.1)))
        if np.isfinite(turn_sh[a:b]).any():
            im = a + int(np.nanargmax(turn_sh[a:b]))
            put("gs_shoulder_turn", turn_sh[im], im)
            put("gs_hip_turn", _nanmax(turn_hip[a:b]), im)
            put("gs_rotation", turn_sh[im] - turn_sh[ic], ic)
            put("gs_nonhit_arm", nw[im, 0], im)
        # front/back: projected shoulder width relative to its maximum within this swing
        shw = np.linalg.norm(hs - ns, axis=1)
        wa, wb = ev.get("window_start", {}).get("i", i0), i1
        wmax = _nanmax(shw[min(wa, i0):wb + 1])
        if wmax and wmax > 0.2:
            seg = shw[i0:b]
            if np.isfinite(seg).any():
                ir = i0 + int(np.nanargmin(seg))
                put("gs_turn_fb", 1 - shw[ir] / wmax, ir)
                put("gs_rotation_fb", (shw[ic] - shw[ir]) / wmax, ic)
        put("gs_spacing_fb", hw[ic, 0], ic)
        if i1 > ic:
            put("gs_cross_fb", _nanmin(hw[ic:i1 + 1, 0]), i1)
        stand = np.nanpercentile(hip_above_ankle[i0:i1 + 1], 90) if np.isfinite(hip_above_ankle[i0:i1 + 1]).any() else None
        lowest = _nanmin(hip_above_ankle[i0:ic + 1])
        if stand and lowest is not None and stand > 0.5:
            put("gs_sink", lowest / stand, ib)
        put("gs_contact_front", hw[ic, 0], ic)
        put("gs_arm_ext", elb[ic], ic)
        put("gs_knee_load", _nanmin(knee[i0:ic + 1]), ib)
        lo = _nanmin(hip_above_ankle[_i(t, ib, -0.3):ic + 1])
        if lo is not None:
            put("gs_leg_drive", hip_above_ankle[ic] - lo, ic)
        low = _nanmin(hw[ib:ic + 1, 1])
        high_after = _nanmax(hw[ic:_i(t, ic, 0.25) + 1, 1])
        if low is not None and high_after is not None:
            put("gs_low_to_high", high_after - low, ic)
        if i1 > ic:
            put("gs_follow_height", _nanmax(hw[ic:i1 + 1, 1]), i1)
        sc = (hs + ns) / 2
        put("gs_trunk_lean", np.degrees(np.arctan2(sc[ic, 0], sc[ic, 1])), ic)
        put("gs_backswing_depth", -hw[ib, 0], ib)
        a, b = _i(t, ic, -0.1), _i(t, ic, 0.1)
        nose = C[a:b + 1, NOSE]
        if np.isfinite(nose).all(axis=1).sum() >= 3:
            put("gs_head_still", _nanmax(np.linalg.norm(nose - C[ic, NOSE], axis=1)), ic)
    elif action == "serve":
        it, idp = ev["trophy"]["i"], ev["drop"]["i"]
        put("sv_toss_arm", nw[it, 1] - ns[it, 1], it)
        a = _i(t, it, -0.3)
        put("sv_knee_load", _nanmin(knee[a:max(a + 1, idp + 1)]), it)
        lo = _nanmin(hip_above_ankle[a:max(a + 1, idp + 1)])
        if lo is not None:
            put("sv_leg_drive", hip_above_ankle[ic] - lo, ic)
        put("sv_shoulder_tilt", ns[it, 1] - hs[it, 1], it)
        put("sv_drop", hw[idp, 1] - hs[idp, 1], idp)
        put("sv_contact_reach", hw[ic, 1] - C[ic, NOSE, 1], ic)
        put("sv_arm_ext", elb[ic], ic)
        put("sv_tuck", nw[ic, 1] - ns[ic, 1], ic)
    return out


# ----------------------------------------------------------------------------------------------
ANGLE_TOL = 0.07   # max difference of ready-position projected hip width (torso units) for "same camera angle"


def same_angle(q_view: dict | None, r_view: dict | None) -> bool:
    if not q_view or not r_view or q_view.get("hip_width") is None or r_view.get("hip_width") is None:
        return True
    return abs(float(q_view["hip_width"]) - float(r_view["hip_width"])) <= ANGLE_TOL


# checkpoints measured over a fixed real-time window (e.g. "0.3 s before the trophy position"): pro clips in slow
# motion with an unknown factor would use the wrong window, so they do not contribute to these
TIME_WINDOW_KEYS = {"sv_knee_load", "sv_leg_drive", "gs_leg_drive", "gs_low_to_high", "gs_head_still"}
# a view-pooled checkpoint switches to same-view professionals once there are this many of them
VIEW_PREF_MIN = 15


def pro_distribution(key: str, action: str, group: str, pros: list[dict], q_view: dict | None = None,
                     with_scope: bool = False):
    """Professional values of one checkpoint. Angle-dependent checkpoints use only pros filmed from a similar
    angle; view-pooled ones prefer same-view pros when there are enough of them (projection still changes
    vertical measures a little), otherwise pool all views."""
    cp = BY_KEY[key]
    same, pooled = [], []
    for r in pros:
        if r["action"] != action:
            continue
        if key in TIME_WINDOW_KEYS and r.get("time_known") is False:
            continue
        v = (r.get("technique") or {}).get(key)
        if v is None:
            continue
        in_group = view_group(r["view_key"]) == group
        if not cp.pool_views:
            if in_group and same_angle(q_view, r.get("view")):
                same.append(v["value"])
            continue
        pooled.append(v["value"])
        if in_group:
            same.append(v["value"])
    if not cp.pool_views:
        vals, scope = same, "same_angle"
    elif len(same) >= VIEW_PREF_MIN:
        vals, scope = same, "same_view"
    else:
        vals, scope = pooled, "pooled"
    arr = np.asarray(vals, float)
    return (arr, scope) if with_scope else arr


def judge(tech: dict, action: str, view_key: str, pros: list[dict], q_view: dict | None = None) -> list[dict]:
    """Compare one swing's checkpoint values with the professional distribution."""
    g = view_group(view_key)
    res = []
    for cp in CHECKPOINTS:
        if action not in cp.actions:
            continue
        base = {"key": cp.key, "label": cp.label, "unit": cp.unit, "desc": cp.desc, "event": cp.event,
                "importance": cp.importance}
        if g not in cp.views:
            res.append({**base, "status": "not_applicable",
                        "note": "当前机位下无法可靠测量（需要" + "/".join({"side": "侧面", "front": "正面", "back": "背面"}[v]
                                                              for v in cp.views) + "机位）"})
            continue
        m = tech.get(cp.key)
        if m is None:
            res.append({**base, "status": "not_measured", "note": "关键点缺失或阶段事件不可用，未能测量"})
            continue
        vals, scope = pro_distribution(cp.key, action, g, pros, q_view, with_scope=True)
        rec = {**base, "value": m["value"], "i": m["i"], "t": m["t"], "n_pro": int(len(vals)), "pro_scope": scope}
        if len(vals) < MIN_PRO:
            res.append({**rec, "status": "insufficient_reference",
                        "note": (f"同动作、同机位角度的职业参考只有 {len(vals)} 条（至少需要 {MIN_PRO} 条），不做判断"
                                 + ("" if cp.pool_views else "；该检查点依赖机位角度，只与拍摄角度相近（准备姿势髋宽投影差 ≤ 0.07）的职业挥拍比较"))})
            continue
        p10, p50, p90 = (float(x) for x in np.percentile(vals, [10, 50, 90]))
        iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
        scale = max(iqr, cp.tol)
        rec.update({"pro_p10": round(p10, 3), "pro_median": round(p50, 3), "pro_p90": round(p90, 3),
                    "pro_min": round(float(vals.min()), 3), "pro_max": round(float(vals.max()), 3)})
        v = m["value"]
        if v < p10 - cp.tol:
            side, dev = "low", (p10 - v) / scale
        elif v > p90 + cp.tol:
            side, dev = "high", (v - p90) / scale
        else:
            inside = p10 <= v <= p90
            res.append({**rec, "status": "ok" if inside else "borderline", "deviation": 0.0,
                        **({} if inside else {"note": "略超出职业常见范围，但差值在测量误差容差内，不判为问题"})})
            continue
        msg = cp.low if side == "low" else cp.high
        if msg is None:
            res.append({**rec, "status": "outside_not_fault", "direction": side, "deviation": round(dev, 2),
                        "note": "超出职业参考范围，但这个方向通常不被视为技术问题（可能是个人风格或来球不同）"})
            continue
        res.append({**rec, "status": "issue", "direction": side, "deviation": round(dev, 2),
                    "title": msg[0], "explanation": msg[1], "practice": msg[2],
                    "score": round(min(dev, 3.0) * cp.importance, 3)})
    return res


_VALIDATION: dict | None = None


def load_validation() -> dict:
    """Per (action, checkpoint) validation from scripts/eval_pro_compare.py: a checkpoint drives the main
    diagnosis only if it rarely flags professionals in leave-one-player-out testing."""
    global _VALIDATION
    if _VALIDATION is None:
        import json
        from pathlib import Path
        p = Path(__file__).with_name("checkpoint_validation.json")
        _VALIDATION = json.loads(p.read_text(encoding="utf-8")).get("gate", {}) if p.exists() else {}
    return _VALIDATION


VIEW_NAMES = {SIDE: "侧面", FRONT: "正面/斜前方", BACK: "背面/斜后方"}


def pro_compare(q: dict, pros: list[dict], max_issues: int = 3) -> dict:
    """Checkpoint diagnosis of one query swing against the professional library + nearest pro swing."""
    from .classify import ACTION_LABELS
    from .match import match

    action = q["action"]
    if action == "unknown" or not q.get("phases"):
        return {"status": "action_unknown", "message": "未识别为目标动作，不做职业对比", "checks": [], "issues": []}
    same = [r for r in pros if r["action"] == action]
    if not same:
        return {"status": "no_pro_refs", "message": f"职业参考库中还没有「{ACTION_LABELS[action]}」", "checks": [],
                "issues": []}
    checks = judge(q.get("technique") or {}, action, q["view_key"], same, q.get("view"))
    gate = load_validation()
    for c in checks:
        g_ = gate.get(f"{action}:{c['key']}")
        c["validated"] = bool(g_ and g_["validated"])
        c["validation"] = g_
    all_issues = sorted([c for c in checks if c["status"] == "issue"], key=lambda c: -c["score"])
    issues = [c for c in all_issues if c["validated"]]
    tentative = [c for c in all_issues if not c["validated"]]
    ok = [c for c in checks if c["status"] == "ok"]
    measured = [c for c in checks if c["status"] in ("ok", "borderline", "issue", "outside_not_fault")]
    m = match(q, same)
    nearest = None
    if m["status"] == "ok" and m["candidates"]:
        c0 = m["candidates"][0]
        r = next(r for r in same if r["ref_id"] == c0["ref_id"])
        nearest = {"ref_id": r["ref_id"], "player": r.get("player"), "video_id": r.get("video_id"),
                   "swing_index": r.get("swing_index"), "cost": c0["cost"], "path": c0["path"],
                   "segment_costs": c0["segment_costs"]}
    g = view_group(q["view_key"])
    players = sorted({r.get("player") or "?" for r in same})
    n_view = sum(1 for r in same if view_group(r["view_key"]) == g)
    summary = (f"与职业参考库中 {len(players)} 名球员的 {len(same)} 次「{ACTION_LABELS[action]}」对比"
               f"（其中与你同为{VIEW_NAMES[g]}机位的 {n_view} 次）。可测量的检查点 {len(measured)} 项："
               f"{len(ok)} 项在职业范围内，{len(issues)} 项明显偏离" + (f"，另有 {len(tentative)} 项偏离出现在尚未通过验证的检查点上（仅作提示）" if tentative else "") + "。")
    if not measured:
        summary += " 当前机位下缺少足够的同机位职业参考，无法给出可靠判断。"
    return {"status": "ok", "message": summary, "view_group": g, "n_pro_swings": len(same), "pro_players": players,
            "checks": checks, "issues": issues[:max_issues], "more_issues": issues[max_issues:], "tentative": tentative,
            "strengths": sorted(ok, key=lambda c: -c["importance"])[:3], "nearest": nearest,
            "match_status": m["status"], "match_message": m["message"],
            "limits": "仅基于 2D 骨架：看不到球拍、拍面和球，无法判断击球时机与来球的关系；“击球”时刻为估计值；"
                      "职业范围取职业参考的 10%–90% 分位并加测量容差，参考来自训练视频，未经教练审核。"}


def highlight_joints(key: str, hand: str) -> list[int]:
    cp = BY_KEY.get(key)
    if not cp:
        return []
    sj = side_joints(hand)
    return [NOSE if j == "nose" else sj[j] for j in cp.joints if j == "nose" or j in sj]
