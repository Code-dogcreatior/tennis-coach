"""Rule-templated feedback generated only from computed, reliable metric differences.

Each item states: what was observed, how it differs from the reference group, where (time window)
the evidence is, and a practice suggestion. No difference -> no claim. No injury diagnosis.
"""
from __future__ import annotations

from .classify import ACTION_LABELS

# (metric, direction) -> (observation template, practice suggestion)
TEMPLATES = {
    ("dur_backswing", "above"): ("引拍阶段用时比参考更长", "练习单元步+提前转体：喂球时听到对方击球就开始引拍，用节拍器或口令把引拍时间缩短到与参考相近。"),
    ("dur_backswing", "below"): ("引拍阶段用时比参考更短，可能引拍仓促", "慢速影子挥拍，刻意完成转体和引拍再启动向前挥拍，保持动作连贯。"),
    ("dur_forward", "above"): ("向前加速阶段用时比参考更长，加速不够集中", "做“慢-快”挥拍练习：引拍放松，向前时在击球前集中加速，可用毛巾挥拍体会末端加速。"),
    ("dur_forward", "below"): ("向前加速阶段比参考更短", "若击球控制不稳，可适当放慢加速前段，重点保持击球点稳定。"),
    ("tempo_ratio", "above"): ("引拍/加速时长比高于参考，节奏偏“慢引拍+急加速”", "用口令“转-挥”练节奏，让引拍与加速的比例接近参考。"),
    ("tempo_ratio", "below"): ("引拍/加速时长比低于参考，引拍相对仓促", "提前准备，留出完整引拍时间，避免来不及引拍就向前挥。"),
    ("dur_follow", "below"): ("随挥阶段比参考短，可能收拍过早", "练习完整随挥：击球后手臂继续沿击球方向延伸，再收拍到参考的结束位置并停住 1 秒。"),
    ("follow_path", "below"): ("击球后手腕轨迹比参考短，随挥不够充分", "影子挥拍时刻意让手腕在击球后继续向前上方延伸，再完成收拍。"),
    ("follow_height", "below"): ("随挥结束时手腕高度低于参考", "注意收拍高度，正手可练“收拍到对侧肩”，反手收拍到参考的高度。"),
    ("follow_cross", "above"): ("随挥没有像参考那样越过身体中线", "练习收拍越过身体：击球后手臂带动转体，收拍到非持拍侧。"),
    ("backswing_amp", "below"): ("引拍幅度（手腕位移）小于参考", "做转体引拍练习：双手持拍转肩，让引拍幅度接近参考，再加入挥拍。"),
    ("backswing_amp", "above"): ("引拍幅度大于参考", "若时间来不及或击球不稳，可尝试紧凑引拍，靠转体而不是手臂往后拉。"),
    ("backswing_behind", "above"): ("引拍时手腕在身后更远处", "引拍时保持手腕在身体侧后方可视范围内，避免过度往后拉臂。"),
    ("backswing_behind", "below"): ("引拍时手腕在身后的距离小于参考", "加强转肩引拍，让手腕到达身体侧后方再向前挥。"),
    ("backswing_height", "above"): ("引拍终点手腕位置高于参考", "可对照参考调整引拍高度；若追求上旋，注意引拍后手腕要能下落到球的下方再向上挥。"),
    ("backswing_height", "below"): ("引拍终点手腕位置低于参考", "对照参考提高引拍高度，再通过下落-上挥形成挥拍路径。"),
    ("contact_front", "below"): ("估计击球点比参考更靠近身体/靠后", "练习在身体前方击球：喂球时站位调整，让击球点在前脚前方，可用标志物提示击球位置。"),
    ("contact_front", "above"): ("估计击球点比参考更靠前", "注意不要够球，调整步伐让击球点与参考一致。"),
    ("contact_height", "below"): ("估计击球点高度低于参考", "通过步伐调整击球高度，或屈膝降低重心后再击球。"),
    ("contact_height", "above"): ("估计击球点高度高于参考", "通过步伐（后退或前进）调整到参考的击球高度。"),
    ("elbow_contact", "below"): ("击球附近肘部弯曲（投影角）比参考更多", "对照参考的手臂结构，影子挥拍时在击球位置停顿检查肘部角度。"),
    ("elbow_contact", "above"): ("击球附近肘部比参考更伸直（投影角）", "对照参考的手臂结构，影子挥拍时在击球位置停顿检查肘部角度。"),
    ("knee_min", "above"): ("引拍到击球期间屈膝少于参考（投影膝角更大）", "加入屈膝蓄力练习：引拍时降低重心，向前挥拍时蹬伸。"),
    ("elbow_before_wrist", "below"): ("肘部速度峰值相对手腕出现得更晚/更接近（2D 投影观察）", "可练习由近端到远端的发力顺序：先转体带肘，最后甩腕，用慢动作体会。"),
    ("hip_shift", "below"): ("挥拍期间髋部向击球方向移动少于参考", "练习重心转移：从后脚到前脚完成挥拍，可用“踏步击球”练习。"),
    ("peak_speed", "below"): ("手腕相对峰值速度低于参考（相对强度，非真实速度）", "动作稳定后再逐步加速；先保证结构，再做加速挥拍练习。"),
    ("dur_toss", "above"): ("抛球到奖杯姿势用时比参考长", "练习抛球与引拍同步，节奏口令“抛-举”。"),
    ("dur_toss", "below"): ("抛球到奖杯姿势用时比参考短", "抛球手臂缓慢上举，让持拍手有时间到位。"),
    ("toss_height", "below"): ("抛球手最高点低于参考", "抛球手臂充分上伸，指向球的方向，保持到奖杯姿势。"),
    ("trophy_knee", "above"): ("奖杯姿势屈膝少于参考（投影膝角更大）", "练习奖杯姿势定格：屈膝蓄力后再向上蹬伸击球。"),
    ("drop_depth", "above"): ("持拍手腕下落深度小于参考（手腕未降到同等位置）", "练习“抓背”下落：奖杯后放松持拍臂让手腕自然下落，再向上挥。"),
    ("contact_above_head", "below"): ("发球估计击球点相对头部低于参考", "练习向上伸展击球：抛球更高更靠前，击球时手臂充分伸展。"),
    ("leg_drive", "below"): ("奖杯到击球期间髋部上升少于参考（2D 观察）", "练习蹬地发力：奖杯姿势屈膝后向上蹬伸。"),
}


def build_feedback(action: str, match_result: dict, user_view_label: str) -> list[dict]:
    items = []
    if match_result.get("status") != "ok":
        return [{"kind": "info", "title": "未进行参考对比", "observation": match_result.get("message", ""),
                 "confidence": None}]
    comp = match_result["comparison"]
    for key, c in comp.items():
        if c["status"] == "within":
            continue
        if not c["reliable"]:
            continue
        tpl = TEMPLATES.get((key, c["status"]))
        if not tpl:
            continue
        obs, practice = tpl
        unit = c["unit"]
        diff = c["value"] - c["ref_mean"]
        n = c["n_ref"]
        conf = "中" if n >= 3 else "低"
        items.append({
            "kind": "difference", "metric": key, "title": obs,
            "observation": f"{c['label']}：你 {c['value']:.2f}{unit}，参考组均值 {c['ref_mean']:.2f}{unit}"
                           f"（范围 {c['ref_min']:.2f}–{c['ref_max']:.2f}，n={n}），差 {diff:+.2f}{unit}。",
            "evidence": {"t0": c["t0"], "t1": c["t1"],
                         "text": f"证据时间段：{c['t0']:.2f}s – {c['t1']:.2f}s（估计事件，见阶段时间线）"},
            "practice": practice,
            "confidence": conf,
            "confidence_note": ("参考组样本少于 3 条，结论仅供参考" if n < 3 else "基于参考组分布的超出判断") +
                               "；所有数值为 2D 投影观察。",
        })
    if not items:
        items.append({"kind": "info", "title": "未发现超出参考范围的可靠差异",
                      "observation": f"在 {user_view_label} 机位下可用的指标都处于参考组范围内（或差异小于容差）。"
                                     "这不代表动作没有问题，只说明从当前 2D 视频和参考库中找不到可靠证据。",
                      "confidence": None})
    items.append({"kind": "disclaimer", "title": "说明",
                  "observation": "参考为“参考示范”，除非标注为教练审核，否则不代表专业标准；同一动作允许多种合格风格。"
                                 "反馈仅基于 2D 关节投影，不涉及真实力量/功率，也不做伤病判断。",
                  "confidence": None})
    return items


def action_label(a: str) -> str:
    return ACTION_LABELS.get(a, a)
