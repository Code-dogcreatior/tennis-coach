"""Swing localisation, handedness estimation and phase segmentation from body-space keypoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .skeleton import CANON, L_WRI, NOSE, R_WRI

MIN_PEAK_SPEED = 3.0      # torso-lengths / s; below this a wrist movement is not treated as a swing
PRE_WINDOW = 2.5          # s before the speed peak (serves: the 2D speed peak is in the follow-through,
                          # so the toss starts >2 s earlier)
POST_WINDOW = 1.2         # s after the speed peak


def speed(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Speed of a (T,2) trajectory (NaN-safe)."""
    v = np.gradient(X, t, axis=0)
    return np.linalg.norm(v, axis=-1)


def nanfill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    m = np.isnan(x)
    if m.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[m] = np.interp(idx[m], idx[~m], x[~m])
    return x


@dataclass
class SwingWindow:
    i0: int
    i_peak: int
    i1: int
    t0: float
    t_peak: float
    t1: float
    peak_speed: float          # torso/s, body-relative wrist speed
    wrist: str                 # which wrist produced the peak ("left"/"right")
    truncated_start: bool
    truncated_end: bool

    def to_dict(self):
        return asdict(self)


def adaptive_min_speed(P: np.ndarray, t: np.ndarray) -> float:
    """Swing threshold when the clip's time scale is unknown (e.g. slow motion of unknown factor).

    Uses a fraction of the clip's own maximum wrist speed, with an absolute floor that is well above
    keypoint jitter so a static clip still yields no swings.
    """
    s = np.maximum(nanfill(speed(P[:, L_WRI], t)), nanfill(speed(P[:, R_WRI], t)))
    return float(max(0.6, 0.35 * np.percentile(s, 99.5))) if len(s) else MIN_PEAK_SPEED


def shot_boundary_mask(shot: np.ndarray | None, T: int, pad: int = 2) -> np.ndarray:
    m = np.zeros(T, bool)
    if shot is None:
        return m
    for b in np.where(np.diff(shot) != 0)[0] + 1:
        m[max(0, b - pad):min(T, b + pad + 1)] = True
    return m


def detect_swings(P: np.ndarray, t: np.ndarray, min_speed: float = MIN_PEAK_SPEED,
                  min_gap_s: float = 1.2, shot: np.ndarray | None = None) -> list[SwingWindow]:
    """Find swings as prominent peaks of body-relative wrist speed.

    shot: per-sample shot id; speeds next to an edit are ignored and windows never span shots.
    """
    fps = 1 / np.median(np.diff(t))
    sl = nanfill(speed(P[:, L_WRI], t))
    sr = nanfill(speed(P[:, R_WRI], t))
    bm = shot_boundary_mask(shot, len(t))
    sl[bm] = 0; sr[bm] = 0
    s = np.maximum(sl, sr)
    if len(s) < 5:
        return []
    peaks, props = find_peaks(s, height=min_speed, distance=max(1, int(min_gap_s * fps)),
                              prominence=min_speed * 0.6)
    # a small peak close to a much larger one is the take-back / recovery of that swing, not a swing
    # (e.g. the toss-arm rise before a serve, a slow take-back, the recovery after the finish)
    near = int(2.0 * fps)
    peaks = np.array([p for p in peaks if not any(abs(q - p) <= near and s[q] > 1.5 * s[p] for q in peaks)], int)
    out = []
    for k, p in enumerate(peaks):
        lo = int(max(0, p - PRE_WINDOW * fps))
        hi = int(min(len(t) - 1, p + POST_WINDOW * fps))
        # do not overlap the neighbouring swing more than half-way
        if k > 0:
            lo = max(lo, (peaks[k - 1] + p) // 2)
        if k < len(peaks) - 1:
            hi = min(hi, (peaks[k + 1] + p) // 2)
        if shot is not None:
            same = np.where(shot == shot[p])[0]
            lo, hi = max(lo, int(same.min())), min(hi, int(same.max()))
        out.append(SwingWindow(i0=lo, i_peak=int(p), i1=hi, t0=float(t[lo]), t_peak=float(t[p]), t1=float(t[hi]),
                               peak_speed=float(s[p]), wrist="left" if sl[p] > sr[p] else "right",
                               truncated_start=(p - lo) < 0.6 * fps, truncated_end=(hi - p) < 0.3 * fps))
    return out


@dataclass
class HandEstimate:
    hand: str | None           # "right" / "left" / None (undetermined)
    confidence: float
    evidence: str
    two_handed_like: bool = False


def serve_pattern(C_or_P: np.ndarray, t: np.ndarray, w: SwingWindow, hit: int, toss: int):
    """Check the overhead-serve pattern for a given (hitting wrist, tossing wrist) assignment.

    Returns (ok, contact_idx, hit_above_head, toss_above_head, toss_idx). Contact = highest hitting-wrist
    point in the 0.6 s before (and just after) the speed peak (full extension), which is earlier than the
    2D speed peak of a serve (the arm keeps accelerating downwards after contact).
    """
    dt = float(np.median(np.diff(t)))
    ny = C_or_P[:, 0, 1]
    a, b = max(w.i0, w.i_peak - int(0.6 / dt)), min(w.i1, w.i_peak + int(0.06 / dt))
    hy = C_or_P[a:b + 1, hit, 1] - ny[a:b + 1]
    if not np.isfinite(hy).any():
        return False, w.i_peak, -9.0, -9.0, w.i0
    ic = a + int(np.nanargmax(hy))
    hit_above = float(np.nanmax(hy))
    ta, tb = max(w.i0, ic - int(2.0 / dt)), max(w.i0 + 1, ic - int(0.12 / dt))
    ty = C_or_P[ta:tb, toss, 1] - ny[ta:tb]
    if not np.isfinite(ty).any():
        return False, ic, hit_above, -9.0, ta
    it = ta + int(np.nanargmax(ty))
    toss_above = float(np.nanmax(ty))
    ok = hit_above > 0.25 and toss_above > 0.1
    return ok, ic, hit_above, toss_above, it


def serve_hand(P: np.ndarray, t: np.ndarray, w: SwingWindow) -> HandEstimate | None:
    """Hand from the serve pattern (toss arm rises first, hitting arm reaches higher later)."""
    okR = serve_pattern(P, t, w, R_WRI, L_WRI)
    okL = serve_pattern(P, t, w, L_WRI, R_WRI)
    if okR[0] and not okL[0]:
        return HandEstimate("right", 0.8, "发球模式：左手先上举（抛球），右手腕随后在头顶更高处到达最高点")
    if okL[0] and not okR[0]:
        return HandEstimate("left", 0.8, "发球模式：右手先上举（抛球），左手腕随后在头顶更高处到达最高点")
    if okR[0] and okL[0]:
        # both arms go overhead: the hitting arm reaches the higher point, later
        dr, dl = okR[2] - okL[2], okR[1] - okL[1]
        if abs(dr) > 0.15 and np.sign(dr) == np.sign(dl):
            hand = "right" if dr > 0 else "left"
            return HandEstimate(hand, 0.5, f"两臂都举过头顶；{'右' if hand == 'right' else '左'}手腕到达更高点且更晚，判为持拍手（中等置信）")
    return None


def estimate_hand(P: np.ndarray, t: np.ndarray, w: SwingWindow, view: str | None = None) -> HandEstimate:
    """Hitting hand of one swing from two cues that must agree: peak wrist speed and wrist path length
    around the speed peak.

    * Two-handed backhands: both wrists move together -> hand not observable from wrists (low confidence).
    * Back / oblique-back views: the hitting arm moves largely in depth while the non-hitting arm sweeps
      across the image, so the 2D cues are weak -> confidence is reduced.
    """
    dt = float(np.median(np.diff(t)))
    sv = serve_hand(P, t, w)
    if sv is not None:
        return sv
    sl = nanfill(speed(P[:, L_WRI], t))
    sr = nanfill(speed(P[:, R_WRI], t))
    a, b = max(w.i0, w.i_peak - int(0.35 / dt)), min(w.i1, w.i_peak + 3)
    pl, pr = sl[a:b + 1].max(), sr[a:b + 1].max()
    pa, pb = max(w.i0, w.i_peak - int(0.6 / dt)), min(w.i1, w.i_peak + int(0.3 / dt))
    ll, lr_ = float(np.sum(sl[pa:pb] * dt)), float(np.sum(sr[pa:pb] * dt))
    dist = np.linalg.norm(P[a:b + 1, L_WRI] - P[a:b + 1, R_WRI], axis=-1)
    close = float(np.nanmedian(dist)) if np.isfinite(dist).any() else np.inf
    if close < 0.45:
        return HandEstimate(hand=None, confidence=0.2, two_handed_like=True,
                            evidence=f"两手腕在加速段距离很近（中位 {close:.2f} 躯干长），疑似双手击球，无法从手腕判断持拍手")
    r_speed = max(pl, pr) / max(min(pl, pr), 1e-6)
    r_path = max(ll, lr_) / max(min(ll, lr_), 1e-6)
    h_speed = "left" if pl > pr else "right"
    h_path = "left" if ll > lr_ else "right"
    if h_speed != h_path:
        return HandEstimate(hand=None, confidence=0.1,
                            evidence=f"峰值速度（比值 {r_speed:.2f}）与轨迹长度（比值 {r_path:.2f}）指向不同的手，持拍手不确定")
    conf = float(np.clip((min(r_speed, r_path) - 1.0) / 0.6, 0, 1))
    note = ""
    if view in ("back", "oblique_back"):
        conf *= 0.5
        note = "；背面/斜后方机位下持拍臂主要沿深度方向运动，该判断置信度已下调"
    if conf < 0.2:
        return HandEstimate(hand=None, confidence=conf,
                            evidence=f"左右手腕速度/轨迹差异不明显（{r_speed:.2f}/{r_path:.2f}），持拍手不确定{note}")
    return HandEstimate(hand=h_speed, confidence=conf,
                        evidence=f"{'左' if h_speed == 'left' else '右'}手腕峰值速度为另一侧 {r_speed:.2f} 倍、"
                                 f"轨迹长度为 {r_path:.2f} 倍{note}")


def combine_hand(estimates: list[HandEstimate]) -> HandEstimate:
    votes = {"right": 0.0, "left": 0.0}
    for e in estimates:
        if e.hand:
            votes[e.hand] += e.confidence
    tot = votes["right"] + votes["left"]
    if tot == 0:
        return HandEstimate(None, 0.0, "所有挥拍都无法判断持拍手（可能均为双手击球），请人工指定")
    hand = max(votes, key=votes.get)
    conf = votes[hand] / tot * min(1.0, tot / 1.0)
    return HandEstimate(hand, float(conf), f"{len([e for e in estimates if e.hand])} 次单手挥拍投票："
                        f"右 {votes['right']:.2f} / 左 {votes['left']:.2f}")


# ----------------------------------------------------------------------------------------------
# Phase segmentation (canonical space: right-handed, hitting wrist = CANON['h_wri'])
# ----------------------------------------------------------------------------------------------

@dataclass
class Phases:
    action: str
    events: dict                 # name -> {"i": idx, "t": seconds, "estimated": bool, "note": str}
    segments: list = field(default_factory=list)   # [{"name", "label", "t0", "t1", "i0", "i1"}]
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {"action": self.action, "events": self.events, "segments": self.segments, "warnings": self.warnings}

    def idx(self, name):
        return self.events[name]["i"]


PHASE_LABELS = {
    "ready": "准备", "backswing": "引拍/蓄力", "forward": "加速", "contact": "击球附近（估计）",
    "follow": "随挥", "toss": "抛球/蓄力", "drop": "持拍手下落", "finish": "收拍",
}


def _ev(i, t, estimated=True, note=""):
    return {"i": int(i), "t": float(t[i]), "estimated": bool(estimated), "note": note}


def _rest_before(s: np.ndarray, i_from: int, lo: int, frac: float, peak: float, hold: int) -> int:
    """Walk backwards from i_from to the last index where speed stays below frac*peak for `hold` samples."""
    thr = frac * peak
    run = 0
    for i in range(i_from, lo - 1, -1):
        run = run + 1 if s[i] < thr else 0
        if run >= hold:
            return i + hold - 1
    return lo


def _rest_after(s: np.ndarray, i_from: int, hi: int, frac: float, peak: float, hold: int) -> int:
    thr = frac * peak
    run = 0
    for i in range(i_from, hi + 1):
        run = run + 1 if s[i] < thr else 0
        if run >= hold:
            return i - hold + 1
    return hi


def backswing_end(s: np.ndarray, ic: int, lo: int, fps: float, peak: float) -> int | None:
    """End of take-back = speed trough that starts the forward swing.

    Walk backwards from the contact estimate down the acceleration slope; the forward swing (incl. a
    looped racket drop) accelerates almost monotonically, so the first rise in speed when going back
    marks the take-back motion. The trough must be well below the peak to count.
    """
    i = ic
    floor = s[ic]
    tol = 0.04 * peak
    min_i = max(lo, ic - int(1.2 * fps))
    best = ic
    while i > min_i:
        i -= 1
        if s[i] < floor:
            floor, best = s[i], i
        elif s[i] > floor + tol and s[i] > 1.3 * floor:
            break
    if best >= ic - int(0.06 * fps) or floor > 0.5 * peak:
        return None
    return best


def takeback_start(s: np.ndarray, ib: int, lo: int, fps: float, peak: float) -> int:
    """Start of take-back: go back over the take-back speed bump to the preceding rest."""
    a = max(lo, ib - int(1.2 * fps))
    if ib - a < 2:
        return a
    seg = s[a:ib + 1]
    ip = a + int(np.argmax(seg))                     # take-back speed bump
    bump = s[ip]
    thr = max(0.08 * peak, 0.3 * bump, s[ib] * 1.05)
    i = ip
    while i > a and s[i] > thr:
        i -= 1
    return i


def segment_phases(C: np.ndarray, t: np.ndarray, w: SwingWindow, action: str) -> Phases:
    """Segment one swing (canonical body space C) into phases.

    Contact is *estimated* as the hitting-wrist speed peak (wrist, not racket head; no ball
    detection), so it is always flagged ``estimated``.
    """
    fps = 1 / np.median(np.diff(t))
    hw, nw = CANON["h_wri"], CANON["n_wri"]
    W = C[:, hw]
    if action == "backhand_2h":
        W = (C[:, hw] + C[:, nw]) / 2
    s = nanfill(speed(W, t))
    lo, hi = w.i0, w.i1
    # contact estimate: hitting-wrist speed peak near the detected swing peak
    a, b = max(lo, w.i_peak - int(0.1 * fps)), min(hi, w.i_peak + int(0.1 * fps))
    ic = a + int(np.argmax(s[a:b + 1]))
    peak = s[ic]
    hold = max(2, int(0.08 * fps))
    warnings = []
    ev = {}
    if action == "serve":
        # contact for serves: full extension (highest hitting wrist), not the 2D speed peak
        _, ic, _, _, _ = serve_pattern(C, t, w, hw, nw)
        Ny = C[:, nw, 1]
        # trophy / toss peak: highest non-hitting wrist in the 2.0 s before contact
        ta, tb = max(lo, ic - int(2.0 * fps)), max(lo, ic - int(0.15 * fps))
        if tb > ta and np.isfinite(Ny[ta:tb]).any():
            it = ta + int(np.nanargmax(Ny[ta:tb]))
        else:
            it = max(lo, ic - int(0.6 * fps)); warnings.append("抛球手最高点无法可靠定位，使用默认间隔")
        # wrist drop: lowest hitting wrist (relative to hitting shoulder) between trophy and contact
        rel = C[:, hw, 1] - C[:, CANON["h_sho"], 1]
        if ic - it > 2 and np.isfinite(rel[it:ic]).any():
            idrop = it + int(np.nanargmin(rel[it:ic]))
        else:
            idrop = it
        # start of toss: non-hitting wrist begins to rise
        sn = nanfill(speed(C[:, nw], t))
        istart = _rest_before(sn, it, lo, 0.15, max(sn[lo:ic + 1].max(), 1e-6), hold)
        iend = _rest_after(s, ic, hi, 0.2, peak, hold)
        ev = {"start": _ev(istart, t, True, "抛球手开始上举（估计）"),
              "trophy": _ev(it, t, True, "抛球手最高点，近似奖杯姿势"),
              "drop": _ev(idrop, t, True, "持拍手腕相对肩部最低点（手腕，不是拍头）"),
              "contact": _ev(ic, t, True, "持拍手腕速度峰值，估计击球时刻（未检测球/拍）"),
              "end": _ev(iend, t, True, "随挥结束（手腕速度回落）")}
        segs = [("ready", lo, istart), ("toss", istart, it), ("drop", it, idrop), ("forward", idrop, ic),
                ("follow", ic, iend)]
    else:
        ib = backswing_end(s, ic, lo, fps, peak)
        if ib is None:
            # fallback: wrist farthest from its contact position within 1.2 s before contact
            ba, bb = max(lo, ic - int(1.2 * fps)), max(lo, ic - int(0.08 * fps))
            if bb > ba:
                d = np.linalg.norm(W[ba:bb] - W[ic], axis=-1)
                ib = ba + int(np.nanargmax(np.nan_to_num(d, nan=-1)))
            else:
                ib = lo
            warnings.append("引拍终点没有明显的速度低谷，改用“离击球点最远”估计")
        istart = takeback_start(s, ib, lo, fps, peak)
        if istart <= lo + 1 and (w.truncated_start or lo == 0):
            warnings.append("视频开头距离引拍太近，准备阶段可能不完整")
        iend = _rest_after(s, ic, hi, 0.2, peak, hold)
        if iend == hi and w.truncated_end:
            warnings.append("视频在随挥结束前截断，随挥阶段可能不完整")
        ev = {"start": _ev(istart, t, True, "手腕开始引拍（速度离开静止，估计）"),
              "backswing_end": _ev(ib, t, True, "手腕离击球位置最远处（引拍终点）"),
              "contact": _ev(ic, t, True, "持拍手腕速度峰值，估计击球时刻（未检测球/拍）"),
              "end": _ev(iend, t, True, "随挥结束（手腕速度回落）")}
        segs = [("ready", lo, istart), ("backswing", istart, ib), ("forward", ib, ic), ("follow", ic, iend)]
    segments = []
    for name, i0, i1 in segs:
        if i1 < i0:
            i1 = i0
        segments.append({"name": name, "label": PHASE_LABELS[name], "i0": int(i0), "i1": int(i1),
                         "t0": float(t[i0]), "t1": float(t[i1])})
    ev["window_start"] = _ev(lo, t, False, "分析窗口起点")
    ev["window_end"] = _ev(hi, t, False, "分析窗口终点")
    return Phases(action=action, events=ev, segments=segments, warnings=warnings)


def head_y(C: np.ndarray) -> np.ndarray:
    return C[:, NOSE, 1]


def wrist_ids():
    return L_WRI, R_WRI
