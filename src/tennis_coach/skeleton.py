"""Keypoint cleaning, body-centred normalisation, and left/right mirroring with label semantics.

Coordinate conventions
----------------------
* Pixel space: x right, y down (as delivered by the pose model).
* Body space ("normalised"): origin at the hip centre of each frame, unit = median torso length
  (shoulder-centre to hip-centre) over the clip, x right, **y up**.
* Canonical space: body space after (1) handedness canonicalisation - left-handers are mirrored
  (x -> -x **and** left/right joint labels swapped) so every player is analysed as right-handed -
  and (2) a sign convention on x chosen per view (see ``view.canonical_x_sign``). (2) is only a
  sign convention; comparisons are restricted to the same view class, so it never pretends that a
  different camera position is the same view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .config import KPT_THR

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# index permutation that swaps every left joint with its right counterpart
FLIP_PERM = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror_lr(P: np.ndarray, scores: np.ndarray | None = None):
    """Mirror a body-space sequence horizontally AND swap left/right labels.

    This turns a left-hander into an equivalent right-hander: the joint that was the left wrist
    (the hitting wrist of a left-hander) becomes the right wrist at the mirrored position.
    Works on (..., 17, 2) arrays. Applying it twice is the identity.
    """
    Q = P[..., FLIP_PERM, :].copy()
    Q[..., 0] = -Q[..., 0]
    if scores is None:
        return Q
    return Q, scores[..., FLIP_PERM].copy()


def side_joints(hand: str) -> dict:
    """Map semantic joints (hitting / non-hitting side) to COCO indices for a given hand."""
    if hand == "right":
        return dict(h_sho=R_SHO, h_elb=R_ELB, h_wri=R_WRI, h_hip=R_HIP, h_knee=R_KNEE, h_ank=R_ANK,
                    n_sho=L_SHO, n_elb=L_ELB, n_wri=L_WRI, n_hip=L_HIP, n_knee=L_KNEE, n_ank=L_ANK)
    if hand == "left":
        return dict(h_sho=L_SHO, h_elb=L_ELB, h_wri=L_WRI, h_hip=L_HIP, h_knee=L_KNEE, h_ank=L_ANK,
                    n_sho=R_SHO, n_elb=R_ELB, n_wri=R_WRI, n_hip=R_HIP, n_knee=R_KNEE, n_ank=R_ANK)
    raise ValueError(hand)


CANON = side_joints("right")  # after canonicalisation every player is right-handed


@dataclass
class CleanSeq:
    t: np.ndarray        # (T,) uniform time grid in seconds (real time preserved)
    K: np.ndarray        # (T,17,2) cleaned pixel coordinates (NaN where unreliable)
    S: np.ndarray        # (T,17) scores resampled (0 where missing)
    valid: np.ndarray    # (T,17) bool - joint measured or interpolated over a short gap
    fps: float
    src_index: np.ndarray  # (T,) nearest original frame index for each grid sample
    lr_swaps: int = 0


def fix_lr_swaps(K: np.ndarray, S: np.ndarray, thr: float = KPT_THR) -> tuple[np.ndarray, np.ndarray, int]:
    """Correct frame-wise left/right label flips of whole limbs using temporal continuity.

    Pose models occasionally swap left and right limbs (most often in side views). For each
    frame and limb group (arms, legs) we compare the cost of keeping vs swapping the labels with
    respect to the previous frame and swap when it is clearly cheaper.
    """
    K = K.copy(); S = S.copy()
    groups = [[(5, 6), (7, 8), (9, 10)], [(11, 12), (13, 14), (15, 16)]]
    max_run = 5
    T = len(K)

    def costs(i, ref, g):
        torso = np.nanmean([np.linalg.norm(ref[5] - ref[11]), np.linalg.norm(ref[6] - ref[12])])
        if not np.isfinite(torso) or torso <= 1:
            torso = 50.0
        keep = swap = 0.0
        n = 0
        for a, b in g:
            if S[i, a] < thr or S[i, b] < thr or np.isnan(ref[[a, b]]).any():
                continue
            keep += np.linalg.norm(K[i, a] - ref[a]) + np.linalg.norm(K[i, b] - ref[b])
            swap += np.linalg.norm(K[i, a] - ref[b]) + np.linalg.norm(K[i, b] - ref[a])
            n += 1
        return keep, swap, n, torso

    n_swaps = 0
    for g in groups:
        i = 1
        while i < T:
            ref = K[i - 1]
            if np.isnan(K[i]).all() or np.isnan(ref).all():
                i += 1; continue
            keep, swap, n, torso = costs(i, ref, g)
            if n >= 2 and swap < 0.5 * keep and keep > 0.25 * torso * n:
                # candidate glitch: only accept if the original labelling resumes within max_run frames
                end = None
                for j in range(i + 1, min(T, i + max_run + 1)):
                    k2, s2, n2, _ = costs(j, ref, g)
                    if n2 >= 2 and k2 < 0.7 * s2:
                        end = j
                        break
                if end is not None:
                    for f in range(i, end):
                        for a, b in g:
                            K[f, [a, b]] = K[f, [b, a]]
                            S[f, [a, b]] = S[f, [b, a]]
                    n_swaps += end - i
                    i = end
                    continue
            i += 1
    return K, S, n_swaps


def clean(t: np.ndarray, kpts: np.ndarray, scores: np.ndarray, thr: float = KPT_THR,
          max_gap_s: float = 0.25, fix_swaps: bool = True) -> CleanSeq:
    """Resample onto a uniform grid (preserving real time), fill short gaps, smooth."""
    t = np.asarray(t, float)
    K = np.asarray(kpts, float).copy()
    S = np.asarray(scores, float).copy()
    n_swaps = 0
    if fix_swaps:
        K, S, n_swaps = fix_lr_swaps(K, S, thr)
    dts = np.diff(t)
    dts = dts[dts > 0]
    dt = float(np.median(dts)) if len(dts) else 1 / 30
    fps = 1.0 / dt
    grid = np.arange(t[0], t[-1] + dt * 0.5, dt)
    T = len(grid)
    Kg = np.full((T, 17, 2), np.nan)
    Sg = np.zeros((T, 17))
    valid = np.zeros((T, 17), bool)
    for j in range(17):
        ok = (S[:, j] >= thr) & ~np.isnan(K[:, j, 0])
        if ok.sum() < 2:
            continue
        tj = t[ok]
        for c in range(2):
            Kg[:, j, c] = np.interp(grid, tj, K[ok, j, c])
        Sg[:, j] = np.interp(grid, t, np.nan_to_num(S[:, j]))
        # distance (in time) to the nearest real observation -> invalidate long gaps / extrapolation
        idx = np.searchsorted(tj, grid)
        lo = tj[np.clip(idx - 1, 0, len(tj) - 1)]
        hi = tj[np.clip(idx, 0, len(tj) - 1)]
        gap_ok = ((grid >= tj[0] - dt) & (grid <= tj[-1] + dt)) & ((hi - lo) <= max_gap_s + 1e-9)
        exact = np.minimum(np.abs(grid - lo), np.abs(grid - hi)) < dt * 0.51
        v = gap_ok | exact
        valid[:, j] = v
        Kg[~v, j] = np.nan
    # smoothing (Savitzky-Golay over ~70 ms, order 3 keeps swing peaks)
    win = int(round(0.07 * fps)) | 1
    win = max(5, win)
    if T >= win + 2:
        for j in range(17):
            for c in range(2):
                x = Kg[:, j, c]
                m = ~np.isnan(x)
                if m.sum() < win:
                    continue
                # smooth contiguous runs only
                runs = np.split(np.arange(T), np.where(np.diff(m.astype(int)) != 0)[0] + 1)
                for r in runs:
                    if m[r[0]] and len(r) >= win:
                        x[r] = savgol_filter(x[r], win, 3)
    src = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    return CleanSeq(t=grid - grid[0] + t[0], K=Kg, S=Sg, valid=valid, fps=fps, src_index=src, lr_swaps=n_swaps)


def torso_length(K: np.ndarray) -> float:
    sc = (K[:, L_SHO] + K[:, R_SHO]) / 2
    hc = (K[:, L_HIP] + K[:, R_HIP]) / 2
    d = np.linalg.norm(sc - hc, axis=1)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    return float(np.median(d))


def torso_track(K: np.ndarray, fps: float, win_s: float = 2.0) -> tuple[np.ndarray, bool]:
    """Per-frame torso scale (pixels). Constant (global median) for a fixed camera; a rolling median over
    ``win_s`` when the apparent body size changes a lot (zooming hand-held camera, player walking towards
    the camera). Returns (scale[T], adaptive)."""
    g = torso_length(K)
    T = len(K)
    sc = (K[:, L_SHO] + K[:, R_SHO]) / 2
    hc = (K[:, L_HIP] + K[:, R_HIP]) / 2
    d = np.linalg.norm(sc - hc, axis=1)
    w = max(3, int(win_s * fps) | 1)
    h = w // 2
    roll = np.full(T, np.nan)
    for i in range(T):
        seg = d[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(3, w // 4):
            roll[i] = np.median(seg)
    ok = np.isfinite(roll)
    if not ok.any() or not np.isfinite(g):
        return np.full(T, g), False
    roll = np.interp(np.arange(T), np.where(ok)[0], roll[ok])
    p5, p95 = np.percentile(roll, [5, 95])
    if p95 / max(p5, 1e-6) < 1.25:
        return np.full(T, g), False
    return roll, True


def normalise(K: np.ndarray, torso: float | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixel -> body space. Returns (P, hip_track, torso).

    P: (T,17,2) hip-centred, torso-scaled, y-up. hip_track: (T,2) hip-centre trajectory in torso units
    (y-up), relative to its first valid value - footwork / global movement kept separately.
    ``torso`` may be a per-frame array (see ``torso_track``); the returned torso is then its median.
    """
    if torso is None:
        torso = torso_length(K)
    if isinstance(torso, np.ndarray):
        tarr = torso[:, None, None]
        tvec = torso[:, None]
        torso_out = float(np.median(torso))
    else:
        tarr = tvec = torso
        torso_out = torso
    hc = (K[:, L_HIP] + K[:, R_HIP]) / 2
    P = (K - hc[:, None, :]) / tarr
    P[..., 1] *= -1
    H = hc / tvec
    H[:, 1] *= -1
    first = np.where(np.isfinite(H[:, 0]))[0]
    if len(first):
        H = H - H[first[0]]
    return P, H, torso_out


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Projected 2D angle ABC in degrees (vectorised over leading dims)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))
