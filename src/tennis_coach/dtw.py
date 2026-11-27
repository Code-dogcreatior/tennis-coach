"""Dynamic time warping with a Sakoe-Chiba band, plus phase-anchored DTW."""
from __future__ import annotations

import numpy as np


def dtw(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None, band: float = 0.3):
    """Multivariate DTW. A:(n,F) B:(m,F). Returns (mean aligned cost, path [(i,j),...]).

    The cost is normalised by the path length so sequences of different duration are comparable;
    durations themselves are reported separately (tempo metrics), so warping does not hide them.
    """
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return float("inf"), []
    w = np.ones(A.shape[1]) if weights is None else np.asarray(weights, float)
    diff = A[:, None, :] - B[None, :, :]
    cost = np.sqrt(np.sum(w * diff ** 2, axis=-1))
    r = max(int(band * max(n, m)), abs(n - m)) + 1
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jc = int(round(i * m / n))
        j0, j1 = max(1, jc - r), min(m, jc + r)
        for j in range(j0, j1 + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    if not np.isfinite(D[n, m]):
        return float("inf"), []
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        cands = []
        if i > 1 and j > 1:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 1:
            cands.append((D[i - 1, j], i - 1, j))
        if j > 1:
            cands.append((D[i, j - 1], i, j - 1))
        _, i, j = min(cands)
        path.append((i - 1, j - 1))
    path.reverse()
    return float(D[n, m] / len(path)), path


def phase_dtw(A: np.ndarray, B: np.ndarray, segA: list[tuple[int, int]], segB: list[tuple[int, int]],
              weights: np.ndarray | None = None, band: float = 0.35):
    """Run DTW separately inside each pair of corresponding phase segments and stitch the paths.

    Anchoring on detected phase events prevents DTW from e.g. matching a backswing frame to a
    follow-through frame that happens to look similar in 2D.
    Returns (overall mean cost, stitched path, per-segment costs).
    """
    assert len(segA) == len(segB)
    path, costs, total, nsteps = [], [], 0.0, 0
    for (a0, a1), (b0, b1) in zip(segA, segB):
        a1, b1 = max(a1, a0 + 1), max(b1, b0 + 1)
        c, p = dtw(A[a0:a1], B[b0:b1], weights, band)
        costs.append(c)
        if p:
            path.extend((i + a0, j + b0) for i, j in p)
            total += c * len(p); nsteps += len(p)
    return (total / nsteps if nsteps else float("inf")), path, costs
