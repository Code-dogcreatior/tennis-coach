from math import erf

import numpy as np
import pytest

from tennis_coach.dtw import dtw, phase_dtw
from tennis_coach.skeleton import (FLIP_PERM, L_ELB, L_SHO, L_WRI, R_ELB, R_SHO, R_WRI, clean, mirror_lr,
                                   normalise, side_joints)
from tennis_coach.swing import detect_swings, estimate_hand, segment_phases
from tennis_coach.view import ViewEstimate, canonicalise, view_key


def synthetic_swing(hand="right", fps=60.0, dur=3.0, contact_t=1.8, tempo=1.0):
    """A crude 2D right/left-handed forehand-like swing in pixel space (y down).

    The hitting wrist rests, takes back (+ pause), then accelerates forward through 'contact'.
    """
    t = np.arange(0, dur, 1 / fps)
    T = len(t)
    K = np.zeros((T, 17, 2))
    base = {0: (500, 300), 1: (495, 295), 2: (505, 295), 3: (490, 300), 4: (510, 300),
            5: (470, 360), 6: (530, 360), 7: (460, 430), 8: (540, 430), 9: (455, 490), 10: (545, 490),
            11: (480, 520), 12: (520, 520), 13: (478, 620), 14: (522, 620), 15: (476, 720), 16: (524, 720)}
    for j, (x, y) in base.items():
        K[:, j] = (x, y)
    hw = R_WRI if hand == "right" else L_WRI
    he = R_ELB if hand == "right" else L_ELB
    side = 1 if hand == "right" else -1
    for i, ti in enumerate(t):
        u = (ti - contact_t) / tempo
        if u < -1.0:
            off = (0, 0)
        elif u < -0.35:           # take-back: wrist goes out and back (to hitting side), slow
            a = (u + 1.0) / 0.65
            off = (side * 120 * np.sin(a * np.pi / 2), -60 * a)
        elif u < -0.25:           # pause at top
            off = (side * 120, -60)
        else:                     # forward swing: smooth S-curve whose speed peaks exactly at contact (u=0)
            a = 0.5 * (1 + erf(u / 0.1))
            off = (side * (120 - 300 * a), -60 + 40 * a)
        K[i, hw] = K[i, hw] + off
        K[i, he] = K[i, he] + np.array(off) * 0.5
    S = np.ones((T, 17))
    return t, K, S


def test_mirror_is_involution_and_swaps_labels():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(10, 17, 2))
    Q = mirror_lr(P)
    assert np.allclose(mirror_lr(Q), P)
    # left wrist of the mirrored skeleton = mirrored right wrist of the original
    assert np.allclose(Q[:, L_WRI, 0], -P[:, R_WRI, 0])
    assert np.allclose(Q[:, L_WRI, 1], P[:, R_WRI, 1])
    assert np.allclose(Q[:, 0, 0], -P[:, 0, 0])            # nose stays nose
    assert sorted(FLIP_PERM.tolist()) == list(range(17))


def test_mirror_scores_follow_labels():
    P = np.zeros((1, 17, 2))
    S = np.arange(17, dtype=float)[None]
    _, S2 = mirror_lr(P, S)
    assert S2[0, L_SHO] == S[0, R_SHO] and S2[0, R_SHO] == S[0, L_SHO]


def test_left_hander_mirrors_to_right_hander_features():
    """A left-handed swing, after canonicalisation, must match the right-handed one."""
    t, Kr, S = synthetic_swing("right")
    _, Kl, _ = synthetic_swing("left")
    # make the left-hander an exact mirror image of the right-hander about x=500
    Kl = Kr[:, FLIP_PERM].copy()
    Kl[..., 0] = 1000 - Kl[..., 0]
    Pr, _, _ = normalise(Kr)
    Pl, _, _ = normalise(Kl)
    v = ViewEstimate("front", "towards", 1, 0.5, 1, 1, [])
    Cr, _, kr, _ = canonicalise(Pr, S, "right", v)
    Cl, _, kl, _ = canonicalise(Pl, S, "left", v)
    assert kr == kl
    assert np.allclose(Cr, Cl, atol=1e-9)


def test_side_joints_semantics():
    r, l = side_joints("right"), side_joints("left")
    assert r["h_wri"] == R_WRI and l["h_wri"] == L_WRI and r["n_sho"] == L_SHO and l["n_sho"] == R_SHO


def test_view_key_side_depends_on_facing():
    assert view_key("side", "screen_right") == "side_hitting"
    assert view_key("side", "screen_left") == "side_nonhitting"


def test_hand_detection_and_phase_order():
    for hand in ("right", "left"):
        t, K, S = synthetic_swing(hand)
        cs = clean(t, K, S, fix_swaps=False)
        P, _, _ = normalise(cs.K)
        sw = detect_swings(P, cs.t, min_speed=2.0)
        assert len(sw) == 1
        h = estimate_hand(P, cs.t, sw[0])
        assert h.hand == hand
        v = ViewEstimate("front", "towards", 1, 0.5, 1, 1, [])
        C, _, _, _ = canonicalise(P, cs.S, hand, v)
        ph = segment_phases(C, cs.t, sw[0], "forehand")
        e = ph.events
        assert e["start"]["t"] <= e["backswing_end"]["t"] < e["contact"]["t"] <= e["end"]["t"]
        assert abs(e["contact"]["t"] - 1.8) < 0.25          # wrist speed peak near designed contact
        assert e["contact"]["estimated"] is True


def test_tempo_preserved():
    """A slower swing must produce a longer forward-swing duration (durations are not normalised away)."""
    out = []
    for tempo in (1.0, 1.5):
        t, K, S = synthetic_swing("right", tempo=tempo, dur=4.0, contact_t=2.4)
        cs = clean(t, K, S, fix_swaps=False)
        P, _, _ = normalise(cs.K)
        sw = detect_swings(P, cs.t, min_speed=1.0)
        v = ViewEstimate("front", "towards", 1, 0.5, 1, 1, [])
        C, _, _, _ = canonicalise(P, cs.S, "right", v)
        ph = segment_phases(C, cs.t, sw[0], "forehand")
        out.append(ph.events["contact"]["t"] - ph.events["backswing_end"]["t"])
    assert out[1] > out[0] * 1.2


def test_dtw_identity_and_shift():
    x = np.sin(np.linspace(0, 6, 80))[:, None]
    c0, p = dtw(x, x)
    assert c0 == pytest.approx(0, abs=1e-9) and p[0] == (0, 0) and p[-1] == (79, 79)
    y = np.sin(np.linspace(0, 6, 120))[:, None]      # same shape, different duration
    c1, _ = dtw(x, y)
    z = np.cos(np.linspace(0, 6, 80))[:, None]
    c2, _ = dtw(x, z)
    assert c1 < c2


def test_phase_dtw_respects_anchors():
    x = np.concatenate([np.zeros(20), np.linspace(0, 1, 20), np.ones(20)])[:, None]
    y = np.concatenate([np.zeros(30), np.linspace(0, 1, 10), np.ones(25)])[:, None]
    cost, path, segc = phase_dtw(x, y, [(0, 20), (20, 40), (40, 60)], [(0, 30), (30, 40), (40, 65)])
    assert len(segc) == 3
    # every aligned pair stays within its own segment
    for i, j in path:
        si = 0 if i < 20 else 1 if i < 40 else 2
        sj = 0 if j < 30 else 1 if j < 40 else 2
        assert si == sj


def test_clean_preserves_real_time_and_gaps():
    t = np.array([0, 0.033, 0.066, 0.1, 0.2, 0.233, 0.266])   # irregular (VFR) with a 100 ms gap
    K = np.zeros((7, 17, 2)) + np.arange(7)[:, None, None]
    S = np.ones((7, 17))
    cs = clean(t, K, S, fix_swaps=False)
    assert cs.t[0] == pytest.approx(0) and cs.t[-1] == pytest.approx(0.266, abs=0.034)
    assert np.isfinite(cs.K).all()


def test_lr_swap_fix():
    from tennis_coach.skeleton import fix_lr_swaps
    t, K, S = synthetic_swing("right", dur=1.0)
    K2 = K.copy()
    K2[10, [5, 6, 7, 8, 9, 10]] = K[10, [6, 5, 8, 7, 10, 9]]     # inject a label flip on one frame
    fixed, _, n = fix_lr_swaps(K2, S)
    assert n >= 1
    assert np.allclose(fixed[10], K[10])


def test_knn_vote_source_and_balanced_strategies():
    """One player with many near-identical swings must not out-vote other actions by sheer count."""
    from tennis_coach.match import knn_vote
    costs = [(0.50, "serve", 1, "A"), (0.51, "serve", 2, "A"), (0.52, "serve", 3, "A"),
             (0.55, "forehand", 4, "B"), (0.56, "forehand", 5, "C"), (0.90, "serve", 6, "D")]
    q = {"view_key": "front"}
    assert knn_vote(q, [], strategy="plain", costs=costs)["label"] == "serve"        # A fills all 3 slots
    src = knn_vote(q, [], strategy="source", costs=costs)
    assert src["votes"] == {"serve": 1, "forehand": 2} and src["label"] == "forehand"
    bal = knn_vote(q, [], strategy="balanced", costs=costs)
    assert bal["label"] == "forehand" and bal["class_score"]["serve"] == 0.7
    assert bal["nearest_cost"] == 0.5          # rejection still uses the nearest swing


def test_toss_hand_low_flags_smash_like_preparation():
    """Serve: free wrist starts below the hips (ball held at the waist) before the toss; smash: already up."""
    import numpy as np
    from tennis_coach.classify import (Classification, SMASH_LIKE_THR, combine_with_knn, toss_hand_low)
    from tennis_coach.skeleton import CANON as H
    t = np.arange(90) / 30.0
    C = np.zeros((90, 17, 2))
    C[:, H["h_hip"], 1] = C[:, H["n_hip"], 1] = 0.0
    serve, smash = C.copy(), C.copy()
    serve[:, H["n_wri"], 1] = np.linspace(-0.3, 1.5, 90)      # rises from below the hips
    smash[:, H["n_wri"], 1] = np.linspace(0.6, 1.5, 90)       # arm already up
    assert toss_hand_low(serve, t, 85, 70) < SMASH_LIKE_THR <= toss_hand_low(smash, t, 85, 70)
    assert toss_hand_low(serve, t, 85, 5) is None              # < 0.3 s visible before the toss
    rule = Classification("serve", 0.35, [], "serve", 0.35, scores={"smash_like": 1.0})
    kv = {"label": "serve", "votes": {"serve": 3}, "k": 3, "nearest_cost": 0.5, "cost_threshold": 1.6}
    assert combine_with_knn(rule, kv).confidence <= 0.35      # kNN agreement must not hide the smash flag
