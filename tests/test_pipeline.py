"""Pipeline-level tests on synthetic pose sequences (no video decoding / no model needed)."""
import numpy as np

from tennis_coach.analysis import analyse_video, swing_record
from tennis_coach.feedback import build_feedback
from tennis_coach.match import match
from tennis_coach.pose import PoseSeq
from tennis_coach.quality import check as quality_check
from tennis_coach.skeleton import FLIP_PERM

from test_core import synthetic_swing


def make_seq(K, S, t, w=1000, h=900):
    T = len(t)
    bbox = np.column_stack([np.nanmin(K[..., 0], 1) - 20, np.nanmin(K[..., 1], 1) - 20,
                            np.nanmax(K[..., 0], 1) + 20, np.nanmax(K[..., 1], 1) + 20])
    return PoseSeq(t=t, kpts=K, scores=S * 0.9, bbox=bbox, n_persons=np.where(np.arange(T) % 6 == 0, 1, -1),
                   other_area=np.zeros(T), width=w, height=h, fps=1 / np.median(np.diff(t)))


def analyse(K, S, t, **kw):
    return analyse_video("nonexistent.mp4", pose=make_seq(K, S, t), **kw)


def test_static_person_has_no_swing():
    t, K, S = synthetic_swing("right", dur=3.0)
    K[:] = K[0]                                     # nobody moves (non-tennis / idle clip)
    r = analyse(K, S, t, time_scale=1.0)
    assert r["n_swings"] == 0
    assert any(i["code"] == "no_swing" for i in r["quality"]["issues"])


def test_non_stroke_arm_wave_is_unknown_or_absent():
    t, K, S = synthetic_swing("right", dur=3.0)
    K2 = np.repeat(K[:1], len(t), axis=0).copy()
    K2[:, 9, 0] += 60 * np.sin(2 * np.pi * 0.7 * t)   # slow waving of the non-hitting arm only
    r = analyse(K2, S, t, time_scale=1.0, hand_override="right")
    assert all(s["action"] == "unknown" for s in r["swings"])


def test_feet_cut_and_occlusion_flagged():
    t, K, S = synthetic_swing("right", dur=3.0)
    seq = make_seq(K, S, t)
    seq.scores[:, [15, 16]] = 0.05                    # ankles not visible (cut off / occluded)
    seq.scores[:, 5:11] = 0.2                         # arms poorly visible
    q = quality_check(seq)
    codes = {i["code"] for i in q["issues"]}
    assert "feet_cut" in codes and "arm_occluded" in codes
    assert q["level"] in ("warn", "bad")


def test_unknown_time_scale_disables_time_metrics():
    t, K, S = synthetic_swing("right", dur=3.0)
    r = analyse(K, S, t, time_scale=None, hand_override="right", view_override={"view": "front", "facing": "towards"},
                action_overrides={0: "forehand"})
    m = r["swings"][0]["metrics"]
    assert not r["time_known"]
    assert m["dur_forward"]["reliable"] is False and m["peak_speed"]["reliable"] is False
    assert m["contact_height"]["reliable"] is True


def _ref(r, rid, view_key=None):
    sw = dict(r["swings"][0])
    if view_key:
        sw["view_key"] = view_key
    return swing_record(sw, rid)


def test_view_mismatch_refuses_to_score():
    t, K, S = synthetic_swing("right", dur=3.0)
    kw = dict(time_scale=1.0, hand_override="right", view_override={"view": "front", "facing": "towards"},
              action_overrides={0: "forehand"})
    r = analyse(K, S, t, **kw)
    ref = _ref(r, 1, view_key="side_hitting")
    m = match(swing_record(r["swings"][0], None), [ref])
    assert m["status"] == "view_mismatch" and not m["candidates"]
    fb = build_feedback("forehand", m, "正面")
    assert fb[0]["kind"] == "info"                    # no scores / no technique claims


def test_left_hander_matches_right_handed_reference():
    """Mirrored left-hander vs right-handed reference of the same motion -> ~zero DTW cost, no flags."""
    t, K, S = synthetic_swing("right", dur=3.0)
    Kl = K[:, FLIP_PERM].copy()
    Kl[..., 0] = 1000 - Kl[..., 0]
    kw = dict(time_scale=1.0, view_override={"view": "front", "facing": "towards"}, action_overrides={0: "forehand"})
    rr = analyse(K, S, t, hand_override="right", **kw)
    rl = analyse(Kl, S, t, hand_override="left", **kw)
    m = match(swing_record(rl["swings"][0], None), [_ref(rr, 1), _ref(rr, 2), _ref(rr, 3)])
    assert m["status"] == "ok"
    assert m["candidates"][0]["cost"] < 1e-6
    assert all(c["status"] == "within" for c in m["comparison"].values())


def test_slower_swing_is_reported_against_reference():
    kw = dict(time_scale=1.0, hand_override="right", view_override={"view": "front", "facing": "towards"},
              action_overrides={0: "forehand"})
    refs = []
    for i, tempo in enumerate((1.0, 1.02, 0.98)):
        t, K, S = synthetic_swing("right", dur=4.0, contact_t=2.4, tempo=tempo)
        refs.append(_ref(analyse(K, S, t, **kw), i))
    t, K, S = synthetic_swing("right", dur=4.0, contact_t=2.4, tempo=1.8)
    q = analyse(K, S, t, **kw)
    m = match(swing_record(q["swings"][0], None), refs)
    assert m["comparison"]["dur_forward"]["status"] == "above"
    fb = build_feedback("forehand", m, "正面")
    assert any(f.get("metric") == "dur_forward" for f in fb)
