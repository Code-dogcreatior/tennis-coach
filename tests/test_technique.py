"""Technique checkpoints and the professional comparison (synthetic data)."""
import numpy as np

from tennis_coach import technique
from tennis_coach.analysis import swing_record
from tennis_coach.skeleton import normalise, torso_track

from test_core import synthetic_swing
from test_pipeline import analyse

KW = dict(time_scale=1.0, hand_override="right", view_override={"view": "front", "facing": "towards"},
          action_overrides={0: "forehand"})


def _pros(base_tech, key, values, view_key="front", hipw=0.4):
    out = []
    for k, v in enumerate(values):
        tech = {kk: dict(vv) for kk, vv in base_tech.items()}
        tech[key] = {**tech[key], "value": v}
        out.append({"ref_id": k, "action": "forehand", "view_key": view_key, "technique": tech,
                    "view": {"hip_width": hipw}, "player": f"P{k % 4}"})
    return out


def test_checkpoints_computed_for_view():
    t, K, S = synthetic_swing("right", dur=3.0)
    r = analyse(K, S, t, **KW)
    tech = r["swings"][0]["technique"]
    # front view: front/back variants and vertical checkpoints, never the side-only ones
    assert "gs_follow_height" in tech and "gs_sink" in tech
    assert "gs_contact_front" not in tech and "gs_shoulder_turn" not in tech


def test_judge_flags_only_fault_direction_outside_pro_range():
    t, K, S = synthetic_swing("right", dur=3.0)
    sw = analyse(K, S, t, **KW)["swings"][0]
    tech = sw["technique"]
    v = tech["gs_follow_height"]["value"]
    # professionals finish much higher -> "follow-through too low" is an issue
    pros = _pros(tech, "gs_follow_height", list(np.linspace(v + 1.0, v + 1.4, 10)))
    res = {c["key"]: c for c in technique.judge(tech, "forehand", "front", pros, {"hip_width": 0.4})}
    assert res["gs_follow_height"]["status"] == "issue" and res["gs_follow_height"]["direction"] == "low"
    # professionals finish lower -> outside the range, but 'higher' is not a fault direction
    pros = _pros(tech, "gs_follow_height", list(np.linspace(v - 1.4, v - 1.0, 10)))
    res = {c["key"]: c for c in technique.judge(tech, "forehand", "front", pros, {"hip_width": 0.4})}
    assert res["gs_follow_height"]["status"] == "outside_not_fault"


def test_too_few_pros_means_no_verdict():
    t, K, S = synthetic_swing("right", dur=3.0)
    tech = analyse(K, S, t, **KW)["swings"][0]["technique"]
    pros = _pros(tech, "gs_follow_height", [9.0, 9.1, 9.2])
    res = {c["key"]: c for c in technique.judge(tech, "forehand", "front", pros, {"hip_width": 0.4})}
    assert res["gs_follow_height"]["status"] == "insufficient_reference"


def test_angle_dependent_checkpoint_ignores_pros_filmed_from_other_angle():
    t, K, S = synthetic_swing("right", dur=3.0)
    tech = analyse(K, S, t, **KW)["swings"][0]["technique"]
    v = tech["gs_spacing_fb"]["value"]
    pros = _pros(tech, "gs_spacing_fb", list(np.linspace(v + 1, v + 1.5, 10)), hipw=0.15)  # very different angle
    res = {c["key"]: c for c in technique.judge(tech, "forehand", "front", pros, {"hip_width": 0.40})}
    assert res["gs_spacing_fb"]["status"] == "insufficient_reference"
    # vertical checkpoints pool all angles
    assert res["gs_follow_height"]["status"] != "insufficient_reference"


def test_pro_compare_status_and_gating():
    t, K, S = synthetic_swing("right", dur=3.0)
    sw = analyse(K, S, t, **KW)["swings"][0]
    q = swing_record(sw, None)
    assert technique.pro_compare(q, [])["status"] == "no_pro_refs"
    v = sw["technique"]["gs_follow_height"]["value"]
    pros = _pros(sw["technique"], "gs_follow_height", list(np.linspace(v + 1.0, v + 1.4, 12)))
    for p in pros:
        p.update(feat=q["feat"], feat_offset=q["feat_offset"], contact_i=q["contact_i"], fps=q["fps"],
                 phases=q["phases"], metrics=q["metrics"])
    old = technique._VALIDATION
    try:
        technique._VALIDATION = {"forehand:gs_follow_height": {"validated": True, "pro_false_alarm": 0.0, "pro_measured": 30}}
        r = technique.pro_compare(q, pros)
        assert r["status"] == "ok" and [c["key"] for c in r["issues"]] == ["gs_follow_height"]
        technique._VALIDATION = {}
        r = technique.pro_compare(q, pros)
        assert not r["issues"] and [c["key"] for c in r["tentative"]] == ["gs_follow_height"]
    finally:
        technique._VALIDATION = old


def test_adaptive_torso_scale_for_zooming_camera():
    t, K, S = synthetic_swing("right", dur=4.0)
    z = np.linspace(1.0, 2.0, len(t))[:, None, None]       # camera zooms in 2x
    cx = np.nanmean(K[..., 0])
    Kz = K.copy()
    Kz[..., 0] = cx + (K[..., 0] - cx) * z[..., 0]
    Kz[..., 1] = K[..., 1] * z[..., 0]
    fps = 1 / np.median(np.diff(t))
    tt, adaptive = torso_track(Kz, fps)
    assert adaptive
    P, _, _ = normalise(Kz, tt)
    P0, _, _ = normalise(K)
    # body-space coordinates stay (approximately) those of the un-zoomed sequence
    assert np.nanmedian(np.abs(P - P0)) < 0.1
    _, adaptive0 = torso_track(K, fps)
    assert not adaptive0


def _rec(key, value, view_key, time_known=True, action="serve"):
    return {"action": action, "view_key": view_key, "time_known": time_known, "view": {"hip_width": 0.4},
            "technique": {key: {"value": value}}}


def test_pooled_checkpoint_prefers_same_view_when_enough():
    key = "sv_contact_reach"   # view-pooled vertical measure
    side = [_rec(key, 1.0, "side_hitting") for _ in range(technique.VIEW_PREF_MIN)]
    back = [_rec(key, 0.2, "back") for _ in range(20)]
    vals, scope = technique.pro_distribution(key, "serve", "side", side + back, with_scope=True)
    assert scope == "same_view" and len(vals) == technique.VIEW_PREF_MIN and np.all(vals == 1.0)
    # too few same-view pros -> pool every view
    vals, scope = technique.pro_distribution(key, "serve", "side", side[:3] + back, with_scope=True)
    assert scope == "pooled" and len(vals) == 23


def test_slow_motion_pros_excluded_from_time_window_checkpoints():
    pros = [_rec("sv_leg_drive", 0.3, "back", time_known=False) for _ in range(8)] + \
           [_rec("sv_contact_reach", 1.0, "back", time_known=False) for _ in range(8)]
    assert len(technique.pro_distribution("sv_leg_drive", "serve", "back", pros)) == 0
    assert len(technique.pro_distribution("sv_contact_reach", "serve", "back", pros)) == 8
