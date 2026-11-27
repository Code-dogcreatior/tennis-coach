"""Application services: ingest, asynchronous processing jobs, reference library, deletion."""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from . import config, db
from .analysis import analyse_video, strip_arrays, swing_record
from .feedback import build_feedback
from .match import match
from .technique import pro_compare
from .pose import PoseSeq, extract_cached, file_sha1
from .video import transcode, validate

log = logging.getLogger(__name__)

ACTIONS = {"forehand", "backhand_1h", "backhand_2h", "serve", "unknown"}


def video_dir(vid: str) -> Path:
    return config.STORE_DIR / vid


def derived_dir(vid: str) -> Path:
    return config.DERIVED_DIR / vid


# ----------------------------------------------------------------------------------------------
# job queue (single worker: pose inference is the bottleneck and the GPU/CPU is shared)
# ----------------------------------------------------------------------------------------------
_q: "queue.Queue[tuple[str, str, dict]]" = queue.Queue()
_worker: threading.Thread | None = None


def _ensure_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, daemon=True, name="analysis-worker")
        _worker.start()


def _loop():
    while True:
        vid, task, kw = _q.get()
        try:
            if db.get_video(vid, full=False) is None:
                continue  # deleted while queued
            if task == "process":
                process(vid, **kw)
        except Exception as e:  # noqa: BLE001
            log.exception("job failed")
            db.update_video(vid, status="failed", error=f"{type(e).__name__}: {e}", stage="failed")
            (derived_dir(vid)).mkdir(parents=True, exist_ok=True)
            (derived_dir(vid) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            _q.task_done()


def enqueue(vid: str, task: str = "process", **kw):
    db.update_video(vid, status="queued", stage="queued", progress=0.0, error=None)
    _q.put((vid, task, kw))
    _ensure_worker()


# ----------------------------------------------------------------------------------------------
def ingest(src_path: Path, filename: str, kind: str, meta: dict) -> str:
    """Validate an uploaded file, store it and queue processing. Raises VideoError on invalid input."""
    if kind not in ("reference", "query"):
        raise ValueError("kind must be 'reference' or 'query'")
    validate(src_path)
    vid = uuid.uuid4().hex[:12]
    d = video_dir(vid)
    d.mkdir(parents=True, exist_ok=True)
    orig = d / ("original" + Path(filename).suffix.lower())
    shutil.move(str(src_path), orig)
    meta = dict(meta)
    if kind == "reference":
        # demo clips are never presented as a professional standard unless a coach review is recorded
        meta.setdefault("review_status", "demo")
        if meta.get("review_status") == "coach_reviewed" and not meta.get("reviewer"):
            meta["review_status"] = "demo"
    db.insert_video(vid, kind, filename, str(orig), meta)
    enqueue(vid)
    return vid


def _progress(vid: str):
    last = [0.0]

    def cb(stage: str, p: float):
        now = time.time()
        if now - last[0] > 0.5 or p >= 1:
            last[0] = now
            weights = {"transcode": (0.0, 0.05), "pose": (0.05, 0.85), "analyse": (0.85, 0.95), "match": (0.95, 1.0)}
            a, b = weights.get(stage, (0, 1))
            db.update_video(vid, stage=stage, progress=round(a + (b - a) * p, 3))
    return cb


def process(vid: str, reanalyse_only: bool = False):
    v = db.get_video(vid)
    meta = v["meta"]
    prog = _progress(vid)
    db.update_video(vid, status="running", error=None)
    d = video_dir(vid)
    mp4 = d / "video.mp4"
    if not mp4.exists():
        prog("transcode", 0)
        info = transcode(v["orig_path"], mp4)
        from .video import probe
        orig_info = probe(v["orig_path"])
        db.update_video(vid, path=str(mp4), sha1=file_sha1(mp4),
                        info={**info.to_dict(), "original": orig_info.to_dict()})
    pose, hit = extract_cached(mp4, progress=lambda p: prog("pose", p))
    # keep a per-video copy of the pose for the UI / deletion bookkeeping
    dd = derived_dir(vid)
    dd.mkdir(parents=True, exist_ok=True)
    pose.save(dd / "pose.npz")
    ts = meta.get("time_scale", 1.0)
    ts = None if ts in (None, "unknown", "") else float(ts)
    refs = load_refs(exclude_video=vid)
    kw = dict(hand_override=meta.get("hand") or None, view_override=meta.get("view_override") or None,
              refs=refs, progress=prog, pose=pose, time_scale=ts)
    act_over = {int(k): a for k, a in (meta.get("action_overrides") or {}).items() if a in ACTIONS}
    res = analyse_video(str(mp4), action_overrides=act_over or None, **kw)
    auto = {sw["index"]: sw["classification"]["rule_action"] for sw in res["swings"]}
    declared = meta.get("action") if meta.get("action") in ACTIONS - {"unknown"} else None
    if declared:
        # single-action upload: the declared action labels every swing recognised as a stroke-like movement
        # (swings judged non-stroke stay unknown so they never enter the reference library); per-swing
        # user overrides still win.
        extra = {sw["index"]: declared for sw in res["swings"]
                 if not sw["classification"]["scores"].get("non_stroke") and sw["index"] not in act_over}
        if extra:
            res = analyse_video(str(mp4), action_overrides={**extra, **act_over}, **kw)
    for sw in res["swings"]:
        sw["auto_action"] = auto.get(sw["index"])
    prog("match", 0)
    np.savez_compressed(dd / "swings.npz", **{f"feat_{sw['index']}": sw["feat"] for sw in res["swings"]})
    if v["kind"] == "reference":
        kept = set()
        for sw in res["swings"]:
            if sw["action"] == "unknown" or sw["quality"]["level"] == "bad" or not sw.get("phases"):
                continue
            if any("不完整" in wmsg for wmsg in sw["phases"].get("warnings", [])):
                continue  # truncated swings are not used as references
            # 'video_label': labelled through the video's declared action / per-swing overrides (refreshed on every
            # re-analysis); 'user' is reserved for edits of an individual reference entry, which are preserved
            src = "video_label" if (meta.get("action") or meta.get("action_overrides")) else "auto"
            db.upsert_ref(vid, sw["index"], sw["action"], sw["view_key"], sw["hand"], src)
            kept.add(sw["index"])
        db.prune_refs(vid, kept)
    else:
        for sw in res["swings"]:
            if sw["action"] == "unknown" or not sw.get("phases"):
                sw["match"] = {"status": "action_unknown", "message": "未识别为目标动作，不做匹配", "candidates": []}
                sw["feedback"] = build_feedback(sw["action"], sw["match"], sw["view_key_label"])
                continue
            q = swing_record(sw, None)
            m = match(q, [r for r in refs if r.get("tier") != "pro"] or refs)
            sw["match"] = m
            sw["feedback"] = build_feedback(sw["action"], m, sw["view_key_label"])
            sw["pro"] = pro_compare(q, trusted_pros(refs))
        targets = [s for s in res["swings"] if s["action"] != "unknown"]
        res["primary_swing"] = max(targets, key=lambda s: s["window"]["peak_speed"])["index"] if targets else None
    res["n_refs_available"] = len(refs)
    db.update_video(vid, analysis=strip_arrays(res), status="done", stage="done", progress=1.0,
                    model_version=pose.model_id, analysis_version=config.ANALYSIS_VERSION)


def trusted_pros(refs: list[dict]) -> list[dict]:
    """Professional swings usable as comparison references.

    Pro labels come from the rule classifier (+ spot checks), so weak, quality-flagged swings - typically
    half-swings, shadow swings or follow-throughs that the rules mislabelled - are left out: a swing whose peak
    wrist speed is below 70% of its action's pro median AND whose quality is not 'good'."""
    pros = [r for r in refs if r.get("tier") == "pro"]
    med = {}
    for a in {r["action"] for r in pros}:
        med[a] = float(np.median([r["window"].get("peak_speed", 0) for r in pros if r["action"] == a]))
    return [r for r in pros
            if not (r["window"].get("peak_speed", 0) < 0.7 * med[r["action"]] and (r.get("quality") or {}).get("level") != "good")]


def load_refs(exclude_video: str | None = None) -> list[dict]:
    """Enabled, non-rejected reference swings with their feature arrays."""
    out = []
    cache: dict = {}
    for r in db.list_refs(enabled_only=True):
        if r["video_id"] == exclude_video:
            continue
        if r["video_id"] not in cache:
            v = db.get_video(r["video_id"])
            p = derived_dir(r["video_id"]) / "swings.npz"
            if not v or not v.get("analysis") or not p.exists():
                cache[r["video_id"]] = None
                continue
            cache[r["video_id"]] = (v, np.load(p))
        entry = cache[r["video_id"]]
        if entry is None:
            continue
        v, z = entry
        sws = [s for s in v["analysis"]["swings"] if s["index"] == r["swing_index"]]
        if not sws or f"feat_{r['swing_index']}" not in z:
            continue
        sw = dict(sws[0])
        sw["feat"] = z[f"feat_{r['swing_index']}"]
        sw["action"] = r["action"]
        if not sw.get("phases") or not sw.get("metrics"):
            continue
        out.append(swing_record(sw, r["id"], {"review_status": r["review_status"], "player": r["meta"].get("player"),
                                              "style": r.get("style"), "video_id": r["video_id"],
                                              "swing_index": r["swing_index"], "time_known": v["analysis"].get("time_known", True),
                                              "tier": v["meta"].get("tier", "demo")}))
    return out


def get_ref_detail(rid: int) -> dict | None:
    r = db.get_ref(rid)
    if not r:
        return None
    v = db.get_video(r["video_id"])
    sw = next((s for s in (v.get("analysis") or {}).get("swings", []) if s["index"] == r["swing_index"]), None)
    return {"ref": r, "video": {k: v[k] for k in ("id", "filename", "meta", "info")}, "swing": sw,
            "time_scale": (v.get("analysis") or {}).get("time_scale", 1.0)}


def _event_video_time(an: dict, sw: dict, cp_key: str | None, event: str) -> float | None:
    """Video timestamp (pose clock) of a checkpoint measurement, or of a phase event as fallback."""
    tv = None
    m = (sw.get("technique") or {}).get(cp_key) if cp_key else None
    if m and "tv" in m:
        tv = m["tv"]
    elif sw.get("phases") and event in sw["phases"]["events"]:
        tv = sw["phases"]["events"][event].get("tv")
    return None if tv is None else tv + an.get("video_t0", 0.0)


def _panel(vid: str, sw: dict, an: dict, cp_key: str | None, event: str, flip: bool):
    from .render import keyframe_panel
    from .technique import highlight_joints
    from .video import read_frame_at
    v = db.get_video(vid, full=False)
    pose = load_pose(vid)
    tv = _event_video_time(an, sw, cp_key, event)
    if pose is None or tv is None:
        return None
    fi = int(np.argmin(np.abs(pose.t - tv)))
    img = read_frame_at(v["path"], pose.video_frame(fi))
    if img is None:
        return None
    hand = sw.get("hand") or an["hand"]["hand"]
    measure = "horizontal" if cp_key in ("gs_contact_front", "gs_backswing_depth", "gs_nonhit_arm") else None
    return keyframe_panel(img, pose.kpts[fi], pose.scores[fi], highlight_joints(cp_key, hand) if cp_key else [],
                          flip=flip, measure=measure, hand=hand)


def compare_image(vid: str, k: int, cp_key: str | None = None, event: str | None = None,
                  ref_id: int | None = None) -> bytes | None:
    """Side-by-side JPEG: the user's frame (left) and the professional's frame (right) at the same checkpoint
    / phase event, both cropped to the player, the pro mirrored if needed so both face the same way."""
    import cv2
    from .render import side_by_side
    from .technique import BY_KEY
    v = db.get_video(vid)
    if not v or not v.get("analysis"):
        return None
    an = v["analysis"]
    sw = next((s for s in an["swings"] if s["index"] == k), None)
    if sw is None:
        return None
    event = event or (BY_KEY[cp_key].event if cp_key in BY_KEY else "contact")
    left = _panel(vid, sw, an, cp_key, event, False)
    if left is None:
        return None
    if ref_id is None:
        ref_id = ((sw.get("pro") or {}).get("nearest") or {}).get("ref_id")
    right = None
    if ref_id is not None:
        r = db.get_ref(int(ref_id))
        rv = db.get_video(r["video_id"]) if r else None
        if rv and rv.get("analysis"):
            ran = rv["analysis"]
            rsw = next((s for s in ran["swings"] if s["index"] == r["swing_index"]), None)
            if rsw:
                flip = (np.sign(rsw.get("x_sign", 1)) != np.sign(sw.get("x_sign", 1)))
                right = _panel(r["video_id"], rsw, ran, cp_key, event, bool(flip))
    img = side_by_side(left, right) if right is not None else left
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def delete_video(vid: str):
    v = db.get_video(vid, full=False)
    if not v:
        return False
    # pose cache entry for this file (content-addressed)
    pose_npz = derived_dir(vid) / "pose.npz"
    if v.get("path") and Path(v["path"]).exists():
        try:
            import hashlib
            key = hashlib.sha1((file_sha1(v["path"]) + config.POSE_MODEL_ID + str(config.DET_EVERY)).encode()).hexdigest()[:20]
            (config.DERIVED_DIR / "pose_cache" / f"{key}.npz").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(video_dir(vid), ignore_errors=True)
    shutil.rmtree(derived_dir(vid), ignore_errors=True)
    _ = pose_npz
    db.delete_video(vid)
    return True


def load_pose(vid: str) -> PoseSeq | None:
    p = derived_dir(vid) / "pose.npz"
    return PoseSeq.load(p) if p.exists() else None


def resume_pending():
    """Re-queue jobs interrupted by a restart."""
    for v in db.list_videos():
        if v["status"] in ("queued", "running"):
            enqueue(v["id"])
