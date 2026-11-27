"""FastAPI app: upload / library / analysis / correction / deletion endpoints + static web UI."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, service
from .classify import ACTION_LABELS
from .video import VideoError
from .view import VIEW_KEY_LABELS, VIEW_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="网球视频动作训练助手", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"


@app.on_event("startup")
def _startup():
    db.init()
    service.resume_pending()


# ---------------------------------------------------------------------------------------------- meta
@app.get("/api/meta")
def meta():
    return {"actions": ACTION_LABELS, "views": VIEW_LABELS, "view_keys": VIEW_KEY_LABELS,
            "review_status": {"demo": "参考示范（未经教练审核）", "coach_reviewed": "教练审核", "rejected": "已停用"},
            "pose_model": config.POSE_MODEL_ID, "analysis_version": config.ANALYSIS_VERSION}


# ---------------------------------------------------------------------------------------------- videos
@app.post("/api/videos")
async def upload(file: UploadFile = File(...), kind: str = Form(...), meta: str = Form("{}")):
    try:
        m = json.loads(meta or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "meta must be JSON")
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    tmp = Path(tempfile.mkdtemp(dir=config.STORE_DIR)) / ("upload" + suffix)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        vid = service.ingest(tmp, file.filename or tmp.name, kind, m)
    except (VideoError, ValueError) as e:
        raise HTTPException(400, str(e))
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return {"id": vid}


@app.get("/api/videos")
def list_videos(kind: str | None = None):
    return db.list_videos(kind)


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    v = db.get_video(vid)
    if not v:
        raise HTTPException(404, "not found")
    return v


@app.get("/api/videos/{vid}/status")
def status(vid: str):
    v = db.get_video(vid, full=False)
    if not v:
        raise HTTPException(404, "not found")
    return {k: v[k] for k in ("id", "status", "stage", "progress", "error")}


class VideoPatch(BaseModel):
    hand: str | None = None                 # right | left | "" (auto)
    view_override: dict | None = None       # {"view": ..., "facing": ...} or {} to clear
    time_scale: float | str | None = None   # 1.0 real time, 0.25 = 4x slow motion, "unknown"
    action: str | None = None               # declared single action for the whole clip ("" to clear)
    action_overrides: dict | None = None    # {swing_index: action}
    player: str | None = None
    source: str | None = None
    license: str | None = None
    license_commercial: str | None = None   # yes | no | unknown
    redistribution: str | None = None       # yes | no | unknown
    camera: str | None = None
    notes: str | None = None
    review_status: str | None = None
    reviewer: str | None = None


ANALYSIS_FIELDS = {"hand", "view_override", "time_scale", "action", "action_overrides"}


@app.patch("/api/videos/{vid}")
def patch_video(vid: str, p: VideoPatch):
    v = db.get_video(vid, full=False)
    if not v:
        raise HTTPException(404, "not found")
    meta = dict(v["meta"])
    changes = p.model_dump(exclude_unset=True)
    if "hand" in changes and changes["hand"] not in (None, "", "right", "left"):
        raise HTTPException(400, "hand must be right/left/empty")
    if changes.get("review_status") == "coach_reviewed" and not (changes.get("reviewer") or meta.get("reviewer")):
        raise HTTPException(400, "coach_reviewed requires a reviewer name")
    for k, val in changes.items():
        if meta.get(k) != val:
            db.log_correction("video", vid, k, meta.get(k), val)
        meta[k] = val
    db.update_video(vid, meta=meta)
    if ANALYSIS_FIELDS & set(changes):
        service.enqueue(vid)
    return {"ok": True, "requeued": bool(ANALYSIS_FIELDS & set(changes))}


@app.post("/api/videos/{vid}/reanalyse")
def reanalyse(vid: str):
    if not db.get_video(vid, full=False):
        raise HTTPException(404, "not found")
    service.enqueue(vid)
    return {"ok": True}


@app.delete("/api/videos/{vid}")
def delete_video(vid: str):
    if not service.delete_video(vid):
        raise HTTPException(404, "not found")
    return {"ok": True, "deleted": vid, "note": "原视频、转码文件、姿态缓存、分析结果和由它产生的参考条目已删除"}


@app.get("/api/videos/{vid}/swings/{k}/compare.jpg")
def compare_jpg(vid: str, k: int, cp: str | None = None, event: str | None = None, ref: int | None = None):
    from fastapi import Response
    data = service.compare_image(vid, k, cp, event, ref)
    if data is None:
        raise HTTPException(404, "comparison frame not available")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=60"})


@app.get("/api/videos/{vid}/pose")
def pose(vid: str):
    seq = service.load_pose(vid)
    if seq is None:
        raise HTTPException(404, "pose not available yet")
    return JSONResponse(seq.to_json())


@app.get("/api/videos/{vid}/file")
def video_file(vid: str):
    v = db.get_video(vid, full=False)
    if not v or not v.get("path") or not Path(v["path"]).exists():
        raise HTTPException(404, "video not available")
    return FileResponse(v["path"], media_type="video/mp4")


@app.get("/api/videos/{vid}/corrections")
def corrections(vid: str):
    return db.list_corrections(vid)


# ---------------------------------------------------------------------------------------------- references
@app.get("/api/references")
def list_refs():
    return db.list_refs()


@app.get("/api/references/{rid}")
def get_ref(rid: int):
    d = service.get_ref_detail(rid)
    if not d:
        raise HTTPException(404, "not found")
    return d


class RefPatch(BaseModel):
    action: str | None = None
    review_status: str | None = None
    reviewer: str | None = None
    style: str | None = None
    notes: str | None = None
    enabled: bool | None = None


@app.patch("/api/references/{rid}")
def patch_ref(rid: int, p: RefPatch):
    r = db.get_ref(rid)
    if not r:
        raise HTTPException(404, "not found")
    ch = p.model_dump(exclude_unset=True)
    if "action" in ch and ch["action"] not in ACTION_LABELS:
        raise HTTPException(400, "invalid action")
    if ch.get("review_status") == "coach_reviewed" and not ch.get("reviewer"):
        raise HTTPException(400, "coach_reviewed requires a reviewer name")
    if "action" in ch:
        ch["label_source"] = "user"
    if "enabled" in ch:
        ch["enabled"] = int(bool(ch["enabled"]))
    reviewer = ch.pop("reviewer", None)
    for k, val in ch.items():
        db.log_correction("reference", rid, k, r.get(k), val)
    if reviewer:
        ch["notes"] = ((r.get("notes") or "") + f" [审核人: {reviewer}]").strip()
    db.update_ref(rid, **ch)
    return {"ok": True}


@app.delete("/api/references/{rid}")
def delete_ref(rid: int):
    if not db.get_ref(rid):
        raise HTTPException(404, "not found")
    db.delete_ref(rid)
    return {"ok": True, "note": "仅从参考库移除，源视频保留；如需删除视频请在视频列表中删除"}


# ---------------------------------------------------------------------------------------------- web
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
