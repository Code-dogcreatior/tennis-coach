"""Bringing videos into the annotation tool, and system hints for labelling.

Videos stay on this machine: uploads are converted locally; YouTube links are downloaded with yt-dlp one at a
time on explicit request (public videos only, no login / cookies). Each import gets its own folder
data/raw/annotate/<slug>/ holding <slug>.mp4 (+ <slug>.info.json for YouTube, + <slug>.link.json when the
video is analysed in the app).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import annotprep, config
from .annotate import IMPORT_ROOT, VIDEO_EXT, _key, _link, _path, _probe, label_path, router

log = logging.getLogger(__name__)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
SLUG_RE = re.compile(r"[^A-Za-z0-9_\-]+")
UPLOAD_EXT = VIDEO_EXT | {".avi", ".mkv"}


def _slug(s: str, n: int = 40) -> str:
    return (SLUG_RE.sub("_", s).strip("_") or "video")[:n]


def _new_job(kind: str, **kw) -> str:
    jid = uuid.uuid4().hex[:10]
    with _jobs_lock:
        _jobs[jid] = {"id": jid, "kind": kind, "state": "running", "stage": "排队", "key": None, "error": None,
                      "video_id": None, "created": time.time(), **kw}
    return jid


def _upd(jid: str, **kw):
    with _jobs_lock:
        _jobs[jid].update(kw)


def _job(jid: str) -> dict:
    with _jobs_lock:
        return dict(_jobs[jid])


def _run(jid: str, fn):
    def go():
        try:
            fn()
            _upd(jid, state="done", stage="完成")
        except Exception as e:  # noqa: BLE001
            log.exception("annotation import failed")
            _upd(jid, state="failed", stage="失败", error=str(e)[:400])
    threading.Thread(target=go, daemon=True, name=f"annot-import-{jid}").start()


def _unique_dir(slug: str) -> Path:
    d, k = IMPORT_ROOT / slug, 2
    while d.exists():
        d, k = IMPORT_ROOT / f"{slug}_{k}", k + 1
    d.mkdir(parents=True)
    return d


def _cleanup_failed(out_dir: Path):
    if not any(q for q in out_dir.glob("*.mp4") if not q.name.endswith(".tmp.mp4")):
        shutil.rmtree(out_dir, ignore_errors=True)


def _finish_import(jid: str, src: Path, out_dir: Path, analyse: bool, info: Path | None = None):
    """CFR conversion into out_dir/<name>.mp4; optionally queue an app analysis of that very copy (its times then
    coincide with the copy: offset 0) so the timeline can show the system's swings."""
    from . import service
    name = out_dir.name
    _upd(jid, stage="转换为恒定帧率")
    out = annotprep.prep(src, out_dir / f"{name}.mp4")
    if info is not None:
        shutil.move(str(info), out_dir / f"{name}.info.json")
    key = _key(out)
    _upd(jid, key=key)
    if analyse:
        _upd(jid, stage="提交系统分析")
        tmp = Path(tempfile.mkdtemp(dir=config.STORE_DIR)) / "upload.mp4"
        shutil.copyfile(out, tmp)
        try:
            vid = service.ingest(tmp, f"{name}.mp4", "query", {"annotation_key": key})
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        out.with_name(out.stem + ".link.json").write_text(json.dumps({"video_id": vid, "offset": 0.0, "origin": "annotate"}), encoding="utf-8")
        _upd(jid, video_id=vid)


@router.post("/import/upload")
def import_upload(file: UploadFile = File(...), analyse: bool = Form(True)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in UPLOAD_EXT:
        raise HTTPException(400, f"不支持的文件类型 {suffix or '(无扩展名)'}")
    out_dir = _unique_dir("up_" + _slug(Path(file.filename or "video").stem))
    src_dir = out_dir / "_src"
    src_dir.mkdir()
    src = src_dir / ("upload" + suffix)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jid = _new_job("upload", name=file.filename)

    def work():
        from .video import validate
        try:
            _upd(jid, stage="检查视频")
            validate(src)
            _finish_import(jid, src, out_dir, analyse)
        except Exception:
            _cleanup_failed(out_dir)
            raise
        finally:
            shutil.rmtree(src_dir, ignore_errors=True)
    _run(jid, work)
    return _job(jid)


class YoutubeImport(BaseModel):
    url: str
    start: float | None = None
    end: float | None = None
    analyse: bool = True


@router.post("/import/youtube")
def import_youtube(r: YoutubeImport):
    yid = annotprep.youtube_id(r.url)
    if not yid:
        raise HTTPException(400, "只支持单个 YouTube 视频链接（youtube.com/watch?v=…、youtu.be/…、shorts/…）")
    if (r.start is not None and r.start < 0) or (r.end is not None and r.end <= (r.start or 0)):
        raise HTTPException(400, "时间段不合法：需要 0 ≤ 开始 < 结束")
    sec = "" if r.start is None and r.end is None else f"_{int(r.start or 0)}-{int(r.end) if r.end is not None else 'end'}"
    out_dir = _unique_dir(f"yt_{yid}{sec}")
    jid = _new_job("youtube", name=yid)

    def work():
        src_dir = out_dir / "_src"
        try:
            _upd(jid, stage="从 YouTube 下载")
            src, info = annotprep.youtube_download(yid, src_dir, r.start, r.end)
            _finish_import(jid, src, out_dir, r.analyse, info)
        except Exception:
            _cleanup_failed(out_dir)
            raise
        finally:
            shutil.rmtree(src_dir, ignore_errors=True)
    _run(jid, work)
    return _job(jid)


@router.post("/from_video/{vid}")
def from_app_video(vid: str):
    """Annotation copy (CFR, t0 = 0) of a video already analysed in the app; its analysis gives the hints."""
    from . import db
    v = db.get_video(vid, full=False)
    if not v or not v.get("path") or not Path(v["path"]).exists():
        raise HTTPException(404, "该视频没有可用的视频文件（参考包条目只有关键点）")
    if IMPORT_ROOT.exists():
        for p in IMPORT_ROOT.glob("*/*.mp4"):
            if _link(p).get("video_id") == vid:
                return {"id": None, "state": "done", "stage": "完成", "key": _key(p), "video_id": vid, "error": None}
    out_dir = _unique_dir(f"app_{_slug(Path(v.get('filename') or vid).stem, 30)}_{vid[:6]}")
    jid = _new_job("app", video_id=vid, name=v.get("filename"))

    def work():
        src = Path(v["path"])
        try:
            _upd(jid, stage="转换为恒定帧率")
            out = annotprep.prep(src, out_dir / f"{out_dir.name}.mp4")
        except Exception:
            _cleanup_failed(out_dir)
            raise
        out.with_name(out.stem + ".link.json").write_text(
            json.dumps({"video_id": vid, "offset": annotprep.start_time(src), "origin": "app"}), encoding="utf-8")
        _upd(jid, key=_key(out))
    _run(jid, work)
    return _job(jid)


@router.get("/jobs/{jid}")
def job(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "job not found")
    return _job(jid)


def suggestion_swings(an: dict, fps: float, offset: float) -> list[dict]:
    """System swings in the annotation copy's time / frames (copy time = presentation time - offset)."""
    from .classify import ACTION_LABELS
    t0 = float(an.get("video_t0", 0.0))
    out = []
    for sw in an.get("swings", []):
        ev = (sw.get("phases") or {}).get("events", {}).get("contact")
        tv = ev["tv"] if ev else sw["window"]["tv_peak"]
        t = tv + t0 - offset
        sc = sw["classification"].get("scores", {})
        out.append({"index": sw["index"], "t": round(t, 4), "frame": max(0, int(t * fps + 1e-6)), "action": sw["action"],
                    "label": ACTION_LABELS.get(sw["action"], sw["action"]),
                    "confidence": round(float(sw["classification"]["confidence"]), 2),
                    "non_stroke": bool(sc.get("non_stroke")), "contact_estimated": bool(ev)})
    return out


@router.get("/suggestions")
def suggestions(key: str):
    """Display-only hints: never written into the labels, so the labels stay an independent check of the system."""
    from . import db
    p = _path(key)
    lk = _link(p)
    if not lk.get("video_id"):
        return {"status": "none", "swings": []}
    v = db.get_video(lk["video_id"])
    if not v:
        return {"status": "missing", "swings": []}
    if v["status"] != "done" or not v.get("analysis"):
        return {"status": v["status"], "progress": v.get("progress"), "stage": v.get("stage"),
                "video_id": lk["video_id"], "swings": []}
    return {"status": "done", "video_id": lk["video_id"],
            "swings": suggestion_swings(v["analysis"], _probe(p)["fps"], float(lk.get("offset", 0.0)))}


@router.delete("/video")
def delete_video(key: str, labels: bool = False, app_video: bool = False):
    """Delete an imported video with everything derived from it (its folder). The labels (committed data) and the
    linked app analysis only on request. Evaluation clips (eval_real) are not deleted here."""
    from . import service
    p = _path(key)
    if not p.is_relative_to(IMPORT_ROOT.resolve()):
        raise HTTPException(400, "只能删除自己导入的视频（data/raw/annotate/ 下）")
    lk = _link(p)
    folder = p.parent
    own_folder = folder.parent == IMPORT_ROOT.resolve() and \
        [q for q in folder.iterdir() if q.suffix.lower() in VIDEO_EXT and not q.name.endswith(".tmp.mp4")] == [p]
    if own_folder:
        shutil.rmtree(folder, ignore_errors=True)
    else:
        for q in (p, p.with_name(p.stem + ".link.json"), p.with_name(p.stem + ".info.json")):
            q.unlink(missing_ok=True)
    removed = {"video": True, "labels": False, "app_video": False}
    if labels and label_path(key).exists():
        label_path(key).unlink()
        removed["labels"] = True
    if app_video and lk.get("video_id"):
        removed["app_video"] = bool(service.delete_video(lk["video_id"]))
    return removed


