"""Ground-truth annotation of real videos (evaluation set): which strokes happen and their contact frames.

Videos are read from ANNOT_ROOTS (never uploaded anywhere); labels are JSON files under data/labels/annotations/
(committed to git, small). One file per video, one block per annotator, so two people can label the same
video and their agreement can be measured. Frame numbers are the ground truth; seconds are kept for
convenience.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config

RAW = config.DATA_DIR / "raw"
ANNOT_ROOTS = [RAW / "eval_real", RAW / "annotate"]   # put your own videos in data/raw/annotate/
LABEL_DIR = config.ROOT / "data" / "labels" / "annotations"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
STROKES = ["forehand", "backhand_1h", "backhand_2h", "serve", "smash", "volley", "other"]
ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")

router = APIRouter(prefix="/api/annot")


def _key(p: Path) -> str:
    return p.relative_to(RAW).as_posix()


def _path(key: str) -> Path:
    p = (RAW / key).resolve()
    if not any(p.is_relative_to(r.resolve()) for r in ANNOT_ROOTS) or p.suffix.lower() not in VIDEO_EXT or not p.is_file():
        raise HTTPException(404, "video not found")
    return p


def label_path(key: str) -> Path:
    return LABEL_DIR / (key.replace("/", "__").rsplit(".", 1)[0] + ".json")


def _source(p: Path) -> dict:
    """Source URL / licence from a yt-dlp info.json next to the video, if any."""
    for info in p.parent.glob("*.info.json"):
        try:
            j = json.loads(info.read_text(encoding="utf-8"))
            return {"url": j.get("webpage_url"), "license": j.get("license") or "standard YouTube licence",
                    "uploader": j.get("uploader"), "title": j.get("title")}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _probe(p: Path) -> dict:
    import subprocess
    from .video import probe
    i = probe(p)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate,start_time",
                        "-of", "json", str(p)], capture_output=True, text=True)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    t0 = float(st.get("start_time") or 0)
    cfr = st.get("r_frame_rate") == st.get("avg_frame_rate")
    # frame n is addressed as [n/fps, (n+1)/fps): only exact for constant-rate video starting at 0
    ok = abs(t0) < 1e-3 and cfr
    return {"fps": i.fps, "width": i.width, "height": i.height, "duration": i.duration, "n_frames": i.n_frames,
            "start_time": t0, "cfr": cfr, "frame_exact": ok}


def load_labels(key: str) -> dict:
    lp = label_path(key)
    return json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {"video": key, "annotations": {}}


@router.get("/videos")
def list_videos():
    out = []
    for root in ANNOT_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXT and p.is_file():
                lab = load_labels(_key(p))
                out.append({"key": _key(p), "name": p.name, "annotators": {
                    a: {"n_events": len(b.get("events", [])), "status": b.get("status", "in_progress")}
                    for a, b in lab.get("annotations", {}).items()}})
    return out


@router.get("/video")
def video_info(key: str):
    p = _path(key)
    lab = load_labels(key)
    return {"key": key, "probe": _probe(p), "source": _source(p), "labels": lab, "strokes": STROKES}


@router.get("/file")
def video_file(key: str):
    return FileResponse(_path(key), media_type="video/mp4")


class Annotation(BaseModel):
    annotator: str
    status: str = "in_progress"          # in_progress | done
    view: str | None = None
    hand: str | None = None
    level: str | None = None
    labelled_range: list[float] | None = None   # [t0, t1] seconds actually checked (None = whole video)
    notes: str = ""
    events: list[dict] = []


@router.put("/labels")
def save_labels(key: str, a: Annotation):
    p = _path(key)
    if not ANNOTATOR_RE.match(a.annotator):
        raise HTTPException(400, "annotator: letters, digits, _ or - only")
    events = []
    for e in a.events:
        if e.get("stroke") not in STROKES or not isinstance(e.get("frame"), int) or e["frame"] < 0:
            raise HTTPException(400, f"bad event {e}")
        events.append({"frame": e["frame"], "t": round(float(e.get("t", 0.0)), 4), "stroke": e["stroke"],
                       "unsure": bool(e.get("unsure", False)), "note": str(e.get("note", ""))[:200]})
    events.sort(key=lambda e: e["frame"])
    lab = load_labels(key)
    lab["video"] = key
    lab["source"] = _source(p)
    lab["probe"] = _probe(p)
    lab.setdefault("annotations", {})[a.annotator] = {
        **a.model_dump(exclude={"events", "annotator"}), "events": events, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    label_path(key).write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "n_events": len(events)}
