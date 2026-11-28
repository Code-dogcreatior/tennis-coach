"""Ground-truth annotation of real videos (evaluation set): which strokes happen and their contact frames.

Videos are read from ANNOT_ROOTS (never uploaded anywhere); labels are JSON files under data/labels/annotations/
(committed to git, small): one file per video and annotator (<video>__<annotator>.json), so two people can label
the same video without git conflicts and their agreement can be measured. Frame numbers are the ground truth; seconds are kept for
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
IMPORT_ROOT = RAW / "annotate"                       # own videos: one folder per video (imported via the page)
ANNOT_ROOTS = [RAW / "eval_real", IMPORT_ROOT]
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


def _label_base(key: str) -> str:
    return key.replace("/", "__").rsplit(".", 1)[0]


def label_path(key: str, annotator: str) -> Path:
    """One file per video *and annotator*: two people never edit the same file, so git never has to merge them."""
    return LABEL_DIR / f"{_label_base(key)}__{annotator}.json"


def legacy_label_path(key: str) -> Path:
    """Before 2026-11-28 all annotators of a video shared one file (still read, rewritten on the next save)."""
    return LABEL_DIR / f"{_label_base(key)}.json"


def label_files(key: str) -> list[Path]:
    """All label files of a video (per-annotator files; the legacy shared file, if any, first)."""
    out = [legacy_label_path(key)] if legacy_label_path(key).exists() else []
    for f in sorted(LABEL_DIR.glob(f"{_label_base(key)}__*.json")):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("video") == key:   # not another video with this prefix
                out.append(f)
        except Exception:  # noqa: BLE001
            pass
    return out


def labelled_videos() -> list[str]:
    keys = set()
    for f in LABEL_DIR.glob("*.json") if LABEL_DIR.exists() else []:
        try:
            v = json.loads(f.read_text(encoding="utf-8")).get("video")
        except Exception:  # noqa: BLE001
            continue
        if v:
            keys.add(v)
    return sorted(keys)


def _source(p: Path) -> dict:
    """Source URL / licence from a yt-dlp info.json next to the video, if any."""
    for info in p.parent.glob(p.stem + "*.info.json"):
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
    from .video import _ffbin
    r = subprocess.run([_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate,start_time",
                        "-of", "json", str(p)], capture_output=True, text=True)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    t0 = float(st.get("start_time") or 0)
    cfr = st.get("r_frame_rate") == st.get("avg_frame_rate")
    # frame n is addressed as [n/fps, (n+1)/fps): only exact for constant-rate video starting at 0
    ok = abs(t0) < 1e-3 and cfr
    return {"fps": i.fps, "width": i.width, "height": i.height, "duration": i.duration, "n_frames": i.n_frames,
            "start_time": t0, "cfr": cfr, "frame_exact": ok}


def _link(p: Path) -> dict:
    """<video>.link.json: which app video (analysis) belongs to this annotation copy, and the time offset
    (copy time = app-video presentation time - offset)."""
    lp = p.with_name(p.stem + ".link.json")
    try:
        return json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


ANNOT_META = ("video", "annotator", "source", "probe")


def load_labels(key: str) -> dict:
    """All annotators of a video merged: {"video", "source", "probe", "annotations": {annotator: block}}."""
    lab = {"video": key, "annotations": {}}
    for f in label_files(key):
        j = json.loads(f.read_text(encoding="utf-8"))
        for k in ("source", "probe"):
            if j.get(k):
                lab[k] = j[k]
        if "annotations" in j:                       # legacy shared file
            lab["annotations"].update(j["annotations"])
        elif j.get("annotator"):
            lab["annotations"][j["annotator"]] = {k: v for k, v in j.items() if k not in ANNOT_META}
    return lab


def write_annotation(key: str, annotator: str, block: dict, source: dict, probe: dict) -> Path:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    lp = label_path(key, annotator)
    lp.write_text(json.dumps({"video": key, "annotator": annotator, "source": source, "probe": probe, **block},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    return lp


def split_legacy(key: str):
    """Move the blocks of a legacy shared file into per-annotator files (existing per-annotator files win)."""
    lp = legacy_label_path(key)
    if not lp.exists():
        return
    j = json.loads(lp.read_text(encoding="utf-8"))
    for who, block in (j.get("annotations") or {}).items():
        if not label_path(key, who).exists():
            write_annotation(key, who, block, j.get("source") or {}, j.get("probe") or {})
    lp.unlink()


@router.get("/videos")
def list_videos():
    out = []
    for root in ANNOT_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXT and p.is_file():
                lab = load_labels(_key(p))
                if p.name.endswith(".tmp.mp4") or "_src" in p.parts:
                    continue
                lk = _link(p)
                out.append({"key": _key(p), "name": p.name, "root": root.name, "linked_video": lk.get("video_id"),
                            "deletable": root == IMPORT_ROOT, "annotators": {
                    a: {"n_events": len(b.get("events", [])), "status": b.get("status", "in_progress")}
                    for a, b in lab.get("annotations", {}).items()}})
    return out


@router.get("/video")
def video_info(key: str):
    p = _path(key)
    lab = load_labels(key)
    return {"key": key, "probe": _probe(p), "source": _source(p), "labels": lab, "strokes": STROKES,
            "linked_video": _link(p).get("video_id"), "deletable": p.is_relative_to(IMPORT_ROOT.resolve()),
            # evaluation clips: hints off by default so the labels stay independent of the system
            "suggest_default": not p.is_relative_to((RAW / "eval_real").resolve())}


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
    split_legacy(key)
    block = {**a.model_dump(exclude={"events", "annotator"}), "events": events, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_annotation(key, a.annotator, block, _source(p), _probe(p))
    return {"ok": True, "n_events": len(events)}
