"""Video probing, validation, rotation-aware transcoding and timestamped frame iteration."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp"}
MAX_DURATION_S = 600
MAX_LONG_SIDE = 1920


class VideoError(Exception):
    pass


def _ffbin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise VideoError(f"{name} not found on PATH")
    return p


@dataclass
class VideoInfo:
    width: int            # display width (after rotation)
    height: int           # display height (after rotation)
    rotation: int         # rotation metadata in degrees (0/90/180/270)
    fps: float            # average frame rate
    duration: float       # seconds
    n_frames: int | None
    codec: str
    vfr: bool             # r_frame_rate != avg_frame_rate (likely variable frame rate)

    def to_dict(self):
        return asdict(self)


def _rate(s: str) -> float:
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def probe(path: str | Path) -> VideoInfo:
    cmd = [_ffbin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise VideoError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    j = json.loads(r.stdout)
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise VideoError("no video stream found")
    s = vs[0]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if "rotate" in (s.get("tags") or {}):
        rot = int(s["tags"]["rotate"])
    rot = rot % 360
    w, h = int(s["width"]), int(s["height"])
    if rot in (90, 270):
        w, h = h, w
    avg = _rate(s.get("avg_frame_rate", "0/0"))
    rfr = _rate(s.get("r_frame_rate", "0/0"))
    dur = float(s.get("duration") or j.get("format", {}).get("duration") or 0)
    nf = int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None
    return VideoInfo(width=w, height=h, rotation=rot, fps=avg or rfr, duration=dur, n_frames=nf,
                     codec=s.get("codec_name", "?"), vfr=bool(avg and rfr and abs(avg - rfr) / rfr > 0.02))


def validate(path: str | Path) -> VideoInfo:
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_EXT:
        raise VideoError(f"unsupported file type {p.suffix}; allowed: {sorted(ALLOWED_EXT)}")
    info = probe(p)
    if info.duration <= 0.3:
        raise VideoError("video too short (<0.3 s)")
    if info.duration > MAX_DURATION_S:
        raise VideoError(f"video too long ({info.duration:.0f}s > {MAX_DURATION_S}s); please trim to the swing")
    if info.fps < 10:
        raise VideoError(f"frame rate too low ({info.fps:.1f} fps)")
    return info


def transcode(src: str | Path, dst: str | Path) -> VideoInfo:
    """Transcode to browser-friendly H.264 MP4.

    ffmpeg applies the rotation metadata (autorotate) so pixels end up upright and the
    output carries no rotation flag. Frame timestamps are passed through unchanged
    (no constant-frame-rate resampling) so real capture timing is preserved.
    """
    info = probe(src)
    long_side = max(info.width, info.height)
    vf = []
    if long_side > MAX_LONG_SIDE:
        if info.width >= info.height:
            vf.append(f"scale={MAX_LONG_SIDE}:-2")
        else:
            vf.append(f"scale=-2:{MAX_LONG_SIDE}")
    cmd = [_ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
           "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd.append(str(dst))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise VideoError(f"transcode failed: {r.stderr.strip()[:300]}")
    return probe(dst)


def frame_timestamps(path: str | Path) -> np.ndarray:
    """Presentation timestamps (seconds) of every video frame, read from the container."""
    cmd = [_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
           "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise VideoError(f"ffprobe timestamps failed: {r.stderr[:200]}")
    ts = []
    for line in r.stdout.splitlines():
        line = line.strip().strip(",")
        if line and line != "N/A":
            try:
                ts.append(float(line))
            except ValueError:
                pass
    t = np.asarray(ts, dtype=np.float64)
    if len(t):
        t = t - t[0]
    return t


def iter_frames(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


CUT_THR = 0.45   # 1 - histogram correlation above which consecutive frames belong to different shots


def shot_hist(frame: np.ndarray) -> np.ndarray:
    """Hue/saturation histogram of a downscaled frame (shot-change signature)."""
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def is_cut(prev_hist: np.ndarray | None, hist: np.ndarray, thr: float = CUT_THR) -> bool:
    return prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 1 - thr


def scene_cuts(path: str | Path, thr: float = CUT_THR) -> np.ndarray:
    """Boolean array over *all* video frames: True at frames that start a new shot (hard cut or strong dissolve).

    Uses the colour-histogram correlation between consecutive downscaled frames. Cheap compared
    to pose inference; used so that swings and speeds are never computed across an edit.
    """
    cuts, prev = [], None
    for _, f in iter_frames(path):
        h = shot_hist(f)
        cuts.append(is_cut(prev, h, thr))
        prev = h
    return np.asarray(cuts, bool)


def read_frame_at(path: str | Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    return f if ok else None
