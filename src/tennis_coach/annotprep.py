"""Convert videos into the format the annotation tool and the evaluation assume:
constant frame rate, first frame at t = 0, H.264 MP4, no audio.

Why: the browser addresses frames by time (frame n = [n/fps, (n+1)/fps)). Clips cut from YouTube often start at
one frame past zero, and phone videos are often variable-frame-rate; both would shift the labelled contact
frame against the frame the analysis sees. After this conversion "frame n" means the same image everywhere.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from .video import _ffbin

STD_RATES = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30), Fraction(50),
             Fraction(60000, 1001), Fraction(60), Fraction(120), Fraction(240))


def _frac(x) -> Fraction:
    try:
        return Fraction(x or "0/1")
    except (ZeroDivisionError, ValueError):   # "0/0" when ffprobe cannot tell
        return Fraction(0)


def native_fps(src: Path) -> Fraction:
    r = subprocess.run([_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(src)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    fps = _frac(s.get("avg_frame_rate")) or _frac(s.get("r_frame_rate")) or Fraction(30)
    # snap to the closest usual rate (29.97 -> 30000/1001, 60 -> 60, ...) so the output is exactly CFR
    best = min(STD_RATES, key=lambda x: abs(float(fps) - float(x)))
    return best if abs(float(fps) - float(best)) < 0.005 * float(best) else fps.limit_denominator(1001)


def start_time(src: Path) -> float:
    """First video presentation timestamp (s). The CFR copy starts at 0, i.e. copy time = source pts - this."""
    r = subprocess.run([_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
                        "-of", "json", str(src)], capture_output=True, text=True)
    try:
        return float((json.loads(r.stdout or "{}").get("streams") or [{}])[0].get("start_time") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def prep(src: Path, out: Path) -> Path:
    src, out = Path(src), Path(out)
    if out.resolve() == src.resolve():
        raise ValueError(f"refusing to overwrite the input in place: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = native_fps(src)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [_ffbin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
           "-vf", f"setpts=PTS-STARTPTS,fps={fps.numerator}/{fps.denominator}", "-fps_mode", "cfr",
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------------------------- YouTube
YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
YT_PATTERNS = [re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/watch\?(?:.*&)?v=([A-Za-z0-9_-]{11})"),
               re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/(?:shorts|live|embed)/([A-Za-z0-9_-]{11})"),
               re.compile(r"^https?://youtu\.be/([A-Za-z0-9_-]{11})")]


def youtube_id(url: str) -> str | None:
    """Only YouTube video links (or a bare 11-character id) are accepted - no playlists, channels or other sites."""
    url = (url or "").strip()
    if YT_ID.match(url):
        return url
    for p in YT_PATTERNS:
        m = p.match(url)
        if m:
            return m.group(1)
    return None


def youtube_download(yid: str, dst_dir: Path, start: float | None, end: float | None,
                     max_full_s: int = 900) -> tuple[Path, Path | None]:
    """Download one video (or one section of it) with yt-dlp. Returns (video file, info.json or None).
    Public videos only: no cookies / login are passed, so private or age-restricted videos simply fail."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist",
           "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/bv*[height<=1080]+ba/b",
           "--merge-output-format", "mp4", "--write-info-json", "-o", str(dst_dir / f"{yid}.%(ext)s")]
    if start is not None or end is not None:
        a = f"{float(start or 0):g}"
        b = f"{float(end):g}" if end is not None else "inf"
        cmd += ["--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts"]
    else:
        cmd += ["--match-filter", f"duration<={max_full_s}"]
    cmd.append(f"https://www.youtube.com/watch?v={yid}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    vids = [p for p in dst_dir.glob(f"{yid}.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm") and ".info" not in p.name]
    if r.returncode != 0 or not vids:
        msg = (r.stderr or r.stdout or "").strip().splitlines()
        hint = "" if (start is not None or end is not None) else f"（未指定时间段时只下载 {max_full_s // 60} 分钟以内的视频）"
        raise RuntimeError("YouTube 下载失败" + hint + (": " + msg[-1][:300] if msg else ""))
    info = dst_dir / f"{yid}.info.json"
    return vids[0], (info if info.exists() else None)
