"""Convert videos into the format the annotation tool and the evaluation assume:
constant frame rate, first frame at t = 0, H.264 MP4, no audio.

Why: the browser addresses frames by time (frame n = [n/fps, (n+1)/fps)). Clips cut from YouTube often start at
one frame past zero, and phone videos are often variable-frame-rate; both would shift the labelled contact
frame against the frame the analysis sees. After this conversion "frame n" means the same image everywhere.

  uv run python scripts/prep_annotate.py my_serves/*.mov                 # -> data/raw/annotate/<name>.mp4
  uv run python scripts/prep_annotate.py clip.mp4 --out data/raw/annotate/serves/clip.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

OUT_DIR = Path("data/raw/annotate")


def native_fps(src: Path) -> Fraction:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                        "-of", "json", str(src)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    def frac(x):
        try:
            return Fraction(x or "0/1")
        except ZeroDivisionError:   # "0/0" when ffprobe cannot tell
            return Fraction(0)
    fps = frac(s.get("avg_frame_rate")) or frac(s.get("r_frame_rate")) or Fraction(30)
    # snap to the usual rates (29.97 -> 30000/1001, 59.94 -> 60000/1001, ...) so the output is exactly CFR
    stds = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30), Fraction(50),
            Fraction(60000, 1001), Fraction(60), Fraction(120), Fraction(240))
    best = min(stds, key=lambda x: abs(float(fps) - float(x)))
    return best if abs(float(fps) - float(best)) < 0.005 * float(best) else fps.limit_denominator(1001)


def prep(src: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = native_fps(src)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
           "-vf", f"setpts=PTS-STARTPTS,fps={fps.numerator}/{fps.denominator}", "-fps_mode", "cfr",
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="output file (only with a single input)")
    a = ap.parse_args()
    if a.out and len(a.inputs) != 1:
        sys.exit("--out needs exactly one input")
    for src in a.inputs:
        out = a.out or OUT_DIR / (src.stem + ".mp4")
        if out.resolve() == src.resolve():
            sys.exit(f"refusing to overwrite the input in place: {src}")
        prep(src, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
