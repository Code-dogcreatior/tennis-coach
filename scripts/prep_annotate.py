"""Convert videos into the format the annotation tool and the evaluation assume (constant frame rate, first
frame at t = 0, H.264 MP4, no audio); see tennis_coach.annotprep. The annotation page can also do this itself
("导入视频"), this script is for batches.

  uv run python scripts/prep_annotate.py my_serves/*.mov                 # -> data/raw/annotate/<name>/<name>.mp4
  uv run python scripts/prep_annotate.py clip.mp4 --out data/raw/annotate/serves/clip.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tennis_coach.annotprep import native_fps, prep  # noqa: F401  (re-exported for fetch_eval_real)

OUT_DIR = Path("data/raw/annotate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="output file (only with a single input)")
    a = ap.parse_args()
    if a.out and len(a.inputs) != 1:
        sys.exit("--out needs exactly one input")
    for src in a.inputs:
        out = a.out or OUT_DIR / src.stem / (src.stem + ".mp4")
        prep(src, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
