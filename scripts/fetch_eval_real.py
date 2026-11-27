"""Download the real-world evaluation clips listed in data/labels/eval_real_manifest.csv (CC BY YouTube videos,
public, no login) and convert them with prep_annotate (CFR, t0 = 0, no audio) so the committed frame labels in
data/labels/annotations/ line up exactly.

  uv run python scripts/fetch_eval_real.py [--only ID ...]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_annotate import prep  # noqa: E402

MANIFEST = Path("data/labels/eval_real_manifest.csv")
ROOT = Path("data/raw/eval_real")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        yid = row["youtube_id"]
        if a.only and yid not in a.only:
            continue
        out = ROOT / yid / f"{yid}.mp4"
        if out.exists():
            continue
        src_dir = ROOT / yid / "_src"
        src_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
               "--merge-output-format", "mp4", "--write-info-json", "--no-playlist",
               "--download-sections", f"*{row['section']}", "--force-keyframes-at-cuts",
               "-o", str(src_dir / f"{yid}.%(ext)s"), f"https://www.youtube.com/watch?v={yid}"]
        if subprocess.run(cmd).returncode != 0:
            print("FAILED", yid); continue
        prep(src_dir / f"{yid}.mp4", out)
        shutil.move(str(src_dir / f"{yid}.info.json"), ROOT / yid / f"{yid}.info.json")
        shutil.rmtree(src_dir, ignore_errors=True)
        print("ok", yid, flush=True)


if __name__ == "__main__":
    main()
