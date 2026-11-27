"""Download the professional practice clips listed in data/labels/pro_manifest.csv (public videos, no login /
cookies; standard YouTube licence -> local analysis only, never redistributed). Optional 'section' column
("START-END" seconds) limits long videos. Files land in data/raw/pro/<id>/<id>.mp4 with an .info.json.

uv run python scripts/fetch_pro.py [--only ID ...]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

MANIFEST = Path("data/labels/pro_manifest.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
        yid = row["youtube_id"]
        if a.only and yid not in a.only:
            continue
        out = Path(f"data/raw/pro/{yid}/{yid}.mp4")
        url_id = yid.split("#")[0]   # "ID#name" = a named section of one YouTube video
        if out.exists():
            continue
        cmd = [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
               "--merge-output-format", "mp4", "--write-info-json", "--no-playlist", "--no-warnings",
               "-o", f"data/raw/pro/{yid}/{yid}.%(ext)s"]
        if (row.get("section") or "").strip():
            cmd += ["--download-sections", f"*{row['section'].strip()}", "--force-keyframes-at-cuts"]
        cmd.append(f"https://www.youtube.com/watch?v={url_id}")
        print("downloading", yid, row["player"], row.get("section") or "", flush=True)
        r = subprocess.run(cmd)
        print(yid, "ok" if r.returncode == 0 and out.exists() else f"FAILED ({r.returncode})", flush=True)


if __name__ == "__main__":
    main()
