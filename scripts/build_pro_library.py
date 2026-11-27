"""Upload the professional practice clips listed in data/labels/pro_manifest.csv as reference videos
(tier=pro) to the running app, then apply the per-swing review labels from data/labels/pro_review.csv
(if present): action relabels and exclusions.

uv run python scripts/build_pro_library.py [--base http://127.0.0.1:8765] [--review-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_demo import req, upload, wait  # noqa: E402

MANIFEST = Path("data/labels/pro_manifest.csv")
REVIEW = Path("data/labels/pro_review.csv")


def existing(base) -> dict:
    return {v["meta"].get("youtube_id"): v for v in req(base, "/api/videos?kind=reference") if v["meta"].get("youtube_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--review-only", action="store_true")
    a = ap.parse_args()
    have = existing(a.base)
    if not a.review_only:
        ids = []
        for row in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
            yid = row["youtube_id"]
            if yid in have:
                continue
            f = Path(f"data/raw/pro/{yid}/{yid}.mp4")
            if not f.exists():
                print("missing", f)
                continue
            meta = {"tier": "pro", "youtube_id": yid, "player": row["player"], "hand": row["hand"],
                    "time_scale": "unknown" if row["time_scale"] == "unknown" else float(row["time_scale"]), "source": f"https://www.youtube.com/watch?v={yid.split('#')[0]}" + (f" ({row['section']} s)" if row.get("section") else ""),
                    "camera": row["camera"], "content": row["content"], "note": row["note"],
                    "review_status": "demo", "redistribution": "no", "license": "YouTube standard (local analysis only)"}
            ids.append(upload(a.base, f, "reference", meta))
            print("uploaded", yid, ids[-1], flush=True)
        if ids:
            wait(a.base, ids, timeout=4 * 3600)
        have = existing(a.base)
    if REVIEW.exists():
        apply_review(a.base, have)


def apply_review(base, have):
    """pro_review.csv: youtube_id,swing_index,action  (action 'exclude' disables the swing as a reference)."""
    by_vid: dict = {}
    for row in csv.DictReader(open(REVIEW, encoding="utf-8")):
        by_vid.setdefault(row["youtube_id"], {})[int(row["swing_index"])] = row["action"].strip()
    for yid, labels in by_vid.items():
        v = have.get(yid)
        if not v:
            continue
        # 'exclude' -> action 'unknown': the swing is analysed but never enters the reference library
        overrides = {str(k): ("unknown" if a == "exclude" else a) for k, a in labels.items()}
        if v["meta"].get("action_overrides") != overrides:
            body = json.dumps({"action_overrides": overrides}).encode()
            req(base, f"/api/videos/{v['id']}", "PATCH", body, {"Content-Type": "application/json"})
            print("relabelled", yid, sum(a != "unknown" for a in overrides.values()),
                  "excluded", sum(a == "unknown" for a in overrides.values()))
    wait(base, [v["id"] for v in have.values()], timeout=3600)


if __name__ == "__main__":
    main()
