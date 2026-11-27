"""Re-run analysis for stored videos in this process (poses are cached), optionally applying
data/labels/pro_review.csv first. Use while the server's single worker is busy with pose extraction.

uv run python scripts/reprocess_offline.py [--apply-review] [--pro-done] [--queries] [VIDEO_ID ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tennis_coach import db, service  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--apply-review", action="store_true")
    ap.add_argument("--pro-done", action="store_true", help="all finished pro reference videos")
    ap.add_argument("--queries", action="store_true", help="all finished query videos")
    a = ap.parse_args()
    ids = list(a.ids)
    vids = {v["id"]: v for v in db.list_videos()}
    if a.pro_done:
        ids += [i for i, v in vids.items() if v["meta"].get("tier") == "pro" and v["status"] == "done"]
    if a.queries:
        ids += [i for i, v in vids.items() if v["kind"] == "query" and v["status"] == "done"]
    review = {}
    if a.apply_review and Path("data/labels/pro_review.csv").exists():
        for r in csv.DictReader(open("data/labels/pro_review.csv", encoding="utf-8")):
            review.setdefault(r["youtube_id"], {})[r["swing_index"]] = "unknown" if r["action"] == "exclude" else r["action"]
    for vid in ids:
        meta = dict(vids[vid]["meta"])
        yid = meta.get("youtube_id")
        if a.apply_review and yid in review:
            meta["action_overrides"] = review[yid]
            db.update_video(vid, meta=meta)
        service.process(vid)
        print("processed", vid, yid or vids[vid]["filename"], flush=True)


if __name__ == "__main__":
    main()
