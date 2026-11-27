"""One-off repair: pose files written by the strided extractor before PoseSeq.frame_index existed.
For every stored video whose pose has fewer frames than the video, rebuild frame_index by matching pose
timestamps to the video's frame timestamps, and rewrite both the per-video pose and the pose-cache entry.

uv run python scripts/repair_frame_index.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tennis_coach import config, db, service  # noqa: E402
from tennis_coach.pose import PoseSeq, file_sha1  # noqa: E402
from tennis_coach.video import frame_timestamps  # noqa: E402

for v in db.list_videos():
    full = db.get_video(v["id"], full=False)
    path = full.get("path")
    p = service.derived_dir(v["id"]) / "pose.npz"
    if not path or not p.exists():
        continue
    seq = PoseSeq.load(p)
    ts = frame_timestamps(path)
    if len(ts) == len(seq.t):
        continue
    idx = np.clip(np.searchsorted(ts, seq.t), 0, len(ts) - 1)
    prev = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[prev] - seq.t) < np.abs(ts[idx] - seq.t), prev, idx)
    seq.frame_index = idx
    seq.save(p)
    key = hashlib.sha1((file_sha1(path) + config.POSE_MODEL_ID + str(config.DET_EVERY)).encode()).hexdigest()[:20]
    c = config.DERIVED_DIR / "pose_cache" / f"{key}.npz"
    if c.exists():
        seq.save(c)
    print("repaired", v["id"], v["meta"].get("youtube_id") or v["filename"], len(seq.t), "of", len(ts), "frames",
          "step", np.median(np.diff(idx)))
