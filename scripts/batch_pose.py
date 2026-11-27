"""Extract (and cache) poses for several videos sequentially.  uv run python scripts/batch_pose.py v1.mp4 v2.mp4 ..."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tennis_coach.pose import extract_cached  # noqa: E402

for v in sys.argv[1:]:
    t0 = time.perf_counter()
    seq, hit = extract_cached(v)
    print(f"{v}: frames={len(seq.t)} hit={hit} {time.perf_counter() - t0:.0f}s meta={seq.meta}", flush=True)
