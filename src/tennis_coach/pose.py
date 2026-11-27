"""2D human pose extraction (RTMPose-m via rtmlib/OpenVINO) with single-target tracking and caching."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .video import frame_timestamps, is_cut, iter_frames, probe, shot_hist

log = logging.getLogger(__name__)

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb",
          "l_wri", "r_wri", "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]


@dataclass
class PoseSeq:
    t: np.ndarray            # (T,) seconds, real presentation timestamps
    kpts: np.ndarray         # (T,17,2) pixel coords (NaN when no person)
    scores: np.ndarray       # (T,17)
    bbox: np.ndarray         # (T,4) xyxy of tracked person (NaN when none)
    n_persons: np.ndarray    # (T,) persons found on detection frames (-1 = not a detection frame)
    other_area: np.ndarray   # (T,) largest other-person bbox area / target area on detection frames
    width: int
    height: int
    fps: float
    model_id: str = config.POSE_MODEL_ID
    meta: dict = field(default_factory=dict)
    frame_index: np.ndarray | None = None   # (T,) source video frame number of each pose frame (frame striding)

    def __post_init__(self):
        if self.frame_index is None:
            self.frame_index = np.arange(len(self.t))

    def video_frame(self, i: int) -> int:
        """Video frame number of pose frame i (differs from i when high-frame-rate video was strided)."""
        return int(self.frame_index[int(i)])

    def save(self, path: Path):
        np.savez_compressed(path, t=self.t, kpts=self.kpts, scores=self.scores, bbox=self.bbox,
                            n_persons=self.n_persons, other_area=self.other_area,
                            info=np.array([self.width, self.height, self.fps]), model_id=np.array(self.model_id),
                            frame_index=self.frame_index)

    @classmethod
    def load(cls, path: Path) -> "PoseSeq":
        z = np.load(path, allow_pickle=False)
        w, h, fps = z["info"]
        return cls(t=z["t"], kpts=z["kpts"], scores=z["scores"], bbox=z["bbox"], n_persons=z["n_persons"],
                   other_area=z["other_area"], width=int(w), height=int(h), fps=float(fps),
                   model_id=str(z["model_id"]), frame_index=z["frame_index"] if "frame_index" in z.files else None)

    def slice(self, i0: int, i1: int) -> "PoseSeq":
        return PoseSeq(t=self.t[i0:i1], kpts=self.kpts[i0:i1], scores=self.scores[i0:i1], bbox=self.bbox[i0:i1],
                       n_persons=self.n_persons[i0:i1], other_area=self.other_area[i0:i1], width=self.width,
                       height=self.height, fps=self.fps, model_id=self.model_id,
                       frame_index=self.frame_index[i0:i1])

    def to_json(self, decimals: int = 1) -> dict:
        k = np.where(np.isnan(self.kpts), -1, np.round(self.kpts, decimals))
        return {"t": np.round(self.t, 4).tolist(), "kpts": k.tolist(),
                "scores": np.round(np.nan_to_num(self.scores), 3).tolist(),
                "width": self.width, "height": self.height, "fps": self.fps, "names": COCO17}


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _kpt_bbox(k: np.ndarray, s: np.ndarray, expand: float = 1.2) -> np.ndarray | None:
    ok = s > config.KPT_THR
    if ok.sum() < 6:
        return None
    x, y = k[ok, 0], k[ok, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    hw, hh = (x.max() - x.min()) / 2 * expand, (y.max() - y.min()) / 2 * expand
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


LOST_RESET = 3   # detections in a row without an IoU match before the target is re-selected


class PoseEstimator:
    _instance: "PoseEstimator | None" = None

    def __init__(self, backend: str = config.POSE_BACKEND, device: str = config.POSE_DEVICE):
        import sys
        if backend == "openvino" and "onnxruntime" not in sys.modules:
            # rtmlib probes onnxruntime at import time even when another backend is used. On Windows with an
            # outdated System32 msvcp140.dll (< 14.40) that import crashes the whole process (no exception),
            # so make it a clean ImportError, which rtmlib handles. OpenVINO does not need onnxruntime.
            sys.modules["onnxruntime"] = None
        from rtmlib import YOLOX, RTMPose

        last_err = None
        for dev in ([device, "cpu"] if device != "cpu" else ["cpu"]):
            try:
                self.det = YOLOX(config.DET_MODEL, model_input_size=config.DET_INPUT, backend=backend, device=dev,
                                 score_thr=0.5)
                self.pose = RTMPose(config.POSE_MODEL, model_input_size=config.POSE_INPUT, backend=backend, device=dev)
                self.device = dev
                self.backend = backend
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("pose backend %s/%s unavailable: %s", backend, dev, e)
        else:
            raise RuntimeError(f"no pose backend available: {last_err}")

    @classmethod
    def get(cls) -> "PoseEstimator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(self, video_path: str | Path, progress: Callable[[float], None] | None = None,
            target_hint: str = "largest") -> PoseSeq:
        info = probe(video_path)
        ts = frame_timestamps(video_path)
        n_est = len(ts) or (info.n_frames or int(info.duration * info.fps))
        kpts, scores, bboxes, nper, other = [], [], [], [], []
        target = None
        since_det = 10 ** 9
        lost_wait = 0
        misses, weak = 0, False   # consecutive detections without an IoU match; last pose was low-confidence
        # high-frame-rate input: analyse at most ~MAX_POSE_FPS (every k-th frame; real timestamps are kept)
        stride = max(1, int(round((info.fps or 30.0) / config.MAX_POSE_FPS))) if info.fps and info.fps > config.MAX_POSE_FPS * 1.3 else 1
        keep_idx = []
        prev_hist = None
        t0 = time.perf_counter()
        for i, frame in iter_frames(video_path):
            if i % stride:
                continue
            keep_idx.append(i)
            # edited footage: after a camera cut the old box points at whatever is there now (often a spectator)
            # and IoU matching would then follow that person for the rest of the video -> pick the target afresh
            h = shot_hist(frame)
            if is_cut(prev_hist, h):
                target, since_det, lost_wait = None, 10 ** 9, 0
            prev_hist = h
            if lost_wait > 0:          # target lost: re-detect only every few frames (far / tiny players)
                lost_wait -= 1
                kpts.append(np.full((17, 2), np.nan)); scores.append(np.zeros(17)); bboxes.append(np.full(4, np.nan))
                nper.append(-1); other.append(0.0)
                continue
            need_det = target is None or since_det >= config.DET_EVERY
            n_here, other_ratio = -1, 0.0
            if need_det:
                dets = self.det(frame)
                dets = [d for d in np.asarray(dets).reshape(-1, 4) if _area(d) > 400]
                n_here = len(dets)
                if dets:
                    cx = frame.shape[1] / 2

                    def score(d):   # largest person, slightly favouring the image centre
                        c = (d[0] + d[2]) / 2
                        return _area(d) * (1.0 - 0.3 * abs(c - cx) / cx)
                    if target is None:
                        best = max(dets, key=score)
                    else:
                        ious = [_iou(d, target) for d in dets]
                        j = int(np.argmax(ious))
                        best = dets[j] if ious[j] > 0.2 else None
                        misses = 0 if best is not None else misses + 1
                        if best is None:
                            # lost: keep the keypoint-propagated box for a moment (occlusion), but a box that no
                            # detection overlaps while the pose there is weak is dead - without this it can stay
                            # frozen on an empty spot for minutes. Pick the target afresh instead.
                            best = max(dets, key=score) if (weak or misses >= LOST_RESET) else target
                            if best is not target:
                                misses = 0
                    others = [_area(d) for d in dets if _iou(d, best) < 0.5]
                    other_ratio = (max(others) / max(_area(best), 1.0)) if others else 0.0
                    target = np.asarray(best, dtype=float)
                else:
                    target = None          # nobody found: back off (see lost_wait) instead of detecting every frame
                since_det = 0
            if target is None:
                kpts.append(np.full((17, 2), np.nan)); scores.append(np.zeros(17)); bboxes.append(np.full(4, np.nan))
                lost_wait = config.LOST_REDETECT_EVERY - 1
            else:
                k, s = self.pose(frame, bboxes=[target])
                k, s = k[0], s[0]
                kpts.append(k); scores.append(s); bboxes.append(target.copy())
                weak = bool(s.mean() <= 0.25)
                nb = _kpt_bbox(k, s)
                if nb is not None and s.mean() > 0.25:
                    target = nb
                else:
                    since_det = config.DET_EVERY  # force re-detection next frame
            nper.append(n_here); other.append(other_ratio)
            since_det += 1
            if progress and i % 10 == 0:
                progress(min(0.999, (i + 1) / max(n_est, 1)))
        T = len(kpts)
        if T == 0:
            raise RuntimeError("no frames decoded")
        if stride > 1 and len(ts) >= keep_idx[-1] + 1:
            ts = ts[np.asarray(keep_idx)]
        if len(ts) != T:
            log.warning("timestamp count %d != frame count %d; falling back to fps grid", len(ts), T)
            ts = np.arange(T) * stride / (info.fps or 30.0)
        dt = time.perf_counter() - t0
        seq = PoseSeq(t=ts, kpts=np.asarray(kpts, float), scores=np.asarray(scores, float),
                      bbox=np.asarray(bboxes, float), n_persons=np.asarray(nper), other_area=np.asarray(other),
                      width=info.width, height=info.height, fps=info.fps, frame_index=np.asarray(keep_idx))
        seq.meta = {"pose_seconds": dt, "ms_per_frame": 1000 * dt / T, "device": f"{self.backend}/{self.device}"}
        return seq


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def cuts_cached(video_path: str | Path, cache_dir: Path | None = None) -> np.ndarray:
    from .video import scene_cuts
    cache_dir = cache_dir or (config.DERIVED_DIR / "pose_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{file_sha1(video_path)[:20]}_cuts.npy"
    if p.exists():
        return np.load(p)
    c = scene_cuts(video_path)
    np.save(p, c)
    return c


def extract_cached(video_path: str | Path, cache_dir: Path | None = None,
                   progress: Callable[[float], None] | None = None) -> tuple[PoseSeq, bool]:
    """Return (pose sequence, cache_hit). Cache key = file content hash + pose model id."""
    cache_dir = cache_dir or (config.DERIVED_DIR / "pose_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((file_sha1(video_path) + config.POSE_MODEL_ID + str(config.DET_EVERY)).encode()).hexdigest()[:20]
    p = cache_dir / f"{key}.npz"
    if p.exists():
        return PoseSeq.load(p), True
    seq = PoseEstimator.get().run(video_path, progress=progress)
    seq.save(p)
    return seq, False
