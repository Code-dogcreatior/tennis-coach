"""Benchmark candidate 2D pose backends on this machine (CPU / Intel iGPU).

Usage:  uv run python scripts/bench_pose.py <video> [--frames 120]
Reports per-frame latency and mean keypoint confidence for each backend so the
pose model choice is based on measurements on the actual deployment hardware.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def read_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def bench(name, fn, frames):
    fn(frames[0])  # warm-up (model load / compile)
    t0 = time.perf_counter()
    confs = []
    for f in frames:
        c = fn(f)
        if c is not None:
            confs.append(c)
    dt = (time.perf_counter() - t0) / len(frames)
    print(f"{name:40s} {dt*1000:7.1f} ms/frame  {1/dt:6.1f} fps  "
          f"mean_conf={np.mean(confs) if confs else float('nan'):.3f}  detected={len(confs)}/{len(frames)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    frames = read_frames(args.video, args.frames)
    print(f"{len(frames)} frames, size={frames[0].shape[1]}x{frames[0].shape[0]}")

    from rtmlib import Body

    for backend, device in [("onnxruntime", "cpu"), ("openvino", "cpu"), ("openvino", "gpu")]:
        for mode in ["lightweight", "balanced"]:
            try:
                model = Body(mode=mode, to_openpose=False, backend=backend, device=device)
            except Exception as e:  # noqa: BLE001
                print(f"rtmlib {mode} {backend}/{device}: FAILED {e}")
                continue

            def run(f, model=model):
                kpts, scores = model(f)
                return float(scores[0].mean()) if len(scores) else None

            try:
                bench(f"rtmlib {mode} {backend}/{device}", run, frames)
            except Exception as e:  # noqa: BLE001
                print(f"rtmlib {mode} {backend}/{device}: FAILED {e}")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        import urllib.request, os
        mpath = "models/pose_landmarker_full.task"
        if not os.path.exists(mpath):
            os.makedirs("models", exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                mpath)
        opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=mpath),
                                            running_mode=vision.RunningMode.IMAGE)
        lm = vision.PoseLandmarker.create_from_options(opts)

        def run_mp(f):
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            if not r.pose_landmarks:
                return None
            return float(np.mean([p.visibility for p in r.pose_landmarks[0]]))

        bench("mediapipe pose_landmarker_full cpu", run_mp, frames)
    except Exception as e:  # noqa: BLE001
        print(f"mediapipe: FAILED {e}")


if __name__ == "__main__":
    main()
