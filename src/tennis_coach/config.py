from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root (src/tennis_coach/config.py)
DATA_DIR = Path(os.environ.get("TC_DATA_DIR", ROOT / "data"))
STORE_DIR = DATA_DIR / "store"          # uploaded originals + transcoded copies
DERIVED_DIR = DATA_DIR / "derived"      # pose caches, analysis artefacts
DB_PATH = DATA_DIR / "tennis_coach.sqlite3"

# Pose model configuration (measured on i5-1235U / Iris Xe, see docs/MODELS.md)
POSE_BACKEND = os.environ.get("TC_POSE_BACKEND", "openvino")
POSE_DEVICE = os.environ.get("TC_POSE_DEVICE", "gpu")  # falls back to cpu automatically
MODELS_DIR = ROOT / "models"   # weights shipped with the repo (Git LFS); download URL as fallback


def _weights(name: str) -> str:
    local = MODELS_DIR / f"{name}.onnx"
    # a clone without git-lfs leaves a ~130-byte pointer file: fall back to the download then
    return str(local) if local.exists() and local.stat().st_size > 1_000_000 else (
        f"https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/{name}.zip")


DET_MODEL = _weights("yolox_m_8xb8-300e_humanart-c2c7a14a")
DET_INPUT = (640, 640)
POSE_MODEL = _weights("rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504")
POSE_INPUT = (192, 256)
POSE_MODEL_ID = "rtmpose-m_body7_256x192+yolox-m_humanart"
DET_EVERY = 6            # run person detector every N frames (and on track loss)
KPT_THR = 0.3            # RTMPose SimCC score below which a keypoint is treated as unreliable

# Analysis version — bump when the analysis logic changes so cached results are recomputed.
ANALYSIS_VERSION = "0.1.1"

for d in (STORE_DIR, DERIVED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# pose extraction speed-ups (cache keys unchanged: results at the analysed frames are identical in method)
MAX_POSE_FPS = 30          # high-frame-rate video is analysed at every k-th frame (timestamps kept)
LOST_REDETECT_EVERY = 3    # when nobody is found, run the person detector only every N analysed frames
