"""Professional reference pack: share the pro reference library WITHOUT the source videos.

The pack holds, per pro reference video, only derived data: the 2D keypoint sequence (pose.npz), the per-swing
feature arrays (swings.npz), the camera-cut flags (cuts.npy), the analysis result and the reference labels.
No video frames are included. After import, comparison, recognition (kNN), DTW matching and the pro
checkpoint ranges work as with the full library; key-frame comparison images and the reference player show
the pro as a skeleton instead of video.

  uv run python scripts/ref_pack.py export [--out refpack/pro_reference_pack.zip]   # from the local library
  uv run python scripts/ref_pack.py import [--pack refpack/pro_reference_pack.zip]  # into a (new) install
  uv run python scripts/ref_pack.py remove                                          # delete imported entries

Run import/remove with the app stopped, or restart it afterwards. To use real video frames instead, run
`remove`, then fetch_pro.py + build_pro_library.py (they skip videos that are already in the library).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from tennis_coach import config, db, service
from tennis_coach.pose import cuts_cached

PACK_VERSION = 1
DEFAULT = Path("refpack/pro_reference_pack.zip")
REF_FIELDS = ("swing_index", "action", "view_key", "hand", "label_source", "review_status", "style", "notes", "enabled")
DROP_META = ("reviewer",)   # nothing personal or machine-specific goes into the pack


def export(out: Path):
    vids = [v for v in db.list_videos("reference") if (v.get("meta") or {}).get("tier") == "pro"]
    refs = db.list_refs()
    entries = []
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # npz members are already compressed
        for v in vids:
            vrefs = [r for r in refs if r["video_id"] == v["id"]]
            if not vrefs:
                continue   # e.g. excluded slow-motion clips: nothing to compare against
            full = db.get_video(v["id"])
            dd = service.derived_dir(v["id"])
            if not full.get("analysis") or not (dd / "pose.npz").exists() or not (dd / "swings.npz").exists():
                print("skip (not analysed)", v["id"], v["meta"].get("player"))
                continue
            if (dd / "cuts.npy").exists():
                cuts = np.load(dd / "cuts.npy")
            elif service.has_video(full):
                cuts = cuts_cached(full["path"])
            else:
                cuts = np.zeros(0, bool)
            buf = io.BytesIO(); np.save(buf, cuts)
            z.writestr(f"{v['id']}/cuts.npy", buf.getvalue())
            z.write(dd / "pose.npz", f"{v['id']}/pose.npz")
            z.write(dd / "swings.npz", f"{v['id']}/swings.npz")
            meta = {k: x for k, x in full["meta"].items() if k not in DROP_META}
            info = {k: x for k, x in (full.get("info") or {}).items() if k != "original"}
            entries.append({"id": v["id"], "filename": full["filename"], "meta": meta, "info": info,
                            "analysis": full["analysis"], "model_version": full.get("model_version"),
                            "analysis_version": full.get("analysis_version"),
                            "refs": [{k: r[k] for k in REF_FIELDS} for r in vrefs]})
        manifest = {"pack_version": PACK_VERSION, "created": time.strftime("%Y-%m-%d"),
                    "pose_model": config.POSE_MODEL_ID, "analysis_version": config.ANALYSIS_VERSION,
                    "content": "2D keypoints + derived features of public YouTube practice videos; no video frames",
                    "videos": entries}
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, default=float))
    tmp.replace(out)
    n = sum(len(e["refs"]) for e in entries)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(entries)} videos, {n} reference swings")


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def import_(pack: Path):
    db.init()
    with zipfile.ZipFile(pack) as z:
        man = json.loads(z.read("manifest.json"))
        if man.get("pack_version") != PACK_VERSION:
            sys.exit(f"unsupported pack version {man.get('pack_version')}")
        if man.get("pose_model") != config.POSE_MODEL_ID:
            print(f"warning: pack pose model {man.get('pose_model')} != {config.POSE_MODEL_ID}")
        added = skipped = nref = 0
        for e in man["videos"]:
            vid = e["id"]
            if db.get_video(vid, full=False):
                skipped += 1
                continue
            dd = service.derived_dir(vid)
            dd.mkdir(parents=True, exist_ok=True)
            for f in ("pose.npz", "swings.npz", "cuts.npy"):
                name = f"{vid}/{f}"
                if name in z.namelist() and _safe_member(name):
                    (dd / f).write_bytes(z.read(name))
            meta = {**e["meta"], "ref_pack": PACK_VERSION}
            db.insert_video(vid, "reference", e["filename"], "", meta)
            db.update_video(vid, info=e["info"], analysis=e["analysis"], status="done", stage="done", progress=1.0,
                            model_version=e.get("model_version"), analysis_version=e.get("analysis_version"))
            for r in e["refs"]:
                db.upsert_ref(vid, r["swing_index"], r["action"], r["view_key"], r["hand"], r["label_source"])
                rid = next(x["id"] for x in db.list_refs() if x["video_id"] == vid and x["swing_index"] == r["swing_index"])
                db.update_ref(rid, review_status=r["review_status"], style=r["style"], notes=r["notes"],
                              enabled=r["enabled"], label_source=r["label_source"])
                nref += 1
            added += 1
    print(f"imported {added} videos / {nref} reference swings ({skipped} already present)")


def remove():
    n = 0
    for v in db.list_videos("reference"):
        if (v.get("meta") or {}).get("ref_pack"):
            service.delete_video(v["id"]); n += 1
    print(f"removed {n} reference-pack videos")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--out", type=Path, default=DEFAULT)
    i = sub.add_parser("import"); i.add_argument("--pack", type=Path, default=DEFAULT)
    sub.add_parser("remove")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.out)
    elif a.cmd == "import":
        import_(a.pack)
    else:
        remove()


if __name__ == "__main__":
    main()
