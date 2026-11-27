"""Reproducible download of a THETIS RGB subset from the official GitHub repository.

Source: https://github.com/THETIS-dataset/dataset (redirect target of http://thetis.image.ece.ntua.gr/)
Licence: the repository does not state a licence (checked 2026-11-17) -> treat as "unspecified":
research/evaluation use only, no redistribution, no commercial use assumed.

uv run python scripts/fetch_thetis.py --out data/raw/thetis [--classes ...] [--players p1,p3,...] [--all-players]
Writes <out>/manifest.csv. Only public, unauthenticated raw URLs are used; downloads are resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/THETIS-dataset/dataset/contents/VIDEO_RGB/{cls}"
UA = {"User-Agent": "thetis-subset-fetch"}
CLASS_MAP = {  # THETIS folder -> (our action label, note)
    "forehand_flat": ("forehand", ""), "forehand_openstands": ("forehand", "open stance"),
    "backhand": ("backhand_1h", ""), "backhand2hands": ("backhand_2h", ""),
    "flat_service": ("serve", "flat"), "kick_service": ("serve", "kick"), "slice_service": ("serve", "slice"),
    "smash": ("unknown", "non-target: smash"), "forehand_volley": ("unknown", "non-target: volley"),
    "backhand_volley": ("unknown", "non-target: volley"), "forehand_slice": ("forehand", "slice"),
    "backhand_slice": ("backhand_1h", "slice"),
}
DEFAULT_CLASSES = ["forehand_flat", "forehand_openstands", "backhand", "backhand2hands", "flat_service", "smash",
                   "forehand_volley"]
DEFAULT_PLAYERS = [f"p{i}" for i in range(1, 22, 2)] + [f"p{i}" for i in range(33, 54, 2)]  # 11 beginners + 11 experts


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url, dst: Path, size: int):
    if dst.exists() and dst.stat().st_size == size:
        return "cached"
    tmp = dst.with_suffix(".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/thetis")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--players", default=",".join(DEFAULT_PLAYERS))
    ap.add_argument("--all-players", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    players = None if a.all_players else set(a.players.split(","))
    rows = []
    for cls in a.classes.split(","):
        listing = get_json(API.format(cls=cls))
        (out / cls).mkdir(exist_ok=True)
        for e in listing:
            name = e["name"]
            pl = name.split("_")[0]
            if players is not None and pl not in players:
                continue
            dst = out / cls / name
            for attempt in range(3):
                try:
                    st = download(e["download_url"], dst, e["size"])
                    break
                except Exception as ex:  # noqa: BLE001
                    st = f"error: {ex}"
                    time.sleep(3)
            idx = int(pl[1:])
            rows.append({"file": str(dst.as_posix()), "player": pl, "level": "beginner" if idx <= 31 else "expert",
                         "thetis_class": cls, "action": CLASS_MAP[cls][0], "note": CLASS_MAP[cls][1],
                         "seq": name.rsplit("_", 1)[-1].split(".")[0], "size": e["size"], "status": st,
                         "source": e["download_url"], "license": "unspecified (no licence in repository)"})
            print(cls, name, st, flush=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] in ("ok", "cached") for r in rows)
    print(f"done: {ok}/{len(rows)} files -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
