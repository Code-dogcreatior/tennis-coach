"""SQLite persistence: videos (+metadata, licence, analysis), reference swings, corrections log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from .config import DB_PATH

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('reference','query')),
    filename TEXT, orig_path TEXT, path TEXT, sha1 TEXT,
    created_at REAL, updated_at REAL,
    info_json TEXT,           -- probe info of the transcoded file (fps, size, rotation of original...)
    meta_json TEXT,           -- user metadata: source, licence, player, hand, view, action, time_scale, notes...
    status TEXT, stage TEXT, progress REAL, error TEXT,
    analysis_json TEXT,       -- full analysis result (without feature arrays)
    model_version TEXT, analysis_version TEXT
);
CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    swing_index INTEGER NOT NULL,
    action TEXT NOT NULL, view_key TEXT, hand TEXT,
    label_source TEXT,        -- auto | user
    review_status TEXT NOT NULL DEFAULT 'demo',   -- demo (参考示范) | coach_reviewed | rejected
    style TEXT, notes TEXT, enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL, UNIQUE(video_id, swing_index)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target TEXT, target_id TEXT, field TEXT, old_value TEXT, new_value TEXT, at REAL
);
"""


@contextmanager
def conn():
    with _lock:
        c = sqlite3.connect(DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def _row_video(r: sqlite3.Row | None, full: bool = False) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("info_json", "meta_json"):
        d[k[:-5]] = json.loads(d.pop(k) or "{}")
    a = d.pop("analysis_json")
    if full:
        d["analysis"] = json.loads(a) if a else None
    else:
        d["has_analysis"] = bool(a)
        if a:
            j = json.loads(a)
            d["summary"] = {"n_swings": j.get("n_swings"), "hand": j.get("hand", {}).get("hand"),
                            "view": j.get("view_overall", {}).get("label"), "quality": j.get("quality", {}).get("level"),
                            "actions": [s.get("action") for s in j.get("swings", [])]}
    return d


def insert_video(vid: str, kind: str, filename: str, orig_path: str, meta: dict):
    now = time.time()
    with conn() as c:
        c.execute("INSERT INTO videos(id,kind,filename,orig_path,created_at,updated_at,meta_json,status,stage,progress)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (vid, kind, filename, orig_path, now, now, json.dumps(meta, ensure_ascii=False), "queued", "queued", 0.0))


def update_video(vid: str, **fields):
    if not fields:
        return
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("info", "meta", "analysis"):
            k, v = k + "_json", json.dumps(v, ensure_ascii=False, default=float)
        cols.append(f"{k}=?"); vals.append(v)
    cols.append("updated_at=?"); vals.append(time.time())
    with conn() as c:
        c.execute(f"UPDATE videos SET {', '.join(cols)} WHERE id=?", (*vals, vid))


def get_video(vid: str, full: bool = True) -> dict | None:
    with conn() as c:
        return _row_video(c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone(), full)


def list_videos(kind: str | None = None) -> list[dict]:
    with conn() as c:
        q = "SELECT * FROM videos" + (" WHERE kind=?" if kind else "") + " ORDER BY created_at DESC"
        return [_row_video(r) for r in c.execute(q, (kind,) if kind else ())]


def delete_video(vid: str):
    with conn() as c:
        c.execute("DELETE FROM refs WHERE video_id=?", (vid,))
        c.execute("DELETE FROM videos WHERE id=?", (vid,))


def upsert_ref(video_id: str, swing_index: int, action: str, view_key: str, hand: str, label_source: str):
    with conn() as c:
        old = c.execute("SELECT * FROM refs WHERE video_id=? AND swing_index=?", (video_id, swing_index)).fetchone()
        if old:
            # keep user-set fields; refresh automatic ones unless the user labelled the action
            if old["label_source"] == "user":
                c.execute("UPDATE refs SET view_key=?, hand=? WHERE id=?", (view_key, hand, old["id"]))
            else:
                c.execute("UPDATE refs SET action=?, view_key=?, hand=? WHERE id=?", (action, view_key, hand, old["id"]))
        else:
            c.execute("INSERT INTO refs(video_id,swing_index,action,view_key,hand,label_source,created_at)"
                      " VALUES (?,?,?,?,?,?,?)", (video_id, swing_index, action, view_key, hand, label_source, time.time()))


def prune_refs(video_id: str, keep_indices: set[int]):
    """Remove reference entries of a re-analysed video whose swing no longer qualifies (unknown / truncated /
    bad quality / relabelled as excluded). Entries edited individually by a user (label_source='user') stay."""
    with conn() as c:
        rows = c.execute("SELECT id, swing_index, label_source FROM refs WHERE video_id=?", (video_id,)).fetchall()
        for r in rows:
            if r["swing_index"] not in keep_indices and r["label_source"] != "user":
                c.execute("DELETE FROM refs WHERE id=?", (r["id"],))


def list_refs(enabled_only: bool = False) -> list[dict]:
    with conn() as c:
        q = ("SELECT r.*, v.filename, v.meta_json FROM refs r JOIN videos v ON v.id=r.video_id"
             + (" WHERE r.enabled=1 AND r.review_status!='rejected'" if enabled_only else "") + " ORDER BY r.created_at")
        out = []
        for r in c.execute(q):
            d = dict(r)
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            out.append(d)
        return out


def get_ref(rid: int) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM refs WHERE id=?", (rid,)).fetchone()
        return dict(r) if r else None


def update_ref(rid: int, **fields):
    allowed = {"action", "review_status", "style", "notes", "enabled", "label_source"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    with conn() as c:
        c.execute(f"UPDATE refs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), rid))


def delete_ref(rid: int):
    with conn() as c:
        c.execute("DELETE FROM refs WHERE id=?", (rid,))


def log_correction(target: str, target_id: str, field: str, old, new):
    with conn() as c:
        c.execute("INSERT INTO corrections(target,target_id,field,old_value,new_value,at) VALUES (?,?,?,?,?,?)",
                  (target, str(target_id), field, json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False),
                   time.time()))


def list_corrections(target_id: str | None = None) -> list[dict]:
    with conn() as c:
        q = "SELECT * FROM corrections" + (" WHERE target_id=?" if target_id else "") + " ORDER BY at DESC"
        return [dict(r) for r in c.execute(q, (target_id,) if target_id else ())]
