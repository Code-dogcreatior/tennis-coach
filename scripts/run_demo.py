"""End-to-end demo against the running app (uv run tennis-coach): reference upload -> query upload -> results.

References: THETIS *train* players (front view, apparent hand from data/labels) + one CC BY amateur forehand clip.
Queries:    THETIS *test* players (one clip per class incl. non-target smash/volley) + Nadal (left-handed, CC BY).
Writes reports/demo/summary.json and reports/demo/summary.md. Nothing leaves the machine.

uv run python scripts/run_demo.py [--base http://127.0.0.1:8765] [--fresh]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ACT = {"forehand_flat": "forehand", "backhand": "backhand_1h", "backhand2hands": "backhand_2h", "flat_service": "serve"}
THETIS_SRC = "https://github.com/THETIS-dataset/dataset (licence unspecified; local evaluation only)"


def req(base, path, method="GET", body=None, headers=None):
    r = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())


def upload(base, path: Path, kind: str, meta: dict) -> str:
    """Upload, or - when a video of this name and kind is already in the app - re-analyse that one instead
    (re-running the demo used to add another copy of every reference each time)."""
    for v in req(base, f"/api/videos?kind={kind}"):
        if v["filename"] == path.name:
            req(base, f"/api/videos/{v['id']}/reanalyse", "POST")
            return v["id"]
    b = uuid.uuid4().hex
    parts = []
    for name, val in (("kind", kind), ("meta", json.dumps(meta, ensure_ascii=False))):
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode())
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return req(base, "/api/videos", "POST", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={b}"})["id"]


def wait(base, ids, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = [req(base, f"/api/videos/{i}/status") for i in ids]
        busy = [s for s in st if s["status"] in ("queued", "running")]
        print(f"\r{len(ids) - len(busy)}/{len(ids)} done", end="", flush=True)
        if not busy:
            print()
            return st
        time.sleep(3)
    raise TimeoutError


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console (cp1252) cannot print the Chinese summary
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--fresh", action="store_true", help="delete existing videos first")
    ap.add_argument("--summary-only", action="store_true", help="re-analyse existing query videos and re-summarise")
    ap.add_argument("--no-reanalyse", action="store_true", help="with --summary-only: only re-summarise (e.g. after reprocess_offline.py)")
    a = ap.parse_args()
    base = a.base
    hands = {r["player"]: r["apparent_hand"] for r in csv.DictReader(open("data/labels/thetis_apparent_hand.csv", encoding="utf-8"))}
    if a.summary_only:
        tok = {"foreflat": "forehand_flat", "backhand2h": "backhand2hands", "backhand": "backhand", "serflat": "flat_service",
               "smash": "smash", "fvolley": "forehand_volley"}
        refs = [v["id"] for v in req(base, "/api/videos?kind=reference")]
        qs = [v for v in req(base, "/api/videos?kind=query")    # demo clips only, not the user's own uploads
              if v["filename"].startswith("Rw3ZDfTTD9M") or v["filename"].split("_")[0] in ("p3", "p35", "p7")]
        if not a.no_reanalyse:
            for v in qs:
                req(base, f"/api/videos/{v['id']}/reanalyse", "POST")
            wait(base, [v["id"] for v in qs])
        qids = []
        for v in qs:
            name = v["filename"]
            if name.startswith("Rw3ZDfTTD9M"):
                qids.append((v["id"], Path(name), "forehand (left-handed, multiple)", "left"))
            else:
                pl, kind = name.split("_")[0], name.split("_")[1]
                qids.append((v["id"], Path(name), tok.get(kind, kind), hands.get(pl)))
        return summarise(base, refs, qids, None, None)
    if a.fresh:
        for v in req(base, "/api/videos"):
            req(base, f"/api/videos/{v['id']}", "DELETE")
    root = Path("data/raw/thetis")
    ref_players = ["p1", "p13", "p33", "p41", "p5"]           # train split
    q_players = ["p3", "p35", "p7"]                           # test split (p7 appears right-handed)
    t_start = time.time()
    ref_ids = []
    for pl in ref_players:
        for cls, act in ACT.items():
            f = root / cls / f"{pl}_{'foreflat' if cls == 'forehand_flat' else 'backhand2h' if cls == 'backhand2hands' else 'serflat' if cls == 'flat_service' else 'backhand'}_s1.avi"
            if not f.exists():
                f = next(iter(sorted((root / cls).glob(f"{pl}_*.avi"))), None)
            if f is None:
                continue
            ref_ids.append(upload(base, f, "reference", {
                "action": act, "hand": hands.get(pl), "time_scale": 1, "player": f"THETIS {pl}", "source": THETIS_SRC,
                "license": "unspecified", "license_commercial": "unknown", "redistribution": "no",
                "camera": "front, Kinect RGB (mirrored)", "review_status": "demo"}))
    cc = Path("data/raw/youtube_cc/nW5QJN-0EeQ/nW5QJN-0EeQ.mp4")
    if cc.exists():
        ref_ids.append(upload(base, cc, "reference", {
            "action": "forehand", "hand": "right", "time_scale": 1, "player": "amateur (YouTube nW5QJN-0EeQ)",
            "source": "https://www.youtube.com/watch?v=nW5QJN-0EeQ", "license": "CC BY", "license_commercial": "yes",
            "redistribution": "yes", "review_status": "demo"}))
    wait(base, ref_ids)
    t_refs = time.time() - t_start
    queries = []
    for pl in q_players:
        for cls in ["forehand_flat", "backhand", "backhand2hands", "flat_service", "smash", "forehand_volley"]:
            f = next(iter(sorted((root / cls).glob(f"{pl}_*.avi"))), None)
            if f:
                queries.append((f, {"time_scale": 1, "player": f"THETIS {pl}", "source": THETIS_SRC}, cls, hands.get(pl)))
    nadal = Path("data/raw/youtube_cc/Rw3ZDfTTD9M/Rw3ZDfTTD9M.mp4")
    if nadal.exists():
        queries.append((nadal, {"time_scale": 1, "player": "Nadal practice (CC BY)"}, "forehand (left-handed, multiple)", "left"))
    qids = []
    t_q0 = time.time()
    for f, meta, gt, hand in queries:
        qids.append((upload(base, f, "query", meta), f, gt, hand))
    wait(base, [q[0] for q in qids])
    t_q = time.time() - t_q0
    summarise(base, ref_ids, qids, t_refs, t_q)


def summarise(base, ref_ids, qids, t_refs, t_q):
    out = []
    for vid, f, gt, hand in qids:
        v = req(base, f"/api/videos/{vid}")
        an = v["analysis"]
        pi = an.get("primary_swing")
        sw = next((s for s in an["swings"] if s["index"] == pi), an["swings"][0] if an["swings"] else None)
        rec = {"video_id": vid, "file": f.as_posix(), "ground_truth": gt, "gt_apparent_hand": hand,
               "pose_ms_per_frame": an["pose_meta"].get("ms_per_frame"), "n_swings": an["n_swings"],
               "hand": an["hand"], "view_overall": an["view_overall"]["label"], "quality": an["quality"]["level"],
               "quality_issues": [i["message"] for i in an["quality"]["issues"]]}
        if sw:
            rec.update({"primary_swing": {
                "action": sw["action"], "confidence": sw["classification"]["confidence"],
                "evidence": sw["classification"]["evidence"], "view": sw["view_key_label"],
                "phases": [(s["label"], round(s["tv0"], 2), round(s["tv1"], 2)) for s in (sw["phases"] or {}).get("segments", [])],
                "match_status": sw["match"]["status"], "match_message": sw["match"]["message"],
                "top_refs": [(c["ref_id"], c["cost"]) for c in sw["match"].get("candidates", [])],
                "feedback": [{k: f2.get(k) for k in ("title", "observation", "practice", "confidence")} for f2 in sw.get("feedback", [])
                             if f2["kind"] != "disclaimer"],
                "pro": {"status": (sw.get("pro") or {}).get("status"), "message": (sw.get("pro") or {}).get("message"),
                        "nearest": ((sw.get("pro") or {}).get("nearest") or {}).get("player"),
                        "issues": [f"{c['title']}（你 {c['value']:.2f}，职业 {c['pro_p10']:.2f}–{c['pro_p90']:.2f}）"
                                   for c in (sw.get("pro") or {}).get("issues", [])],
                        "tentative": [c["title"] for c in (sw.get("pro") or {}).get("tentative", [])]},
            }, "all_swings": [(s["index"], s["action"], round(s["classification"]["confidence"], 2)) for s in an["swings"]]})
        out.append(rec)
    Path("reports/demo").mkdir(parents=True, exist_ok=True)
    summary = {"n_reference_videos": len(ref_ids), "n_reference_swings": len(req(base, "/api/references")),
               "reference_processing_s": round(t_refs, 1) if t_refs else None,
               "query_processing_s": round(t_q, 1) if t_q else None, "queries": out}
    if t_refs is None:   # summary-only run: keep the timings of the original run
        old = Path("reports/demo/summary.json")
        if old.exists():
            o = json.loads(old.read_text(encoding="utf-8"))
            summary["reference_processing_s"], summary["query_processing_s"] = o.get("reference_processing_s"), o.get("query_processing_s")
    Path("reports/demo/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 闭环演示结果（自动生成）", "",
             f"参考视频 {len(ref_ids)} 个 → 参考挥拍 {summary['n_reference_swings']} 条，处理 {summary['reference_processing_s']} s（含转码+姿态）；"
             f"查询 {len(out)} 个，处理 {summary['query_processing_s']} s。", "",
             "| 查询文件 | 真值 | 识别（主挥拍） | 置信度 | 持拍手（来源/置信度） | 机位 | 质量 | 匹配 | 反馈条数 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        p = r.get("primary_swing") or {}
        lines.append(f"| {Path(r['file']).name} | {r['ground_truth']} | {p.get('action', '无挥拍')} | {p.get('confidence', 0):.2f} | "
                     f"{r['hand']['hand']}（{r['hand']['source']}/{r['hand']['confidence']:.2f}；真值 {r['gt_apparent_hand']}） | "
                     f"{p.get('view', r['view_overall'])} | {r['quality']} | {p.get('match_status', '-')} | "
                     f"{len([f for f in p.get('feedback', []) if f.get('confidence')])} |")
    lines += ["", "## 与职业动作对比（主挥拍）", "", "| 查询文件 | 识别 | 最接近的职业挥拍 | 主要问题（已验证检查点） | 参考性提示 |", "|---|---|---|---|---|"]
    for r in out:
        p = r.get("primary_swing") or {}
        pr = p.get("pro") or {}
        if pr.get("status") != "ok":
            lines.append(f"| {Path(r['file']).name} | {p.get('action', '无挥拍')} | – | {pr.get('message') or '–'} | |")
            continue
        lines.append(f"| {Path(r['file']).name} | {p.get('action')} | {pr.get('nearest') or '–'} | "
                     f"{'；'.join(pr['issues']) or '无'} | {'；'.join(pr['tentative']) or '–'} |")
    Path("reports/demo/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
