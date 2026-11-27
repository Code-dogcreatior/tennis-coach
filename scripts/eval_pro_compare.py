"""Validate the professional comparison (technique checkpoints).

1. Pro leave-one-player-out: every professional reference swing is judged against the professionals *other
   than that player*. A good checkpoint should rarely flag a professional (false-alarm proxy).
2. THETIS amateurs: every THETIS clip whose ground-truth stroke is one of the four targets (gt action forced,
   gt apparent hand) is judged against the full professional library. THETIS labels players as beginners
   (p1-p31) or experienced (p32-p55); if the checkpoints measure something real, beginners should be flagged
   more often than experienced players, and both more often than professionals.

uv run python scripts/eval_pro_compare.py [--out reports/pro_compare_eval.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tennis_coach import service  # noqa: E402
from tennis_coach.analysis import analyse_video  # noqa: E402
from tennis_coach.technique import BY_KEY, judge  # noqa: E402

VALIDATION = Path(__file__).resolve().parent.parent / "src" / "tennis_coach" / "checkpoint_validation.json"
GATE_MAX_FA, GATE_MIN_N, GATE_INVERSION = 0.15, 10, 0.20
GT = {"forehand_flat": "forehand", "forehand_openstands": "forehand", "backhand": "backhand_1h",
      "backhand2hands": "backhand_2h", "flat_service": "serve"}


def summarise(judged: list[list[dict]]) -> dict:
    n = len(judged)
    per_cp = defaultdict(Counter)
    n_issues = []
    for checks in judged:
        k = 0
        for c in checks:
            per_cp[c["key"]][c["status"]] += 1
            k += c["status"] == "issue"
        n_issues.append(k)
    rates = {}
    for key, cnt in per_cp.items():
        m = cnt["ok"] + cnt["borderline"] + cnt["issue"] + cnt["outside_not_fault"]
        if m:
            rates[key] = {"label": BY_KEY[key].label, "measured": m, "issue_rate": round(cnt["issue"] / m, 3)}
    return {"n_swings": n, "mean_issues_per_swing": round(float(np.mean(n_issues)), 3) if n else None,
            "share_with_any_issue": round(float(np.mean([x > 0 for x in n_issues])), 3) if n else None,
            "per_checkpoint": dict(sorted(rates.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/pro_compare_eval.json")
    ap.add_argument("--thetis", default="data/raw/thetis")
    a = ap.parse_args()
    pros = service.trusted_pros(service.load_refs())   # same filter as the app
    players = sorted({r["player"] for r in pros})
    print(f"{len(pros)} pro swings, players: {players}")
    print("by action:", dict(Counter(r["action"] for r in pros)))
    # 1. LOPO over professionals
    lopo = defaultdict(list)
    for r in pros:
        others = [x for x in pros if x["player"] != r["player"]]
        lopo[r["action"]].append(judge(r["technique"], r["action"], r["view_key"], others, r.get("view")))
    # 2. THETIS amateurs
    hands = {r["player"]: r["apparent_hand"] for r in csv.DictReader(open("data/labels/thetis_apparent_hand.csv", encoding="utf-8"))}
    am = {"beginner": defaultdict(list), "experienced": defaultdict(list)}
    for f in sorted(Path(a.thetis).glob("*/*.avi")):
        cls = f.parent.name
        if cls not in GT:
            continue
        player = f.name.split("_")[0]
        level = "beginner" if int(player[1:]) <= 31 else "experienced"
        act = GT[cls]
        try:
            res = analyse_video(str(f), hand_override=hands.get(player) if hands.get(player) in ("left", "right") else None,
                                time_scale=1.0, action_overrides={i: act for i in range(8)})
        except Exception as e:  # noqa: BLE001
            print("skip", f, e)
            continue
        sws = [s for s in res["swings"] if s.get("technique")]
        if not sws:
            continue
        sw = max(sws, key=lambda s: s["window"]["peak_speed"])
        am[level][act].append(judge(sw["technique"], act, sw["view_key"], pros, sw.get("view")))
    rep = {"n_pro_swings": len(pros), "pro_players": players, "pro_by_action": dict(Counter(r["action"] for r in pros)),
           "pro_leave_one_player_out": {act: summarise(v) for act, v in lopo.items()},
           "thetis_beginner": {act: summarise(v) for act, v in am["beginner"].items()},
           "thetis_experienced": {act: summarise(v) for act, v in am["experienced"].items()}}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    # gating table used at run time: a checkpoint drives the main diagnosis only if it rarely flags
    # professionals (leave-one-player-out) and was measurable on enough professional swings
    gate = {}
    for act, s in rep["pro_leave_one_player_out"].items():
        for key, c in s["per_checkpoint"].items():
            ok = c["measured"] >= GATE_MIN_N and c["issue_rate"] <= GATE_MAX_FA
            beg = rep["thetis_beginner"].get(act, {}).get("per_checkpoint", {}).get(key)
            exp = rep["thetis_experienced"].get(act, {}).get("per_checkpoint", {}).get(key)
            # a fault detector that fires far more often on experienced players than on beginners is suspect
            inverted = bool(beg and exp and beg["measured"] >= 20 and exp["measured"] >= 20
                            and exp["issue_rate"] - beg["issue_rate"] > GATE_INVERSION)
            ok = ok and not inverted
            gate[f"{act}:{key}"] = {"validated": ok, "pro_false_alarm": c["issue_rate"], "pro_measured": c["measured"],
                                    "inverted_vs_experience": inverted,
                                    "thetis_beginner_rate": beg and beg["issue_rate"],
                                    "thetis_experienced_rate": exp and exp["issue_rate"]}
    VALIDATION.write_text(json.dumps({"criteria": f"pro LOPO false-alarm <= {GATE_MAX_FA} and n >= {GATE_MIN_N} and not "
                                                  f"(THETIS experienced rate - beginner rate > {GATE_INVERSION})",
                                      "n_pro_swings": len(pros), "pro_players": players, "gate": gate},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print("validated:", sorted(k for k, v in gate.items() if v["validated"]))
    for grp in ("pro_leave_one_player_out", "thetis_experienced", "thetis_beginner"):
        for act, s in rep[grp].items():
            print(f"{grp:26s} {act:12s} n={s['n_swings']:3d} issues/swing={s['mean_issues_per_swing']} any={s['share_with_any_issue']}")


if __name__ == "__main__":
    main()
