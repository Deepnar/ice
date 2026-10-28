"""Z2 F17: where turn summaries help and where they hurt — and whether ICE's own policies say so.

Three checks, from saved data only (read-only store query; no model calls):
  A. Is the lossless flag right? The background audit (S8) judged a sample of turn summaries
     complete or not; a turn whose summary is incomplete should carry inject_raw (keep raw).
  B. Does it land? Per prompt piece: was a flagged turn shown as its original?
  C. Are the read policies right? choose_representation prefers the original for some intents
     and the summary for others; per question, did summaries help or hurt (grade-1, per-point
     rescoring) under each policy class, time-question score and lossless share of cited turns?

    uv run --no-sync python experiments/z2/summary_allowance.py --history B --database ice_z2_main_b \
        --units logs/z2-run-main/s9-B/units --grades logs/z2-partial-B/graded.json \
        --occurrences logs/z2-questions-final/B/occurrences.json --audit logs/z2-run-main/s8-B
"""

import argparse
import collections
import json
import math
import os
import re
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREFER_RAW = {"Factual_Retrieval", "Troubleshooting"}
PREFER_SUMMARY = {"Analysis_&_Summarization", "Strategic_Planning", "Ideation"}


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def shingles(text, n=6, step=3):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(0, max(1, len(w) - n + 1), step)}


def store_rows(database):
    """[(row id, inject_raw, lossless_flag, has summary, coverage, raw chars)] in turn order."""
    from sqlalchemy import create_engine, text
    from src.api.config import settings
    url = settings.database_url.rsplit("/", 1)[0] + "/" + database
    with create_engine(url).connect() as c:
        return [tuple(r) for r in c.execute(text(
            "SELECT id::text, inject_raw, lossless_flag, summary_text IS NOT NULL, summary_coverage, "
            "length(raw_text) FROM episodic_memory ORDER BY timestamp, id"))]


def mean_ci(xs):
    if len(xs) < 2:
        return f"{100 * (xs[0] if xs else 0):+.1f} (n={len(xs)})"
    m = S.mean(xs)
    return f"{100 * m:+5.1f} ±{196 * S.stdev(xs) / math.sqrt(len(xs)):4.1f} (n={len(xs)})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for a in ("--history", "--database"):
        parser.add_argument(a, required=True)
    for a in ("--units", "--grades", "--occurrences", "--audit"):
        parser.add_argument(a, type=Path, required=True)
    args = parser.parse_args()
    from experiments.z2.ledger import load_history, role_text

    rows = store_rows(args.database)
    turn_of = {r[0]: n for n, r in enumerate(rows, 1)}
    flag = {n: r[1] for n, r in enumerate(rows, 1)}
    print(f"{args.history}: {len(rows)} stored turns; inject_raw true on {sum(bool(r[1]) for r in rows)}, "
          f"lossless_flag true on {sum(bool(r[2]) for r in rows)}, with a summary {sum(r[3] for r in rows)}")

    # A — the lossless flag against the audit's completeness verdicts
    judged = json.loads((args.audit / "details.json").read_text())["judged"]
    a = collections.defaultdict(list)
    for key, v in judged.items():
        if not key.startswith("turn_summary:") or v.get("status") != "judged":
            continue
        verdict = (v.get("verdicts") or {}).get("complete")
        n = turn_of.get(key.split(":", 1)[1])
        if verdict in ("yes", "no") and n:
            a[f"inject_raw={flag[n]}"].append(verdict == "yes")
            chars = rows[n - 1][5]
            a["turn ≤1,500 chars" if chars <= 1500 else "turn 1,500-4,000" if chars <= 4000 else "turn >4,000 chars"].append(verdict == "yes")
    print("\nA. Audited turn summaries judged COMPLETE, by the turn's lossless flag and length")
    for k in sorted(a):
        print(f"   {k:22} {sum(a[k])}/{len(a[k])} complete ({100 * sum(a[k]) / len(a[k]):.0f}%)")

    # B — does the flag land: how flagged turns reach full ICE's prompts
    originals = load_history(args.history)
    cache = {}
    b = collections.Counter()
    units = {}
    for e in os.scandir(args.units / "full_default"):
        u = json.loads(Path(e.path).read_text())
        if u["ask"]["kind"] != "question":
            continue
        units[u["ask"]["id"]] = u
        for f in u["stage"]["selected"]:
            st = f.get("source_turns") or []
            st = json.loads(st) if isinstance(st, str) else st
            if f["type"] != "episodic" or len(st) != 1:
                continue
            t = st[0]
            if t not in cache:
                cache[t] = norm(role_text(originals, t, "user")) + " " + norm(role_text(originals, t, "assistant"))
            body = norm(re.sub(r"^(\s*\[[^\]]*\])+", "", f["text"]))
            whole = sum(1 for s in shingles(cache[t]) if s in body) / max(1, len(shingles(cache[t])))
            own = sum(1 for s in shingles(body) if s in cache[t]) / max(1, len(shingles(body)))
            kind = "original" if whole >= .9 else "excerpt" if own >= .8 else "summary"
            b[(flag.get(t), kind)] += 1
    print("\nB. Pieces in full ICE's prompts, by the turn's lossless flag (does the flag land?)")
    for fl in (True, False):
        tot = sum(v for (f, _), v in b.items() if f is fl) or 1
        print(f"   inject_raw={fl!s:5} {tot:5} pieces: " +
              ", ".join(f"{k} {100 * b[(fl, k)] / tot:.0f}%" for k in ("original", "summary", "excerpt")))

    # C — are the read policies right: summary effect per question by policy class and signals
    graded = json.loads(args.grades.read_text())
    graded = graded.get("graded", graded)

    def pts(c, q):
        r = graded.get(f"{c}|{q}")
        if not r or r.get("status") != "graded":
            return None
        if not r["items_total"]:
            return float(r["grade"] == "correct")
        return float(len(set(r["items_covered"])) >= r["items_total"] and not r["superseded_used"])

    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads(args.occurrences.read_text())}
    groups = collections.defaultdict(lambda: ([], []))
    for q, u in units.items():
        add = (pts("add_4_summaries", q), pts("add_3_codex", q))
        rem = (pts("full", q), pts("loo_summaries", q))
        intents = set(u["stage"].get("intent_tags") or [])
        gate = u["stage"].get("gate") or {}
        pt = gate.get("classifier_p_temporal")
        gold = {int(c["turn"]) for c in occ.get(q, {}).get("reference", {}).get("citations", []) if c.get("turn")}
        lossless = (sum(bool(flag.get(t)) for t in gold) / len(gold)) if gold else None
        keys = ["all",
                "policy: prefer original" if intents & PREFER_RAW else
                "policy: prefer summary" if intents & PREFER_SUMMARY else "policy: turn's flag",
                ("time score ≥0.5" if pt >= .5 else "time score <0.5") if isinstance(pt, (int, float)) else "time score n/a",
                ("cited turns mostly lossless" if lossless >= .5 else "cited turns mostly not lossless")
                if lossless is not None else "no cited turns",
                "type: " + occ.get(q, {}).get("type", "?")]
        for k in keys:
            if None not in add:
                groups[k][0].append(add[0] - add[1])
            if None not in rem:
                groups[k][1].append(rem[0] - rem[1])
    print("\nC. Summary effect on answers (+ = summaries HELP), grade-1 per-point rescoring")
    print(f"   {'group':34} {'adding them (add_4 − add_3)':>28} {'removing them (full − loo)':>28}")
    for k in sorted(groups, key=lambda k: (k != "all", k)):
        print(f"   {k:34} {mean_ci(groups[k][0]):>28} {mean_ci(groups[k][1]):>28}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
