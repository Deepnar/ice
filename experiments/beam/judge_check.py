"""Judge the BEAM judge (Z2 FINDINGS F27, Z3 judging run 1): BEAM's own protocol under FINAL's
judge, on planted answers with a known score, before any FINAL answer is graded with it.

Plants, from BEAM's public data only (the exported asks of FINAL's BEAM conversations; seeded):
- gold: the question's own ideal answer;
- gold + extra: the ideal answer with another question's ideal answer appended — true, unasked
  detail, the shape of ICE's long answers (not for abstention or instruction-following questions,
  where appended content is itself what the rubric checks);
- off-topic: another question's ideal answer alone;
- event ordering: the rubric's events as a numbered list, in order and reversed.

Pass marks, stated before the run: gold mean >= 0.85; gold + extra within 0.10 of gold; off-topic
mean <= 0.20; ordered tau_norm >= 0.9; reversed <= 0.1. Writes <out>/report.json and report.txt.

    uv run --no-sync python experiments/beam/judge_check.py --out logs/z3-judging/beam-check [--dry-run]
"""

import argparse
import glob
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.beam import adapter as A  # noqa: E402

POLICY = "ice-final-beam-judge-check-1"
PER_TYPE, ORDERING = 3, 10
NO_EXTRA = {"abstention", "instruction_following"}
PASS = dict(gold=0.85, extra_drop=0.10, off_topic=0.20, ordered=0.9, reversed=0.1)


def plants(asks, seed=20261103):
    """[(kind, ask, answer)] — deterministic."""
    rng = random.Random(seed)
    by_conv = {}
    for a in asks:
        by_conv.setdefault(a["id"].rsplit(":", 2)[0], []).append(a)
    usable = [a for a in asks if a["reference"].get("answer") and a["reference"]["nuggets"]]
    out = []
    for qtype in sorted({a["type"] for a in usable} - {"event_ordering"}):
        for ask in rng.sample(sorted((a for a in usable if a["type"] == qtype), key=lambda a: a["id"]), PER_TYPE):
            donors = [d for d in by_conv[ask["id"].rsplit(":", 2)[0]]
                      if d["id"] != ask["id"] and d["type"] not in ("abstention", qtype) and d["reference"].get("answer")]
            donor = rng.choice(sorted(donors, key=lambda d: d["id"]))["reference"]["answer"]
            gold = str(ask["reference"]["answer"])
            out.append(("gold", ask, gold))
            if qtype not in NO_EXTRA:
                out.append(("extra", ask, f"{gold}\n\n{donor}"))
            out.append(("off_topic", ask, str(donor)))
    ordering = sorted((a for a in usable if a["type"] == "event_ordering" and len(a["reference"]["nuggets"]) >= 3),
                      key=lambda a: a["id"])
    for ask in rng.sample(ordering, min(ORDERING, len(ordering))):
        events = ask["reference"]["nuggets"]
        out.append(("ordered", ask, "\n".join(events)))
        out.append(("reversed", ask, "\n".join(reversed(events))))
    return out


def summarise(rows):
    by = {}
    for kind, ask, s in rows:
        by.setdefault(kind, []).append(s["beam_score"])
    mean = {k: round(statistics.fmean(v), 3) for k, v in by.items()}
    gold_extra = [s["beam_score"] for k, a, s in rows if k == "gold" and a["type"] not in NO_EXTRA]
    extra_drop = round(statistics.fmean(gold_extra) - mean["extra"], 3) if gold_extra and "extra" in mean else None
    passed = dict(gold=mean.get("gold", 0) >= PASS["gold"],
                  extra_drop=extra_drop is not None and extra_drop <= PASS["extra_drop"],
                  off_topic=mean.get("off_topic", 1) <= PASS["off_topic"],
                  ordered=mean.get("ordered", 0) >= PASS["ordered"],
                  reversed=mean.get("reversed", 1) <= PASS["reversed"])
    low = [dict(kind=k, id=a["id"], type=a["type"], score=s["beam_score"]) for k, a, s in rows
           if (k in ("gold", "extra", "ordered") and s["beam_score"] < 0.75) or (k in ("off_topic", "reversed") and s["beam_score"] > 0.25)]
    return dict(policy=POLICY, n={k: len(v) for k, v in by.items()}, mean=mean, extra_drop=extra_drop,
                pass_marks=PASS, passed=passed, misses=low)


def run(asks, out, judge, *, dry=False, say=print):
    out.mkdir(parents=True, exist_ok=True)
    todo = plants(asks)
    plan = {k: sum(1 for p in todo if p[0] == k) for k in ("gold", "extra", "off_topic", "ordered", "reversed")}
    (out / "plan.json").write_text(json.dumps(dict(policy=POLICY, plants=plan), indent=1) + "\n")
    if dry:
        say(json.dumps(plan))
        return 0

    import threading
    from experiments.z2.grade import Errors
    errors, lock = Errors(out / "beam-check-errors.json"), threading.Lock()

    def one(p):
        """(kind, ask, score), "unjudged" after MAX_ERRORS failures (excluded and counted), or None
        to retry next run — one failed plant never stops the check (2026-11-03: a cut reply did)."""
        kind, ask, answer = p
        key = f"{kind}|{ask['id']}"
        if errors.given_up(key):
            return "unjudged"
        try:
            return kind, ask, A.score(ask, answer, judge)
        except Exception as exc:                  # noqa: BLE001 — this plant only
            with lock:
                errors.failed(key, exc)
                gave_up = errors.given_up(key)
            say(f"WARNING: BEAM judging failed for {key} ({type(exc).__name__}: {str(exc)[:120]})"
                + ("; given up, recorded unjudged" if gave_up else "; left for the next run"))
            return "unjudged" if gave_up else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, todo))
    pending = sum(r is None for r in rows)
    report = dict(summarise([r for r in rows if isinstance(r, tuple)]), unjudged=sum(r == "unjudged" for r in rows))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = [f"BEAM judge check ({POLICY}): BEAM's own protocol, FINAL's judge", ""]
    lines += [f"  {k}: mean {report['mean'].get(k)} over {report['n'].get(k)}" for k in plan]
    lines += [f"  gold + extra drop: {report['extra_drop']}", "",
              "  pass: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in report["passed"].items()),
              f"  misses: {len(report['misses'])} (report.json); unjudged after repeated failures: {report['unjudged']}"]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending=pending, finished=not pending)) + "\n")
    say("\n".join(lines))
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--asks", default="logs/final-beam/*/*/asks.json", help="glob of FINAL's BEAM asks")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default="muse")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asks = [a for path in sorted(glob.glob(str(ROOT / args.asks))) for a in json.loads(Path(path).read_text())
            if a.get("kind", "question") == "question"]
    if not asks:
        raise SystemExit(f"no BEAM asks at {args.asks} (run prep-beam-export first)")
    judge = None
    if not args.dry_run:
        import hashlib
        from experiments.beam.grade import message_judge
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2 import ledger_merge as M
        from experiments.z2.profiles import generator, metadata
        load_selected_env()
        args.out.mkdir(parents=True, exist_ok=True)
        session = f"ice-final-beam-check-{hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:16]}"
        judge = message_judge(M.Calls(args.out / "beam-grade-raw.jsonl"), generator(args.judge).generate,
                              metadata(args.judge), session)
    return 3 if run(asks, args.out, judge, dry=args.dry_run) else 0


if __name__ == "__main__":
    raise SystemExit(main())
