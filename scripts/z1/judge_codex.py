#!/usr/bin/env python3
"""Judge whether stored TRIPLETS are true of the turn they came from.

This is the OTHER judging, and it asks a different question from
`judge_answers.py`. That one asks "which arm's ANSWER is better". This one asks
"is the arm's MEMORY correct" — the two can disagree, and the disagreement is
the interesting part.

**Categorical, not 1-5.** Same reasoning as the answer judge: numeric ratings
drift between batches so two runs are not comparable. A closed taxonomy is
stable and each label is directly actionable.

**⚑ `reversed` IS THE LOAD-BEARING LABEL.** A hand-read of 28 arm-B triplets
suggested roughly a third have subject and object SWAPPED
(`india --lives_in--> maharashtra`, `krishna --is_used_for--> flute`). That is a
different defect from "wrong": the entities and the relation are right and only
the direction is inverted, which a converse guard could fix and which no current
guard checks outside a curated `_ANTONYM_PAIRS` list. Collapsing it into "wrong"
would hide the most fixable failure in the extractor.

**Grouped by source turn.** Every triplet carries `source_batch`, which joins to
`episodic_memory.batch_id`, so the judge reads the turn ONCE and rules on all
the triplets drawn from it. Fewer calls, and the judge sees the triplets in the
context that produced them rather than one at a time.

**Blind to the arm and to the confidence tier.** The judge is never told which
arm it is reading or whether a triplet was stored as grounded or rejected —
otherwise the tier becomes a prior and the run cannot answer whether confidence
tracks correctness.

Run:
  uv run python scripts/z1/judge_codex.py --arm ner-b-nuner --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx                                       # noqa: E402
from sqlalchemy import text                        # noqa: E402

from src.api.config import settings                # noqa: E402
from src.api.db import SessionLocal                # noqa: E402

OUT = Path("experiments/curation_files/judgements")

LABELS = ("correct", "reversed", "wrong", "vacuous", "malformed", "unjudgeable")

# ⚑ INVARIANT. Byte-identical on every call so a caching provider hits the
# prefix. Never interpolate into this string.
SYSTEM = """You check whether extracted knowledge-graph triplets are true of the \
conversation turn they were extracted from.

You receive one SOURCE turn and a numbered list of TRIPLETS drawn from it. Each \
triplet is written `subject --relation--> object`.

Label every triplet with exactly one of:

correct     - the triplet is true of the source, and the direction is right.
reversed    - the subject and object are SWAPPED. The two entities and the \
relation are right, but the fact runs the other way. Example: source says \
"Maharashtra is in India" and the triplet says `india --lives_in--> \
maharashtra`.
wrong       - not supported by the source, or contradicts it.
vacuous     - technically defensible but carries no information: the relation is \
empty (`have`, `are`, `in`) joining two things in a way that states nothing, or \
the object merely restates the subject.
malformed   - the subject or object is not a thing (a sentence fragment, a \
clause, a dangling phrase), or the relation is a clause rather than a predicate.
unjudgeable - the source does not contain enough to decide.

Rules:
- Judge ONLY against the source text given. Do not use outside knowledge.
- A subject referred to by a number or a nickname is fine if the source uses it \
that way; that is not malformed.
- Prefer `reversed` over `wrong` whenever the entities and relation are right \
and only the direction is inverted. This distinction is the point of the task.
- Prefer `vacuous` over `correct` when the triplet is true but says nothing.

Return ONLY a JSON object, no prose:
{"verdicts": [{"n": <triplet number>, "label": "<label>", "why": "<max 12 words>"}]}
Include every triplet number exactly once."""


def fetch(db, n: int, seed: str, per_turn: int = 0):
    """Live edges joined to their source turn, seeded-random, tier-blind.

    ⚑ SAMPLE BY TURN, NOT BY EDGE (2026-08-23). The flat `order by md5(edge_id)`
    below drew 200 edges from only ~76 turns, and triplets from one turn are NOT
    independent draws: same source text, same extraction call, often the same
    subject. So the effective sample size is the number of TURNS, and the
    interval is set by 76, not 200.

    Measured cost of getting this wrong: two runs of the SAME configuration on
    content-identical stores returned **20.4%** and **10.0%** correct. At n=200
    independent triplets that gap is z=2.91 and reads as a real effect; at
    n≈76 turns it is z=1.80 and is noise. Every graph-quality number this
    project has published — 20%, 15.8%, 11.0%, 20.4%, 10.0% — carries roughly
    ±8 points and none of them are distinguishable from each other.

    ⇒ `per_turn=k` takes k edges from EVERY turn instead of n edges from
    wherever they fall. On this corpus that is ~164 turns of coverage for the
    same order of judge calls, which is a far tighter interval at no extra
    cost. Flat sampling is kept for reproducing older runs, and warns.
    """
    if per_turn:
        rows = db.execute(text("""
            select eid, subj, rel, obj, conf, neg, source, turn_id from (
              select e.id::text as eid, s.canonical_name as subj,
                     e.relation as rel, t.canonical_name as obj,
                     e.extraction_confidence as conf, e.negated as neg,
                     m.raw_text as source, m.id::text as turn_id,
                     row_number() over (
                       partition by m.id
                       order by md5(e.id::text || :seed)
                     ) as rn
              from codex_edges e
              join codex_entities s on s.id = e.source_id
              join codex_entities t on t.id = e.target_id
              join episodic_memory m on m.batch_id = e.source_batch
              where e.valid_until is null
            ) q
            where q.rn <= :k
            order by md5(q.eid || :seed)
            limit :n
        """), {"n": n, "k": per_turn, "seed": seed}).fetchall()
    else:
        rows = db.execute(text("""
            select e.id::text as eid, s.canonical_name as subj, e.relation as rel,
                   t.canonical_name as obj, e.extraction_confidence as conf,
                   e.negated as neg, m.raw_text as source, m.id::text as turn_id
            from codex_edges e
            join codex_entities s on s.id = e.source_id
            join codex_entities t on t.id = e.target_id
            join episodic_memory m on m.batch_id = e.source_batch
            where e.valid_until is null
            order by md5(e.id::text || :seed)
            limit :n
        """), {"n": n, "seed": seed}).fetchall()
    return rows


def report_power(rows, verdicts) -> None:
    """⚑ A CHECK, NOT A NOTE. Print what this sample can actually resolve.

    Every number this project has argued over was a bare percentage with no
    interval, which is how 20.4% and 10.0% got read as different results when
    they were one measurement twice. The tool now states its own resolution, so
    over-reading it takes deliberate effort rather than being the default.
    """
    import math
    n_edges = len(verdicts)
    n_turns = len({r.turn_id for r in rows}) or 1
    if not n_edges:
        return
    p = sum(1 for v in verdicts if v == "correct") / n_edges
    # Turns are the independent unit; edges within a turn are correlated.
    se = math.sqrt(max(p * (1 - p), 1e-9) / n_turns)
    ci = 1.96 * se
    print(f"\nRESOLUTION — what this sample can and cannot tell you")
    print(f"  triplets judged        {n_edges}")
    print(f"  DISTINCT TURNS         {n_turns}   <- the independent unit")
    print(f"  95% interval on a rate ±{100*ci:.1f} points")
    print(f"  ⇒ two results closer than {2*100*ci:.1f} points apart are THE SAME "
          f"measurement.")
    if n_turns < 100:
        print(f"  ⚠ ONLY {n_turns} TURNS. Use --per-turn to spread the same number "
              f"of judge calls across every turn in the store; sampling more "
              f"triplets from these turns will not narrow the interval.")


def call_judge(client, turn_text: str, triplets: list, model: str,
               source_cap: int = 6000) -> dict:
    """⚠ `source_cap` defaults to 6,000 to keep legacy runs reproducible. It is
    a MEASURED DEFECT, not a design choice — see `--source-cap 0`.

    Measured 2026-08-23 over the two control arms: 40% of turns carrying a
    `wrong` verdict exceed 6,000 chars, **30% of all source text was never shown
    to the judge**, and 21 triplets were marked "not supported by the source"
    while the entity being asked about sat past the cut. The identical cap was
    already removed from `judge_answers.py` (`⚑ NO CAP`, there) and was not
    carried across to this file. New runs should pass `--source-cap 0`.
    """
    listing = "\n".join(
        f"{i+1}. {t.subj} --{t.rel}{' [NEGATED]' if t.neg else ''}--> {t.obj}"
        for i, t in enumerate(triplets))
    body = turn_text if source_cap <= 0 else turn_text[:source_cap]
    user = (f"SOURCE TURN:\n{body}\n\n"
            f"TRIPLETS ({len(triplets)}):\n{listing}")
    # ⚑ RETRY ON A MISSING BODY, not just on an HTTP error. Measured
    # 2026-08-23: `muse-spark-1.2-contributor` returned 200 with no `content`
    # key on 3 of 8 calls, and the old code raised KeyError and scored those as
    # unreturned — a provider hiccup silently becoming missing data. Verified
    # not to be a token-budget problem: the same prompt succeeds at 1600, 4000
    # and 8000 max_tokens.
    last = None
    for attempt in range(4):
        try:
            r = client.post("/chat/completions", json={
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": 2400,
                "response_format": {"type": "json_object"},
            }, timeout=180)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content")
            if content:
                return json.loads(content)
            last = "200 with empty/absent content"
        except Exception as exc:                       # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:90]}"
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge call failed after 4 attempts — {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label for the output file")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", default="20260820")
    ap.add_argument("--model", default=None)
    # ⚑ USE THE CORPUS. The store holds ~164 extractable turns and ~7,400
    # triplets; flat sampling touched 76 turns and left the rest unused, which
    # is the whole reason the interval is ±8 points. `--per-turn 3` covers
    # every turn for roughly the same number of judge calls.
    ap.add_argument("--per-turn", type=int, default=0, metavar="K",
                    help="take K triplets from EVERY turn instead of N from "
                         "wherever they fall. Turns are the independent unit, "
                         "so this is what narrows the interval — more triplets "
                         "from the same turns does not.")
    # ⚑ The three flags below default to the OLD behaviour on purpose, so every
    # legacy judgement stays reproducible byte-for-byte. New runs opt in.
    ap.add_argument("--backend", choices=["probe", "coe"], default="probe",
                    help="probe = the pinned cloud judge (deepseek-v4-flash). "
                         "coe = the TCET campus gateway (Qwen3.6-35B-A3B), free "
                         "and ~2s/call, so it affords one call per triplet.")
    ap.add_argument("--single", action="store_true",
                    help="ONE triplet per call instead of batching a turn's "
                         "triplets into one call. Removes the 'include every "
                         "number exactly once' failure mode that silently lost "
                         "6 verdicts per arm, and gives each triplet the whole "
                         "output budget.")
    ap.add_argument("--source-cap", type=int, default=6000, metavar="CHARS",
                    help="truncate the source turn at CHARS. 0 = no cap. The "
                         "6000 default is a measured defect kept for "
                         "reproducibility — pass 0 for any new run.")
    ap.add_argument("--backfill", action="store_true",
                    help="judge ONLY the sampled triplets missing from this "
                         "arm's existing judgement file, then merge and rewrite "
                         "it. For recovering provider outages: a 500 storm cost "
                         "87 of 371 verdicts on one run. Pass the SAME --n / "
                         "--seed / --per-turn as the original or the sample "
                         "differs and the merge is meaningless.")
    args = ap.parse_args()

    if args.backend == "coe":
        base, key = settings.coe_api_base_url, settings.coe_api_key
        model = args.model or settings.coe_model
        if not key or not base:
            print("coe_api_key / coe_api_base_url not set"); return 1
    else:
        base, key = settings.probe_api_base_url, settings.probe_api_key
        model = args.model or settings.probe_model
        if not key or not base:
            print("probe_api_key / probe_api_base_url not set"); return 1

    db = SessionLocal()
    rows = fetch(db, args.n, args.seed, per_turn=args.per_turn)
    db.close()
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r.turn_id].append(r)
    cap_note = "NO CAP" if args.source_cap <= 0 else f"{args.source_cap} chars"
    mode = "1 triplet/call" if args.single else "batched by turn"
    print(f"{len(rows)} triplets from {len(by_turn)} turns · backend {args.backend} "
          f"· model {model} · {mode} · source {cap_note}")

    # ⚑ `rows` gets filtered by --backfill, but the INTERVAL must come from the
    # full sample's turn count, not just the turns that happened to fail.
    all_rows = rows
    prior: list = []
    if args.backfill:
        pf = OUT / f"codex_quality_{args.arm}.json"
        if not pf.exists():
            print(f"--backfill: no existing {pf}"); return 1
        prior = json.load(open(pf))["verdicts"]
        have = {v["edge_id"] for v in prior}
        rows = [r for r in rows if r.eid not in have]
        by_turn = defaultdict(list)
        for r in rows:
            by_turn[r.turn_id].append(r)
        if not rows:
            print("--backfill: nothing missing, file is already complete"); return 0
        # ⚑ Keep a copy before rewriting. The merge is the only step that can
        # destroy a judgement run that took an hour to produce.
        bak = pf.with_suffix(".json.bak")
        bak.write_text(pf.read_text())
        print(f"BACKFILL: {len(prior)} already judged · {len(rows)} missing to judge "
              f"· backup {bak.name}")

    client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"})

    # ⚑ Each unit is one call. Batched = one unit per turn; single = one per
    # triplet. `--single` removes the verdict-loss path: a batched call that
    # omits a number silently drops that triplet (6 lost per arm, measured).
    units = ([(t.turn_id, [t]) for t in rows] if args.single
             else list(by_turn.items()))

    verdicts, failures = [], 0
    for i, (tid, trips) in enumerate(units, 1):
        try:
            out = call_judge(client, trips[0].source, trips, model,
                             source_cap=args.source_cap)
            got = {int(v["n"]): v for v in out.get("verdicts", [])}
            for j, t in enumerate(trips, 1):
                v = got.get(j)
                if not v:
                    failures += 1
                    continue
                lab = str(v.get("label", "")).strip().lower()
                verdicts.append({
                    "edge_id": t.eid, "subj": t.subj, "rel": t.rel, "obj": t.obj,
                    "conf": float(t.conf) if t.conf is not None else None,
                    "negated": bool(t.neg),
                    "label": lab if lab in LABELS else "unjudgeable",
                    "why": str(v.get("why", ""))[:120],
                    "turn_id": t.turn_id,   # ⚑ so a judgement can be re-clustered
                })                          #   by turn AFTERWARDS. It could not be.
        except Exception as exc:
            failures += len(trips)
            print(f"  ! unit {i}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(2)
        if i % 25 == 0:
            print(f"  {i}/{len(units)} · {len(verdicts)} verdicts", flush=True)

    if args.backfill:
        print(f"\nBACKFILL: recovered {len(verdicts)} of {len(rows)} missing "
              f"({failures} still unreturned)")
        verdicts = prior + verdicts

    tally = Counter(v["label"] for v in verdicts)
    n = len(verdicts) or 1
    print(f"\n=== CODEX QUALITY — {args.arm} ===")
    print(f"judged {len(verdicts)} triplets ({failures} unreturned)")
    for lab in LABELS:
        print(f"  {lab:<12} {tally[lab]:>4}  ({100*tally[lab]/n:5.1f}%)")
    report_power(all_rows, [v["label"] for v in verdicts])

    print("\nBY STORED CONFIDENCE — does the tier track correctness?")
    for conf, name in ((0.9, "grounded"), (0.7, "ungrounded"), (0.35, "rejected")):
        sub = [v for v in verdicts if v["conf"] == conf]
        if not sub:
            continue
        c = Counter(v["label"] for v in sub)
        ok = 100 * c["correct"] / len(sub)
        print(f"  {name:<11} n={len(sub):>4}  correct {ok:5.1f}%  "
              f"reversed {100*c['reversed']/len(sub):5.1f}%  "
              f"wrong {100*c['wrong']/len(sub):5.1f}%  "
              f"vacuous {100*c['vacuous']/len(sub):5.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"codex_quality_{args.arm}.json"
    # ⚑ Stamp the PROTOCOL, not just the model. Two judgements are comparable
    # only if backend, batching and source cap all match — a lesson that cost
    # the "+9.4 pts" direction-rule effect (TRAPS #46).
    p.write_text(json.dumps({"arm": args.arm, "model": model, "seed": args.seed,
                             "n_requested": args.n,
                             "backend": args.backend,
                             "single": bool(args.single),
                             "source_cap": args.source_cap,
                             "per_turn": args.per_turn,
                             "tally": dict(tally),
                             "verdicts": verdicts}, indent=1))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
