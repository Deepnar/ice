#!/usr/bin/env python3
"""Z2: judge two arms' answers head-to-head, blind and paired.

**Why paired and not scored.** Asking a model to rate an answer 1-5 drifts
between batches and between prompts, so two runs are not comparable and a small
difference is unreadable. Asking *"here is the question, here is the source
material, here are two answers — which is better?"* is stable, and it is exactly
the question the model comparison asks. Ties are allowed and are informative.

**⚑ `both_failed` IS THE LOAD-BEARING VERDICT.** A tie because both answers are
good and a tie because both are useless are opposite findings. If most probes
land in `both_failed`, memory is not working and the model question is moot —
collapsing that into "TIE" would hide the most important result in the run.

**Blindness.** The judge is never told which arm produced which answer, and the
A/B slot is randomised per probe from a fixed seed, so position bias cannot
align with an arm. The mapping is kept locally and applied when tallying.

**Prompt-cache friendliness.** Everything invariant — the rubric, the taxonomy,
the output shape — lives in the SYSTEM message and is byte-identical on every
call, so a provider that caches by prefix can hit on it. Only the per-probe
block varies, and it goes last. Do not interpolate anything into the system
message: one changed character invalidates the cache for the whole run.

Run:
  uv run python scripts/z1/judge_answers.py --a <arm1.json> --b <arm2.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path("experiments/curation_files/judgements")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")

REASONS = ("more_grounded", "more_complete", "contradicts_source",
           "no_memory_used", "both_failed", "equivalent")

# ⚑ INVARIANT. Byte-identical on every call — that is what makes it cacheable.
SYSTEM = """You judge which of two AI answers better answers a user's question, \
given the source material the answer should have been drawn from.

You will receive:
  QUESTION       - what the user asked
  SOURCE         - the actual earlier conversation text that contains the answer
  ANSWER A       - one system's reply
  ANSWER B       - another system's reply

Both answers came from the SAME model. They differ only in what memory was \
retrieved and put in front of it. You are judging the memory, not the writing.

Decide which answer a person who wrote the SOURCE would find more useful, and \
return ONE JSON object:

  {"verdict": "A" | "B" | "TIE", "reason": "<one of the codes below>", \
"note": "<one short sentence>"}

REASON CODES - pick the single best fit:
  more_grounded      one answer uses specifics from SOURCE; the other hedges, \
generalises, or invents
  more_complete      both are correct, but one covers more of what was asked
  contradicts_source one answer states something SOURCE contradicts
  no_memory_used     one answer is generic or says it does not know, while the \
other clearly used the source material
  both_failed        NEITHER answer answers the question - use this whenever \
both are generic, evasive, or wrong, even if one is slightly better written
  equivalent         both answer it about equally well

Rules:
  - Judge substance, not length or style. A short correct answer beats a long \
vague one.
  - If both answers miss the point, the verdict is TIE and the reason is \
both_failed. Do not pick a winner among two failures.
  - Ignore which answer is longer, more confident, or better formatted.
  - Return ONLY the JSON object. No prose, no code fences."""


def _env(k: str):
    v = os.environ.get(k)
    if v:
        return v
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip().startswith(f"{k}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


_GOLD_IDX = None


def _full_source(rec) -> str:
    """The COMPLETE gold turns for a record, read from the store.

    The `gold_turn_text` baked into an answer file is already truncated (3
    turns, 1,200 chars each), so re-judging an existing run cannot recover the
    source by reading it back — it has to be rebuilt. The answers themselves are
    unaffected: they were generated from RETRIEVED fragments, never from this
    field, so only the judge's view was ever short.

    Keyed on (conversation, turn_number) via the seed manifest rather than on
    episodic ids, because ids are regenerated per seed and differ between arms
    while the corpus text is identical — so one loaded store serves both arms'
    records. Falls back to the stored text if a turn cannot be resolved, and
    says so rather than silently shortening.
    """
    global _GOLD_IDX
    if _GOLD_IDX is None:
        _GOLD_IDX = {}
        try:
            from src.api.db import SessionLocal
            from sqlalchemy import text as _sql
            man = json.loads(Path(
                "experiments/curation_files/seeded_store.json").read_text())
            db = SessionLocal()
            raw = {r[0]: r[1] for r in db.execute(
                _sql("select id::text, raw_text from episodic_memory")).fetchall()}
            db.close()
            for cid, d in man["conversations"].items():
                for turn in d["turns"]:
                    txt = raw.get(turn["episodic_id"])
                    if txt:
                        _GOLD_IDX[(cid, int(turn["turn_number"]))] = txt
        except Exception as exc:                              # noqa: BLE001
            print(f"  ! could not build gold index ({type(exc).__name__}) — "
                  f"falling back to the TRUNCATED stored text")
    conv = rec.get("conversation")
    turns = rec.get("gold_turns") or []
    parts = [_GOLD_IDX.get((conv, int(t))) for t in turns]
    got = [p for p in parts if p]
    if not got:
        return rec.get("gold_turn_text", "")
    return "\n\n".join(got)


def judge_one(question, source, ans_a, ans_b, *, retries=3):
    key, base, model = (_env("PROBE_API_KEY"), _env("PROBE_API_BASE_URL"),
                        _env("PROBE_MODEL"))
    if not key or not base:
        raise SystemExit("PROBE_API_KEY / PROBE_API_BASE_URL missing from .env")
    user = (f"QUESTION\n{question}\n\n"
            # ⚑ NO CAP. This was `source[:4000]`, on top of answer_probes
            # keeping only 3 gold turns at 1,200 chars each. Measured
            # 2026-08-20: the judge saw a MEDIAN 12.7% of the gold material,
            # and the per-type coverage predicted the tie rate monotonically
            # across all five types (summary_synthesis 6.3% coverage -> 92%
            # ties; episodic_lookup 37.5% -> 50%). It was ruling answers
            # unsupported because it had not been shown the support.
            f"SOURCE\n{source}\n\n"
            f"ANSWER A\n{(ans_a or '(empty)')[:2500]}\n\n"
            f"ANSWER B\n{(ans_b or '(empty)')[:2500]}")
    body = {"model": model,
            # System first and unchanged, variable part last: prefix caching.
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            # ⚑ 1500, not 300. This model returns `reasoning_content` — it is a
            # REASONING model, and it spends the budget inside the hidden block
            # first. At 300 the smoke test returned empty content on every real
            # probe while a toy prompt worked fine, which reads as "the judge is
            # broken" and is TRAPS #11 instead.
            # ⚑ 16000 + reasoning ON — effectively no ceiling. Setting reasoning_effort="none" fixed the
            # empty-content bug and CAUSED a worse one: the judge stopped
            # deliberating and over-called both_failed on 30 of 73 — several at
            # 100% content overlap. A comparison task needs the thinking; pay
            # for it with budget instead of switching it off. 4000 still errored on
            # probe 2 of 150, so the ceiling is removed rather than nudged: a
            # judge that returns empty content on the HARDEST probes biases the
            # result toward whatever happens to be easy to judge.
            "temperature": 0.0, "max_tokens": 16000,
            # ⚑ The actual fix, and the repo already knew it: this is a
            # reasoning model, and bg_client_factory.py:23 sends the same flag
            # for exactly this reason. Without it the hidden block eats the
            # budget and `content` comes back empty on long inputs — a toy
            # prompt still works, which is what makes it read as a broken judge
            # rather than an exhausted one (TRAPS #11).
            }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}",
                         # Cloudflare 403/1010 without this — TRAPS #27.
                         "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read())
            txt = (payload["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:                                  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            if attempt == retries - 1:
                return {"verdict": "ERROR", "reason": "api", "note": str(exc)[:120]}
            continue
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception:                                          # noqa: BLE001
            return {"verdict": "ERROR", "reason": "unparseable", "note": txt[:120]}
        if d.get("verdict") not in ("A", "B", "TIE"):
            return {"verdict": "ERROR", "reason": "bad_verdict", "note": txt[:120]}
        if d.get("reason") not in REASONS:
            d["reason"] = "equivalent"
        return d
    return {"verdict": "ERROR", "reason": "exhausted", "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm 1 answers json")
    ap.add_argument("--b", required=True, help="arm 2 answers json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="judge")
    args = ap.parse_args()

    da, db = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    name_a, name_b = da.get("tag", "A"), db.get("tag", "B")
    # Pair on the question text — the two runs used the same probes and seed,
    # but pairing by index would silently mis-align if either skipped a probe.
    by_q = {r["question"]: r for r in db["records"]}
    pairs = [(r, by_q[r["question"]]) for r in da["records"] if r["question"] in by_q]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{name_a}  vs  {name_b}")
    print(f"paired on question text: {len(pairs)} of "
          f"{len(da['records'])}/{len(db['records'])}\n")

    rng = random.Random(args.seed)
    results = []
    global STAMP, PARTIAL
    STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    PARTIAL = OUT / f"{STAMP}_{args.tag}.partial.json"
    for i, (ra, rb) in enumerate(pairs, 1):
        # Randomise the slot so position bias cannot align with an arm.
        a_is_first = rng.random() < 0.5
        first, second = (ra, rb) if a_is_first else (rb, ra)
        v = judge_one(ra["question"], _full_source(ra),
                      first.get("answer", ""), second.get("answer", ""))
        # Translate the blind slot back to the arm.
        winner = v["verdict"]
        if winner in ("A", "B"):
            picked_first = winner == "A"
            arm = name_a if (picked_first == a_is_first) else name_b
        elif winner == "TIE":
            arm = "TIE"
        else:
            # ⚑ An ERROR is not a tie. Folding it into TIE would let a judge
            # that failed on every probe report a clean draw.
            arm = "ERROR"
        results.append({"question": ra["question"],
                        "probe_type": ra.get("probe_type", "untyped"),
                        "winner": arm, "reason": v["reason"],
                        "note": v.get("note", ""), "a_was_first": a_is_first})
        print(f"  {i}/{len(pairs)}  {ra.get('probe_type','?'):18s} "
              f"{arm:26s} {v['reason']}", flush=True)
        # ⚑ Written after EVERY probe, not at the end. This run can be killed by
        # a usage limit at any point, and a partial file with real verdicts is
        # worth far more than a complete file that never got written.
        OUT.mkdir(parents=True, exist_ok=True)
        PARTIAL.write_text(json.dumps(
            {"utc": STAMP, "arm_a": name_a, "arm_b": name_b,
             "complete": False, "judged": len(results),
             "of": len(pairs), "results": results}, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = OUT / f"{stamp}_{args.tag}.json"
    path.write_text(json.dumps({"utc": stamp, "arm_a": name_a, "arm_b": name_b,
                                "results": results}, indent=2))

    print("\n" + "=" * 66)
    print("VERDICT BY PROBE TYPE")
    print("=" * 66)
    by_type = defaultdict(Counter)
    for r in results:
        by_type[r["probe_type"]][r["winner"]] += 1
    print(f"{'type':20s} {name_a[:14]:>14s} {name_b[:14]:>14s} {'TIE':>6s} {'ERR':>5s}")
    for t in sorted(by_type):
        c = by_type[t]
        print(f"{t:20s} {c[name_a]:>14d} {c[name_b]:>14d} "
              f"{c['TIE']:>6d} {c['ERROR']:>5d}")
    tot = Counter(r["winner"] for r in results)
    print(f"{'TOTAL':20s} {tot[name_a]:>14d} {tot[name_b]:>14d} "
          f"{tot['TIE']:>6d} {tot['ERROR']:>5d}")

    print("\nREASONS  (both_failed is the one that matters)")
    for reason, n in Counter(r["reason"] for r in results).most_common():
        flag = "   <-- neither arm answered these" if reason == "both_failed" else ""
        print(f"  {reason:20s} {n:4d}{flag}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
