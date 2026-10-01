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
import hashlib
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path("experiments/curation_files/judgements")
UA = "ice-research/3.0"

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

SYSTEM_V3 = SYSTEM + """

For this v3 probe, a separately reviewed EXPECTED ANSWER is also provided.
Grade EACH answer against the question, expected answer, and original source:
  correct    answers the requested fact(s), with no contradiction to SOURCE
  partial    gives a supported subset, but misses a required part
  incorrect  wrong, contradicts SOURCE, or does not answer
  uncertain  the evidence shown is insufficient to decide
Paraphrases count. Do not require the exact words of EXPECTED ANSWER. A fact
absent from SOURCE is not automatically false; use uncertain if it matters.
Distinguish contradiction from missing evidence. If SOURCE never mentions a
requested private fact and answers assert different values for it, their truth
cannot be verified: grade uncertain, not incorrect. Use TIE / equivalent if
neither unverified answer is better supported. By contrast, SOURCE explicitly
saying no choice was made contradicts an answer claiming a choice was made.
When both grades are incorrect, use TIE / both_failed. When the grades differ
between correct, partial and incorrect, prefer the higher grade. Uncertain
means evidence is insufficient, not that the answer is known to have failed.
Keep the paired preference independent of answer length or polish. Return:
  {"verdict":"A|B|TIE", "reason":"<reason code>",
   "A_grade":"correct|partial|incorrect|uncertain",
   "B_grade":"correct|partial|incorrect|uncertain", "note":"<short reason>"}
Do not mark both answers correct merely because they agree with each other."""

GRADES = {"correct", "partial", "incorrect", "uncertain"}


def validate_verdict(value, *, absolute: bool):
    """A malformed judge response is an error, never a plausible tie."""
    def error(reason):
        return {"verdict": "ERROR", "reason": reason,
                "note": "Judge response failed schema/consistency validation"}
    if not isinstance(value, dict) or value.get("verdict") not in {"A", "B", "TIE"}:
        return error("bad_verdict")
    if absolute:
        if value.get("A_grade") not in GRADES or value.get("B_grade") not in GRADES:
            return error("bad_absolute_grade")
        if value.get("reason") not in REASONS:
            return error("bad_reason")
        a, b = value["A_grade"], value["B_grade"]
        failed = a == b == "incorrect"
        if ((failed and (value["verdict"] != "TIE" or value["reason"] != "both_failed"))
                or (value["reason"] == "both_failed" and not failed)):
            return error("inconsistent_failure_verdict")
        order = {"incorrect": 0, "partial": 1, "correct": 2}
        if a in order and b in order and a != b:
            winner = "A" if order[a] > order[b] else "B"
            if value["verdict"] != winner:
                return error("inconsistent_grade_preference")
    elif value.get("reason") not in REASONS:
        value = {**value, "reason": "equivalent"}
    return value


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
    """The COMPLETE gold turns from the answer record or seeded store.

    Current v3 answer files include the complete gold source. Older files kept
    only three 1,200-character prefixes; for those, rebuild it from the seeded
    store before judging. An unresolved older source is an error, not a prompt
    that silently shows the judge incomplete evidence.

    Keyed on (conversation, turn_number) via the seed manifest rather than on
    episodic ids, because ids are regenerated per seed and differ between arms
    while the corpus text is identical — so one loaded store serves both arms'
    records.
    """
    if rec.get("gold_source_complete"):
        source = rec.get("gold_turn_text")
        if not source:
            raise RuntimeError("Answer record declares complete gold but has none")
        return source
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
                  "old answer records cannot be judged without a complete source")
    conv = rec.get("conversation")
    turns = rec.get("gold_turns") or []
    parts = [_GOLD_IDX.get((conv, int(t))) for t in turns]
    got = [p for p in parts if p]
    if len(got) != len(turns):
        raise RuntimeError("Complete gold source unavailable for answer judgement")
    return "\n\n".join(got)


def judge_one(question, source, ans_a, ans_b, *, expected_answer=None,
              question_time=None, session_id=None, retries=3):
    key, base, model = (_env("PROBE_API_KEY"), _env("PROBE_API_BASE_URL"),
                        _env("PROBE_MODEL"))
    if not key or not base:
        raise SystemExit("PROBE_API_KEY / PROBE_API_BASE_URL missing from .env")
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {key}", "User-Agent": UA}
    if urllib.parse.urlsplit(base).hostname == "opencode.ai":
        headers["x-opencode-session"] = (session_id or
            "ice-z1-judge-" + hashlib.sha256(source.encode()).hexdigest()[:24])
    user = (f"QUESTION\n{question}\n\n"
            + (f"HISTORICAL QUESTION TIME\n{question_time}\n\n"
               if question_time is not None else "")
            + (f"EXPECTED ANSWER\n{expected_answer}\n\n"
               if expected_answer is not None else "")
            # ⚑ NO CAP. This was `source[:4000]`, on top of answer_probes
            # keeping only 3 gold turns at 1,200 chars each. Measured
            # 2026-08-20: the judge saw a MEDIAN 12.7% of the gold material,
            # and the per-type coverage predicted the tie rate monotonically
            # across all five types (summary_synthesis 6.3% coverage -> 92%
            # ties; episodic_lookup 37.5% -> 50%). It was ruling answers
            # unsupported because it had not been shown the support.
            # ⚑ G56: THE ANSWERS ARE NOT TRUNCATED EITHER, and they used to be.
            # The SOURCE cap above was removed on 2026-08-20 with the note that
            # a judge cannot rule on support it was never shown — and an
            # identical `[:2500]` on both answers was left in place directly
            # underneath it. The same argument applies with more force: the
            # SOURCE is evidence, the ANSWERS are the thing being judged.
            #
            # Measured across every recorded answer run, the cap was not an
            # edge case but the normal case: median answer 2,576-3,285 chars,
            # and on the paired codex ablation 80 of 104 pairs had BOTH answers
            # cut, 88 of 104 at least one. The system prompt asks the model to
            # answer "accurately, thoroughly and in deep detail", so the
            # instrument was amputating the conclusion of the behaviour it had
            # just requested — and doing it to whichever answer went deepest.
            + f"SOURCE\n{source}\n\n"
            f"ANSWER A\n{ans_a or '(empty)'}\n\n"
            f"ANSWER B\n{ans_b or '(empty)'}")
    body = {"model": model,
            # System first and unchanged, variable part last: prefix caching.
            "messages": [{"role": "system", "content":
                          SYSTEM_V3 if expected_answer is not None else SYSTEM},
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
                headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read())
            txt = (payload["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:                                  # noqa: BLE001
            note = str(exc)[:120]
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    provider_error = json.loads(exc.read()).get("error", {})
                    kind = provider_error.get("type") or provider_error.get("code")
                    note = f"HTTP {exc.code}" + (f" ({kind})" if kind else "")
                except (ValueError, AttributeError):
                    note = f"HTTP {exc.code}"
            time.sleep(2 * (attempt + 1))
            if attempt == retries - 1:
                return {"verdict": "ERROR", "reason": "api", "note": note}
            continue
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception:                                          # noqa: BLE001
            return {"verdict": "ERROR", "reason": "unparseable", "note": txt[:120]}
        return validate_verdict(d, absolute=expected_answer is not None)
    return {"verdict": "ERROR", "reason": "exhausted", "note": ""}


def probe_key(record):
    """A historical checkpoint is part of identity, even if wording repeats."""
    if record.get("probe_id") is not None:
        if record.get("split_turn") is None:
            raise ValueError("v3 answer is missing its historical cutoff")
        return ("v3", record["probe_id"], record["conversation"],
                record["split_turn"])
    return ("legacy", record.get("conversation"),
            tuple(record.get("gold_turns") or ()), record["question"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm 1 answers json")
    ap.add_argument("--b", required=True, help="arm 2 answers json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="judge")
    args = ap.parse_args()
    if args.limit < 0:
        raise ValueError("judge limit cannot be negative")

    a_bytes, b_bytes = Path(args.a).read_bytes(), Path(args.b).read_bytes()
    da, db = json.loads(a_bytes), json.loads(b_bytes)
    name_a, name_b = da.get("tag", "A"), db.get("tag", "B")
    is_v3 = da.get("version") == "v3" or db.get("version") == "v3"
    if is_v3:
        if (da.get("version") != db.get("version")
                or not da.get("complete") or not db.get("complete")
                or not da.get("trace_sha256")
                or da["trace_sha256"] != db.get("trace_sha256")
                or da.get("seed_clock_policy") != db.get("seed_clock_policy")
                or da.get("decoding") != db.get("decoding")
                or da.get("probe_ids") != db.get("probe_ids")
                or da.get("label_audit_sha256") != db.get("label_audit_sha256")):
            raise ValueError("v3 answer arms have different trace, labels, probes, or incomplete answers")
        for data in (da, db):
            declared = data.get("probe_ids") or []
            observed = [r.get("probe_id") for r in data["records"]]
            if (not declared or len(set(declared)) != len(declared)
                    or len(observed) != len(set(observed)) or set(observed) != set(declared)):
                raise ValueError("v3 answer arm does not contain every declared probe exactly once")
            if not data.get("label_audit_sha256") and not data.get("development_partial"):
                raise ValueError("v3 campaign has no reviewed label identity")
        if name_a == name_b or name_a in {"TIE", "ERROR"} or name_b in {"TIE", "ERROR"}:
            raise ValueError("v3 answer arms need distinct, unambiguous names")

    def index(records):
        keyed = {}
        for record in records:
            key = probe_key(record)
            if key in keyed:
                raise ValueError("Duplicate answer probe identity in one arm")
            keyed[key] = record
        return keyed

    arm_a, arm_b = index(da["records"]), index(db["records"])
    if set(arm_a) != set(arm_b):
        raise ValueError("Answer arms have different probe identities")
    if not arm_a:
        raise ValueError("no answer pairs to judge")
    pairs = [(record, arm_b[probe_key(record)]) for record in da["records"]]
    sources = []
    for left, right in pairs:
        if (left.get("error") or right.get("error") or not left.get("answer")
                or not right.get("answer")):
            raise ValueError("Answer arm contains a failed or empty answer")
        if (left.get("answer_model") != right.get("answer_model")
                or left.get("answer_profile") != right.get("answer_profile")):
            raise ValueError("Answer arms used different answering models")
        if (left.get("question") != right.get("question")
                or left.get("gold_turns") != right.get("gold_turns")
                or left.get("question_time") != right.get("question_time")
                or left.get("expected_answer") != right.get("expected_answer")):
            raise ValueError("Answer arms disagree on the question or validated label")
        if is_v3 and (not isinstance(left.get("expected_answer"), str)
                      or not left["expected_answer"].strip()):
            raise ValueError("v3 judged answer lacks its reviewed expected answer")
        source = _full_source(left)
        if source != _full_source(right):
            raise ValueError("Answer arms disagree on the complete gold source")
        sources.append(source)
    if args.limit:
        pairs, sources = pairs[:args.limit], sources[:args.limit]
    print(f"{name_a}  vs  {name_b}")
    print(f"paired on probe identity: {len(pairs)} of "
          f"{len(da['records'])}/{len(db['records'])}\n")

    rng = random.Random(args.seed)
    results = []
    judge_identity = {"judge_model": _env("PROBE_MODEL"),
                      "calibration_status": "no_human_real_pair_calibration" if is_v3 else "legacy_not_asserted",
                      "score_of_record": False,
                      "judge_prompt_version": "v3_expected_answer_absolute_and_paired"
                      if da.get("version") == "v3" else "legacy_paired",
                      "answer_file_sha256": {"a": hashlib.sha256(a_bytes).hexdigest(),
                                             "b": hashlib.sha256(b_bytes).hexdigest()},
                      "shuffle_seed": args.seed}
    global STAMP, PARTIAL
    STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    PARTIAL = OUT / f"{STAMP}_{args.tag}.partial.json"
    for i, ((ra, rb), source) in enumerate(zip(pairs, sources), 1):
        # Randomise the slot so position bias cannot align with an arm.
        a_is_first = rng.random() < 0.5
        first, second = (ra, rb) if a_is_first else (rb, ra)
        expected = ra.get("expected_answer")
        judge_kwargs = ({"expected_answer": expected, "question_time": ra.get("question_time"),
                         "session_id": f"ice-z1-{da['trace_sha256'][:24]}-{ra['conversation']}"}
                        if da.get("version") == "v3" else {})
        v = judge_one(ra["question"], source,
                      first.get("answer", ""), second.get("answer", ""),
                      **judge_kwargs)
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
        grades = ({"arm_a_grade": v.get("A_grade") if a_is_first else v.get("B_grade"),
                   "arm_b_grade": v.get("B_grade") if a_is_first else v.get("A_grade")}
                  if da.get("version") == "v3" else {})
        results.append({"probe_id": ra.get("probe_id"),
                        "conversation": ra.get("conversation"),
                        "split_turn": ra.get("split_turn"),
                        "question": ra["question"],
                        "probe_type": ra.get("probe_type", "untyped"),
                        "winner": arm, "reason": v["reason"],
                        "note": v.get("note", ""), "a_was_first": a_is_first,
                        "arm_a_prompt_tokens_est": ra.get("prompt_tokens"),
                        "arm_b_prompt_tokens_est": rb.get("prompt_tokens"),
                        "arm_a_selected_tokens_est": ra.get("selected_tokens"),
                        "arm_b_selected_tokens_est": rb.get("selected_tokens"),
                        **grades})
        print(f"  {i}/{len(pairs)}  {ra.get('probe_type','?'):18s} "
              f"{arm:26s} {v['reason']}", flush=True)
        # ⚑ Written after EVERY probe, not at the end. This run can be killed by
        # a usage limit at any point, and a partial file with real verdicts is
        # worth far more than a complete file that never got written.
        OUT.mkdir(parents=True, exist_ok=True)
        PARTIAL.write_text(json.dumps(
            {"utc": STAMP, "arm_a": name_a, "arm_b": name_b,
             **judge_identity,
             "seed_clock_policy": da.get("seed_clock_policy"),
             "complete": False, "development_partial": bool(args.limit or da.get("development_partial")),
             "trace_sha256": da.get("trace_sha256"), "judged": len(results),
             "of": len(pairs), "results": results}, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = OUT / f"{stamp}_{args.tag}.json"
    costs = [(r["arm_a_prompt_tokens_est"], r["arm_b_prompt_tokens_est"])
             for r in results if isinstance(r["arm_a_prompt_tokens_est"], (int, float))
             and isinstance(r["arm_b_prompt_tokens_est"], (int, float))]
    token_summary = ({"pairs_with_estimates": len(costs),
                      "arm_a_median_prompt_tokens_est": statistics.median(a for a, _ in costs),
                      "arm_b_median_prompt_tokens_est": statistics.median(b for _, b in costs),
                      "median_a_minus_b_tokens_est": statistics.median(a - b for a, b in costs),
                      "a_fewer_prompt_tokens_fraction": sum(a < b for a, b in costs) / len(costs)}
                     if costs else {"pairs_with_estimates": 0})
    absolute_by_type = {}
    if da.get("version") == "v3":
        for kind in sorted({r["probe_type"] for r in results}):
            group = [r for r in results if r["probe_type"] == kind]
            absolute_by_type[kind] = {
                "probes": len(group),
                "arm_a": dict(Counter(r["arm_a_grade"] for r in group)),
                "arm_b": dict(Counter(r["arm_b_grade"] for r in group)),
            }
    path.write_text(json.dumps({"utc": stamp, "arm_a": name_a, "arm_b": name_b,
                                **judge_identity,
                                "seed_clock_policy": da.get("seed_clock_policy"),
                                "trace_sha256": da.get("trace_sha256"),
                                "complete": not args.limit and all(r["winner"] != "ERROR" for r in results),
                                "development_partial": bool(args.limit or da.get("development_partial")),
                                "paired_prompt_cost": token_summary,
                                "absolute_by_type": absolute_by_type,
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
    if da.get("version") == "v3":
        for name, field in ((name_a, "arm_a_grade"), (name_b, "arm_b_grade")):
            print(f"{name} absolute grades: {dict(Counter(r[field] for r in results))}")
    print(f"\nwrote {path}")
    return 1 if any(r["winner"] == "ERROR" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
