#!/usr/bin/env python3
"""Z1: generate retrieval probes FROM real turns, with the answer key by construction.

**Why this exists, and why it is not the other script.** `derive_retrieval_gt.py`
runs the hard direction: given a written answer, work backwards to the turns that
support it. Hand-verification on 2026-08-11 measured what that costs — a lexical
shortlist plus a model confirmation still mis-grounded ~20% of claims, and worse,
**29 of the 91 curated probes turned out usable for tuning at all** (the rest ask
for synthesis, which no single-turn key can score). 29 probes gives a standard
error of ~0.093 against a keep-rule of +0.05. Unresolvable.

This script runs the EASY direction instead: **pick a real turn, then write a
question whose answer is in it.** The gold turn is not derived, inferred or
confirmed — it is *chosen first*, so it is correct by construction. No shortlist,
no circularity, no confirmation step to be wrong.

**⚑ WHAT THIS DELIBERATELY IS NOT: synthetic conversations.** The alternative was
to script facts and have a model write dialogue containing them. Rejected by the
user, correctly: it bets on the model writing realistic dialogue, embedding the
scripted facts where the script says, and hitting the focal points — three
unvalidated assumptions, expensive to discover wrong. Here the conversation is
**real, unmodified human text**; only the questions are generated, and a question
is a far smaller thing to get right than a conversation.

**The two checks that make a generated probe usable:**

  1. **Answerable from its own turn** — the model is told to quote the evidence,
     and a probe whose quote is not actually in the turn is dropped. Cheap
     protection against a question about something the model imagined.
  2. **NOT better answered by a different turn.** This is the load-bearing one
     and it catches two distinct failures with one test:
       * *ambiguity* — a generic question ("what did we decide?") that a dozen
         turns answer equally well, which would score retrieval as wrong for
         returning a perfectly good turn;
       * *superseded facts* — turn 10 says "I'm using Postgres", turn 50 says
         "I switched to SQLite". A probe from turn 10 has a stale answer, and
         retrieval returning turn 50 would be marked wrong for being right.
         We are not running LSREP, so nothing else tracks fact evolution.

Output: `experiments/curation_files/generated_probes.json` (gitignored — it
quotes personal conversation text).

Run:
  uv run python scripts/z1/generate_probes.py --limit 5      # smoke, one conv
  uv run python scripts/z1/generate_probes.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.derive_retrieval_gt import (  # noqa: E402
    CURATION, build_idf, load_conversations, turn_text, words,
)

OUT = CURATION / "generated_probes.json"

_SCHEMA = {
    "type": "object",
    "properties": {
        "probes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "evidence": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "evidence", "answer"],
            },
        }
    },
    "required": ["probes"],
}


def make_prompt(turn: dict, n: int) -> str:
    return (
        "Below is one turn from a real conversation between a person and an AI "
        "assistant.\n\n"
        f"--- TURN ---\n{turn_text(turn)[:5000]}\n--- END TURN ---\n\n"
        f"Write {n} questions this person might ask LATER in the conversation, "
        "whose answer is contained in this turn.\n\n"
        "Rules:\n"
        "1. Write the question the way THIS PERSON types — their register, their "
        "   informality, lowercase if that is how they write. Do not write like "
        "   a quiz or an exam question.\n"
        "2. NEVER refer to the turn itself. No 'according to the above', no "
        "   'in this turn', no 'as mentioned'. The person asking has forgotten "
        "   the details and is asking from memory.\n"
        "3. The question must be answerable from THIS turn specifically — not "
        "   from general knowledge, and not from any conversation.\n"
        "4. Ask about something concrete and stated: a fact, a decision, a name, "
        "   a number, a reason that is actually written here.\n"
        "5. `evidence` must be a VERBATIM span copied from the turn that "
        "   contains the answer. Copy it exactly; do not paraphrase.\n"
        "6. `answer` is a short direct answer.\n\n"
        "If the turn contains nothing concrete enough to ask about, return an "
        "empty list. A short filler turn should produce zero probes."
    )


def evidence_present(evidence: str, turn: dict, min_overlap: float) -> bool:
    """Is the quoted evidence really in the turn?

    Exact containment first (the instruction says verbatim), then a word-overlap
    fallback — models normalise whitespace and quotes even when told not to, and
    failing an otherwise-good probe on a smart-quote would be pedantry.
    """
    body = turn_text(turn)
    if evidence and evidence.strip().lower() in body.lower():
        return True
    ev = set(words(evidence))
    if not ev:
        return False
    return len(ev & set(words(body))) / len(ev) >= min_overlap


def better_elsewhere(question: str, answer: str, gold_turn: int,
                     turns: list, idf: dict) -> int | None:
    """Return a turn that answers *question* at least as well as the gold one.

    **⚑ SCORED ON THE QUESTION ALONE, and the fix is the whole point (G46).**
    This used to score `question | answer` on the reasoning that a question's own
    words are often generic while the thing asked about is not. That reasoning is
    about what a *reader* knows. **Retrieval never sees the answer** — it gets the
    question and nothing else.

    The consequence was not subtle: the answer's distinctive words live in the
    gold turn by construction, so including them handed the gold turn a score no
    rival could match, and the guard passed almost everything.

    **Replayed over the existing 592 probes, 2026-08-13 — the two rules side by
    side on the same set:** the old rule rejects **5 probes (0.8%)**, this one
    rejects **385 (65.0%)**. So **380 probes — 64% of the set now in use — are
    there only because the guard scored words retrieval never sees.** Each is a
    probe where retrieval can be marked wrong for returning an equally good
    answer, and the 0.508 recall measured over that set inherits it.

    So: score exactly what retrieval is given. A hit here means the probe is
    unusable — either several turns answer it, or a later turn supersedes it.

    *answer* is still accepted (callers pass it) and deliberately unused; it is
    kept in the signature so the reason for its absence stays visible here rather
    than becoming a silent omission somebody re-adds.
    """
    terms = set(words(question))
    if not terms:
        return None

    def score(t):
        return sum(idf.get(w, 1.0) for w in terms & set(words(turn_text(t))))

    by_num = {t.get("turn_number"): t for t in turns}
    gold = by_num.get(gold_turn)
    if gold is None:
        return None
    gold_score = score(gold)
    if gold_score <= 0:
        return gold_turn
    for t in turns:
        tn = t.get("turn_number")
        if tn == gold_turn:
            continue
        if score(t) >= gold_score:
            return tn
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-turn", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N turns per conv")
    ap.add_argument("--only", default=None)
    ap.add_argument("--min-evidence-overlap", type=float, default=0.6)
    args = ap.parse_args()

    from src.workers.bg_client_factory import (get_bg_client, get_bg_model_name,
                                               json_schema)

    convs = load_conversations()
    if args.only:
        convs = {k: v for k, v in convs.items() if k == args.only}
    if not convs:
        print("no conversations loaded")
        return 1

    client, model = get_bg_client(), get_bg_model_name()
    print(f"generating with {model}\n")

    out = {"model": model, "per_turn": args.per_turn, "probes": []}
    stats = Counter()

    for cid, meta in sorted(convs.items()):
        turns = meta["turns"]
        idf = build_idf(turns)
        todo = turns[:args.limit] if args.limit else turns
        print(f"── {cid}: {len(todo)} turns")
        for i, turn in enumerate(todo, 1):
            tn = turn.get("turn_number")
            if len(turn_text(turn).split()) < 40:
                stats["turn_too_short"] += 1
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": make_prompt(turn, args.per_turn)}],
                    temperature=0.3, max_tokens=1200,
                    response_format=json_schema("probes", _SCHEMA))
                got = (json.loads(resp.choices[0].message.content or "{}")
                       .get("probes") or [])
            except Exception as exc:
                print(f"    ! turn {tn}: {type(exc).__name__}: {exc}")
                stats["call_failed"] += 1
                continue

            for p in got:
                q = (p.get("question") or "").strip()
                ev = (p.get("evidence") or "").strip()
                ans = (p.get("answer") or "").strip()
                stats["generated"] += 1
                if not q or not ev:
                    stats["rejected_empty"] += 1
                    continue
                if not evidence_present(ev, turn, args.min_evidence_overlap):
                    stats["rejected_evidence_not_in_turn"] += 1
                    continue
                clash = better_elsewhere(q, ans, tn, turns, idf)
                if clash is not None:
                    stats["rejected_ambiguous_or_superseded"] += 1
                    continue
                stats["kept"] += 1
                out["probes"].append({
                    "conversation": cid, "conversation_id": meta["conv"],
                    "gold_turn": tn, "question": q, "answer": ans,
                    "evidence": ev,
                })
            if i % 20 == 0:
                print(f"    {i}/{len(todo)}  kept {stats['kept']}")

    OUT.write_text(json.dumps(out, indent=2))
    g = max(1, stats["generated"])
    print(f"\ngenerated {stats['generated']} · KEPT {stats['kept']} "
          f"({100*stats['kept']//g}%)")
    print(f"  rejected, evidence not in turn : {stats['rejected_evidence_not_in_turn']}")
    print(f"  rejected, ambiguous/superseded : {stats['rejected_ambiguous_or_superseded']}")
    print(f"  rejected, empty                : {stats['rejected_empty']}")
    print(f"  turns skipped as too short     : {stats['turn_too_short']}")
    print(f"  call failures                  : {stats['call_failed']}")
    per_conv = Counter(p["conversation"] for p in out["probes"])
    for c, n in sorted(per_conv.items()):
        print(f"  {c}: {n}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
