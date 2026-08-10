#!/usr/bin/env python3
"""Z1: derive the retrieval ground truth the probes were always missing.

**The gap this fills.** `experiments/curation_files/` carries 681 probes, each
with a written `expected_answer` and an EMPTY `ground_truth_expected_fragments`
list — the field is written as `[]` by both generators
(`phase1_generate_curation_files.py:92`, `generate_mega_curation.py:67`) and
read by nothing. So the corpus can say *what the answer is* but not *which turn
holds it*, and every evaluation the project has ever run was therefore scored by
hand (`experiments/mature/manual_evaluate.py`: "User enters absolute scores").

Retrieval tuning cannot use a human judge — the whole point is to sweep settings
overnight. It needs a key: for each question, which turns are the right answer.
This script derives that key once, so scoring afterwards is exact, deterministic
and free.

**⚑ THE CIRCULARITY TRAP, and why the candidate stage is deliberately lexical.**
If the key were derived with ICE's own embedder and ranking, scoring ICE's
retrieval against it would grade the exam with its own answer sheet: the vector
leg would appear near-perfect by construction and tuning would optimise toward a
fixed point rather than toward relevance. Nothing downstream would reveal it.
So:

  * candidates are shortlisted by **IDF-weighted lexical overlap only** — the
    embedder is never consulted here;
  * the overlap is computed against the **expected_answer**, which retrieval
    never sees (retrieval maps question -> memory; this maps answer -> memory,
    a different direction with different information);
  * a local model confirms which candidates genuinely support the answer, so
    the key is not a pure keyword artifact either.

**The key is not trusted until it is checked.** `--sample N` prints a random N
for hand-verification. If the key is wrong, every tuning number afterwards is
wrong invisibly, which is the one failure this whole instrument exists to avoid.

Output goes to `experiments/curation_files/derived_gt.json` — inside the
gitignored corpus directory, because the probe text it echoes is personal.

Run:
  uv run python scripts/z1/derive_retrieval_gt.py --dry-run     # no model, lexical only
  uv run python scripts/z1/derive_retrieval_gt.py
  uv run python scripts/z1/derive_retrieval_gt.py --sample 30   # hand-verification set
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CURATION = Path("experiments/curation_files")
MATURE_PROBES = Path("experiments/mature/intermediates/generated_probes.json")
OUT = CURATION / "derived_gt.json"

# The Z1 tuning corpus, chosen 2026-08-10 by a classifier variety sweep over all
# 19 curated conversations (topic+intent label entropy, measured with the LIVE
# v2 checkpoint — v1 and v2 rank variety almost oppositely, so this selection is
# pinned to ice_classifier_v4_schema2.pt).
#   ecc64aab  best variety of the four MATURE conversations (5.78), taken at a
#             low split so the store stays ~145 turns instead of 1000+.
#   cca73c87  Creative_&_Media-led, near-disjoint from ecc64aab's Business lead.
#   355a5709  most probes of any conversation (33).
# Each is the others' distractor in the store.
CORPUS = {
    "ecc64aab": 145,
    "cca73c87": None,   # widest available
    "355a5709": None,
}

_WORD = re.compile(r"[a-z0-9']+")
# Deliberately small: this is a candidate-shortlisting aid, not an intent
# lexicon, so it carries none of the G28 "decision keyed on wording" risk.
_STOP = frozenset("""
a an the and or but if then than that this these those of to in on at for with
from by as is are was were be been being do does did doing have has had having
i me my we our you your he she it they them his her its their what which who
whom when where why how all any both each few more most other some such no nor
not only own same so too very can will just don should now about into over
under again further once here there
""".split())


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


def load_conversations() -> dict:
    """Widest history block per conversation in CORPUS, honouring a split cap."""
    picked: dict[str, dict] = {}
    for path in sorted(CURATION.glob("EC-*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cid = (d.get("original_conversation_id") or "")[:8]
        if cid not in CORPUS:
            continue
        hb = d.get("historical_context_block") or []
        cap = CORPUS[cid]
        if cap is not None and len(hb) > cap:
            continue
        cur = picked.get(cid)
        if cur is None or len(hb) > len(cur["turns"]):
            picked[cid] = {"conv": d.get("original_conversation_id"), "turns": hb,
                           "total": d.get("total_turns"), "file": path.name}
    return picked


def collect_probes(convs: dict) -> dict:
    """Probes per conversation, deduped by question text.

    Two sources: the hand-written curated probes, and the mature experiment's
    generated probes (which additionally carry `target_leg`). Mature probes are
    included only when their split fits inside the history we actually loaded —
    a probe about turn 200 is unanswerable from 145 turns and would score as a
    retrieval failure that is really a fixture bug.
    """
    out: dict[str, dict] = {c: {} for c in convs}

    for path in sorted(CURATION.glob("EC-*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cid = (d.get("original_conversation_id") or "")[:8]
        if cid not in out:
            continue
        if d.get("split_turn_index") and d["split_turn_index"] > len(convs[cid]["turns"]):
            continue
        for pr in d.get("evaluation_probes") or []:
            q = (pr.get("user_injected_prompt") or "").strip()
            a = (pr.get("expected_answer") or "").strip()
            if not q or not a or q in out[cid]:
                continue
            out[cid][q] = {"probe_id": pr.get("probe_id"), "question": q,
                           "expected_answer": a, "source": "curated",
                           "target_leg": None}

    if MATURE_PROBES.exists():
        mature = json.loads(MATURE_PROBES.read_text())
        for conv_id, splits in mature.items():
            cid = conv_id[:8]
            if cid not in out:
                continue
            horizon = len(convs[cid]["turns"])
            for split, probes in splits.items():
                if int(split) > horizon:
                    continue
                for pr in probes:
                    q = (pr.get("user_injected_prompt") or "").strip()
                    a = (pr.get("expected_answer") or "").strip()
                    if not q or not a or q in out[cid]:
                        continue
                    out[cid][q] = {"probe_id": pr.get("probe_id"), "question": q,
                                   "expected_answer": a, "source": "mature",
                                   "target_leg": pr.get("target_leg")}
    return out


def turn_text(t: dict) -> str:
    return f"{t.get('user_input') or ''}\n{t.get('ai_response') or ''}"


def build_idf(turns: list[dict]) -> dict:
    """IDF over the conversation's own turns.

    Conversation-local rather than corpus-wide on purpose: the discriminating
    words here are the ones rare *within this conversation*, and a global IDF
    would be dominated by whatever else happens to be in the corpus.
    """
    df: Counter = Counter()
    for t in turns:
        df.update(set(words(turn_text(t))))
    n = max(1, len(turns))
    return {w: math.log(1 + n / (1 + c)) for w, c in df.items()}


def shortlist(answer: str, turns: list[dict], idf: dict, k: int) -> list[tuple[float, dict]]:
    """Top-k turns by IDF-weighted containment of the ANSWER's vocabulary.

    Containment rather than Jaccard: a long turn that fully contains the answer's
    distinctive terms should not be penalised for also containing other things,
    and the turns here vary in length by two orders of magnitude.
    """
    a_terms = set(words(answer))
    if not a_terms:
        return []
    denom = sum(idf.get(w, 1.0) for w in a_terms) or 1.0
    scored = []
    for t in turns:
        overlap = a_terms & set(words(turn_text(t)))
        if not overlap:
            continue
        scored.append((sum(idf.get(w, 1.0) for w in overlap) / denom, t))
    scored.sort(key=lambda st: -st[0])
    return scored[:k]


_CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "supporting_turns": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["supporting_turns"],
}


def confirm(client, model, question: str, answer: str,
            candidates: list[tuple[float, dict]], char_cap: int) -> list[int]:
    """Ask the local model which candidate turns genuinely support the answer."""
    from src.workers.bg_client_factory import json_schema

    listing = "\n\n".join(
        f"[TURN {t.get('turn_number')}]\n{turn_text(t)[:char_cap]}"
        for _, t in candidates)
    nums = [t.get("turn_number") for _, t in candidates]
    prompt = (
        "You are building an evaluation key for a memory system.\n\n"
        f"QUESTION ASKED LATER:\n{question}\n\n"
        f"THE CORRECT ANSWER (written by a human from the full conversation):\n"
        f"{answer[:2500]}\n\n"
        f"CANDIDATE EARLIER TURNS:\n{listing}\n\n"
        "Return the turn numbers whose content actually supports the correct "
        "answer above — the turns a memory system would have to retrieve for "
        "someone to answer the question. Include a turn only if its own text "
        "carries the supporting information. Return an empty list if none do. "
        f"Choose only from: {nums}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format=json_schema("supporting_turns", _CONFIRM_SCHEMA),
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        got = [int(n) for n in (data.get("supporting_turns") or [])]
        return [n for n in got if n in nums]      # never invent a turn
    except Exception as exc:                       # loud, never silent
        print(f"    ! confirm failed: {type(exc).__name__}: {exc}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="lexical shortlist only, no model calls")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--char-cap", type=int, default=1200)
    ap.add_argument("--sample", type=int, default=0,
                    help="print N derived keys for hand-verification")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-only", action="store_true",
                    help="print the hand-verification sample from the SAVED key, "
                         "no model calls and no re-derivation")
    args = ap.parse_args()

    convs = load_conversations()
    if not convs:
        print("no conversations found — is experiments/curation_files/ present?")
        return 1
    probes = collect_probes(convs)

    if args.verify_only:
        if not OUT.exists():
            print(f"no saved key at {OUT} — run without --verify-only first")
            return 1
        saved = json.loads(OUT.read_text())
        random.seed(args.seed)
        withgold = [p for p in saved["probes"] if p["gold_turns"]]
        picks = random.sample(withgold, min(args.sample or 30, len(withgold)))
        by_turn = {cid: {t.get("turn_number"): t for t in m["turns"]}
                   for cid, m in convs.items()}
        print(f"{'='*70}\nHAND-VERIFICATION SAMPLE ({len(picks)} of {len(withgold)})")
        print("For each: does the GOLD TURN actually contain what the question asks for?")
        print(f"{'='*70}")
        for i, p in enumerate(picks, 1):
            print(f"\n--- {i}. [{p['conversation']} {p['probe_id']}] ---")
            print(f"Q: {p['question']}")
            print(f"Expected answer: {p['expected_answer'][:260].strip()}...")
            print(f"GOLD TURNS: {p['gold_turns']}")
            for tn in p["gold_turns"]:
                t = by_turn.get(p["conversation"], {}).get(tn)
                if t:
                    print(f"  turn {tn}: {turn_text(t)[:300].strip()}...")
            print("  VERDICT (yours): correct / wrong / partial")
        return 0

    print("Z1 tuning corpus")
    for cid, meta in sorted(convs.items()):
        print(f"  {cid}  turns={len(meta['turns']):4}  total={meta['total']}  "
              f"probes={len(probes[cid]):3}  ({meta['file']})")
    total_probes = sum(len(v) for v in probes.values())
    print(f"  → {len(convs)} conversations, {total_probes} unique probes\n")

    client = model = None
    if not args.dry_run:
        from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
        client, model = get_bg_client(), get_bg_model_name()
        print(f"confirming with {model}\n")

    out = {"corpus": {c: {"conversation_id": m["conv"], "turns": len(m["turns"]),
                          "file": m["file"]} for c, m in convs.items()},
           "probes": []}
    stats = Counter()

    for cid, meta in sorted(convs.items()):
        idf = build_idf(meta["turns"])
        items = list(probes[cid].values())
        print(f"── {cid}: {len(items)} probes")
        for i, pr in enumerate(items, 1):
            cands = shortlist(pr["expected_answer"], meta["turns"], idf, args.top_k)
            if not cands:
                stats["no_candidates"] += 1
                gold: list[int] = []
            elif args.dry_run:
                gold = [t.get("turn_number") for _, t in cands[:2]]
            else:
                gold = confirm(client, model, pr["question"],
                               pr["expected_answer"], cands, args.char_cap)
            stats["with_gold" if gold else "no_gold"] += 1
            stats["gold_turns"] += len(gold)
            out["probes"].append({
                **pr,
                "conversation": cid,
                "conversation_id": meta["conv"],
                "gold_turns": sorted(gold),
                "candidates": [{"turn": t.get("turn_number"), "lex": round(s, 4)}
                               for s, t in cands],
            })
            if i % 10 == 0:
                print(f"    {i}/{len(items)}")

    n = len(out["probes"])
    with_gold = stats["with_gold"]
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}derived {n} probes")
    print(f"  with >=1 gold turn : {with_gold}/{n} ({100*with_gold//max(1,n)}%)")
    print(f"  no gold turn       : {stats['no_gold']}  "
          f"(unanswerable from this history, or the derivation missed it)")
    print(f"  no lexical candidate: {stats['no_candidates']}")
    print(f"  mean gold turns    : {stats['gold_turns']/max(1,with_gold):.2f}")

    if not args.dry_run:
        OUT.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {OUT}  (gitignored — it echoes personal probe text)")

    if args.sample:
        random.seed(args.seed)
        picks = random.sample([p for p in out["probes"] if p["gold_turns"]],
                              min(args.sample, with_gold))
        by_turn = {cid: {t.get("turn_number"): t for t in m["turns"]}
                   for cid, m in convs.items()}
        print(f"\n{'='*70}\nHAND-VERIFICATION SAMPLE ({len(picks)})\n{'='*70}")
        for p in picks:
            print(f"\n[{p['conversation']} {p['probe_id']}] {p['question']}")
            print(f"  expected: {p['expected_answer'][:200]}...")
            print(f"  GOLD TURNS: {p['gold_turns']}")
            for tn in p["gold_turns"]:
                t = by_turn[p["conversation"]].get(tn)
                if t:
                    print(f"    turn {tn}: {turn_text(t)[:220].strip()}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
