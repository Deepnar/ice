"""Z2 D4 audit: v2's forward walk as an independent check of the ledger's changes.

The ledger records changes when they are extracted (S1/S3b). This audit checks them
from the other side, as v2 did: for every occurrence, a verifier from a third model
family (Muse — neither ledger writer) reads the new turns between the evidence the
reference rests on and the moment the question is asked, and answers "does anything
here change this answer? cite it", under v2's temporal-anchor guard (a different,
later event is not a change to this one).

The turns shown are the original messages in that interval most similar to the
question and its reference (TOP_K by embedding), never the whole interval. Cited
quotes are checked against the originals. Then:
- walk finds a change, ledger has none in the interval  → review queue (ledger miss?)
- ledger has a change there, walk finds none           → review queue (walk miss?)
Agreement counts and every disagreement go to walk-audit.json; the queue is added to
the S5 review packet.

    uv run --no-sync python experiments/z2/walk_audit.py --history C \
        --ledger logs/z2-ledger-detail-<date>-r2 --out logs/z2-questions-final
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger as L
from experiments.z2 import ledger_merge as M
from experiments.z2 import questions as Q

POLICY = "ice-z2-walk-audit-1"
TOP_K = 6
VERIFIER = "muse"
WALK_SYSTEM = """A QUESTION about a long conversation between a user and an AI assistant has a
CURRENT_ANSWER that was true as of turn ANSWER_TURN. NEW_TURNS are later messages from the same
conversation (turn numbers given), up to the moment the question is asked.
Does anything in NEW_TURNS change the answer to QUESTION — a corrected or updated value, a
reversed decision, a stopped habit, a replaced item, or new information from the user that makes
CURRENT_ANSWER no longer the right answer (e.g. a fact the earlier advice relied on turns out
different, or the user decides on a different goal the question is about)?
Temporal-anchor rule: only a change to the SAME thing the question asks about counts. A different
event, a new item of the same kind that does not replace the old one, or a later topic is NOT a
change. Restating or repeating the answer is NOT a change; a question or request that leaves the
facts as they were is NOT a change.
If it changes, copy an exact quote (3-40 words, character for character) from the message that
changes it.
Return only JSON: {"changes": true|false, "turn": <int or null>, "role": "user|assistant|null",
"quote": "<exact quote or null>", "new_answer": "<the answer after the change, or null>"}"""


def interval(occurrence):
    """(start, end]: after the latest cited turn of the reference, up to the ask."""
    cited = [c["turn"] for c in occurrence["reference"].get("citations", [])]
    return (max(cited) if cited else 0), occurrence["checkpoint_turn"]


def ledger_changed(family, entries_by_id, start, end):
    """Did the ledger's computed reference change between turn *start* and *end*?"""
    before = Q.ground_truth(family, entries_by_id, start)["required"] if start else []
    after = Q.ground_truth(family, entries_by_id, end)["required"]
    return bool(before) and before != after


def candidates(index, query_vector, start, end, top_k=TOP_K):
    import numpy as np
    rows, vectors = index
    keep = [i for i, (turn, _, _) in enumerate(rows) if start < turn <= end]
    if not keep:
        return []
    v = np.asarray(vectors)[keep]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    q = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    order = np.argsort(-(v @ q))[:top_k]
    return sorted((rows[keep[i]] for i in order), key=lambda r: (r[0], r[1]))


def audit(history, families, occurrences, entries_by_id, originals, index, out, *, models, verifier=VERIFIER):
    calls = M.Calls(out / history / "walk-raw.jsonl")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}"))
    by_family = {f["id"]: f for f in families}
    queries = [o for o in occurrences if not o.get("expected_abstention") and o["family"] in by_family]
    vectors = Q.embed([o["question"] + " " + json.dumps(o["reference"].get("required", []), ensure_ascii=False)
                       for o in queries])
    results, queue = [], []
    for o, qv in zip(queries, vectors):
        fam = by_family[o["family"]]
        base = fam.get("base_family")
        if base:                                   # a messy variant shares its base's evidence
            continue
        start, end = interval(o)
        shown = candidates(index, qv, start, end)
        led = ledger_changed(fam, entries_by_id, start, end)
        if not shown:
            results.append(dict(occurrence=o, walk=False, ledger=led, shown=0, note="no new turns"))
            if led:
                queue.append(dict(kind="ledger_change_without_turns", family=o["family"],
                                  checkpoint=o["checkpoint_turn"]))
            continue
        prompt = json.dumps(dict(QUESTION=o["question"], CURRENT_ANSWER=o["reference"].get("required", []),
                                 ANSWER_TURN=start,
                                 NEW_TURNS=[dict(turn=t, role=r, text=text) for t, r, text in shown]),
                            ensure_ascii=False)
        try:
            reply = Q.ask_object(calls, verifier, models[verifier][1], models[verifier][0], WALK_SYSTEM, prompt,
                                 session, max_output_tokens=8000) or {}
        except Exception as exc:                   # this occurrence only; rerun resumes
            results.append(dict(occurrence=o, walk=None, ledger=led, error=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        walk = reply.get("changes") is True
        quote_ok = None
        if walk:
            turn, role, quote = reply.get("turn"), reply.get("role"), reply.get("quote") or ""
            quote_ok = bool(isinstance(turn, int) and start < turn <= end and role in ("user", "assistant")
                            and L.resolve(originals, turn, role, quote))
        row = dict(family=o["family"], checkpoint=o["checkpoint_turn"], type=o["type"], interval=[start, end],
                   walk=walk, ledger=led, quote_ok=quote_ok, reply=reply, shown=len(shown))
        results.append(row)
        if walk and not led:
            queue.append(dict(kind="walk_change_not_in_ledger", **row))
        elif led and not walk:
            queue.append(dict(kind="ledger_change_walk_missed", **row))
    judged = [r for r in results if r.get("walk") is not None]
    summary = dict(policy=POLICY, history=history, occurrences=len(results), judged=len(judged),
                   errors=len(results) - len(judged),
                   agree=sum(1 for r in judged if bool(r["walk"]) == bool(r["ledger"])),
                   walk_only=sum(1 for r in judged if r["walk"] and not r["ledger"]),
                   ledger_only=sum(1 for r in judged if r["ledger"] and not r["walk"]),
                   walk_quote_unverified=sum(1 for r in judged if r.get("walk") and r.get("quote_ok") is False))
    (out / history / "walk-audit.json").write_text(json.dumps(dict(summary, queue=queue, results=results),
                                                             ensure_ascii=False, indent=1, default=str) + "\n")
    return summary, queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="families")
    parser.add_argument("--verifier", default=VERIFIER, help="FINAL: mimo (Muse writes no ground truth, Z3 D13)")
    args = parser.parse_args()
    from scripts.z1.private_logs import under_logs
    for path in (args.ledger, args.out):
        if not under_logs(path):
            raise ValueError("ledger data is private: keep it under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    directory = args.out / args.history
    families = json.loads((directory / f"{args.name}.json").read_text())["kept"]
    occurrences = json.loads((directory / "occurrences.json").read_text())
    entries = Q.usable(args.history, args.ledger)
    changes = directory / "changes.json"
    if changes.exists():
        entries = Q.with_changes(entries, json.loads(changes.read_text()))
    originals = L.load_history(args.history)
    models = {args.verifier: (generator(args.verifier).generate, metadata(args.verifier))}
    summary, queue = audit(args.history, families, occurrences, {e["id"]: e for e in entries}, originals,
                           Q.passage_index(originals, cache=directory / "passages-index.npz"), args.out,
                           models=models, verifier=args.verifier)
    print(json.dumps(dict(summary, queue=len(queue))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
