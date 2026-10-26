"""Z2 S11: grading — absolute grades, 1-5 scores, flags, invented-content checks,
blind ranking, and no-memory controls (spec D11).

Muse grades; the condition is never disclosed. Per answered context:
- absolute grade (correct / partial / incorrect / uncertain) and the anchored 1-5
  score against the occurrence's required items (5 all items and qualifiers, no
  errors; 4 one minor omission; 3 partial; 2 mostly missing or stale; 1 wrong or
  invented), per-item coverage (item coverage for cumulative/aggregation), and flags:
  used a superseded value, abstained (right when the occurrence expects abstention);
- every claim the grader finds unsupported by the reference is checked further (the
  maintainer's rules, 2026-10-18): stated by a ledger entry of ANY status → not
  invented (an `unconfirmed` entry only → "unconfirmed source"); else searched in the
  originals up to the checkpoint (embedding top-k plus exact keywords) and read →
  found = "ledger miss" (counted against the ledger, not the system); not found =
  "confirmed invented";
- ranking: per occurrence, the distinct answers of all conditions, blind, order
  randomised, judged in two orders, ties allowed;
- controls (no reference): each condition's answer compared blind with recent-only's.
Incomplete answers are ungraded, never wrong. Identical (answer, reference) pairs are
graded once. Every call is durable and resumable; an item whose call errors is
retried next attempt and recorded after three.

    uv run --no-sync python experiments/z2/grade.py --history B --units logs/z2-run-main/s9-B \
        --answers logs/z2-run-main/s10-B --questions logs/z2-questions-final \
        --ledger <ledger.json> --out logs/z2-run-main/s11-B
"""

import argparse
import hashlib
import json
import random
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-grade-1"
JUDGE = "muse"
OUTPUT_TOKENS = 8_000
MAX_ERRORS = 3
SEARCH_K = 8

GRADE_SYSTEM = """You grade an AI assistant's ANSWER to a user's QUESTION about their own long
conversation with the assistant. REQUIRED lists what a correct answer must contain (meaning, not
wording). SOURCES are the conversation passages REQUIRED comes from: use them for the exact values
(numbers, names, items, directions) — an answer that states a different value than SOURCES is
wrong on that item even if it sounds right. SUPERSEDED lists values that were true earlier but were
changed — presenting one of them as current is an error. EXPECT_ABSTENTION true means the
conversation never stated the answer and the right response says so.
Return only JSON:
{"items": [{"item": <index into REQUIRED>, "covered": true|false}],
 "grade": "correct|partial|incorrect|uncertain",
 "score": 1-5,   (5 every required item and qualifier, no errors; 4 one minor omission;
                  3 partial; 2 mostly missing or stale; 1 wrong or invented)
 "superseded_used": true|false, "abstained": true|false,
 "unsupported_claims": ["each specific claim in ANSWER about the conversation that REQUIRED does
                         not support, copied briefly"],
 "reason": "one sentence"}"""

CLAIM_LEDGER_SYSTEM = """CLAIM is a statement an AI assistant made about a conversation. ENTRIES are
facts recorded from that conversation. Which entries state CLAIM's content (meaning, not wording)?
Return only JSON: {"stated_by": ["id", ...]}"""

CLAIM_SOURCE_SYSTEM = """CLAIM is a statement an AI assistant made about a conversation with a user.
PASSAGES are original messages from that conversation. Does any passage state or directly support
CLAIM? Return only JSON: {"found": true|false, "passage": <index or null>, "reason": "..."}"""

RANK_SYSTEM = """Several ANSWERS respond to the same QUESTION about a user's long conversation with an
AI assistant. REQUIRED lists what a correct answer must contain; SOURCES are the passages it comes
from (use them for exact values). Rank the answers from best to worst on correctness and
completeness against REQUIRED (then clarity); equal answers share a rank.
Return only JSON: {"ranks": {"<answer label>": <rank, 1 = best>, ...}, "reason": "..."}"""

CONTROL_SYSTEM = """A user sent MESSAGE to an AI assistant; it needs no memory of earlier
conversations. Compare the two replies A and B for how well they answer MESSAGE.
Return only JSON: {"better": "A|B|tie", "reason": "..."}"""


def sources(reference):
    """The cited passages of a reference (turn, role, exact quote), for the judge."""
    return [dict(turn=c.get("turn"), role=c.get("role"), text=c.get("quote")) for c in reference.get("citations", [])
            if c.get("quote")]


def _sha(*parts):
    return hashlib.sha256("|".join(json.dumps(p, ensure_ascii=False, sort_keys=True) for p in parts).encode()).hexdigest()


class Errors:
    def __init__(self, path):
        self.path = path
        self.counts = json.loads(path.read_text()) if path.exists() else {}

    def failed(self, key, exc):
        entry = self.counts.setdefault(key, dict(n=0))
        entry.update(n=entry["n"] + 1, last=f"{type(exc).__name__}: {exc}"[:400])
        self.path.write_text(json.dumps(self.counts, indent=1))

    def given_up(self, key):
        return self.counts.get(key, {}).get("n", 0) >= MAX_ERRORS


def load(units_dir, answers_dir, questions_dir, history):
    """[(unit, answer row, occurrence or None)] for every answered context."""
    answers_dir = Path(answers_dir)
    index = json.loads((answers_dir / "index.json").read_text())["index"]
    rows = {}
    for line in (answers_dir / "answers.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["prompt_sha256"]] = r
    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads((Path(questions_dir) / history / "occurrences.json").read_text())}
    out = []
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        sha = index.get(f"{unit['condition']}|{unit['ask']['id']}")
        answer = rows.get(sha)
        if answer is None:
            continue
        out.append((unit, answer, occ.get(unit["ask"]["id"])))
    return out


def grade_answers(contexts, out, ask, *, say=print):
    """Absolute grade + 1-5 + flags for every complete answer to a question (4 in flight)."""
    from experiments.z2.bg_audit import _parallel
    errors = Errors(out / "grade-errors.json")
    results, pending, todo = {}, 0, {}
    keys = {}
    for unit, answer, occ in contexts:
        if unit["ask"]["kind"] != "question" or occ is None:
            continue
        key = f"{unit['condition']}|{unit['ask']['id']}"
        if answer["finish"] != "complete":
            results[key] = dict(status="ungraded", reason=f"answer {answer['finish']}")
            continue
        ref = occ["reference"]
        body = dict(QUESTION=unit["ask"]["question"], REQUIRED=ref.get("required", []), SOURCES=sources(ref),
                    SUPERSEDED=ref.get("superseded", []), EXPECT_ABSTENTION=bool(occ.get("expected_abstention")),
                    ANSWER=answer["text"])
        item_key = _sha(body)                      # identical (answer, reference): graded once
        keys[key] = (item_key, occ)
        if errors.given_up(item_key):
            results[key] = dict(status="error", reason=errors.counts[item_key]["last"])
        else:
            todo[item_key] = body
    values = {}
    for (item_key, body), value, exc in _parallel(list(todo.items()),
                                                  lambda kb: ask(GRADE_SYSTEM, json.dumps(kb[1], ensure_ascii=False))):
        if exc is not None:
            errors.failed(item_key, exc)
            pending += 1
        else:
            values[item_key] = value
    for key, (item_key, occ) in keys.items():
        if key in results or item_key not in values:
            continue
        value, ref = values[item_key], occ["reference"]
        if not value or value.get("grade") not in ("correct", "partial", "incorrect", "uncertain"):
            results[key] = dict(status="unparseable")
            continue
        covered = [i.get("item") for i in value.get("items") or [] if i.get("covered") is True]
        results[key] = dict(status="graded", grade=value["grade"], score=value.get("score"),
                            items_covered=sorted(set(c for c in covered if isinstance(c, int))),
                            items_total=len(ref.get("required", [])),
                            superseded_used=bool(value.get("superseded_used")),
                            abstained=bool(value.get("abstained")),
                            abstention_expected=bool(occ.get("expected_abstention")),
                            unsupported_claims=[c for c in value.get("unsupported_claims") or [] if isinstance(c, str)],
                            reason=value.get("reason"))
    return results, pending


def check_claims(results, contexts, ledger, originals, index, ask, out):
    """Each unsupported claim: ledger (any status) → originals → verdict, per the maintainer's rules."""
    import numpy as np
    from experiments.z2.questions import embed
    errors = Errors(out / "claims-errors.json")
    checkpoint = {f"{u['condition']}|{u['ask']['id']}": u["checkpoint"] for u, _, _ in contexts}
    rows, vectors = index
    vectors = np.asarray(vectors)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    verdicts, pending = {}, 0
    claims = sorted({(c, checkpoint[k]) for k, r in results.items() if r.get("status") == "graded"
                     for c in r["unsupported_claims"]})
    claim_vecs = embed([c for c, _ in claims]) if claims else []
    for (claim, cp), qv in zip(claims, claim_vecs):
        key = _sha(claim, cp)
        if key in verdicts or errors.given_up(key):
            continue
        try:
            entries = [e for e in ledger if e["valid_from_turn"] <= cp]
            near = sorted(entries, key=lambda e: -_overlap(claim, e["value"]))[:SEARCH_K]
            alias = {f"E{i}": e for i, e in enumerate(near, 1)}
            stated = (ask(CLAIM_LEDGER_SYSTEM, json.dumps(dict(CLAIM=claim, ENTRIES=[dict(id=a, text=e["value"])
                                                                              for a, e in alias.items()]),
                                                     ensure_ascii=False)) or {}).get("stated_by") or []
            hits = [alias[a] for a in stated if a in alias]
            if any(e.get("usable_for_questions") for e in hits):
                verdicts[key] = "in_ledger"
                continue
            if hits:
                verdicts[key] = "unconfirmed_source"
                continue
            q = np.asarray(qv) / (np.linalg.norm(qv) + 1e-12)
            within = [i for i, (turn, _, _) in enumerate(rows) if turn <= cp]
            by_vec = sorted(within, key=lambda i: -float(vectors[i] @ q))[:SEARCH_K]
            words = [w for w in re.findall(r"[A-Za-z0-9][\w'-]{3,}", claim)][:6]
            by_word = [i for i in within if words and sum(w.lower() in rows[i][2].lower() for w in words) >= min(3, len(words))][:SEARCH_K]
            shown = list(dict.fromkeys(by_vec + by_word))
            passages = [dict(index=n, turn=rows[i][0], role=rows[i][1], text=rows[i][2][:3000])
                        for n, i in enumerate(shown)]
            found = (ask(CLAIM_SOURCE_SYSTEM, json.dumps(dict(CLAIM=claim, PASSAGES=passages), ensure_ascii=False))
                     or {}).get("found") is True
            verdicts[key] = "ledger_miss" if found else "confirmed_invented"
        except Exception as exc:
            errors.failed(key, exc)
            pending += 1
    for k, r in results.items():
        if r.get("status") == "graded":
            r["claims"] = {c: verdicts.get(_sha(c, checkpoint[k]), "pending") for c in r["unsupported_claims"]}
            r["invented"] = sum(v == "confirmed_invented" for v in r["claims"].values())
    return verdicts, pending


def _overlap(a, b):
    wa, wb = set(re.findall(r"\w{4,}", a.lower())), set(re.findall(r"\w{4,}", b.lower()))
    return len(wa & wb) / (len(wa) or 1)


# Ranked together (v2 ranked four answers); ablation conditions are compared through
# their absolute scores, since one prompt cannot rank dozens of answers well.
RANKED = ("full_default", "full", "warm_vector", "bare_vector", "recent_only")


def rank(contexts, ask, out, seed_key="rank", ranked=RANKED):
    """Blind ranking of the distinct answers per question occurrence, in two orders."""
    errors = Errors(out / "rank-errors.json")
    by_ask = {}
    for unit, answer, occ in contexts:
        if (unit["ask"]["kind"] == "question" and occ is not None and answer["finish"] == "complete"
                and unit["condition"] in ranked):
            by_ask.setdefault(unit["ask"]["id"], (unit, occ, {}))[2][unit["condition"]] = answer["text"]
    ranks, pending = {}, 0
    for ask_id, (unit, occ, texts) in sorted(by_ask.items()):
        distinct = sorted(set(texts.values()))
        if len(distinct) < 2:
            ranks[ask_id] = dict(status="single_answer", conditions=texts and {c: 1 for c in texts})
            continue
        order = sorted(distinct, key=lambda t: _sha(seed_key, ask_id, t))
        orders = [order, list(reversed(order))]
        got = []
        for o in orders:
            labels = {chr(65 + i): t for i, t in enumerate(o)}
            body = dict(QUESTION=unit["ask"]["question"], REQUIRED=occ["reference"].get("required", []),
                        SOURCES=sources(occ["reference"]), ANSWERS=[dict(label=l, text=t) for l, t in labels.items()])
            key = _sha(body)
            if errors.given_up(key):
                break
            try:
                value = ask(RANK_SYSTEM, json.dumps(body, ensure_ascii=False)) or {}
            except Exception as exc:
                errors.failed(key, exc)
                pending += 1
                break
            r = value.get("ranks") or {}
            if set(r) != set(labels):
                break
            got.append({labels[l]: r[l] for l in labels})
        if len(got) != 2:
            ranks[ask_id] = dict(status="unranked")
            continue
        mean = {t: (got[0][t] + got[1][t]) / 2 for t in distinct}
        ranks[ask_id] = dict(status="ranked", order_agreement=got[0] == got[1],
                             conditions={c: mean[t] for c, t in texts.items()})
    return ranks, pending


def controls(contexts, ask, out, baseline="recent_only"):
    """Each condition's control reply vs recent-only's, blind, both orders."""
    errors = Errors(out / "control-errors.json")
    by_ask = {}
    for unit, answer, _ in contexts:
        if unit["ask"]["kind"] == "control" and answer["finish"] == "complete":
            by_ask.setdefault(unit["ask"]["id"], (unit, {}))[1][unit["condition"]] = answer["text"]
    out_rows, pending = {}, 0
    for ask_id, (unit, texts) in sorted(by_ask.items()):
        base = texts.get(baseline)
        for cond, text in texts.items():
            if cond == baseline or base is None:
                continue
            if text == base:
                out_rows[f"{cond}|{ask_id}"] = "same_reply"
                continue
            votes = []
            for a, b, flip in ((text, base, False), (base, text, True)):
                body = dict(MESSAGE=unit["ask"]["question"], A=a, B=b)
                key = _sha(body)
                if errors.given_up(key):
                    break
                try:
                    v = (ask(CONTROL_SYSTEM, json.dumps(body, ensure_ascii=False)) or {}).get("better")
                except Exception as exc:
                    errors.failed(key, exc)
                    pending += 1
                    break
                votes.append({"A": "B", "B": "A"}.get(v, v) if flip else v)
            out_rows[f"{cond}|{ask_id}"] = (votes[0] if len(votes) == 2 and votes[0] == votes[1]
                                            else "tie_or_split" if len(votes) == 2 else "pending")
    return out_rows, pending


SEMANTIC_SYSTEM = """REQUIRED lists what a correct answer to QUESTION must contain. FRAGMENTS are pieces
of memory a system put in its prompt (facts, notes, summaries, excerpts), each drawn from the
conversation turns that hold the answer. For each required item, list the fragment ids that carry it
(meaning, not wording). Also list fragment ids that state something wrong about it (a contradicted or
stale value).
Return only JSON: {"carried": {"<item index>": ["F1", ...]}, "wrong": ["F2", ...]}"""
SEMANTIC_CONDITIONS = ("full_default", "full", "warm_vector", "bare_vector")   # bare_vector: precision, D6e


def semantic_credit(contexts, ask, out, conditions=SEMANTIC_CONDITIONS):
    """Spec D11 retrieval diagnostic: of the prompt fragments citing a gold turn, which
    carry each required item and which are wrong — so compressed memory gets credit."""
    from experiments.z2.bg_audit import _parallel
    errors = Errors(out / "semantic-errors.json")
    todo, result, pending = [], {}, 0
    for unit, _answer, occ in contexts:
        if unit["condition"] not in conditions or unit["ask"]["kind"] != "question" or occ is None:
            continue
        gold = {c.get("turn") for c in occ["reference"].get("citations", [])}
        cited = [f for f in unit["stage"].get("selected") or [] if gold & set(f.get("source_turns") or [])]
        key = f"{unit['condition']}|{unit['ask']['id']}"
        total = len(unit["stage"].get("selected") or [])
        if not cited:
            result[key] = dict(gold_fragments=0, carried={}, wrong=0, items=len(occ["reference"].get("required", [])),
                               fragments_total=total, fragments_carrying=0)
            continue
        alias = {f"F{n}": f for n, f in enumerate(cited, 1)}
        body = dict(QUESTION=unit["ask"]["question"], REQUIRED=occ["reference"].get("required", []),
                    FRAGMENTS=[dict(id=a, kind=f.get("leg") or f.get("type"), text=f.get("text")) for a, f in alias.items()])
        if errors.given_up(_sha(body)):
            result[key] = dict(status="unjudged")
        else:
            todo.append((key, body, alias, occ, total))
    for (key, body, alias, occ, total), value, exc in _parallel(todo, lambda job: ask(SEMANTIC_SYSTEM,
                                                                               json.dumps(job[1], ensure_ascii=False))):
        if exc is not None:
            errors.failed(_sha(body), exc)
            pending += 1
            continue
        carried_ids = {str(i): [a for a in ids if a in alias] for i, ids in ((value or {}).get("carried") or {}).items()}
        carried = {i: [alias[a].get("leg") or alias[a].get("type") for a in ids] for i, ids in carried_ids.items()}
        result[key] = dict(gold_fragments=len(alias), items=len(occ["reference"].get("required", [])),
                           items_carried=sum(1 for v in carried.values() if v), carried_by_kind=carried,
                           wrong=len([a for a in (value or {}).get("wrong") or [] if a in alias]),
                           fragments_total=total, fragments_carrying=len({a for ids in carried_ids.values() for a in ids}))
    return result, pending


def run(history, units, answers, questions, ledger_path, out, *, ask, changes=None, audit=None, replay=None):
    from experiments.z2 import bg_audit
    from experiments.z2.ledger import load_history
    from experiments.z2.questions import originals_index
    out.mkdir(parents=True, exist_ok=True)
    contexts = load(units, answers, questions, history)
    graded, p1 = grade_answers(contexts, out, ask)
    ledger = bg_audit.load_ledger(ledger_path, changes)
    originals = load_history(history)
    verdicts, p2 = check_claims(graded, contexts, ledger, originals,
                                originals_index(originals, cache=out / "originals-index.npz"), ask, out)
    ranks, p3 = rank(contexts, ask, out)
    ctl, p4 = controls(contexts, ask, out)
    semantic, p5 = semantic_credit(contexts, ask, out)
    pending = p1 + p2 + p3 + p4 + p5
    (out / "grades.json").write_text(json.dumps(dict(policy=POLICY, history=history, graded=graded,
                                                     claims=verdicts, ranks=ranks, controls=ctl,
                                                     semantic_credit=semantic),
                                                ensure_ascii=False, indent=1) + "\n")
    if not pending:
        from experiments.z2 import report
        grades = json.loads((out / "grades.json").read_text())
        extras = report.scoring_inputs(history, questions, answers, audit=audit, replay=replay)
        (out / "report.json").write_text(json.dumps(report.build(history, report.unit_index(units), grades, extras),
                                                    indent=1, ensure_ascii=False) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending_errors=pending, finished=not pending)) + "\n")
    return pending


def judge_caller(calls, generate, meta, history):
    from experiments.z2.questions import ask_object
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}"))
    return lambda system, prompt: ask_object(calls, JUDGE, meta, generate, system, prompt, session,
                                             max_output_tokens=OUTPUT_TOKENS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--audit", type=Path, help="S8's bg-audit.json (fidelity audit)")
    parser.add_argument("--replay", type=Path, help="S7's progress.jsonl (write cost per turn)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades are private: keep them under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2 import ledger_merge as M
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    args.out.mkdir(parents=True, exist_ok=True)
    ask = judge_caller(M.Calls(args.out / "grade-raw.jsonl"), generator(JUDGE).generate, metadata(JUDGE), args.history)
    pending = run(args.history, args.units, args.answers, args.questions, args.ledger, args.out, ask=ask,
                  changes=args.changes, audit=args.audit, replay=args.replay)
    print(json.dumps(dict(pending_errors=pending)))
    return 3 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
