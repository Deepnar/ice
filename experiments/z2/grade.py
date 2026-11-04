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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# grade-2 (2026-10-29, Z2 FINDINGS F12): the grade is decided by REQUIRED alone — B's grade-1
# verdicts marked 65% of partials and 206 "incorrect" answers down although every required
# point was covered, for details the judge could not see (SOURCES showed only the cited
# quotes, without dates). Extra details go to unsupported_claims, which the claim check
# verifies against the whole conversation. grade-1 and grade-2 grades are never mixed.
POLICY = "ice-z2-grade-2"
JUDGE = "muse"
OUTPUT_TOKENS = 16_000        # was 8,000: one answer's claims checked together cut it (2026-10-28)
MAX_ERRORS = 3
SEARCH_K = 8
CLAIM_WORKERS = 4                  # claim checks side by side, like the answers (were one at a time)
# per call (one answer's claims, B: median 7, at most 23): each claim's best entries and passages
# taken in turn up to these, so every claim keeps its strongest evidence and no call balloons
CLAIM_ENTRIES, CLAIM_PASSAGES = 32, 36       # passages are 1,500 characters (were whole messages cut at 3,000)
# The claim check searches and shows the conversation as overlapping whole passages
# (questions.passage_index); a dated claim shows its minute's closest passages.
DATED_PASSAGES = 4

GRADE_SYSTEM = """You grade an AI assistant's ANSWER to a user's QUESTION about their own long
conversation with the assistant. REQUIRED lists what a correct answer must contain (meaning, not
wording). SOURCES are the conversation passages REQUIRED comes from: use them for the exact values
(numbers, names, items, directions) — an answer that states a different value than SOURCES is
wrong on that item even if it sounds right. SUPERSEDED lists values that were true earlier but were
changed — presenting one of them as current is an error. EXPECT_ABSTENTION true means the
conversation never stated the answer and the right response says so.
Grade against REQUIRED only:
- correct: every REQUIRED item is present with the values SOURCES give, and no SUPERSEDED value
  is presented as current (with EXPECT_ABSTENTION: the answer says it was never stated);
- partial: some REQUIRED items are present, or all are but one lacks a qualifier REQUIRED states;
- incorrect: no REQUIRED item is present, a value contradicts SOURCES, or a SUPERSEDED value is
  presented as current;
- uncertain: you cannot tell.
SOURCES show only the passages REQUIRED comes from, not the rest of the conversation. Details in
ANSWER beyond REQUIRED are therefore not errors by themselves and never lower the grade or score
unless they contradict SOURCES: list them in unsupported_claims, where they are checked against
the whole conversation separately. A date that matches a passage's date in SOURCES is supported.
Return only JSON:
{"items": [{"item": <index into REQUIRED>, "covered": true|false}],
 "grade": "correct|partial|incorrect|uncertain",
 "score": 1-5,   (5 every required item and qualifier; 4 one minor qualifier missing;
                  3 some items missing; 2 most items missing or one stale; 1 none, or contradicted)
 "superseded_used": true|false, "abstained": true|false,
 "unsupported_claims": ["each specific claim in ANSWER about the conversation that REQUIRED does
                         not support, copied briefly"],
 "reason": "one sentence"}"""

# One call per answer, all its claims together (maintainer, 2026-10-28: one or two calls per claim
# made B's 19,063 claims 19,000-38,000 calls); per claim, the same two questions as before.
# Each passage carries the date the assistant's memory showed for it (Z2 F27: without them, 60 of
# ICE's 63 dated claims marked invented on B named the exact minute of a real turn).
CLAIMS_SYSTEM = """CLAIMS are statements an AI assistant made about a conversation with a user. ENTRIES are
facts recorded from that conversation; PASSAGES are original messages from it, each with the date
the assistant's memory showed for it. For EACH claim: which entries state its content (meaning,
not wording), and does any passage state or directly support it? A date or time in a claim is
supported when a passage with that date states what the claim says happened then; a date that
matches no such passage is not. Return only JSON: {"claims": [{"id": "C1", "stated_by": ["E3", ...],
"found": true|false, "passage": <index or null>}, ...]} — one item for every claim."""

# A claim naming a turn's minute (ICE's own timestamp labels) is decided apart (run 1, 2026-11-03:
# with dated passages the batched check still got ~1 dated claim in 4 wrong, both ways — 17/24 true
# plants supported, 18/24 date-shifted twins rejected). Code decides whether a turn exists at that
# minute; the judge sees only that turn and is asked about content alone.
DATED_CLAIM_SYSTEM = """CLAIM is a statement an AI assistant made about a conversation with a user, naming the
moment something was said. PASSAGES are the messages recorded at that moment. Does any passage state
what CLAIM says was said or happened then (meaning, not wording; who said it must match)? Judge the
content only — the moment is already matched. Return only JSON: {"stated": true|false, "reason": "..."}"""
MINUTE = re.compile(r"(20\d\d-\d\d-\d\d)[T ](\d\d:\d\d)")

RANK_SYSTEM = """Several ANSWERS respond to the same QUESTION about a user's long conversation with an
AI assistant. REQUIRED lists what a correct answer must contain; SOURCES are the passages it comes
from (use them for exact values). Rank the answers from best to worst on correctness and
completeness against REQUIRED (then clarity); equal answers share a rank.
Return only JSON: {"ranks": {"<answer label>": <rank, 1 = best>, ...}, "reason": "..."}"""

# The assistant may remember earlier conversations the judge never sees (Z2 F27: 17 of full ICE's 21
# losses on B called true memory "fabricated"). The judge rates the reply, not the memory's truth.
CONTROL_SYSTEM = """A user sent MESSAGE to an AI assistant that remembers its earlier conversations with
this user; MESSAGE itself can be answered without that memory. Compare the two replies A and B for
how well they answer MESSAGE. You cannot see the earlier conversations, so never count a mention
of them as invented or false. Judge whether the reply serves MESSAGE: a mention of the user's past
counts for a reply when it makes the answer more useful to this user, and against it when it is
unrelated to MESSAGE, crowds out the answer, or lengthens it without adding to it.
Return only JSON: {"better": "A|B|tie", "reason": "..."}"""


def sources(reference, dates=None):
    """The cited passages of a reference (turn, role, date, exact quote), for the judge. The
    date is the one the system saw for that turn (the replay clock), so a dated answer can be
    checked instead of read as invented (F12)."""
    dates = dates or {}
    return [dict(turn=c.get("turn"), role=c.get("role"), **({"date": dates[c["turn"]]} if c.get("turn") in dates else {}),
                 text=c.get("quote")) for c in reference.get("citations", []) if c.get("quote")]


def turn_dates(history):
    """{turn: 'YYYY-MM-DD HH:MM UTC'} on the history's replay clock, in UTC as ICE's own fragment
    labels show it (C's clock is written at +08:00: slicing the text gave the judge a local time 8
    hours from the prompt's, 2026-10-29); {} with a warning when it has none."""
    from datetime import datetime, timezone
    from experiments.z2 import clock
    try:
        times = clock.table(history, clock.load_history(history))["turn_times"]
    except Exception as exc:                                  # noqa: BLE001
        print(f"WARNING: grade: no turn dates for {history} ({type(exc).__name__}: {exc}); "
              "SOURCES go to the judge undated", flush=True)
        return {}
    return {n: datetime.fromisoformat(t).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            for n, t in enumerate(times, 1)}


def _sha(*parts):
    return hashlib.sha256("|".join(json.dumps(p, ensure_ascii=False, sort_keys=True) for p in parts).encode()).hexdigest()


class Errors:
    def __init__(self, path):
        self.path = path
        self.counts = json.loads(path.read_text()) if path.exists() else {}

    def failed(self, key, exc):
        entry = self.counts.setdefault(key, dict(n=0))
        entry.update(n=entry["n"] + 1, last=f"{type(exc).__name__}: {exc}"[:400])
        tmp = self.path.with_suffix(".tmp")                     # a stop mid-write never leaves a broken file
        tmp.write_text(json.dumps(self.counts, indent=1))
        tmp.replace(self.path)

    def given_up(self, key):
        return self.counts.get(key, {}).get("n", 0) >= MAX_ERRORS


def jsonl_lines(path):
    """The complete lines of a durable JSONL file: a last line cut by a stop mid-write is skipped (warned),
    never fatal — the call it records is simply made again."""
    from experiments.z2.ledger_merge import complete_lines
    return complete_lines(path)


def load(units_dir, answers_dir, questions_dir, history, occurrences=None):
    """[(unit, answer row, occurrence or None)] for every answered context. *occurrences*
    (a file) replaces questions_dir/<history>/occurrences.json (FINAL synthetic sets)."""
    answers_dir = Path(answers_dir)
    index = json.loads((answers_dir / "index.json").read_text())["index"]
    rows = {}
    for line in jsonl_lines(answers_dir / "answers.jsonl"):
        if line.strip():
            r = json.loads(line)
            rows[r["prompt_sha256"]] = r
    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads(Path(occurrences or Path(questions_dir) / history / "occurrences.json").read_text())}
    out = []
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        sha = index.get(f"{unit['condition']}|{unit['ask']['id']}")
        answer = rows.get(sha)
        if answer is None:
            continue
        # grading reads only the selected fragments of a stage (context precision); the rest —
        # candidates per leg, ranked and budgeted lists, the prompt, ~900 KB a unit — is the
        # report's, which loads units on its own: kept, it took the grader to ~16 GB on B (2026-10-28)
        unit["stage"] = {"selected": (unit.get("stage") or {}).get("selected") or []}
        out.append((unit, answer, occ.get(unit["ask"]["id"])))
    return out


def grade_answers(contexts, out, ask, *, dates=None, say=print):
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
        body = dict(QUESTION=unit["ask"]["question"], REQUIRED=ref.get("required", []), SOURCES=sources(ref, dates),
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
                            reason=value.get("reason"), reference=ref.get("required", []))
    return results, pending


def check_claims(results, contexts, ledger, originals, index, ask, out, dates=None):
    """Each unsupported claim: ledger (any status) → originals → verdict, per the maintainer's rules —
    one judge call per answer (its claims at its checkpoint), answers side by side. *dates*
    ({turn: 'YYYY-MM-DD HH:MM UTC'}, turn_dates) label each passage, and a claim naming a turn's
    time is shown that turn."""
    import numpy as np
    from experiments.z2.questions import embed
    errors = Errors(out / "claims-errors.json")
    checkpoint = {f"{u['condition']}|{u['ask']['id']}": u["checkpoint"] for u, _, _ in contexts}
    rows, vectors = index
    vectors = np.asarray(vectors)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    # an answer's claims at its checkpoint; identical answers (one prompt in several conditions)
    # are one group
    groups = sorted({(checkpoint[k], tuple(sorted(set(r["unsupported_claims"]))))
                     for k, r in results.items() if r.get("status") == "graded" and r["unsupported_claims"]})
    texts =sorted({c for _, claims in groups for c in claims})
    vec = dict(zip(texts, claim_vectors(out, texts, embed))) if texts else {}
    lock = threading.Lock()

    row_words = [_words(text) for _, _, text in rows]

    def passages_for(claim, cp):
        """The claim's dated passages, then the best by shared words and the best by meaning in
        turn — Z3 2026-11-04: the word list took the first matches in conversation order on six
        words, and came after eight meaning matches, so the cap never reached it."""
        q = np.asarray(vec[claim]) / (np.linalg.norm(vec[claim]) + 1e-12)
        within = [i for i, (turn, _, _) in enumerate(rows) if turn <= cp]
        by_vec = sorted(within, key=lambda i: -float(vectors[i] @ q))[:SEARCH_K]
        mine = _words(claim)
        by_word = sorted((i for i in within if mine & row_words[i]),
                         key=lambda i: (-len(mine & row_words[i]), i))[:SEARCH_K] if mine else []
        return claim_dated(claim, within, lambda i: -float(vectors[i] @ q)) + in_turn([by_word, by_vec], 2 * SEARCH_K)

    rows_of_turn = {}
    for i, (turn, _, _) in enumerate(rows):
        rows_of_turn.setdefault(turn, []).append(i)
    minute_rows, day_rows = {}, {}
    for turn, label in (dates or {}).items():
        minute_rows.setdefault(label[:16], []).extend(rows_of_turn.get(turn, []))
        day_rows.setdefault(label[:10], set()).update(rows_of_turn.get(turn, []))
    # the real call count for the meter: one batched call per answer with an undated claim, plus
    # one per minute-dated claim with a message at its minute (one call per answer alone undercounted
    # B by ~950 — the meter sat at its plan for two hours, Z3 2026-11-04)
    def at_minute(claim, cp):
        return tuple(dict.fromkeys(i for d, h in MINUTE.findall(claim) for i in minute_rows.get(f"{d} {h}", [])
                                   if rows[i][0] <= cp))

    batched = sum(any(not (dates and MINUTE.search(c)) for c in claims) for _, claims in groups)
    timed = {(c, at_minute(c, cp)) for cp, claims in groups for c in claims if dates and MINUTE.search(c)}
    write_meter(out, claims=batched + sum(1 for _, at in timed if at), claims_estimated=False)

    def claim_dated(claim, within, distance):
        """The passages at a time the claim names: every message at its minute; on a day alone, the
        day's two messages nearest the claim."""
        allowed, picked = set(within), []
        for day, hour in re.findall(r"(20\d\d-\d\d-\d\d)(?:[T ](\d\d:\d\d))?", claim):
            if hour:
                picked += sorted(minute_rows.get(f"{day} {hour}", []), key=distance)[:DATED_PASSAGES]
            else:
                picked += sorted(day_rows.get(day, ()), key=distance)[:2]
        return [i for i in dict.fromkeys(picked) if i in allowed]

    def dated(claim, cp):
        """A minute-dated claim's verdict: no turn at that minute (by the checkpoint) is invented,
        decided here; otherwise the judge reads only the messages at that minute."""
        at = at_minute(claim, cp)
        if not at:
            return "confirmed_invented"
        q = np.asarray(vec[claim]) / (np.linalg.norm(vec[claim]) + 1e-12)
        at = sorted(at, key=lambda i: -float(vectors[i] @ q))[:DATED_PASSAGES]
        reply = ask(DATED_CLAIM_SYSTEM, json.dumps(dict(CLAIM=claim, PASSAGES=[
            dict(turn=rows[i][0], role=rows[i][1], date=dates[rows[i][0]], text=rows[i][2])
            for i in at]), ensure_ascii=False)) or {}
        if not isinstance(reply.get("stated"), bool):
            raise ValueError("no stated verdict for a dated claim")
        return "ledger_miss" if reply["stated"] else "confirmed_invented"

    def one(group):
        """{claim key: verdict} for one answer, or "pending" after a failed or incomplete call."""
        cp, all_claims = group
        key = _sha(cp, *all_claims)
        if errors.given_up(key):
            return {}
        try:
            decided = {}
            claims = [c for c in all_claims if not (dates and MINUTE.search(c))]
            for c in all_claims:
                if c not in claims:
                    decided[_sha(c, cp)] = dated(c, cp)
            if not claims:
                return decided
            entries = [e for e in ledger if e["valid_from_turn"] <= cp]
            near = in_turn([sorted(range(len(entries)), key=lambda i: -_overlap(c, entries[i]["value"]))[:SEARCH_K]
                            for c in claims], CLAIM_ENTRIES)
            alias = {f"E{n}": entries[i] for n, i in enumerate(near, 1)}
            shown = in_turn([passages_for(c, cp) for c in claims], CLAIM_PASSAGES)
            ids = {f"C{n}": c for n, c in enumerate(claims, 1)}
            reply = ask(CLAIMS_SYSTEM, json.dumps(dict(
                CLAIMS=[dict(id=i, text=c) for i, c in ids.items()],
                ENTRIES=[dict(id=a, text=e["value"]) for a, e in alias.items()],
                PASSAGES=[dict(index=n, turn=rows[i][0], role=rows[i][1],
                               **({"date": dates[rows[i][0]]} if rows[i][0] in (dates or {}) else {}),
                               text=rows[i][2]) for n, i in enumerate(shown)]),
                ensure_ascii=False)) or {}
            items = {str(item.get("id")): item for item in reply.get("claims") or [] if isinstance(item, dict)}
            if set(items) < set(ids):                 # every claim must be answered; else retried
                raise ValueError(f"verdicts for {len(set(items) & set(ids))} of {len(ids)} claims")
            for cid, claim in ids.items():
                hits = [alias[a] for a in items[cid].get("stated_by") or [] if a in alias]
                decided[_sha(claim, cp)] = ("in_ledger" if any(e.get("usable_for_questions") for e in hits)
                                        else "unconfirmed_source" if hits
                                        else "ledger_miss" if items[cid].get("found") is True else "confirmed_invented")
            return decided
        except Exception as exc:
            with lock:
                errors.failed(key, exc)
            return "pending"

    verdicts, pending = {}, 0
    # answers side by side; every call is durable, so a restart keeps what was checked
    with ThreadPoolExecutor(max_workers=CLAIM_WORKERS) as pool:
        for result in pool.map(one, groups):
            if result == "pending":
                pending += 1
            else:
                verdicts.update(result)
    for k, r in results.items():
        if r.get("status") == "graded":
            r["claims"] = {c: verdicts.get(_sha(c, checkpoint[k]), "pending") for c in r["unsupported_claims"]}
            r["invented"] = sum(v == "confirmed_invented" for v in r["claims"].values())
    return verdicts, pending


def in_turn(lists, cap):
    """Each list's first item, then each one's second, ... — distinct, at most *cap*."""
    picked = []
    for rank in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if rank < len(x) and x[rank] not in picked:
                picked.append(x[rank])
                if len(picked) == cap:
                    return picked
    return picked


def claim_vectors(out, texts, embed):
    """The claims' embeddings, kept in <out>/claim-vectors.npz for a restart (B's 19,063 claims
    took ~40 CPU minutes to embed)."""
    import numpy as np
    path, digest = out / "claim-vectors.npz", hashlib.sha256("\n".join(texts).encode()).hexdigest()
    if path.exists():
        saved = np.load(path)
        if str(saved["digest"]) == digest:
            return saved["vectors"]
    vectors = embed(texts)
    np.savez(path, digest=digest, vectors=vectors)
    return vectors


STOP_WORDS = set("""that this with from have were what when which your about there their they them then than
been also into just like more some such only very will would could should after before over under because while
where these those does doing being much many make made really thing things other each both same still even well
back here time said says user assistant""".split())


def _words(text):
    """A text's content words (four letters or more) and numbers, for the claim check's word search."""
    return {w for w in re.findall(r"[a-z][a-z'-]{3,}|\d[\d.,:]*", text.lower()) if w not in STOP_WORDS}


def _overlap(a, b):
    wa, wb = set(re.findall(r"\w{4,}", a.lower())), set(re.findall(r"\w{4,}", b.lower()))
    return len(wa & wb) / (len(wa) or 1)


# Ranked together (v2 ranked four answers); ablation conditions are compared through
# their absolute scores, since one prompt cannot rank dozens of answers well.
RANKED = ("full_default", "full", "warm_vector", "bare_vector", "recent_only")


def rank(contexts, ask, out, seed_key="rank", ranked=RANKED, dates=None):
    """Blind ranking of the distinct answers per question occurrence, in two orders; occurrences
    side by side (one at a time, B's 348 ranking calls were over an hour)."""
    errors, lock = Errors(out / "rank-errors.json"), threading.Lock()
    by_ask = {}
    for unit, answer, occ in contexts:
        if (unit["ask"]["kind"] == "question" and occ is not None and answer["finish"] == "complete"
                and unit["condition"] in ranked):
            by_ask.setdefault(unit["ask"]["id"], (unit, occ, {}))[2][unit["condition"]] = answer["text"]

    def one(item):
        """(ask id, its ranks, pending calls)."""
        ask_id, (unit, occ, texts) = item
        distinct = sorted(set(texts.values()))
        if len(distinct) < 2:
            return ask_id, dict(status="single_answer", conditions=texts and {c: 1 for c in texts}), 0
        order = sorted(distinct, key=lambda t: _sha(seed_key, ask_id, t))
        got = []
        for o in (order, list(reversed(order))):
            labels = {chr(65 + i): t for i, t in enumerate(o)}
            body = dict(QUESTION=unit["ask"]["question"], REQUIRED=occ["reference"].get("required", []),
                        SOURCES=sources(occ["reference"], dates), ANSWERS=[dict(label=l, text=t) for l, t in labels.items()])
            key = _sha(body)
            if errors.given_up(key):
                break
            try:
                value = ask(RANK_SYSTEM, json.dumps(body, ensure_ascii=False)) or {}
            except Exception as exc:
                with lock:
                    errors.failed(key, exc)
                return ask_id, dict(status="unranked"), 1
            r = value.get("ranks") or {}
            if set(r) != set(labels):
                break
            got.append({labels[l]: r[l] for l in labels})
        if len(got) != 2:
            return ask_id, dict(status="unranked"), 0
        mean = {t: (got[0][t] + got[1][t]) / 2 for t in distinct}
        return ask_id, dict(status="ranked", order_agreement=got[0] == got[1],
                            conditions={c: mean[t] for c, t in texts.items()}), 0

    ranks, pending = {}, 0
    with ThreadPoolExecutor(max_workers=CLAIM_WORKERS) as pool:
        for ask_id, value, failed in pool.map(one, sorted(by_ask.items())):
            ranks[ask_id] = value
            pending += failed
    return ranks, pending


def controls(contexts, ask, out, baseline="recent_only"):
    """Each condition's control reply vs recent-only's, blind, both orders; side by side (one at a
    time, B's 1,630 control calls were over six hours)."""
    errors, lock = Errors(out / "control-errors.json"), threading.Lock()
    by_ask = {}
    for unit, answer, _ in contexts:
        if unit["ask"]["kind"] == "control" and answer["finish"] == "complete":
            by_ask.setdefault(unit["ask"]["id"], (unit, {}))[1][unit["condition"]] = answer["text"]
    jobs = []
    for ask_id, (unit, texts) in sorted(by_ask.items()):
        base = texts.get(baseline)
        for cond, text in texts.items():
            if cond != baseline and base is not None:
                jobs.append((ask_id, unit, cond, text, base))

    def one(job):
        """(row key, verdict, pending calls)."""
        ask_id, unit, cond, text, base = job
        if text == base:
            return f"{cond}|{ask_id}", "same_reply", 0
        votes = []
        for a, b, flip in ((text, base, False), (base, text, True)):
            body = dict(MESSAGE=unit["ask"]["question"], A=a, B=b)
            key = _sha(body)
            if errors.given_up(key):
                break
            try:
                v = (ask(CONTROL_SYSTEM, json.dumps(body, ensure_ascii=False)) or {}).get("better")
            except Exception as exc:
                with lock:
                    errors.failed(key, exc)
                return f"{cond}|{ask_id}", "pending", 1
            votes.append({"A": "B", "B": "A"}.get(v, v) if flip else v)
        return f"{cond}|{ask_id}", (votes[0] if len(votes) == 2 and votes[0] == votes[1]
                                    else "tie_or_split" if len(votes) == 2 else "pending"), 0

    out_rows, pending = {}, 0
    with ThreadPoolExecutor(max_workers=CLAIM_WORKERS) as pool:
        for key, verdict, failed in pool.map(one, jobs):
            out_rows[key] = verdict
            pending += failed
    return out_rows, pending


SEMANTIC_SYSTEM = """REQUIRED lists what a correct answer to QUESTION must contain. FRAGMENTS are pieces
of memory a system put in its prompt (facts, notes, summaries, excerpts), each drawn from the
conversation turns that hold the answer. For each required item, list the fragment ids that carry it
(meaning, not wording). Also list fragment ids that state something wrong about it (a contradicted or
stale value).
Return only JSON: {"carried": {"<item index>": ["F1", ...]}, "wrong": ["F2", ...]}"""
SEMANTIC_CONDITIONS = ("full_default", "full", "warm_vector", "bare_vector", "r7_frozen")   # bare_vector: precision, D6e


def semantic_credit(contexts, ask, out, conditions=SEMANTIC_CONDITIONS):
    """Spec D11 retrieval diagnostic: of the prompt fragments citing a gold turn, which
    carry each required item and which are wrong — so compressed memory gets credit. Token
    counts beside fragment counts: context precision is a share of prompt tokens, so it does not
    depend on how a system cuts its memory into pieces (Z2 F27: plain search packs ~250 small
    chunks where ICE packs ~54)."""
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
        tokens = sum(int(f.get("tokens") or 0) for f in unit["stage"].get("selected") or [])
        if not cited:
            result[key] = dict(gold_fragments=0, carried={}, wrong=0, items=len(occ["reference"].get("required", [])),
                               fragments_total=total, fragments_carrying=0, tokens_total=tokens, tokens_carrying=0,
                               tokens_wrong=0)
            continue
        alias = {f"F{n}": f for n, f in enumerate(cited, 1)}
        body = dict(QUESTION=unit["ask"]["question"], REQUIRED=occ["reference"].get("required", []),
                    FRAGMENTS=[dict(id=a, kind=f.get("leg") or f.get("type"), text=f.get("text")) for a, f in alias.items()])
        if errors.given_up(_sha(body)):
            result[key] = dict(status="unjudged")
        else:
            todo.append((key, body, alias, occ, total, tokens))
    for (key, body, alias, occ, total, tokens), value, exc in _parallel(todo, lambda job: ask(SEMANTIC_SYSTEM,
                                                                               json.dumps(job[1], ensure_ascii=False))):
        if exc is not None:
            errors.failed(_sha(body), exc)
            pending += 1
            continue
        carried_ids = {str(i): [a for a in ids if a in alias] for i, ids in ((value or {}).get("carried") or {}).items()}
        carried = {i: [alias[a].get("leg") or alias[a].get("type") for a in ids] for i, ids in carried_ids.items()}
        carrying = {a for ids in carried_ids.values() for a in ids}
        wrong = {a for a in (value or {}).get("wrong") or [] if a in alias}
        result[key] = dict(gold_fragments=len(alias), items=len(occ["reference"].get("required", [])),
                           items_carried=sum(1 for v in carried.values() if v), carried_by_kind=carried,
                           wrong=len(wrong), fragments_total=total, fragments_carrying=len(carrying),
                           tokens_total=tokens, tokens_carrying=sum(int(alias[a].get("tokens") or 0) for a in carrying),
                           tokens_wrong=sum(int(alias[a].get("tokens") or 0) for a in wrong))
    return result, pending


CLAIM_SHARE = 0.73          # B, 2026-10-28: 3,120 of 4,300 graded answers had unsupported claims


def planned_calls(contexts, ranked=RANKED, semantic=SEMANTIC_CONDITIONS, baseline="recent_only", dates=None):
    """How many judge calls each phase makes, from the contexts alone (the runner's meter guessed
    contexts x 1.4 and B needed far more): grading one per distinct (answer, reference); claims
    estimated until grading is done (then check_claims writes the real count); ranking two orders
    per question with two or more distinct answers; controls two per condition that differs from
    the baseline; context precision one per unit with a fragment citing a gold turn."""
    graded, by_rank, by_control, precision = set(), {}, {}, 0
    for unit, answer, occ in contexts:
        kind, done = unit["ask"]["kind"], answer["finish"] == "complete"
        if kind == "question" and occ is not None and done:
            ref = occ["reference"]
            graded.add(_sha(dict(QUESTION=unit["ask"]["question"], REQUIRED=ref.get("required", []), SOURCES=sources(ref, dates),
                                 SUPERSEDED=ref.get("superseded", []), EXPECT_ABSTENTION=bool(occ.get("expected_abstention")),
                                 ANSWER=answer["text"])))
            if unit["condition"] in ranked:
                by_rank.setdefault(unit["ask"]["id"], set()).add(answer["text"])
        if kind == "control" and done:
            by_control.setdefault(unit["ask"]["id"], {})[unit["condition"]] = answer["text"]
        if kind == "question" and occ is not None and unit["condition"] in semantic:
            gold = {c.get("turn") for c in occ["reference"].get("citations", [])}
            precision += any(gold & set(f.get("source_turns") or []) for f in unit["stage"].get("selected") or [])
    controls_n = sum(2 for texts in by_control.values() for c, t in texts.items()
                     if c != baseline and texts.get(baseline) is not None and t != texts[baseline])
    return dict(grade=len(graded), claims=round(len(graded) * CLAIM_SHARE), claims_estimated=True,
                rank=sum(2 for answers in by_rank.values() if len(answers) >= 2), controls=controls_n, semantic=precision)


def write_meter(out, **update):
    """<out>/meter.json: each phase's judge calls, for run_z2's meter."""
    path = out / "meter.json"
    meter = json.loads(path.read_text()) if path.exists() else {}
    meter.update(update)
    path.write_text(json.dumps(meter, indent=1) + "\n")


def run(history, units, answers, questions, ledger_path, out, *, ask, changes=None, occurrences=None, audit=None,
        replay=None, ranking=True, verdicts_only=False, partial=False, primary=None):
    """Grade one history's answers. FINAL passes ranking=False (pre-registration r2 §4 drops the
    multi-answer ranking); the cross-family judge passes verdicts_only=True (the judge's verdict
    per answer, nothing else)."""
    from experiments.z2 import bg_audit
    from experiments.z2 import questions as Q
    from experiments.z2.ledger import load_history
    out.mkdir(parents=True, exist_ok=True)
    contexts = load(units, answers, questions, history, occurrences)
    dates = turn_dates(history)
    plan = planned_calls(contexts, dates=dates)
    if verdicts_only:                              # the cross-family judge: its verdicts only
        plan.update(claims=0, rank=0, controls=0, semantic=0, claims_estimated=False)
    elif not ranking:                              # FINAL: no multi-answer ranking (r2 §4)
        plan.update(rank=0)
    write_meter(out, **plan)
    graded, p1 = grade_answers(contexts, out, ask, dates=dates)
    if verdicts_only:
        (out / "grades.json").write_text(json.dumps(dict(policy=POLICY, history=history, graded=graded,
                                                         verdicts_only=True), ensure_ascii=False, indent=1) + "\n")
        (out / "status.json").write_text(json.dumps(dict(pending_errors=p1, finished=not p1 and not partial,
                                                          partial=partial)) + "\n")
        return p1
    ledger = bg_audit.load_ledger(ledger_path, changes)
    originals = load_history(history)
    verdicts, p2 = check_claims(graded, contexts, ledger, originals,
                                Q.passage_index(originals, cache=out / "claim-passages.npz"), ask, out, dates=dates)
    ranks, p3 = rank(contexts, ask, out, dates=dates) if ranking else ({}, 0)
    ctl, p4 = controls(contexts, ask, out)
    semantic, p5 = semantic_credit(contexts, ask, out)
    pending = p1 + p2 + p3 + p4 + p5
    (out / "grades.json").write_text(json.dumps(dict(policy=POLICY, history=history, graded=graded,
                                                     claims=verdicts, ranks=ranks, controls=ctl,
                                                     semantic_credit=semantic),
                                                ensure_ascii=False, indent=1) + "\n")
    if not pending and not partial:              # a streaming pass grades what is answered; no report yet
        from experiments.z2 import report
        grades = json.loads((out / "grades.json").read_text())
        extras = report.scoring_inputs(history, questions, answers, audit=audit, replay=replay, occurrences=occurrences)
        (out / "report.json").write_text(json.dumps(report.build(history, report.unit_index(units), grades, extras,
                                                                 primary=primary),
                                                    indent=1, ensure_ascii=False) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending_errors=pending, finished=not pending and not partial,
                                                     partial=partial)) + "\n")
    return pending


def waiting(generate, refresh=None, *, sleep=time.sleep, say=print):
    """*generate* that waits through the plan's limits and a rejected key, as the answerer does, instead of
    failing: an item's error count (MAX_ERRORS gives it up for good) must never be spent by a quota pause
    (FINAL streaming, 2026-10-31 — partial passes every 30 min would give items up during a long pause)."""
    from experiments.lme.cloud_provider import ProviderAccessError
    from experiments.z2.answers import AUTH_WAIT, QUOTA_WAIT
    from experiments.z2.profiles import transient
    state = {"generate": generate}

    def call(*args, **kwargs):
        while True:
            try:
                return state["generate"](*args, **kwargs)
            except ProviderAccessError as exc:
                if exc.kind != "authentication" and transient(exc):
                    raise                                 # provider load: the retry wrapper's job
                wait = AUTH_WAIT if exc.kind == "authentication" else QUOTA_WAIT
                say(f"WARNING: judge {exc.kind or 'quota'} ({exc.status_code}); waiting {wait} s, then retrying "
                    "the same call (a key swapped into .env is picked up)")
                sleep(wait)
                if refresh:
                    state["generate"] = refresh() or state["generate"]
    return call


def judge_caller(calls, generate, meta, history, role=JUDGE):
    from experiments.z2.questions import ask_object
    # Z2's judge keeps its original session id (cached calls stay valid); another judge its own.
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}" if role == JUDGE else f"{POLICY}|{history}|{role}"))
    return lambda system, prompt: ask_object(calls, role, meta, generate, system, prompt, session,
                                             max_output_tokens=OUTPUT_TOKENS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--questions", type=Path, help="Z2 questions root (<root>/<history>/occurrences.json)")
    parser.add_argument("--occurrences", type=Path, help="FINAL: an occurrence file instead of --questions")
    parser.add_argument("--judge", default=JUDGE, help="judge role (FINAL's second judge writes its own --out)")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--audit", type=Path, help="S8's bg-audit.json (fidelity audit)")
    parser.add_argument("--replay", type=Path, help="S7's progress.jsonl (write cost per turn)")
    parser.add_argument("--no-ranking", action="store_true", help="FINAL: no multi-answer ranking (r2 §4)")
    parser.add_argument("--primary", help="the condition the report tests (default full_default; R7: r7_frozen)")
    parser.add_argument("--verdicts-only", action="store_true", help="the cross-family judge: verdicts only")
    parser.add_argument("--partial", action="store_true",
                        help="FINAL streaming: grade what is answered so far; never marked finished, no report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades are private: keep them under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2 import ledger_merge as M
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    writer = M.one_writer(args.out)  # noqa: F841 — held until exit; before Calls reads the saved calls
    if not (args.questions or args.occurrences):
        parser.error("one of --questions / --occurrences is required")
    def refresh():
        load_selected_env()
        return generator(args.judge).generate
    ask = judge_caller(M.Calls(args.out / "grade-raw.jsonl"), waiting(generator(args.judge).generate, refresh),
                       metadata(args.judge), args.history, role=args.judge)
    pending = run(args.history, args.units, args.answers, args.questions, args.ledger, args.out, ask=ask,
                  changes=args.changes, occurrences=args.occurrences,
                  audit=args.audit, replay=args.replay, ranking=not args.no_ranking,
                  verdicts_only=args.verdicts_only, partial=args.partial, primary=args.primary)
    print(json.dumps(dict(pending_errors=pending)))
    return 3 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
