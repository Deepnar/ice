"""Z2 S3: question families from the verified ledger (spec Z2_EVALUATION D4).

S3a (this part, local): entries about the same thing are found by meaning, not by
subject name — the writers name subjects uniquely, so the same item rarely
repeats verbatim (B: 13, C: 11 change links captured). Embeddings of
"subject: value" give

- candidate change pairs: for each usable entry, its nearest EARLIER usable
  entries of the same speaker at cosine >= PAIR_FLOOR (S3b asks whether the
  newer one changes the older);
- threads: connected groups at cosine >= THREAD_FLOOR, for cumulative and
  aggregation questions.

Runs on CPU by default: a long local replay may own the GPU (TRAPS 129).

    uv run --no-sync python experiments/z2/questions.py threads --history B \
        --ledger logs/z2-ledger-2026-10-18-r1 --out logs/z2-questions-<date>-r1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger as L
from experiments.z2 import ledger_merge as M

POLICY = "ice-z2-questions-1"
PAIR_FLOOR, PAIR_K = 0.70, 5
THREAD_FLOOR = 0.78
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"


def usable(history, ledger_root):
    entries = json.loads((ledger_root / history / "ledger.json").read_text())["entries"]
    return [e for e in M.chain(entries) if e["usable_for_questions"]]


def default_device():
    """The GPU when there is one: B's 2,238 originals took 20+ minutes at full CPU
    (2026-10-18) against seconds on the GPU; the encoder needs ~1.2 GB."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


_MODELS = {}


def _model(device):
    from sentence_transformers import SentenceTransformer
    if device not in _MODELS:                      # once per process, not per call
        _MODELS[device] = SentenceTransformer(EMBEDDER, device=device)
    return _MODELS[device]


def embed(texts, device=None, batch_size=32):
    import numpy as np
    vectors = _model(device or default_device()).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                        convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def entry_text(e):
    return f"{e['subject']}: {e['value']}"


def candidate_pairs(entries, vectors, *, floor=PAIR_FLOOR, k=PAIR_K):
    """(older_index, newer_index, cosine) for same-speaker entries, older first."""
    import numpy as np
    sims = vectors @ vectors.T
    pairs = []
    for j, newer in enumerate(entries):
        earlier = [i for i, e in enumerate(entries)
                   if e["speaker"] == newer["speaker"] and e["valid_from_turn"] < newer["valid_from_turn"]]
        if not earlier:
            continue
        ranked = sorted(earlier, key=lambda i: -sims[i, j])[:k]
        pairs += [(i, j, round(float(sims[i, j]), 4)) for i in ranked if sims[i, j] >= floor]
    return pairs


def threads(entries, vectors, *, floor=THREAD_FLOOR):
    """Connected components of the cosine >= floor graph, as lists of indices
    ordered by turn; singletons dropped."""
    sims = vectors @ vectors.T
    parent = list(range(len(entries)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= floor:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [sorted(g, key=lambda i: entries[i]["valid_from_turn"]) for g in groups.values() if len(g) > 1]


def run_threads(history, ledger_root, out, device=None):
    import numpy as np
    entries = usable(history, ledger_root)
    vectors = embed([entry_text(e) for e in entries], device=device)
    pairs = candidate_pairs(entries, vectors)
    groups = threads(entries, vectors)
    directory = out / history
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "entry-vectors.npy", vectors)
    span = lambda g: entries[g[-1]]["valid_from_turn"] - entries[g[0]]["valid_from_turn"]
    result = dict(policy=POLICY, history=history, embedder=EMBEDDER, pair_floor=PAIR_FLOOR,
                  pair_k=PAIR_K, thread_floor=THREAD_FLOOR, entry_ids=[e["id"] for e in entries],
                  pairs=[dict(older=entries[i]["id"], newer=entries[j]["id"], cosine=c) for i, j, c in pairs],
                  threads=[dict(entries=[entries[i]["id"] for i in g], span_turns=span(g)) for g in groups])
    (directory / "threads.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    long = [g for g in groups if span(g) >= 100]
    print(f"{history}: usable {len(entries)}, candidate change pairs {len(pairs)}, threads {len(groups)} "
          f"(spanning >= 100 turns: {len(long)}, largest {max(map(len, groups), default=0)})")
    return result


# ── S3b: change detection inside threads ─────────────────────────────────

CHANGE_POLICY = "ice-z2-changes-1"
CHANGE_SYSTEM = (
    "You compare ledger entries from one conversation: one NEWER entry and a list of EARLIER "
    "entries said by the same speaker. For EACH earlier entry, answer two questions about what "
    "the sentences say:\n"
    "1. same_information: do the earlier and the newer sentence state the same thing, with "
    "neither adding or changing any information?\n"
    "2. both_can_be_true: could the earlier and the newer sentence both be true at the same "
    "time, right now? Answer false only if the newer one says the earlier one is no longer so - "
    "a changed value, a cancelled or reversed decision, a stopped habit, a correction, an "
    "updated count or status. Different details, extra detail, items of a list, a question, a "
    "suggestion, a possibility or a different direction of a relation can all be true "
    "together: answer true.\n"
    "Judge each earlier entry on its own against the newer one. Use only what the sentences say.\n"
    'Return only JSON: {"answers": [{"id": "E1", "same_information": true|false, '
    '"both_can_be_true": true|false}]} with one object per earlier entry.')


def _change_prompt(newer, earlier):
    return json.dumps({"newer": newer["value"], "earlier": [{"id": a, "text": e["value"]}
                                                            for a, e in earlier.items()]},
                      ensure_ascii=False)


def _verdicts(calls, role, meta, generate, newer, earlier, session):
    """{alias: 'same'|'replaces'|'both_true'} for *earlier* (alias -> entry); a
    cut reply is asked in halves, a missing answer is left out."""
    answers, _ = M.ask_covering(calls, role, meta, generate, CHANGE_SYSTEM,
                                lambda sub: _change_prompt(newer, sub), "answers", session, earlier)
    out = {}
    for alias, a in answers.items():
        same, both = a.get("same_information"), a.get("both_can_be_true")
        if isinstance(same, bool) and isinstance(both, bool):
            out[alias] = "same" if same else ("both_true" if both else "replaces")
    return out


def detect_changes(history, ledger_root, out, *, models, asker="deepseek", checker="mimo",
                   tie_breaker="muse"):
    """Changes = pairs both families call 'replaces' (or the tie-breaker does when they
    disagree). An older entry never replaces a newer one: pairs are older-first."""
    import uuid
    directory = out / history
    found = json.loads((directory / "threads.json").read_text())
    entries = {e["id"]: e for e in usable(history, ledger_root)}
    calls = M.Calls(directory / "changes-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{CHANGE_POLICY}|{history}|{role}"))
    by_newer = {}
    for p in found["pairs"]:
        by_newer.setdefault(p["newer"], []).append(p["older"])
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=4)           # same calls, same order of results
    first, flagged = {}, []
    asked = list(by_newer.items())
    earliers = [{f"E{n}": entries[i] for n, i in enumerate(older_ids, 1)} for _, older_ids in asked]
    replies = pool.map(lambda job: _verdicts(calls, asker, models[asker][1], models[asker][0],
                                             entries[job[0]], job[1], session(asker)),
                       [(newer_id, earlier) for (newer_id, _), earlier in zip(asked, earliers)])
    for (newer_id, _), earlier, v in zip(asked, earliers, replies):
        for alias, verdict in v.items():
            first[(earlier[alias]["id"], newer_id)] = verdict
            if verdict == "replaces":
                flagged.append((earlier[alias]["id"], newer_id))
    confirmed, disputed = [], []
    checks = pool.map(lambda pair: _verdicts(calls, checker, models[checker][1], models[checker][0],
                                             entries[pair[1]], {"E1": entries[pair[0]]}, session(checker)).get("E1"),
                      flagged)
    for (older_id, newer_id), v in zip(flagged, checks):
        (confirmed if v == "replaces" else disputed).append(dict(older=older_id, newer=newer_id,
                                                                checker=v))
    ties = pool.map(lambda d: _verdicts(calls, tie_breaker, models[tie_breaker][1], models[tie_breaker][0],
                                        entries[d["newer"]], {"E1": entries[d["older"]]},
                                        session(tie_breaker)).get("E1"), disputed)
    for d, v in zip(disputed, ties):
        d["tie_break"] = v
        if v == "replaces":
            confirmed.append(dict(older=d["older"], newer=d["newer"], checker=d["checker"], tie_break=v))
    pool.shutdown()
    same = [dict(older=o, newer=n) for (o, n), v in first.items() if v == "same"]
    result = dict(policy=CHANGE_POLICY, history=history, asker=asker, checker=checker,
                  tie_breaker=tie_breaker, models={r: models[r][1] for r in models},
                  pairs_asked=len(first), flagged=len(flagged), confirmed=confirmed,
                  disputed=disputed, duplicates=same)
    (directory / "changes.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"{history}: pairs {len(first)}, flagged {len(flagged)}, confirmed changes {len(confirmed)}, "
          f"disputed {len(disputed)}, duplicates {len(same)}")
    return result


def with_changes(entries, changes):
    """Ledger entries with confirmed changes added as supersedes links, re-chained."""
    member = {e["id"]: e["members"][0] for e in entries}
    extra = {}
    for c in changes["confirmed"]:
        extra.setdefault(c["newer"], []).append(member[c["older"]])
    updated = [dict(e, supersedes_raw=sorted(set(e.get("supersedes_raw") or []) | set(extra.get(e["id"], []))))
               for e in entries]
    return M.chain(updated)


# ── S3c: family candidate pools by type (no model calls) ─────────────────

# Spec D4 share of question families per history.
QUOTAS = {"fact_recall": .15, "update": .15, "temporal": .12, "cumulative": .10,
          "aggregation": .08, "procedure_advice": .10, "negative": .10, "preference": .05,
          "multi_hop": .05, "messy_variant": .10}
RECENT_TURNS, REPEAT_GAP = 40, 150
# Per-history targets live in experiments/z2/histories.py (Z2 D13).
from experiments.z2.histories import HISTORIES  # noqa: E402
TARGET_OCCURRENCES = {k: h.target_occurrences for k, h in HISTORIES.items()}
ASKS_PER_FAMILY = {k: h.asks_per_family for k, h in HISTORIES.items()}
MEGA_SPAN, CLUSTER_FLOOR, CLUSTER_MAX = 100, 0.84, 25
TEMPORAL_MAX = 300
MULTI_HOP_MAX = 0.90        # two entries this similar are one fact, not two hops


def checkpoints(history):
    """Session-end turns on the declared clock (D3/D4)."""
    from experiments.z2.clock import load_history, table
    return [s["last_turn"] for s in table(history, load_history(history))["sessions"]]


def first_checkpoint_after(turn, points, gap=RECENT_TURNS):
    """The first checkpoint at which *turn* has left the recent window."""
    return next((c for c in points if c - turn >= gap), None)


def tight_clusters(entries, vectors, *, floor=CLUSTER_FLOOR, max_size=CLUSTER_MAX):
    """Average-linkage-style greedy clusters capped in size (one-project histories
    chain everything at a loose floor)."""
    import numpy as np
    sims = vectors @ vectors.T
    order = sorted(range(len(entries)), key=lambda i: entries[i]["valid_from_turn"])
    clusters = []
    for i in order:
        best, best_sim = None, floor
        for c in clusters:
            if len(c) >= max_size:
                continue
            mean = float(np.mean([sims[i, j] for j in c]))
            if mean >= best_sim:
                best, best_sim = c, mean
        if best is None:
            clusters.append([i])
        else:
            best.append(i)
    return [c for c in clusters if len(c) > 1]


def topic_groups(found, entries, vectors, *, max_size=CLUSTER_MAX * 2):
    """S3a threads as index lists; a thread larger than *max_size* (one-project
    histories chain into one giant component) is split with tight_clusters."""
    index = {e["id"]: i for i, e in enumerate(entries)}
    groups = []
    for thread in found["threads"]:
        members = [index[x] for x in thread["entries"] if x in index]
        if len(members) <= max_size:
            groups.append(sorted(members, key=lambda i: entries[i]["valid_from_turn"]))
            continue
        sub = tight_clusters([entries[i] for i in members], vectors[members])
        groups += [sorted((members[j] for j in c), key=lambda i: entries[i]["valid_from_turn"]) for c in sub]
    return [g for g in groups if len(g) > 1]


DUPLICATE_FLOOR = 0.95


def duplicate_map(entries, vectors, changes=None, *, floor=DUPLICATE_FLOOR):
    """entry id -> canonical (earliest) id for entries that say the same thing:
    change detection's "same" pairs, or cosine >= *floor*. The same fact is often
    extracted in several windows (pilot: one aggregation set held one entry 3x)."""
    sims = vectors @ vectors.T
    parent = {e["id"]: e["id"] for e in entries}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            first = min((ra, rb), key=lambda x: (by_id[x]["valid_from_turn"], x))
            parent[ra if first == rb else rb] = first

    by_id = {e["id"]: e for e in entries}
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if sims[i, j] >= floor and entries[i]["speaker"] == entries[j]["speaker"]:
                join(entries[i]["id"], entries[j]["id"])
    for d in (changes or {}).get("duplicates", []):
        if d["older"] in by_id and d["newer"] in by_id:
            join(d["older"], d["newer"])
    return {x: root(x) for x in parent}


def candidate_pools(history, ledger_root, out):
    """Entry sets per question type, each with its first eligible checkpoint."""
    import numpy as np
    directory = out / history
    found = json.loads((directory / "threads.json").read_text())
    vectors = np.load(directory / "entry-vectors.npy")
    entries = usable(history, ledger_root)
    if [e["id"] for e in entries] != found["entry_ids"]:
        raise ValueError("ledger changed since threads were built")
    changes_path = directory / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else None
    canonical = duplicate_map(entries, vectors, changes)
    if changes:
        entries = with_changes(entries, changes)
    keep = [i for i, e in enumerate(entries) if canonical[e["id"]] == e["id"]]
    entries, vectors = [entries[i] for i in keep], vectors[keep]
    found = dict(found, threads=[dict(t, entries=list(dict.fromkeys(canonical[x] for x in t["entries"])))
                                 for t in found["threads"]])
    points = checkpoints(history)
    by_id = {e["id"]: e for e in entries}
    last = points[-1]

    def eligible(*es):
        start = max(e["valid_from_turn"] for e in es)
        c = first_checkpoint_after(start, points)
        return c is not None and c <= last, c

    pools = {k: [] for k in QUOTAS}
    for e in entries:
        ok, c = eligible(e)
        if not ok or e["modality"] != "asserted" and e["kind"] not in ("advice", "procedure"):
            continue
        if e["kind"] in ("advice", "procedure"):
            pools["procedure_advice"].append(dict(entries=[e["id"]], first=c))
        elif e["kind"] == "preference":
            pools["preference"].append(dict(entries=[e["id"]], first=c))
        elif e.get("valid_until_turn") is None:
            pools["fact_recall"].append(dict(entries=[e["id"]], first=c))
    for e in entries:
        if e.get("superseded_by"):
            newer_id = canonical.get(e["superseded_by"], e["superseded_by"])
            if newer_id not in by_id or newer_id == e["id"]:
                continue
            newer = by_id[newer_id]
            ok, c = eligible(e)
            if ok:
                pools["update"].append(dict(entries=[e["id"], newer["id"]], first=c,
                                            change_turn=newer["valid_from_turn"]))
    position = {e["id"]: i for i, e in enumerate(entries)}
    similarity = lambda a, b: float(vectors[position[a["id"]]] @ vectors[position[b["id"]]])
    clusters = topic_groups(found, entries, vectors)
    for cl in clusters:
        es = [entries[i] for i in cl]
        span = es[-1]["valid_from_turn"] - es[0]["valid_from_turn"]
        sessions = {first_checkpoint_after(e["valid_from_turn"], points, gap=0) for e in es}
        # First asked only once every fact in the group exists and has left the recent
        # window: the writer sees all of them (pilot: asked at turn 50 about turns 137-152).
        ok, c = eligible(*es)
        if not ok:
            continue
        if span >= MEGA_SPAN and len(es) >= 4:
            pools["cumulative"].append(dict(entries=[e["id"] for e in es], first=c, span=span))
        if len(es) >= 3 and len(sessions) >= 2:
            pools["aggregation"].append(dict(entries=[e["id"] for e in es], first=c, sessions=len(sessions)))
        for a, b in zip(es, es[1:]):
            ok, c = eligible(a, b)
            if not ok:
                continue
            if a["subject"].lower() != b["subject"].lower() and similarity(a, b) < MULTI_HOP_MAX:
                pools["multi_hop"].append(dict(entries=[a["id"], b["id"]], first=c))
    # Temporal: each event/decision/plan paired with the next one 40-300 turns later
    # (any thread) — "which did I do first" needs two happenings, not two static facts.
    happenings = sorted((e for e in entries if e["kind"] in ("event", "decision", "plan")
                         and e["modality"] == "asserted"), key=lambda e: e["valid_from_turn"])
    for i, a in enumerate(happenings):
        b = next((x for x in happenings[i + 1:]
                  if RECENT_TURNS <= x["valid_from_turn"] - a["valid_from_turn"] <= TEMPORAL_MAX), None)
        if b is not None:
            ok, c = eligible(a, b)
            if ok:
                pools["temporal"].append(dict(entries=[a["id"], b["id"]], first=c))
    pools["negative"] = [dict(cluster=[entries[i]["id"] for i in cl][:3]) for cl in clusters]
    families = round(TARGET_OCCURRENCES.get(history, 0) / ASKS_PER_FAMILY.get(history, 2.3))
    need = {k: round(v * families) for k, v in QUOTAS.items()}
    return pools, need


# ── S3d: select families, phrase questions, verify (cloud) ───────────────

PHRASE_POLICY = "ice-z2-phrase-3"
PHRASE_SYSTEM = """You write one memory question for a test of an AI assistant's long-term memory.
The USER of a long conversation asks it, later in that same conversation. Write it the way
this user writes (STYLE shows some of their real messages): their casual tone, spelling and
length, and their habits — lowercase, abbreviations, missing punctuation, the odd typo are all
fine. Short and natural: refer to things the way the user would, never restate the facts. The question must be answerable from the ENTRIES alone, unambiguous at the time it
is asked, and must NOT contain or hint at the answer: no name, number, item, value or choice
that the answer consists of. TASK says what kind of question.
Each entry has a speaker. "user" entries are what the user said: the user asks about them in
the first person ("what did I decide…", "what's my…"). "assistant" entries are what the AI
said: the user asks "what did you suggest/say…". Never attribute one speaker's words to the other.
Return only JSON: {"question": "..."}"""
TASKS = {
    "fact_recall": "Ask for the information in the entry.",
    "update": ("Ask for the CURRENT state of the thing these entries are about. The same question "
               "must make sense whichever entry is current; never say that it changed."),
    "temporal": ("Ask which of the two things came first (or which happened earlier/later). Describe "
                 "both without giving away the order."),
    "cumulative": ("Ask the assistant to list or reconstruct everything about this topic so far. "
                   "Name the topic only; never list its parts."),
    "aggregation": ("Ask how many, or ask to list all, of the kind of thing these facts are about, as "
                    "it came up across the conversation. Name the kind of thing only; never name the "
                    "items, and never mention 'entries', 'facts' or 'memory'."),
    "procedure_advice": "Ask what advice or steps the assistant gave about this.",
    "preference": "Ask about the user's own like, dislike or preference, in the first person (\"what do I prefer…\").",
    "multi_hop": "Ask one question whose answer needs both entries together.",
}
VERIFY_Q_POLICY = "ice-z2-verify-question-3"
VERIFY_Q_SYSTEM = """You check a memory question written for a test. Given the ENTRIES (the only
facts the answer may use, each with the turn it was said at), the turn the question is ASKED_AT,
and the QUESTION, report (only entries said at or before ASKED_AT count; the question may not
presuppose anything said later):
- answerable: true if the entries said by ASKED_AT give the answer (the order in which things
  were said is part of the entries)
- unambiguous: true if only one reading is reasonable and only these entries answer it; false
  if the entries give conflicting or overlapping numbers, plans or totals and the question does
  not say which one it means. An entry marked "replaced_at_turn" was changed by a later entry at
  that turn: for a question about the current value that is expected, not a conflict — the
  current value is the replacing entry
- leaks_answer: true if the question contains any specific detail of the answer (a name,
  number, item, value or choice the answer consists of) or strongly hints at it
- speaker_right: true if the question attributes every fact to the right person: entries
  with speaker "user" are the user's own words (asked as "I/my"), "assistant" entries are the
  AI's words (asked as "you")
Return only JSON: {"answerable": true|false, "unambiguous": true|false, "leaks_answer": true|false,
"speaker_right": true|false, "reason": "..."}"""
WRITERS_CYCLE = ("deepseek", "mimo")


def select_families(pools, need, history):
    """Deterministic, spread over the history: candidates sorted by their first
    checkpoint, then evenly spaced picks; hash-seeded tie order."""
    import hashlib
    chosen = []
    for kind, n in need.items():
        if kind in ("messy_variant", "negative"):
            continue
        cands = sorted(pools.get(kind, []), key=lambda c: (c["first"], hashlib.sha256(
            f"{history}|{kind}|{'|'.join(c['entries'])}".encode()).hexdigest()))
        if len(cands) <= n:
            picks = cands
        else:
            step = len(cands) / n
            picks = [cands[int(i * step)] for i in range(n)]
        chosen += [dict(c, type=kind) for c in picks]
    for i, f in enumerate(chosen):
        f["id"] = f"{history}:fam{i:03d}"
        f["writer"] = WRITERS_CYCLE[i % 2]
    return chosen


def style_examples(originals, turn, n=4, chars=300):
    """The user's own messages just before *turn* (real wording for the writer)."""
    picks = [originals[t - 1]["prompt"] for t in range(max(1, turn - 12), turn + 1)
             if originals[t - 1]["prompt"].strip()][-n:]
    return [p[:chars] for p in picks]


def ask_object(calls, role, meta, generate, system, prompt, session, *, max_output_tokens=2000):
    """One durable, resumable call whose reply is a JSON object; None when the
    reply is not one (never repaired)."""
    identity = L.sha256(json.dumps(dict(policy=M.CALL_POLICY, role=role, meta=meta,
                                        system=L.sha256(system), prompt=L.sha256(prompt)),
                                   sort_keys=True))

    def run():
        result = generate([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          temperature=0, max_output_tokens=max_output_tokens, session_id=session)
        return result.text, result.usage

    try:
        body = calls.get(identity, run)["text"].strip()
    except Exception as exc:                                  # noqa: BLE001
        if not M._cut(exc):
            raise
        # A reply cut by its output cap is no answer — the same as an unparseable one —
        # for this item only; it is not cached, so a rerun asks again.
        print(f"WARNING: {role} reply cut by its output cap; item treated as unanswered", flush=True)
        return None
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        value = json.loads(body, strict=False)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def phrase_and_verify(history, families, entries_by_id, originals, out, *, models):
    """Writer phrases each family's question; the other writer family verifies it.
    Returns (kept, rejected); every call is durable and resumable."""
    import uuid
    directory = out / history
    directory.mkdir(parents=True, exist_ok=True)
    calls = M.Calls(directory / "questions-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{PHRASE_POLICY}|{history}|{role}"))
    kept, rejected = [], []
    for fam in families:
        ent = [dict(speaker=entries_by_id[x]["speaker"], text=entries_by_id[x]["value"])
               for x in fam["entries"]]
        writer = fam["writer"]
        prompt = json.dumps(dict(TASK=TASKS[fam["type"]], STYLE=style_examples(originals, fam["first"]),
                                 ENTRIES=ent), ensure_ascii=False)
        reply = ask_object(calls, writer, models[writer][1], models[writer][0], PHRASE_SYSTEM, prompt,
                           session(writer)) or {}
        question = reply.get("question")
        if not isinstance(question, str) or not question.strip():
            rejected.append(dict(fam, reason="no_question"))
            continue
        checker = "mimo" if writer == "deepseek" else "deepseek"
        # Only the verifier sees when each fact was said and when the question is asked:
        # it checks grounding (nothing presupposed from later) and, for temporal pairs,
        # the order; the writer never sees turns, so it cannot leak an order.
        # A change known by ASKED_AT is shown (2026-10-19: without it the checker called
        # 15 of 20 C update questions "conflicting values").
        checked = []
        for e, x in zip(ent, fam["entries"]):
            row = dict(e, said_at_turn=entries_by_id[x]["valid_from_turn"])
            until = entries_by_id[x].get("valid_until_turn")
            if until is not None and until <= fam["first"]:
                row["replaced_at_turn"] = until
            checked.append(row)
        verdict = ask_object(calls, checker, models[checker][1], models[checker][0], VERIFY_Q_SYSTEM,
                             json.dumps(dict(ENTRIES=checked, ASKED_AT=fam["first"], QUESTION=question),
                                        ensure_ascii=False), session(checker)) or {}
        ok = (verdict.get("answerable") is True and verdict.get("unambiguous") is True
              and verdict.get("leaks_answer") is False and verdict.get("speaker_right") is True)
        row = dict(fam, question=question.strip(), verifier=checker, verdict=verdict)
        if ok:
            kept.append(row)
        else:
            rejected.append(dict(row, reason="verifier"))
    return kept, rejected


# ── S3e: messy-phrasing variants (style invariance, spec D4 / G28) ───────

MESSY_POLICY = "ice-z2-messy-1"
MESSY_STYLES = ("typos and lowercase, no punctuation",
                "terse: as few words as possible",
                "rambling: filler like 'ok so', 'like', 'idk', asking around the point before getting to it",
                "the question buried mid-message after an unrelated remark")
MESSY_SYSTEM = """Rewrite a user's question in a different writing style, the way a real person might
type it on another day. Keep EXACTLY the same meaning and the same thing being asked; add no
facts, drop no condition, give away nothing new. STYLE says how to write it.
Return only JSON: {"question": "..."}"""
MESSY_VERIFY_SYSTEM = """Two questions are given. Report whether they ask for exactly the same
information, so that the same answer is correct for both (wording and style may differ).
Return only JSON: {"same_meaning": true|false, "reason": "..."}"""


def messy_variants(history, kept, n, out, *, models):
    """*n* variants of kept families (deterministic choice, styles cycled): same
    entries and therefore the same computed answers; the other writer family checks
    the meaning is unchanged. A variant that flips an answer is a G28 finding."""
    import hashlib
    import uuid
    (out / history).mkdir(parents=True, exist_ok=True)
    calls = M.Calls(out / history / "questions-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{MESSY_POLICY}|{history}|{role}"))
    base = sorted((f for f in kept if f["type"] in ("fact_recall", "update", "procedure_advice",
                                                      "preference", "temporal")),
                  key=lambda f: hashlib.sha256(f"{MESSY_POLICY}|{f['id']}".encode()).hexdigest())[:n]
    variants, rejected = [], []
    for i, fam in enumerate(base):
        writer = "mimo" if fam["writer"] == "deepseek" else "deepseek"
        style = MESSY_STYLES[i % len(MESSY_STYLES)]
        reply = ask_object(calls, writer, models[writer][1], models[writer][0], MESSY_SYSTEM,
                           json.dumps(dict(STYLE=style, QUESTION=fam["question"]), ensure_ascii=False),
                           session(writer)) or {}
        q = reply.get("question")
        if not isinstance(q, str) or not q.strip():
            rejected.append(dict(base=fam["id"], reason="no_question"))
            continue
        checker = fam["writer"]
        v = ask_object(calls, checker, models[checker][1], models[checker][0], MESSY_VERIFY_SYSTEM,
                       json.dumps(dict(A=fam["question"], B=q), ensure_ascii=False), session(checker)) or {}
        row = dict(fam, id=f"{fam['id']}:messy", type="messy_variant", base_family=fam["id"],
                   base_type=fam["type"], style=style, question=q.strip(), writer=writer,
                   verifier=checker, verdict=v)
        (variants if v.get("same_meaning") is True else rejected).append(row)
    return variants, rejected


# ── S3f: never-stated (abstention) questions ─────────────────────────────

NEG_POLICY = "ice-z2-negative-1"
NEG_SYSTEM = """You write one memory question for a test of an AI assistant's long-term memory, of a
special kind: the user asks about something that fits the TOPIC (facts shown) but that the
conversation NEVER said — a detail nobody mentioned, so the right answer is "you never told me".
It must sound like a normal question this user would ask (STYLE), not a trick, and must not be
answered by any of the facts shown.
Return only JSON: {"question": "...", "why_unanswered": "..."}"""
NEG_CHECK_SYSTEM = """A memory question is meant to be about something the conversation never said.
Given the QUESTION and PASSAGES (facts and original messages from the conversation up to the
time it is asked), report whether any passage answers it, fully or in part.
Return only JSON: {"answered": true|false, "by": ["passage ids"], "reason": "..."}"""


def originals_index(originals, device=None, cache=None):
    """(turn, role, text, vector) for every original message, for the abstention search.
    *cache*: an .npz path reused while the texts and the encoder are unchanged."""
    import hashlib
    import numpy as np
    rows = [(t, role, originals[t - 1][key]) for t in range(1, len(originals) + 1)
            for role, key in (("user", "prompt"), ("assistant", "response"))
            if originals[t - 1][key].strip()]
    texts = [text[:2000] for _, _, text in rows]
    digest = hashlib.sha256(json.dumps([EMBEDDER, texts]).encode()).hexdigest()
    if cache and Path(cache).exists():
        saved = np.load(cache)
        if str(saved["digest"]) == digest:
            return rows, saved["vectors"]
    vectors = embed(texts, device=device)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, digest=np.array(digest), vectors=vectors)
    return rows, vectors


def negative_families(history, pools, n, entries_by_id, all_entries, originals, index, out, *,
                      models, top_k=8):
    """*n* never-stated families; each passes the ledger check (no entry of ANY status
    answers it) and the originals check (no retrieved original passage answers it)."""
    import hashlib
    import uuid
    rows, vectors = index
    (out / history).mkdir(parents=True, exist_ok=True)
    calls = M.Calls(out / history / "questions-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NEG_POLICY}|{history}|{role}"))
    points = checkpoints(history)
    all_vec = embed([entry_text(e) for e in all_entries])      # once, not per question
    seeds = sorted(pools.get("negative", []), key=lambda c: hashlib.sha256(
        f"{NEG_POLICY}|{'|'.join(c['cluster'])}".encode()).hexdigest())
    kept, rejected = [], []
    for i, seed in enumerate(seeds):
        if len(kept) >= n:
            break
        es = [entries_by_id[x] for x in seed["cluster"] if x in entries_by_id]
        if not es:
            continue
        start = max(e["valid_from_turn"] for e in es)
        cp = first_checkpoint_after(start, points)
        if cp is None:
            continue
        writer = WRITERS_CYCLE[i % 2]
        checker = "mimo" if writer == "deepseek" else "deepseek"
        reply = ask_object(calls, writer, models[writer][1], models[writer][0], NEG_SYSTEM, json.dumps(dict(
            TOPIC=[e["value"] for e in es], STYLE=style_examples(originals, cp)), ensure_ascii=False),
            session(writer)) or {}
        q = reply.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        qv = embed([q])[0]
        # Ledger passages: every entry of any status said by the checkpoint, nearest first.
        led_idx = [j for j, e in enumerate(all_entries) if e["valid_from_turn"] <= cp]
        led_top = sorted(led_idx, key=lambda j: -float(all_vec[j] @ qv))[:top_k]
        led = all_entries
        orig = [j for j, (t, _, _) in enumerate(rows) if t <= cp]
        orig_top = sorted(orig, key=lambda j: -float(vectors[j] @ qv))[:top_k]
        passages = ([dict(id=f"L{k}", text=led[j]["value"]) for k, j in enumerate(led_top)]
                    + [dict(id=f"O{k}", text=rows[j][2][:1500]) for k, j in enumerate(orig_top)])
        v = ask_object(calls, checker, models[checker][1], models[checker][0], NEG_CHECK_SYSTEM,
                       json.dumps(dict(QUESTION=q, PASSAGES=passages), ensure_ascii=False),
                       session(checker), max_output_tokens=4000) or {}
        fam = dict(id=f"{history}:neg{i:03d}", type="negative", entries=[e["id"] for e in es], first=cp,
                   writer=writer, verifier=checker, question=q.strip(), why=reply.get("why_unanswered"),
                   verdict=v, searched=dict(ledger=len(led_top), originals=len(orig_top)))
        (kept if v.get("answered") is False else rejected).append(fam)
    return kept, rejected


# ── S3g: no-memory control prompts (spec D4b) ────────────────────────────

CONTROL_POLICY = "ice-z2-controls-1"
CONTROL_KINDS = ("general_knowledge", "self_contained_task", "small_talk", "recent_follow_up")
CONTROL_TASKS = {
    "general_knowledge": "a general-knowledge question that has nothing to do with the conversation",
    "self_contained_task": "a small self-contained request (rewrite, explain, compute, list) whose input is entirely in the message",
    "small_talk": "a bit of small talk or a reaction, the kind people drop between real requests",
    "recent_follow_up": "a follow-up to the RECENT turns, answerable from them alone",
}
CONTROL_SYSTEM = """You write one message the USER might send next in a long conversation with an AI
assistant, for a test of whether the assistant over-uses long-term memory. KIND says what sort of
message; it must need NO memory of anything earlier in the conversation (a recent follow-up may use
only the RECENT turns shown). Write it the way this user writes (STYLE).
Return only JSON: {"prompt": "..."}"""
CONTROL_CHECK_SYSTEM = """A message is meant to need no long-term memory: it is answerable from general
knowledge, from the message itself, or (if RECENT turns are shown) from those turns alone.
Report whether that holds.
Return only JSON: {"needs_no_memory": true|false, "reason": "..."}"""


def control_prompts(history, originals, out, *, models):
    """One control prompt per checkpoint, kinds cycled; the other writer family checks it."""
    import uuid
    (out / history).mkdir(parents=True, exist_ok=True)
    calls = M.Calls(out / history / "questions-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{CONTROL_POLICY}|{history}|{role}"))
    kept, rejected = [], []
    for i, cp in enumerate(checkpoints(history)):
        kind = CONTROL_KINDS[i % len(CONTROL_KINDS)]
        writer = WRITERS_CYCLE[i % 2]
        checker = "mimo" if writer == "deepseek" else "deepseek"
        recent = ([dict(turn=t, user=originals[t - 1]["prompt"][:800], assistant=originals[t - 1]["response"][:800])
                   for t in range(max(1, cp - 2), cp + 1)] if kind == "recent_follow_up" else None)
        body = dict(KIND=CONTROL_TASKS[kind], STYLE=style_examples(originals, cp),
                    **({"RECENT": recent} if recent else {}))
        reply = ask_object(calls, writer, models[writer][1], models[writer][0], CONTROL_SYSTEM,
                           json.dumps(body, ensure_ascii=False), session(writer)) or {}
        prompt = reply.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            rejected.append(dict(checkpoint_turn=cp, kind=kind, reason="no_prompt"))
            continue
        v = ask_object(calls, checker, models[checker][1], models[checker][0], CONTROL_CHECK_SYSTEM,
                       json.dumps(dict(MESSAGE=prompt, **({"RECENT": recent} if recent else {})),
                                  ensure_ascii=False), session(checker)) or {}
        row = dict(id=f"{history}:ctl{i:03d}", type="no_memory_control", kind=kind, checkpoint_turn=cp,
                   prompt=prompt.strip(), writer=writer, verifier=checker, verdict=v)
        (kept if v.get("needs_no_memory") is True else rejected).append(row)
    return kept, rejected


# ── S4: occurrences and computed ground truth (no model calls) ───────────

GT_POLICY = "ice-z2-ground-truth-1"
THIRD_ASK_SHARE = 0.30


def _third_ask(family_id):
    import hashlib
    return int(hashlib.sha256(f"{GT_POLICY}|{family_id}".encode()).hexdigest(), 16) % 100 < THIRD_ASK_SHARE * 100


def ask_points(family, points):
    """Checkpoints at which a family is asked (spec D4 cutoffs and repeats):
    first eligible; again >= REPEAT_GAP turns later; a third ask after the
    longest available gap for ~30% of families; an update family is asked
    before AND after its change."""
    first = family["first"]
    if family.get("base_type", family["type"]) == "update":
        before = [c for c in points if first <= c < family["change_turn"]]
        after = [c for c in points if c >= family["change_turn"] + RECENT_TURNS]
        return ([before[0]] if before else []) + ([after[0]] if after else [])
    asks = [first]
    later = [c for c in points if c - first >= REPEAT_GAP]
    if later:
        asks.append(later[0])
        if _third_ask(family["id"]) and later[-1] != later[0]:
            asks.append(later[-1])
    return asks


def ground_truth(family, entries_by_id, checkpoint):
    """Reference for one occurrence, computed from the ledger state at *checkpoint*."""
    es = [entries_by_id[x] for x in family["entries"]]
    current = lambda e: e["valid_from_turn"] <= checkpoint and (
        e.get("valid_until_turn") is None or e["valid_until_turn"] > checkpoint)
    kind = family.get("base_type", family["type"])          # a messy variant answers like its base
    if kind == "update":
        now = [e for e in es if current(e)]
        old = [e for e in es if e["valid_from_turn"] <= checkpoint and not current(e)]
        return dict(required=[e["value"] for e in now], superseded=[e["value"] for e in old],
                    citations=[c for e in now for c in e["citations"]])
    if kind == "temporal":
        ordered = sorted(es, key=lambda e: e["valid_from_turn"])
        return dict(required=[f"first: {ordered[0]['value']}", f"then: {ordered[-1]['value']}"],
                    order=[e["id"] for e in ordered], citations=[c for e in ordered for c in e["citations"]])
    if kind in ("cumulative", "aggregation"):
        now = [e for e in es if current(e)]
        return dict(required=[e["value"] for e in now], count=len(now),
                    citations=[c for e in now for c in e["citations"]])
    now = [e for e in es if current(e)]
    return dict(required=[e["value"] for e in now], citations=[c for e in now for c in e["citations"]])


def occurrences(families, entries_by_id, points, originals):
    """Every (family, checkpoint) with its reference; refuses future evidence."""
    out = []
    for fam in families:
        if fam["type"] == "negative":
            # Checked never-stated by its first ask only; asked once (a later turn
            # could state it, and only the first ask was searched).
            out.append(dict(family=fam["id"], type="negative", checkpoint_turn=fam["first"],
                            repeat_index=0, question=fam["question"],
                            reference=dict(required=[], citations=[]), expected_abstention=True))
            continue
        for n, cp in enumerate(ask_points(fam, points)):
            gt = ground_truth(fam, entries_by_id, cp)
            if not gt["required"]:
                continue
            for c in gt["citations"]:
                if L.problem(originals, c, cp):
                    raise ValueError(f"{fam['id']}: citation fails at checkpoint {cp}: {L.problem(originals, c, cp)}")
            out.append(dict(family=fam["id"], type=fam["type"], checkpoint_turn=cp, repeat_index=n,
                            question=fam["question"], reference=gt, expected_abstention=False))
    return out


# ── S5 / S6: maintainer review, split, Z2 asks ───────────────────────────

REVIEW_POLICY = "ice-z2-review-1"


def write_review(history, out, name="families", per_type=6):
    """A stratified sample (per_type families of each type, deterministic) with each
    family's question, reference at its first ask and citations, for the maintainer;
    plus review-decisions.json with an empty drop list to fill in (USER-REQUIRED)."""
    import hashlib
    directory = out / history
    built = json.loads((directory / f"{name}.json").read_text())
    occ = json.loads((directory / "occurrences.json").read_text())
    first = {}
    for o in occ:
        first.setdefault(o["family"], o)
    by_type = {}
    for fam in built["kept"]:
        by_type.setdefault(fam["type"], []).append(fam)
    lines = [f"# Z2 question review — history {history}", "",
             f"{len(built['kept'])} families kept, {len(built['rejected'])} rejected by the verifier, "
             f"{len(built.get('controls', []))} no-memory controls. Sample: {per_type} per type.", "",
             "Mark a family to drop by adding its id to `drop` in review-decisions.json "
             "(with a reason in `notes`). Families not listed are kept.", ""]
    for kind in sorted(by_type):
        group = sorted(by_type[kind], key=lambda f: hashlib.sha256(f"{REVIEW_POLICY}|{f['id']}".encode()).hexdigest())
        lines += [f"## {kind} ({len(group)} families)", ""]
        for fam in group[:per_type]:
            o = first.get(fam["id"], {})
            ref = o.get("reference", {})
            lines += [f"### {fam['id']}", f"- **question:** {fam['question']}",
                      f"- **first asked at:** turn {o.get('checkpoint_turn')}"
                      + (" (expects abstention)" if o.get("expected_abstention") else ""),
                      f"- **required:** {json.dumps(ref.get('required', []), ensure_ascii=False)}",
                      f"- **cited turns:** {sorted({c.get('turn') for c in ref.get('citations', [])})}", ""]
    walk = directory / "walk-audit.json"
    if walk.exists():
        audit = json.loads(walk.read_text())
        lines += [f"## Walk vs ledger disagreements ({len(audit['queue'])})", "",
                  f"The forward walk (Muse) agreed with the ledger on {audit['agree']} of {audit['judged']} "
                  f"occurrences; walk-only changes {audit['walk_only']}, ledger-only {audit['ledger_only']}. "
                  "For each: is the reference at the ask still right? Drop the family if not.", ""]
        for item in audit["queue"]:
            reply = item.get("reply") or {}
            lines += [f"- **{item['family']}** at turn {item.get('checkpoint')} — {item['kind']}"
                      + (f"; walk cites turn {reply.get('turn')}: \"{reply.get('quote')}\" → {reply.get('new_answer')}"
                         if reply.get("changes") else "")]
        lines.append("")
    else:
        lines += ["## Walk vs ledger disagreements", "", "NOT RUN — run walk_audit.py before reviewing.", ""]
    (directory / "review-packet.md").write_text("\n".join(lines) + "\n")
    decisions = directory / "review-decisions.json"
    if not decisions.exists():
        decisions.write_text(json.dumps(dict(policy=REVIEW_POLICY, history=history, reviewed=False,
                                             drop=[], notes={}), indent=1) + "\n")
    print(f"wrote {directory / 'review-packet.md'}; fill {decisions} and set reviewed: true")


def finalize(history, out, name="families"):
    """S6 after S5: drop what the maintainer dropped, seal the 60/40 split, and write
    z2-asks.json (Z2 families' occurrences + controls) for the read passes."""
    from experiments.z2 import read_passes, split
    directory = out / history
    decisions = json.loads((directory / "review-decisions.json").read_text())
    if decisions.get("reviewed") is not True:
        raise RuntimeError("S5 review not done: review-decisions.json has reviewed != true (USER-REQUIRED)")
    built = json.loads((directory / f"{name}.json").read_text())
    drop = set(decisions.get("drop", []))
    families = [f for f in built["kept"] if f["id"] not in drop and f.get("base_family") not in drop]
    controls = [c for c in built.get("controls", []) if c["id"] not in drop]
    (directory / "split").mkdir(exist_ok=True)              # write_split itself refuses a re-seal
    seal = split.write_split(directory / "split", history, families, controls)
    asks = read_passes.build_asks(directory / "split", directory / "occurrences.json")
    (directory / "z2-asks.json").write_text(json.dumps(asks, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(seal, dropped=len(drop), z2_asks=len(asks))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Z2 S3 question machinery")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("threads")
    t.add_argument("--history", required=True)
    t.add_argument("--ledger", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default=None, help="default: the GPU when available")
    pl = sub.add_parser("pools")
    pl.add_argument("--history", required=True)
    pl.add_argument("--ledger", type=Path, required=True)
    pl.add_argument("--out", type=Path, required=True)
    ph = sub.add_parser("phrase")
    ph.add_argument("--history", required=True)
    ph.add_argument("--ledger", type=Path, required=True)
    ph.add_argument("--out", type=Path, required=True)
    ph.add_argument("--per-type", type=int, help="pilot: at most N families per type")
    ph.add_argument("--name", default="families", help="output file stem")
    c = sub.add_parser("changes")
    c.add_argument("--history", required=True)
    c.add_argument("--ledger", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    oc = sub.add_parser("occurrences", help="S4: every (family, checkpoint) with its computed reference")
    oc.add_argument("--history", required=True)
    oc.add_argument("--ledger", type=Path, required=True)
    oc.add_argument("--out", type=Path, required=True)
    oc.add_argument("--name", default="families")
    rv = sub.add_parser("review", help="S5: the maintainer's review packet and an empty decisions file")
    rv.add_argument("--history", required=True)
    rv.add_argument("--ledger", type=Path, required=True)
    rv.add_argument("--out", type=Path, required=True)
    rv.add_argument("--name", default="families")
    rv.add_argument("--per-type", type=int, default=6)
    fz = sub.add_parser("finalize", help="S6: apply review decisions, seal the split, write the Z2 asks")
    fz.add_argument("--history", required=True)
    fz.add_argument("--ledger", type=Path, required=True)
    fz.add_argument("--out", type=Path, required=True)
    fz.add_argument("--name", default="families")
    args = parser.parse_args()
    for path in (args.ledger, args.out):
        if not path.resolve().is_relative_to(ROOT / "logs"):
            raise ValueError("ledger data is private: keep it under logs/")
    if args.command == "threads":
        run_threads(args.history, args.ledger, args.out, device=args.device)
    elif args.command == "pools":
        pools, need = candidate_pools(args.history, args.ledger, args.out)
        for k in QUOTAS:
            have = len(pools[k]) if k != "messy_variant" else "derived"
            print(f"  {k:17s} need {need[k]:3d}   candidates {have}")
    elif args.command == "phrase":
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2.profiles import generator, metadata
        load_selected_env()
        pools, need = candidate_pools(args.history, args.ledger, args.out)
        if args.per_type:
            need = {k: min(v, args.per_type) for k, v in need.items()}
        families = select_families(pools, need, args.history)
        entries = usable(args.history, args.ledger)
        changes = args.out / args.history / "changes.json"
        if changes.exists():
            entries = with_changes(entries, json.loads(changes.read_text()))
        models = {r: (generator(r).generate, metadata(r)) for r in WRITERS_CYCLE}
        kept, rejected = phrase_and_verify(args.history, families, {e["id"]: e for e in entries},
                                           L.load_history(args.history), args.out, models=models)
        messy, messy_rejected = messy_variants(args.history, kept, need.get("messy_variant", 0),
                                               args.out, models=models)
        all_entries = M.chain(json.loads((args.ledger / args.history / "ledger.json").read_text())["entries"])
        originals = L.load_history(args.history)
        neg, neg_rejected = negative_families(args.history, pools, need.get("negative", 0),
                                              {e["id"]: e for e in entries}, all_entries, originals,
                                              originals_index(originals, cache=args.out / args.history / "originals-index.npz"),
                                              args.out, models=models)
        kept, rejected = kept + messy + neg, rejected + messy_rejected + neg_rejected
        controls, controls_rejected = control_prompts(args.history, originals, args.out, models=models)
        (args.out / args.history / f"{args.name}.json").write_text(json.dumps(
            dict(policy=PHRASE_POLICY, history=args.history, need=need, kept=kept, rejected=rejected,
                 controls=controls, controls_rejected=controls_rejected),
            ensure_ascii=False, indent=1) + "\n")
        print(f"{args.history}: families {len(families)}, kept {len(kept)}, rejected {len(rejected)}, "
              f"controls {len(controls)} (rejected {len(controls_rejected)})")
    elif args.command == "occurrences":
        built = json.loads((args.out / args.history / f"{args.name}.json").read_text())
        entries = usable(args.history, args.ledger)
        changes = args.out / args.history / "changes.json"
        if changes.exists():
            entries = with_changes(entries, json.loads(changes.read_text()))
        occ = occurrences(built["kept"], {e["id"]: e for e in entries}, checkpoints(args.history),
                          L.load_history(args.history))
        (args.out / args.history / "occurrences.json").write_text(json.dumps(occ, ensure_ascii=False, indent=1) + "\n")
        print(f"{args.history}: {len(occ)} occurrences from {len(built['kept'])} families")
    elif args.command == "review":
        write_review(args.history, args.out, args.name, args.per_type)
    elif args.command == "finalize":
        finalize(args.history, args.out, args.name)
    elif args.command == "changes":
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2.profiles import generator, metadata
        load_selected_env()
        models = {r: (generator(r).generate, metadata(r)) for r in ("deepseek", "mimo", "muse")}
        detect_changes(args.history, args.ledger, args.out, models=models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
