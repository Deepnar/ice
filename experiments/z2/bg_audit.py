"""Z2 S8: background-content quality audit on a replay's final store (spec D9).

Every kind of artifact the background models write is inventoried from the store
(and any table holding rows no kind covers is reported, never ignored), sampled
(all items when fewer than 100, else a deterministic 100 per kind per history),
and judged with its cited sources supplied:

- faithful — supported by its sources; nothing invented or contradicted;
- attributed — the right speaker; an assertion kept apart from a suggestion;
- current — does not present as current what the ledger shows changed later
  (judged only when the ledger has such a change; otherwise "n/a");
- complete — keeps the source's required detail (summaries, notes, folds);
- decision_valid — a merge, supersession, entity link or grouping is right.

Muse judges; DeepSeek V4.1 Flash re-judges a 20% sample and the agreement rate is
reported before any quality number (§5 item 7). Codex facts and ledger entries are
also matched directly (fact precision and recall against the ledger; a fact
matching only an `unconfirmed` entry is "unconfirmed source", not invented — D11).
Reach during the replay (how many history prompts each artifact reached) comes
from the replay trace; reach in graded answers is S11's.

Every call is durable and resumable. An item whose judging call errors is retried
on the next attempt; after three failed attempts it is recorded as unjudged with
the error, counted in the report, and the audit moves on.

    uv run --no-sync python experiments/z2/bg_audit.py --history B --database ice_z2_main_b \
        --replay logs/z2-run-main/s7-B --ledger <ledger.json> [--changes <changes.json>] \
        --out logs/z2-run-main/s8-B
"""

import argparse
import collections
import hashlib
import json
import math
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-bg-audit-1"
PER_KIND = 100
REJUDGE_SHARE = 0.20
SOURCE_CHARS = 240_000          # ~52K tokens; Muse took a 200K-character prompt (43K tokens) fine,
                                # 2026-10-18. Beyond it an item is recorded "too_large", never cut.
QUOTE_WINDOW = 4_000            # characters either side of a cited quote (fact-based kinds)
MAX_ITEM_ERRORS = 3
# Muse reasons in hidden tokens that count against the output limit: 1,500 left
# 12 of ~20 trial replies "incomplete" (2026-10-18), as the merge saw before.
JUDGE_OUTPUT_TOKENS = 8_000
CRITERIA = ("faithful", "attributed", "current", "complete", "decision_valid")
KIND_CRITERIA = {
    "turn_summary": ("faithful", "attributed", "current", "complete"),
    "conversation_note": ("faithful", "attributed", "current", "complete"),
    "batch_note": ("faithful", "attributed", "current", "complete"),
    "session_summary": ("faithful", "attributed", "current", "complete"),
    "codex_fact": ("faithful", "attributed", "current"),
    "fact_supersession": ("decision_valid",),
    "entity_link": ("decision_valid",),
    "entity_merge": ("decision_valid",),
    "cluster": ("faithful", "decision_valid"),
    "procedural": ("faithful", "attributed"),
    "slot_update": ("faithful", "attributed", "current"),
    "applied_decision": ("decision_valid",),
}
# Tables a background writer fills, by the kinds that judge them; and tables that
# hold originals, indexes, links or operations (not judged content). Anything with
# rows in neither list is an UNCOVERED writer and is reported.
COVERED = {"episodic_memory", "conversation_notes", "batch_notes", "session_summaries", "codex_facts",
           "codex_fact_entities", "codex_entities", "context_clusters", "procedural_memory",
           "review_queue", "memory_slots", "conversation_summaries", "batch_summaries"}
NOT_CONTENT = {"conversations", "episodic_chunks", "episodic_cluster_links", "codex_fact_sources",
               "codex_events", "idempotency_keys", "store_meta", "maintenance_ledger", "cold_storage",
               "cold_chunks", "codex_edges", "session_replays", "import_runs", "import_conversations",
               "curated_labels", "projects", "project_state", "tasks", "documents", "document_links",
               "decisions"}

JUDGE_SYSTEM = """You audit what an AI memory system wrote in the background about a conversation
between a user and an AI assistant. ITEM is one stored artifact of kind KIND. SOURCES are the
original conversation turns it was made from. LATER_CHANGES, when present, are values the
conversation changed after those turns (from an independent fact ledger).
Judge ONLY the listed CRITERIA, each "yes", "no" or "unclear", with a one-sentence reason:
- faithful: everything ITEM states is supported by SOURCES; nothing invented or contradicted.
- attributed: statements are credited to the right speaker (the user vs the assistant), and a
  suggestion or hypothetical is not stated as an adopted fact.
- current: ITEM does not present as current a value that LATER_CHANGES shows was changed.
- complete: ITEM keeps the important specifics of SOURCES for its role (names, numbers,
  decisions, constraints) rather than only a vague gist.
- decision_valid: the decision ITEM records (a merge of two entities, one fact replacing
  another, a fact linked to an entity, turns grouped under a name) is correct per SOURCES.
Each source carries the time the memory system recorded for that turn; a date or time in ITEM that
matches its source's is supported by it.
Return only JSON: {"<criterion>": {"verdict": "yes|no|unclear", "reason": "..."}, ...}"""

MATCH_SYSTEM = """FACT is one statement a memory system stored about a conversation. CANDIDATES are
statements from an independent ledger of the same conversation turns. List the candidate ids
that state the same information as FACT (meaning, not wording; same speaker side).
Return only JSON: {"same": ["id", ...], "reason": "..."}"""

RECALL_SYSTEM = """ENTRY is a statement from an independent ledger of a conversation. FACTS are
statements a memory system stored from the same turns. List the fact ids that carry ENTRY's
information (meaning, not wording; same speaker side).
Return only JSON: {"carried_by": ["id", ...], "reason": "..."}"""


# ── store inventory ─────────────────────────────────────────────────────────

def _turn_of(key):
    match = re.search(r"-(\d+)$", key or "")
    return int(match.group(1)) if match else None


def _query(conn, sql, **params):
    from sqlalchemy import text
    return [dict(r._mapping) for r in conn.execute(text(sql), params)]


def inventory(conn):
    """Every judged item with its sources, read from the store. Returns
    (items, counts, uncovered)."""
    from src.memory.models import Base
    turns = {}
    for r in _query(conn, "SELECT id, batch_id, idempotency_key, raw_text, summary_text, abstract_text, "
                          "timestamp, cluster_id, session_id FROM episodic_memory"):
        turns[str(r["id"])] = dict(r, turn=_turn_of(r["idempotency_key"]))
    for r in _query(conn, "SELECT id, batch_id, idempotency_key, raw_text, timestamp FROM cold_storage"):
        turns.setdefault(str(r["id"]), dict(r, turn=_turn_of(r["idempotency_key"]), cold=True))
    by_batch = {str(t["batch_id"]): i for i, t in turns.items() if t.get("batch_id")}

    def source(ids):
        """Each turn once — a note's source_ids and batch_ids name the same turns, and listing
        them twice pushed six single-turn C notes over the judge's capacity into part-wise judging
        (Z2 F27) — with the time the store recorded for it, as ICE's own labels show it."""
        from src.memory.time_format import format_time
        out = {}
        for i in ids:
            t = turns.get(str(i)) or turns.get(by_batch.get(str(i), ""))
            if t and t["turn"] not in out:
                out[t["turn"]] = dict(turn=t["turn"], text=t["raw_text"],
                                      **({"date": format_time(t["timestamp"])} if t.get("timestamp") else {}))
        return [out[k] for k in sorted(out, key=lambda turn: turn or 0)]

    items = []
    add = lambda kind, rid, text, sources, **meta: items.append(dict(
        id=f"{kind}:{rid}", kind=kind, text=text, sources=sources, **meta))
    for i, t in turns.items():
        if t.get("summary_text"):
            body = t["summary_text"] + (f"\n[abstract] {t['abstract_text']}" if t.get("abstract_text") else "")
            add("turn_summary", i, body, source([i]))
    for r in _query(conn, "SELECT conversation_id, ordinal, text, mode, source_ids, batch_ids FROM conversation_notes"):
        add("conversation_note", f"{r['conversation_id']}:{r['ordinal']}", r["text"],
            source((r["source_ids"] or []) + (r["batch_ids"] or [])), mode=r["mode"])
    for r in _query(conn, "SELECT summary_id, ordinal, text, mode, source_ids, batch_ids FROM batch_notes"):
        add("batch_note", f"{r['summary_id']}:{r['ordinal']}", r["text"],
            source((r["source_ids"] or []) + (r["batch_ids"] or [])), mode=r["mode"])
    for r in _query(conn, "SELECT id, conversation_id, session_date, topics_covered, decisions_made, "
                          "unresolved_items, entities_updated, patterns_observed FROM session_summaries"):
        day = r["session_date"]
        ids = [i for i, t in turns.items() if t.get("timestamp") and t["timestamp"].date() == day]
        body = json.dumps({k: r[k] for k in ("topics_covered", "decisions_made", "unresolved_items",
                                             "entities_updated", "patterns_observed")}, default=str)
        add("session_summary", r["id"], body, source(ids))
    facts = {str(r["id"]): r for r in _query(conn, "SELECT id, text, speaker, modality, subject, relation, "
                                                   "object, negated, valid_until, superseded_by FROM codex_facts")}
    fact_sources = collections.defaultdict(list)
    for r in _query(conn, "SELECT fact_id, episodic_id, source_batch, quote FROM codex_fact_sources"):
        fact_sources[str(r["fact_id"])].append(r)

    def fact_line(f):
        return (f"{f['text']}  [speaker={f['speaker']}, modality={f['modality']}, "
                f"{f['subject']} --{f['relation']}--> {f['object']}{', negated' if f['negated'] else ''}"
                f"{', no longer valid' if f['valid_until'] else ''}]")

    def fact_turns(fid):
        """Each cited source as the quote with QUOTE_WINDOW characters around it (a long
        turn is not sent whole); the whole turn when the quote cannot be located."""
        out = []
        for s in fact_sources[fid]:
            t = turns.get(str(s["episodic_id"])) or turns.get(by_batch.get(str(s["source_batch"]), ""))
            if not t:
                continue
            raw, quote = t["raw_text"] or "", s["quote"] or ""
            at = raw.find(quote) if quote else -1
            if at < 0 or len(raw) <= 2 * QUOTE_WINDOW + len(quote):
                out.append(dict(turn=t["turn"], text=raw))
            else:
                lo, hi = max(0, at - QUOTE_WINDOW), min(len(raw), at + len(quote) + QUOTE_WINDOW)
                out.append(dict(turn=t["turn"], text=("…" if lo else "") + raw[lo:hi] + ("…" if hi < len(raw) else "")))
        return sorted(out, key=lambda x: x["turn"] or 0)

    for fid, f in facts.items():
        add("codex_fact", fid, fact_line(f), fact_turns(fid),
            quotes=[s["quote"] for s in fact_sources[fid]], speaker=f["speaker"])
        if f["superseded_by"] and str(f["superseded_by"]) in facts:
            newer = facts[str(f["superseded_by"])]
            add("fact_supersession", fid, f"EARLIER: {fact_line(f)}\nREPLACED BY: {fact_line(newer)}",
                fact_turns(fid) + fact_turns(str(newer["id"])))
    entities = {str(r["id"]): r for r in _query(conn, "SELECT id, canonical_name, aliases, entity_type, "
                                                      "description FROM codex_entities")}
    for r in _query(conn, "SELECT fact_id, entity_id, role FROM codex_fact_entities"):
        f, e = facts.get(str(r["fact_id"])), entities.get(str(r["entity_id"]))
        if f and e:
            add("entity_link", f"{r['fact_id']}:{r['entity_id']}:{r['role']}",
                f"FACT: {fact_line(f)}\nLINKED AS {r['role']} TO ENTITY: {e['canonical_name']} "
                f"(type {e['entity_type']}; aliases {e['aliases'] or []})", fact_turns(str(r["fact_id"])))
    links = collections.defaultdict(list)
    for r in _query(conn, "SELECT fact_id, entity_id FROM codex_fact_entities"):
        links[str(r["entity_id"])].append(str(r["fact_id"]))
    for r in _query(conn, "SELECT id, item_type, item_content, status FROM review_queue WHERE status = 'applied'"):
        content = r["item_content"] or {}
        if r["item_type"] == "entity_merge":
            facts_of = lambda eid: [facts[x] for x in links.get(str(eid), [])[:5] if x in facts]
            kept = facts_of(content.get("keep_id"))
            body = (f"MERGED '{content.get('absorb_name')}' INTO '{content.get('keep_name')}' "
                    f"(evidence {content.get('evidence')}).\nFacts now on the merged entity: "
                    + "; ".join(f["text"] for f in kept))
            add("entity_merge", r["id"], body, [w for f in kept for w in fact_turns(str(f["id"]))])
        elif r["item_type"] == "memory_slot_update":
            claims = content.get("claims") or []
            body = "SLOT TEXT CLAIMS: " + json.dumps([dict(text=c.get("text"), speaker=c.get("speaker"),
                                                           quote=c.get("quote")) for c in claims])
            add("slot_update", r["id"], body, _window_sources(content, turns))
        else:
            add("applied_decision", r["id"], f"{r['item_type']}: {json.dumps(content, default=str)[:4000]}",
                _window_sources(content, turns))
    members = collections.defaultdict(list)
    for r in _query(conn, "SELECT episodic_id, cluster_id FROM episodic_cluster_links"):
        members[str(r["cluster_id"])].append(str(r["episodic_id"]))
    for r in _query(conn, "SELECT id, name, description FROM context_clusters"):
        ids = sorted(members.get(str(r["id"]), []), key=lambda i: turns.get(i, {}).get("turn") or 0)
        sample = ids[:: max(1, len(ids) // 10)][:10]
        add("cluster", r["id"], f"CLUSTER NAME: {r['name']}\nDESCRIPTION: {r['description']}\n"
                                f"MEMBERS: {len(ids)} turns (a spread sample of up to 10 is in SOURCES)",
            source(sample))
    for r in _query(conn, "SELECT id, pattern_name, pattern_description, trigger_conditions, source_batch_ids, "
                          "is_active FROM procedural_memory"):
        add("procedural", r["id"], f"{r['pattern_name']}: {r['pattern_description']}\nTRIGGER: "
                                   f"{json.dumps(r['trigger_conditions'], default=str)}",
            source([str(b) for b in (r["source_batch_ids"] or [])]))
    counts = collections.Counter(i["kind"] for i in items)
    rows = {t: conn.exec_driver_sql(f'SELECT count(*) FROM "{t}"').scalar() for t in Base.metadata.tables}
    uncovered = {t: n for t, n in rows.items() if n and t not in COVERED | NOT_CONTENT}
    return items, dict(counts), dict(table_rows=rows, uncovered=uncovered)


def _window_sources(content, turns):
    """Turns referenced by a background decision's evidence windows (`unit` = episodic id:part)."""
    ids = {str(w.get("unit", "")).split(":")[0] for w in content.get("windows") or []}
    out = [dict(turn=turns[i]["turn"], text=turns[i]["raw_text"]) for i in ids if i in turns]
    return sorted(out, key=lambda s: s["turn"] or 0)


# ── sampling and judging ────────────────────────────────────────────────────

def _rank(history, key):
    return hashlib.sha256(f"{POLICY}|{history}|{key}".encode()).hexdigest()


def sample(items, history, per_kind=PER_KIND):
    by_kind = collections.defaultdict(list)
    for item in items:
        by_kind[item["kind"]].append(item)
    out = []
    for kind, group in sorted(by_kind.items()):
        out += sorted(group, key=lambda i: _rank(history, i["id"]))[:per_kind]
    return out


def rejudged(sampled, history):
    return [i for i in sampled if int(_rank(history, "rejudge|" + i["id"]), 16) % 100 < REJUDGE_SHARE * 100]


def later_changes(item, ledger, end_turn):
    """Ledger changes whose earlier value is cited in the item's source turns and
    that took effect after them, up to the store's last turn."""
    cited = {s["turn"] for s in item["sources"] if s["turn"]}
    if not cited or not ledger:
        return []
    by_id = {e["id"]: e for e in ledger}
    out = []
    for e in ledger:
        newer = by_id.get(e.get("superseded_by"))
        if not newer or not ({c["turn"] for c in e["citations"]} & cited):
            continue
        if max(cited) < e["valid_until_turn"] <= end_turn:
            out.append(dict(earlier=e["value"], later=newer["value"], at_turn=e["valid_until_turn"]))
    return out


def judge_prompt(item, changes):
    criteria = [c for c in KIND_CRITERIA[item["kind"]] if c != "current" or changes]
    sources = [dict(turn=s["turn"], **({"date": s["date"]} if s.get("date") else {}), text=s["text"])
               for s in item["sources"]]
    body = dict(KIND=item["kind"], ITEM=item["text"], CRITERIA=criteria, SOURCES=sources)
    if changes:
        body["LATER_CHANGES"] = changes
    return criteria, json.dumps(body, ensure_ascii=False, default=str)


PART_SYSTEM = """STATEMENTS are numbered statements an AI memory system wrote about a conversation.
PASSAGES are one part of the conversation it was written from (the conversation is too long to show
at once; other parts are judged separately). For EACH statement, say what THIS part shows:
- "supported": this part states it (meaning, not wording), credited to the right speaker;
- "misattributed": this part states it but from the other speaker, or as a suggestion where the
  statement presents it as decided;
- "contradicted": this part states something incompatible with it;
- "absent": this part does not bear on it.
Each passage carries the time the memory system recorded for it; a date or time in a statement that
matches its passage's is supported by it. Give a verdict for EVERY statement number.
Return only JSON: {"<statement number>": "supported|misattributed|contradicted|absent", ...}"""


def statements(item):
    """An item split into numbered statements for part-wise judging (session summaries'
    list fields item by item; any text by sentence)."""
    import re
    sentences = lambda text: [t.strip() for t in re.split(r"(?<=[.!?])\s+", text or "") if t.strip()]
    try:
        body = json.loads(item["text"])
    except (TypeError, ValueError):
        body = None
    if not isinstance(body, dict):
        return sentences(item["text"])
    out = []
    for value in body.values():
        if isinstance(value, list):
            out += [str(v) for v in value if str(v).strip()]
        elif isinstance(value, str):
            out += sentences(value)
    return out


def source_parts(sources, limit=SOURCE_CHARS):
    """Consecutive source turns grouped into parts under the judge's capacity."""
    parts, current, size = [], [], 0
    for src in sources:
        n = len(src["text"] or "")
        if current and size + n > limit:
            parts.append(current)
            current, size = [], 0
        current.append(src)
        size += n
    if current:
        parts.append(current)
    return parts


def combine_parts(statement_count, part_verdicts):
    """faithful: every statement supported in some part, none contradicted;
    attributed: no part shows a statement misattributed."""
    seen = {str(i): set() for i in range(1, statement_count + 1)}
    for verdicts in part_verdicts:
        for k, v in (verdicts or {}).items():
            if k in seen:
                seen[k].add(v)
    contradicted = [k for k, v in seen.items() if "contradicted" in v]
    unsupported = [k for k, v in seen.items() if not v & {"supported", "misattributed"}]
    misattributed = [k for k, v in seen.items() if "misattributed" in v and "supported" not in v]
    faithful = "no" if contradicted or unsupported else "yes"
    attributed = "no" if misattributed else "yes"
    return dict(faithful=faithful, attributed=attributed), dict(
        faithful=f"contradicted {contradicted}, unsupported {unsupported}" if faithful == "no" else "all supported",
        attributed=f"misattributed {misattributed}" if misattributed else "all attributed")


class Errors:
    """Per-item error counts across attempts (an item is retried until MAX_ITEM_ERRORS)."""

    def __init__(self, path):
        self.path = path
        self.counts = json.loads(path.read_text()) if path.exists() else {}

    def failed(self, key, exc):
        entry = self.counts.setdefault(key, dict(n=0, last=""))
        entry["n"] += 1
        entry["last"] = f"{type(exc).__name__}: {exc}"[:500]
        self.path.write_text(json.dumps(self.counts, indent=1))

    def given_up(self, key):
        return self.counts.get(key, {}).get("n", 0) >= MAX_ITEM_ERRORS


WORKERS = 4                     # judging calls in flight at once (Muse/DeepSeek limits are five-digit)


def _parallel(items, work):
    """work(item) for every item on WORKERS threads; yields (item, value, error)."""
    from concurrent.futures import ThreadPoolExecutor
    def safe(item):
        try:
            return item, work(item), None
        except Exception as exc:                    # this item only; retried next attempt
            return item, None, exc
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        yield from pool.map(safe, items)


def judge_all(items, history, ledger, end_turn, out, *, models, judge="muse", tag="judge"):
    from experiments.z2 import ledger_merge as M
    from experiments.z2.questions import ask_object
    calls = M.Calls(out / f"{tag}-raw.jsonl")
    errors = Errors(out / f"{tag}-errors.json")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}|{tag}"))
    results, pending_errors, todo, partwise = {}, 0, [], []
    for item in items:
        key = item["id"]
        if errors.given_up(key):
            results[key] = dict(status="unjudged", reason=errors.counts[key]["last"])
        elif sum(len(s["text"] or "") for s in item["sources"]) > SOURCE_CHARS:
            partwise.append(item)                   # judged in parts (faithful, attributed only)
        elif not item["sources"]:
            results[key] = dict(status="no_sources", reason="the store records no source turn for it")
        else:
            todo.append(item)

    def work_parts(item):
        numbered = statements(item)
        if not numbered:
            return None
        verdicts = []
        for part in source_parts(item["sources"]):
            body = dict(STATEMENTS={str(i): t for i, t in enumerate(numbered, 1)},
                        PASSAGES=[dict(turn=s["turn"], **({"date": s["date"]} if s.get("date") else {}),
                                       text=s["text"]) for s in part])
            # Part-wise reasoning runs longer (4 of 113 qualification calls ran out at 8,000).
            reply = ask_object(calls, judge, models[judge][1], models[judge][0], PART_SYSTEM,
                               json.dumps(body, ensure_ascii=False), session,
                               max_output_tokens=2 * JUDGE_OUTPUT_TOKENS)
            # A cut or unreadable part reply is not "nothing supported": combining it as
            # empty marked every statement unsupported (3 C fold notes, 2026-10-26). The
            # item errors instead - retried, then recorded unjudged, never "no".
            if not isinstance(reply, dict):
                raise ValueError(f"part {len(verdicts) + 1} of {item['id']}: no usable judge reply")
            # A statement the reply skips was not judged — counting it unsupported failed three
            # C notes on every statement (Z2 F27). The item errors instead, as above.
            missing = [str(i) for i in range(1, len(numbered) + 1) if str(i) not in reply]
            if missing:
                raise ValueError(f"part {len(verdicts) + 1} of {item['id']}: no verdict for {len(missing)} "
                                 f"of {len(numbered)} statements")
            verdicts.append(reply)
        return len(numbered), verdicts

    for item, got, exc in _parallel(partwise, work_parts):
        if exc is not None:
            errors.failed(item["id"], exc)
            pending_errors += 1
        elif got is None:
            results[item["id"]] = dict(status="no_statements")
        else:
            verdicts, reasons = combine_parts(*got)
            results[item["id"]] = dict(status="judged", verdicts=verdicts, reasons=reasons, partwise=True,
                                       parts=len(got[1]), not_judged={c: "sources over the judge's capacity; "
                                                                        "judged in parts" for c in
                                                                     KIND_CRITERIA[item["kind"]]
                                                                     if c not in verdicts})

    def work(item):
        changes = later_changes(item, ledger, end_turn)
        criteria, prompt = judge_prompt(item, changes)
        value = ask_object(calls, judge, models[judge][1], models[judge][0], JUDGE_SYSTEM, prompt, session,
                           max_output_tokens=JUDGE_OUTPUT_TOKENS)
        return criteria, changes, value

    for item, got, exc in _parallel(todo, work):
        key = item["id"]
        if exc is not None:
            errors.failed(key, exc)
            pending_errors += 1
            continue
        criteria, changes, value = got
        if value is None:
            results[key] = dict(status="unparseable")
            continue
        verdicts = {c: (value.get(c) or {}).get("verdict") for c in criteria}
        results[key] = dict(status="judged", verdicts=verdicts,
                            reasons={c: (value.get(c) or {}).get("reason") for c in criteria},
                            current="n/a" if "current" in KIND_CRITERIA[item["kind"]] and not changes else None)
    return results, pending_errors


def match_facts(sampled_facts, ledger, history, out, *, models, judge="muse"):
    """Fact precision against the ledger: which ledger entries (any status) state each fact."""
    from experiments.z2 import ledger_merge as M
    from experiments.z2.questions import ask_object
    calls = M.Calls(out / "match-raw.jsonl")
    errors = Errors(out / "match-errors.json")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}|match"))
    by_turn = collections.defaultdict(list)
    for e in ledger:
        for c in e["citations"]:
            by_turn[c["turn"]].append(e)
    result, pending, todo = {}, 0, []
    for f in sampled_facts:
        turns = {s["turn"] for s in f["sources"]}
        candidates = {e["id"]: e for t in turns for e in by_turn.get(t, [])}
        if not candidates:
            result[f["id"]] = dict(match="none", reason="no ledger entry cites its source turns")
        elif errors.given_up(f["id"]):
            result[f["id"]] = dict(match="unjudged", reason=errors.counts[f["id"]]["last"])
        else:
            todo.append((f, {f"L{k}": e for k, e in enumerate(candidates.values(), 1)}))

    def work(pair):
        f, alias = pair
        prompt = json.dumps(dict(FACT=f["text"], CANDIDATES=[dict(id=a, speaker=e["speaker"], text=e["value"])
                                                             for a, e in alias.items()]), ensure_ascii=False)
        return ask_object(calls, judge, models[judge][1], models[judge][0], MATCH_SYSTEM, prompt, session,
                          max_output_tokens=JUDGE_OUTPUT_TOKENS)

    for (f, alias), value, exc in _parallel(todo, work):
        if exc is not None:
            errors.failed(f["id"], exc)
            pending += 1
            continue
        same = [alias[a] for a in (value or {}).get("same", []) if a in alias]
        status = ("verified" if any(e.get("usable_for_questions") for e in same)
                  else "unconfirmed_source" if same else "none")
        result[f["id"]] = dict(match=status, entries=[e["id"] for e in same])
    return result, pending


def recall(ledger, facts_by_turn, history, out, *, models, judge="muse", n=PER_KIND, end_turn=None):
    """Fact recall: of a deterministic sample of usable ledger entries the store could
    hold (every cited turn at or before its last turn), how many a stored fact carries."""
    from experiments.z2 import ledger_merge as M
    from experiments.z2.questions import ask_object
    calls = M.Calls(out / "recall-raw.jsonl")
    errors = Errors(out / "recall-errors.json")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}|recall"))
    within = lambda e: end_turn is None or max(c["turn"] for c in e["citations"]) <= end_turn
    usable = sorted((e for e in ledger if e.get("usable_for_questions") and within(e)),
                    key=lambda e: _rank(history, e["id"]))[:n]
    result, pending, todo = {}, 0, []
    for e in usable:
        candidates = {f["id"]: f for c in e["citations"] for f in facts_by_turn.get(c["turn"], [])}
        if not candidates:
            result[e["id"]] = dict(carried=False, reason="no stored fact from its turns", kind=e["kind"])
        elif errors.given_up(e["id"]):
            result[e["id"]] = dict(carried=None, reason=errors.counts[e["id"]]["last"])
        else:
            todo.append((e, {f"F{k}": f for k, f in enumerate(candidates.values(), 1)}))

    def work(pair):
        e, alias = pair
        prompt = json.dumps(dict(ENTRY=dict(speaker=e["speaker"], text=e["value"]),
                                 FACTS=[dict(id=a, text=f["text"]) for a, f in alias.items()]), ensure_ascii=False)
        return ask_object(calls, judge, models[judge][1], models[judge][0], RECALL_SYSTEM, prompt, session,
                          max_output_tokens=JUDGE_OUTPUT_TOKENS)

    for (e, alias), value, exc in _parallel(todo, work):
        if exc is not None:
            errors.failed(e["id"], exc)
            pending += 1
            continue
        carried = [a for a in (value or {}).get("carried_by", []) if a in alias]
        result[e["id"]] = dict(carried=bool(carried), facts=[alias[a]["id"] for a in carried], kind=e["kind"])
    return result, pending


REACH_TRACEABLE = {"codex_fact", "turn_summary", "procedural"}


def reach(seed_path, summaries=None):
    """How many history prompts each artifact reached in the replay (the final
    `preflight.selected` fragments). Facts count by id; a turn summary counts only
    when its own text was shown (the row id alone cannot tell summary from raw).
    Other kinds are not identifiable in the trace and are reported as such."""
    counts = collections.Counter()
    heads = {rid: text[:200] for rid, text in (summaries or {}).items() if text}
    if not Path(seed_path).exists():
        return counts
    with open(seed_path) as handle:
        for line in handle:
            if '"event": "turn"' not in line:
                continue
            for f in (json.loads(line).get("preflight") or {}).get("selected") or []:
                for fact in f.get("origin_facts") or []:
                    counts[fact] += 1
                for table, rid in f.get("stored_artifact_ids") or []:
                    if table == "procedural_memory":
                        counts[rid] += 1
                row = f.get("source_row")
                if row in heads and heads[row] in (f.get("text") or ""):
                    counts[row] += 1
    return counts


# ── report ─────────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def report(history, counts, store, sampled, results, second, matches, recalled, reached):
    kinds = {}
    for kind in sorted(counts):
        mine = [i for i in sampled if i["kind"] == kind]
        status = collections.Counter(results.get(i["id"], {}).get("status", "pending") for i in mine)
        criteria = {}
        for c in KIND_CRITERIA[kind]:
            votes = [results[i["id"]]["verdicts"].get(c) for i in mine
                     if results.get(i["id"], {}).get("status") == "judged" and c in results[i["id"]]["verdicts"]]
            yes, no = votes.count("yes"), votes.count("no")
            criteria[c] = dict(yes=yes, no=no, unclear=len(votes) - yes - no,
                               yes_rate=round(yes / (yes + no), 3) if yes + no else None, ci95=wilson(yes, yes + no),
                               not_applicable=sum(1 for i in mine if results.get(i["id"], {}).get("current") == "n/a")
                               if c == "current" else None)
        traceable = kind in REACH_TRACEABLE
        reach_counts = [reached.get(i["id"].split(":", 1)[1], 0) for i in mine]
        kinds[kind] = dict(items=counts[kind], sampled=len(mine), status=dict(status), criteria=criteria,
                           sampled_reaching_a_prompt=sum(1 for r in reach_counts if r) if traceable
                           else "not traceable in the replay trace",
                           mean_prompt_reach=(round(sum(reach_counts) / len(mine), 2) if mine else None)
                           if traceable else None)
    agree, disagree = collections.Counter(), []
    for key, other in second.items():
        first = results.get(key, {})
        if first.get("status") != "judged" or other.get("status") != "judged":
            continue
        for c, v in first["verdicts"].items():
            w = other["verdicts"].get(c)
            agree["compared"] += 1
            if v == w:
                agree["same"] += 1
            else:
                disagree.append(dict(item=key, criterion=c, muse=v, deepseek=w,
                                     muse_reason=first["reasons"].get(c), deepseek_reason=other["reasons"].get(c)))
    match_counts = collections.Counter(m["match"] for m in matches.values())
    # A fact the ledger lacks is not therefore wrong (the ledger is selective): cross
    # the match with the faithful verdict — no match + faithful = true but unledgered;
    # no match + not faithful = invented.
    crossed = collections.Counter(
        f"{m['match']}+faithful_{results.get(k, {}).get('verdicts', {}).get('faithful', 'unjudged')}"
        for k, m in matches.items())
    carried = [r["carried"] for r in recalled.values() if r["carried"] is not None]
    return dict(policy=POLICY, history=history,
                agreement=dict(compared=agree["compared"], same=agree["same"],
                               rate=round(agree["same"] / agree["compared"], 3) if agree["compared"] else None),
                kinds=kinds, store=store,
                fact_match_by_faithful=dict(crossed),
                fact_precision=dict(match_counts, rate_verified=round(match_counts["verified"] / sum(match_counts.values()), 3)
                                    if matches else None),
                fact_recall=dict(sampled=len(recalled), carried=sum(carried),
                                 rate=round(sum(carried) / len(carried), 3) if carried else None),
                disagreements=disagree)


# ── driver ─────────────────────────────────────────────────────────────────

def load_ledger(path, changes=None):
    from experiments.z2 import ledger_merge as M
    from experiments.z2.questions import with_changes
    entries = json.loads(Path(path).read_text())["entries"]
    if changes:
        return with_changes(entries, json.loads(Path(changes).read_text()))
    return M.chain(entries)


def freeze(history, database, out, per_kind=PER_KIND):
    """Freeze the store's inventory to items.json (once). FINAL runs this locally right after
    the replay (`--inventory-only`): its read passes then restore checkpoints into the same
    database, so a later inventory would see a checkpoint, not the final store."""
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "items.json"
    if not frozen.exists():
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url
        from src.api.config import settings
        engine = create_engine(make_url(settings.database_url).set(database=database))
        with engine.connect() as conn:
            items, counts, store = inventory(conn)
        engine.dispose()
        tmp = frozen.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(policy=POLICY, history=history, database=database, items=items,
                                       counts=counts, store=store, per_kind=per_kind),
                                  default=str, ensure_ascii=False))
        tmp.replace(frozen)
    return frozen


def run(history, database, replay, ledger_path, out, *, changes=None, models=None, per_kind=PER_KIND):
    """Resumable: the inventory is frozen to items.json on the first run, so later
    attempts judge exactly the same items even if the store has moved on.
    Returns (judged_or_settled, total_units, pending_errors)."""
    frozen = freeze(history, database, out, per_kind)
    data = json.loads(frozen.read_text())
    items, counts, store = data["items"], data["counts"], data["store"]
    per_kind = data.get("per_kind", PER_KIND)          # frozen with the inventory
    ledger = load_ledger(ledger_path, changes)
    end_turn = max((s["turn"] or 0 for i in items for s in i["sources"]), default=0)
    sampled = sample(items, history, per_kind)
    results, e1 = judge_all(sampled, history, ledger, end_turn, out, models=models)
    second, e2 = judge_all(rejudged(sampled, history), history, ledger, end_turn, out, models=models,
                           judge="deepseek", tag="rejudge")
    facts = [i for i in sampled if i["kind"] == "codex_fact"]
    matches, e3 = match_facts(facts, ledger, history, out, models=models)
    facts_by_turn = collections.defaultdict(list)
    for f in (i for i in items if i["kind"] == "codex_fact"):
        for s in f["sources"]:
            facts_by_turn[s["turn"]].append(f)
    recalled, e4 = recall(ledger, facts_by_turn, history, out, models=models, n=per_kind, end_turn=end_turn)
    summaries = {i["id"].split(":", 1)[1]: i["text"].split("\n[abstract]")[0]
                 for i in items if i["kind"] == "turn_summary"}
    reached = reach(Path(replay) / "seed.jsonl", summaries)
    result = report(history, counts, store, sampled, results, second, matches, recalled, reached)
    (out / "bg-audit.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    (out / "details.json").write_text(json.dumps(dict(judged=results, rejudged=second, fact_matches=matches,
                                                      fact_recall=recalled), indent=1, ensure_ascii=False,
                                                 default=str) + "\n")
    pending = e1 + e2 + e3 + e4
    (out / "status.json").write_text(json.dumps(dict(pending_errors=pending, finished=not pending)) + "\n")
    return result, pending


def progress(out):
    """(settled, total) judging units for the runner's meter, from files only."""
    out = Path(out)
    frozen = out / "items.json"
    if not frozen.exists():
        return 0, None
    data = json.loads(frozen.read_text())
    per_kind = data.get("per_kind", PER_KIND)
    sampled = sample(data["items"], data["history"], per_kind)
    total = (len(sampled) + len(rejudged(sampled, data["history"]))
             + sum(1 for i in sampled if i["kind"] == "codex_fact") + per_kind)
    done = 0
    for name in ("judge-raw.jsonl", "rejudge-raw.jsonl", "match-raw.jsonl", "recall-raw.jsonl"):
        path = out / name
        if path.exists():
            with path.open() as handle:
                done += sum(1 for _ in handle)
    return min(done, total), total


def call_seconds(out, last=50):
    """Median seconds per judging call (the runner's rate), from the raw call files."""
    seconds = []
    for name in ("judge-raw.jsonl", "rejudge-raw.jsonl", "match-raw.jsonl", "recall-raw.jsonl"):
        path = Path(out) / name
        if path.exists():
            seconds += [json.loads(line).get("seconds") or 0 for line in path.read_text().splitlines()[-last:]]
    seconds = sorted(x for x in seconds if x)
    return seconds[len(seconds) // 2] if len(seconds) >= 3 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--replay", type=Path, required=True, help="the replay directory (its seed.jsonl)")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=PER_KIND, help="sample size per kind (a trial run)")
    parser.add_argument("--inventory-only", action="store_true",
                        help="freeze the final store's inventory and stop (local; no model calls)")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("audit output is private: keep it under logs/")
    if args.inventory_only:
        print(json.dumps(dict(frozen=str(freeze(args.history, args.database, args.out, args.per_kind)))))
        return 0
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    models = {r: (generator(r).generate, metadata(r)) for r in ("muse", "deepseek")}
    result, pending = run(args.history, args.database, args.replay, args.ledger, args.out,
                          changes=args.changes, models=models, per_kind=args.per_kind)
    print(json.dumps(dict(agreement=result["agreement"], fact_precision=result["fact_precision"],
                          fact_recall=result["fact_recall"], pending_errors=pending)))
    return 3 if pending else 0                      # nonzero: items still to retry; the runner resumes


if __name__ == "__main__":
    raise SystemExit(main())
