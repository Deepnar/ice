"""CODEX_V4 fact writer: verified v4 claims → codex_facts, their sources and entity links.

docs/specs/CODEX_V4.md D4–D7 and §2. Input is a summary-support record whose
assertions already passed the quote/speaker check and NLI (method v4); this
module applies the deterministic post-rules, resolves entities, merges exact
duplicates and writes. Nothing here admits a fact: admission was the claim
pass's job. Same / replaces / both_true (D6) plugs in through *decide*.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import structlog
from sqlalchemy import exists as db_exists
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.api.config import settings
from src.memory.models import CodexEntity, CodexEvent, CodexFact, CodexFactEntity, CodexFactSource
from src.memory.source import digest

logger = structlog.get_logger("ice.memory.facts")

# The namespace codex_extractor.generate_uuid5 used, so a name keeps its entity id.
ENTITY_NAMESPACE = uuid.NAMESPACE_DNS
PARTIES = ("user", "assistant")
# Resolving a pronoun to the party it names is value resolution, not intent
# inference (CLAUDE.md: parsers that RESOLVE a value may stay).
_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
_QUOTES = "\"'`“”‘’*()[]{}<>"
_HUSK = "NOT (properties ? 'merged_into') AND NOT (properties ? 'deleted_reason')"


def _flat(value: str) -> str:
    return " ".join(re.sub(r"[*`#_\"'“”]", " ", value or "").lower().split())


def canonical(name: str) -> str:
    """Lowercase, trimmed, quote-stripped, single-spaced; a leading article dropped."""
    s = " ".join((name or "").strip().strip(_QUOTES).strip(" .,;:").split()).lower()
    return s[4:] if s.startswith("the ") and len(s) > 4 else s


def party(name: str, speaker: str) -> Optional[str]:
    """'user'/'assistant' by name, or a pronoun resolved by who spoke (fixes A3)."""
    n = canonical(name)
    if n in PARTIES:
        return n
    if speaker in PARTIES:
        if n in _FIRST_PERSON:
            return speaker
        if n in _SECOND_PERSON:
            return "assistant" if speaker == "user" else "user"
    return None


def linkable(name: str) -> bool:
    """An entity is a name, not a value: long objects (sentences, tensors, URLs
    with queries) stay as text on the fact."""
    n = canonical(name)
    return (bool(n) and len(n) <= settings.codex_fact_entity_max_chars
            and len(n.split()) <= settings.codex_fact_entity_max_words
            and any(ch.isalpha() for ch in n))


def post_rules(claim: dict, raw_text: str) -> tuple[dict, list, bool]:
    """CODEX_V4 D4 deterministic rules. Returns (claim, flags, drop)."""
    claim, flags = dict(claim), []
    if claim.get("modality") == "shared" and claim.get("speaker") in ("assistant", "document"):
        # Only the user pastes into a conversation; the model ignored "never" 43×.
        claim["modality"] = "asserted"
        flags.append("relabel_shared")
    s, o = canonical(claim.get("subject", "")), canonical(claim.get("object", ""))
    if s and o and s == o:
        return claim, flags + ["drop_self_relation"], True
    hay = _flat(claim.get("quote", "")) + " " + _flat(claim.get("text", ""))
    for field, value in (("subject", s), ("object", o)):
        if value and value not in PARTIES and _flat(value) not in hay:
            flags.append(f"{field}_absent")
    obj = " ".join((claim.get("object") or "").split())
    if obj and raw_text:
        flat_raw = " ".join(raw_text.split())
        i = flat_raw.find(obj)
        nxt = flat_raw[i + len(obj): i + len(obj) + 2] if i >= 0 else ""
        # Cut = the source keeps going inside the same token (".../platform.open" + "ai.com").
        if nxt and (nxt[0].isalnum() or nxt[0] in "/_-"
                    or (nxt[0] == "." and len(nxt) > 1 and nxt[1].isalnum())):
            flags.append("object_cut")
    return claim, flags, False


def _entity_query(db, sql: str, **params):
    row = db.execute(text(sql), params).first()
    return db.get(CodexEntity, row[0]) if row else None


def resolve_entity(db, name: str, *, embedder, entity_type: str = "entity") -> CodexEntity:
    """Deterministic only (D5): exact name/alias → merge_key → new. Near variants
    (plurals, typos) go to the maintenance agent's Tier 2 model check.

    No trigram tier: measured 2026-10-16, a plural sits at 0.885-0.893 and a
    different model's version suffix at 0.882 ("devstral-small" / "-2"), so no
    threshold separates them (the TRAPS 26 shape); at 0.9 even typos (0.75-0.88)
    miss, and anything above is already an exact or merge_key match."""
    from src.workers.maintenance_agent import merge_key

    surface = " ".join(name.split()).strip(_QUOTES)
    key = canonical(name)
    entity = _entity_query(db, f"""
        SELECT id FROM codex_entities WHERE canonical_name = :c
        UNION ALL
        SELECT id FROM codex_entities WHERE source = 'conversation' AND {_HUSK}
          AND EXISTS (SELECT 1 FROM unnest(aliases) a WHERE lower(a) = :c)
        LIMIT 1""", c=key)
    if entity is not None:
        merged_into = (entity.properties or {}).get("merged_into")
        return db.get(CodexEntity, uuid.UUID(merged_into)) if merged_into else entity
    entity = _entity_query(db, f"""
        SELECT id FROM codex_entities WHERE source = 'conversation' AND {_HUSK}
          AND properties->>'merge_key' = :k ORDER BY canonical_name LIMIT 1""", k=merge_key(key))
    if entity is not None:
        if surface and surface not in (entity.aliases or []):
            entity.aliases = list(entity.aliases or []) + [surface]
        return entity
    created = CodexEntity(
        id=uuid.uuid5(ENTITY_NAMESPACE, key), canonical_name=key, aliases=[surface] if surface else [],
        tags=[], entity_type=entity_type, properties={"merge_key": merge_key(key)},
        context_payload="", source="conversation",
        embedding=embedder.encode(key, convert_to_tensor=False).tolist(),
        last_updated=datetime.now(timezone.utc))
    try:
        with db.begin_nested():
            db.add(created)
    except IntegrityError:          # created concurrently under the same deterministic id
        return db.get(CodexEntity, created.id)
    return created


def _party_entity(db, name: str, *, embedder) -> CodexEntity:
    return resolve_entity(db, name, embedder=embedder,
                          entity_type="person" if name == "user" else "assistant")


def _source_row(fact_id, turn, assertion) -> CodexFactSource:
    spans = assertion.get("cited_spans") or []
    return CodexFactSource(
        fact_id=fact_id, source_batch=turn.batch_id,
        window_sha256=digest(json.dumps(spans, sort_keys=True, default=str)),
        episodic_id=getattr(turn, "id", None), conversation_id=getattr(turn, "conversation_id", None),
        window={"sources": assertion.get("sources"), "cited_spans": spans},
        quote=assertion.get("quote", ""), recorded_at=getattr(turn, "timestamp", None),
        verification=assertion.get("verification") or {})


# ── D6: same / replaces / both_true ──────────────────────────────────────────
# Qualified 2026-10-16 (CODEX_V4 §5.2, 99 real same-speaker pairs + 12 planted):
# asking "which verdict" made both gemma4:e4b and a 27B call "replaces" on 18-23
# of 94 both-true pairs; asking "same information?" and "can both be true now?"
# (Graphiti's framing) cut gemma to 6, and the label rule below to 2 — both a
# newer, more detailed version of the same details.
RECONCILE_SYSTEM = (
    "You compare two memory facts said by the same speaker: an EARLIER one and a NEWER one. "
    "Answer two questions about what the sentences say:\n"
    "1. same_information: do both sentences state the same thing, with neither adding or "
    "changing any information?\n"
    "2. both_can_be_true: could both sentences be true at the same time, right now? Answer "
    "false only if the newer one says the earlier one is no longer so - a changed value, a "
    "cancelled or reversed decision, a stopped habit, a correction. Different details, extra "
    "detail, items of a list, a question, a suggestion, a possibility or a different direction "
    "of a relation can all be true together: answer true.\n"
    "3. same_item: are both sentences about the same specific thing - the same item, file, "
    "run, batch, test, event, decision or the same property of one thing? Two runs, two files, "
    "two batches or two lists of the same kind are different things: answer false.\n"
    "Use only what the sentences say.")
# pre-G60 (main, what Z2 measured): the first two questions only
RECONCILE_PRE_G60_SYSTEM = RECONCILE_SYSTEM.split("3. same_item")[0].rstrip() + "\nUse only what the sentences say."
RECONCILE_PRE_G60_SCHEMA = {"type": "object", "additionalProperties": False,
                            "required": ["same_information", "both_can_be_true"],
                            "properties": {"same_information": {"type": "boolean"},
                                           "both_can_be_true": {"type": "boolean"}}}
# Z2 F20 (2026-10-29): with the two sentences alone no design separated correct from wrong
# replacements (S8: the pre-G60 rule kept 44 of 53 wrong ones, G60 lost 35 of 46 correct ones);
# what the audit's judge used was the source passages. Graphiti's rule — contradicted only on
# explicit evidence, never from recency — with each fact's cited quote and date, reason first.
RECONCILE_CONTEXT_SYSTEM = (
    "You compare two memory facts said by the same speaker: an EARLIER one and a NEWER one. Each "
    "comes with the passage it was taken from and when it was said, and may come with the user "
    "message that prompted it and its recorded speaker, modality, subject, relation and object.\n"
    "- about: first, in one sentence, what each fact is about and whether the newer one changes the "
    "earlier one, repeats it, adds to it, or is a separate occurrence.\n"
    "- same_information: do both facts state the same thing, neither adding nor changing anything?\n"
    "- contradicts: does the NEWER fact make the EARLIER one no longer true - it corrects it, updates "
    "the same value, reverses it, or says it stopped? Answer true only with explicit evidence in the "
    "facts and their passages. Never infer it from the newer one being later. A different instance of "
    "the same kind (another run, file, error, batch, list or measurement of a different thing), extra "
    "detail, a restatement, a question, a suggestion or a possibility does not contradict.\n"
    "Judge only from the facts and their passages.")
RECONCILE_CONTEXT_SCHEMA = {"type": "object", "additionalProperties": False,
                            "required": ["about", "same_information", "contradicts"],
                            "properties": {"about": {"type": "string"}, "same_information": {"type": "boolean"},
                                           "contradicts": {"type": "boolean"}}}
RECONCILE_SCHEMA = {"type": "object", "additionalProperties": False,
                    "required": ["same_information", "both_can_be_true", "same_item"],
                    "properties": {"same_information": {"type": "boolean"},
                                   "both_can_be_true": {"type": "boolean"},
                                   "same_item": {"type": "boolean"}}}
# One call screens all of a new fact's candidates with the same two questions; only
# the ones it calls "same" or "replaces" are re-asked alone with RECONCILE_SYSTEM,
# whose answer decides (Z2 S0, 2026-10-18). Calls per new fact fell from one per
# candidate (2.3 per fact by turn 150 of history B, rising toward 28) to one plus
# about one confirmation in five. Qualified on the §5.2 set against single-pair:
# 90/99 vs 88/99 real, planted 10/12 both, false replaces 6 -> 1, missed "same"
# 5 -> 7, 0/60 distractors called "replaces" (logs/z2-s0-reconcile-batch-*).
RECONCILE_BATCH_SYSTEM = (
    "You compare memory facts said by the same speaker: one NEWER fact and a list of EARLIER "
    "facts. For EACH earlier fact, answer two questions about what the sentences say:\n"
    "1. same_information: do the earlier sentence and the newer sentence state the same thing, "
    "with neither adding or changing any information?\n"
    "2. both_can_be_true: could the earlier and the newer sentence both be true at the same time, "
    "right now? Answer false only if the newer one says the earlier one is no longer so - a "
    "changed value, a cancelled or reversed decision, a stopped habit, a correction. Different "
    "details, extra detail, items of a list, a question, a suggestion, a possibility or a "
    "different direction of a relation can all be true together: answer true.\n"
    "3. same_item: are the earlier and the newer sentence about the same specific thing - the "
    "same item, file, run, batch, test, event, decision or the same property of one thing? Two "
    "runs, two files, two batches or two lists of the same kind are different things: answer false.\n"
    "Judge each earlier fact on its own against the newer one. Use only what the sentences say.")


def reconcile_batch_schema(ids) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["answers"],
            "properties": {"answers": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["id", "same_information", "both_can_be_true", "same_item"],
                          "properties": {"id": {"type": "string", "enum": list(ids)},
                                         "same_information": {"type": "boolean"},
                                         "both_can_be_true": {"type": "boolean"},
                                         "same_item": {"type": "boolean"}}}}}}


def screen(claim: dict, earlier: list, *, ask_many: Optional[Callable]) -> list:
    """Which candidates need the single-pair question: all of them, unless one batched
    call says a candidate is plainly both_true. *ask_many(earlier_texts, newer_text)*
    returns one {same_information, both_can_be_true} per text, in order. A failed or
    short batch falls back to asking every candidate alone — warned every time."""
    if ask_many is None or len(earlier) < 2:
        return [True] * len(earlier)
    try:
        answers = ask_many([e.text for e in earlier], claim["text"].strip())
        if len(answers) != len(earlier):
            raise ValueError(f"{len(answers)} answers for {len(earlier)} facts")
        item = settings.codex_fact_reconcile_design == "g60"
        return [bool(a["same_information"])
                or (not bool(a["both_can_be_true"]) and (bool(a.get("same_item")) or not item)) for a in answers]
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("fact_reconcile_batch_failed", candidates=len(earlier),
                       error_type=type(exc).__name__)
        return [True] * len(earlier)


# A label that is not a statement of fact never replaces, and is never replaced
# (the label is ~85-90% reliable; this rule removed 3 of 5 false replacements).
_NO_REPLACE = {"asked", "suggested", "hypothetical"}


def candidates(db, claim: dict, vector, *, exclude=()) -> list:
    """Current facts of the same speaker that share a named entity with the
    claim, or are among the k nearest at cosine >= the floor."""
    from pgvector.sqlalchemy import Vector as PgVector
    from sqlalchemy import bindparam
    from src.workers.maintenance_agent import merge_key
    keys = [merge_key(canonical(claim.get(r) or "")) for r in ("subject", "object")
            if claim.get(r) and party(claim[r], claim["speaker"]) is None and linkable(claim[r])]
    ids = set()
    if keys:
        ids |= {r[0] for r in db.execute(text("""
            SELECT DISTINCT f.id FROM codex_facts f
            JOIN codex_fact_entities fe ON fe.fact_id = f.id
            JOIN codex_entities e ON e.id = fe.entity_id
            WHERE f.valid_until IS NULL AND f.speaker = :sp AND fe.role IN ('subject', 'object')
              AND e.properties->>'merge_key' = ANY(:keys)
            LIMIT 20"""), {"sp": claim["speaker"], "keys": keys})}
    rows = db.execute(text("""
        SELECT id, 1 - (embedding <=> :v) AS sim FROM codex_facts
        WHERE valid_until IS NULL AND speaker = :sp AND embedding IS NOT NULL
        ORDER BY embedding <=> :v LIMIT :k""").bindparams(bindparam("v", type_=PgVector)),
        {"v": [float(x) for x in vector], "sp": claim["speaker"],
         "k": settings.codex_fact_candidate_k}).fetchall()
    ids |= {r.id for r in rows if r.sim >= settings.codex_fact_candidate_cosine}
    ids -= set(exclude)
    return [db.get(CodexFact, i) for i in sorted(ids, key=str)]


def reconcile(claim: dict, earlier, *, ask: Callable, when, passages=None) -> str:
    """'same' | 'replaces' | 'both_true' for one earlier fact. *ask(earlier_text,
    newer_text[, passages])* returns the answer for codex_fact_reconcile_design (passages: each
    fact's cited quote and date, for "context"). An older observation never replaces a newer one;
    unknown or failed answers are both_true (TRAPS 99)."""
    design = settings.codex_fact_reconcile_design
    try:
        if design == "context":
            answer = ask(earlier.text, claim["text"].strip(), passages)
            same, compatible, same_item = bool(answer["same_information"]), not bool(answer["contradicts"]), True
        else:
            answer = ask(earlier.text, claim["text"].strip())
            same, compatible = bool(answer["same_information"]), bool(answer["both_can_be_true"])
            same_item = bool(answer.get("same_item")) if design == "g60" else True
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("fact_reconcile_failed", error_type=type(exc).__name__)
        return "both_true"
    if same:
        return "same"
    # Z2 S8 (2026-10-26): of 99 judged replacements 53 were wrong - restatements, extra
    # detail, and two different runs/files/batches of the same kind (designs: Z2 F20).
    if (compatible or not same_item
            or claim.get("modality") in _NO_REPLACE or earlier.modality in _NO_REPLACE
            or (earlier.mentioned_at and when and when < earlier.mentioned_at)):
        return "both_true"
    return "replaces"


CONTEXT_WINDOW = 1200          # characters of the source turn around a fact's cited quote


def _around(text: str, quote: str, size: int = CONTEXT_WINDOW) -> str:
    """The part of a source turn around a fact's quote (exact match first, else most shared words)."""
    if len(text) <= size:
        return text
    at = text.find(quote[:80]) if quote else -1
    if at >= 0:
        start = max(0, min(at + len(quote) // 2 - size // 2, len(text) - size))
        return text[start:start + size]
    words = set(re.findall(r"[a-z0-9]{3,}", (quote or "").lower()))
    starts = range(0, len(text) - size + 1, size // 3)
    best = max(starts, key=lambda s: len(words & set(re.findall(r"[a-z0-9]{3,}", text[s:s + size].lower()))))
    return text[best:best + size]


def context_passage(mode: str, *, quote: str, said, turn_text: str = "", user_text: str = "",
                    fields: Optional[dict] = None) -> dict:
    """One fact's evidence for the "context" reconcile design, per codex_fact_reconcile_context:
    quote — the cited sentence; window — the source turn around it; turn — the window plus the
    user message of that turn (why it was said); fields — the window plus the fact's stored
    speaker / modality / subject / relation / object (Z2 F20)."""
    out = {"said": said}
    out["passage"] = quote if mode == "quote" or not turn_text else _around(turn_text, quote)
    if mode == "turn" and user_text and user_text[:200] not in out["passage"]:
        out["user_message"] = user_text[:600]
    if mode == "fields" and fields:
        out.update({k: v for k, v in fields.items() if v})
    return out


def utc_stamp(when) -> Optional[str]:
    """'YYYY-MM-DD HH:MM UTC' — one form for every date the model sees (a naive time is taken as UTC)."""
    if when is None:
        return None
    if isinstance(when, str):
        when = datetime.fromisoformat(when.replace("Z", "+00:00"))
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fields(fact) -> dict:
    get = (lambda k: fact.get(k)) if isinstance(fact, dict) else (lambda k: getattr(fact, k, None))
    return {k: get(k) for k in ("speaker", "modality", "subject", "relation", "object")}


def _turn_texts(row):
    """(whole turn text, its user message) of a stored turn."""
    from src.memory.source import source_units
    if row is None:
        return "", ""
    user = " ".join(u.text for u in source_units(row) if u.role == "user")
    return getattr(row, "raw_text", "") or "", user


def passages(db, earlier, assertion, claim, when, turn=None):
    """Each fact's evidence for the "context" reconcile design; None for the other designs."""
    if settings.codex_fact_reconcile_design != "context":
        return None
    from src.memory.models import EpisodicMemory
    mode = settings.codex_fact_reconcile_context
    stamp = lambda t: utc_stamp(t)
    source = (db.query(CodexFactSource).filter(CodexFactSource.fact_id == earlier.id)
              .order_by(CodexFactSource.recorded_at).first())
    row = db.get(EpisodicMemory, source.episodic_id) if source is not None and source.episodic_id and mode != "quote" else None
    e_text, e_user = _turn_texts(row)
    n_text, n_user = _turn_texts(turn) if mode != "quote" else ("", "")
    return dict(
        earlier=context_passage(mode, quote=getattr(source, "quote", None) or earlier.text,
                                said=stamp(getattr(source, "recorded_at", None) or earlier.mentioned_at),
                                turn_text=e_text, user_text=e_user, fields=_fields(earlier)),
        newer=context_passage(mode, quote=assertion.get("quote") or claim["text"], said=stamp(when),
                              turn_text=n_text, user_text=n_user, fields=_fields(claim)))


def write_turn_facts(db, turn, record: dict, *, embedder,
                     ask: Optional[Callable] = None, mentions: Optional[Callable] = None,
                     ask_many: Optional[Callable] = None) -> dict:
    """Write one turn's verified v4 assertions as facts. Caller commits.

    Exact-text duplicates of the same speaker merge without a model. *ask* is
    D6's question (see `reconcile`); None skips it, so every other claim is a
    new fact. *ask_many* screens a claim's candidates in one call (see `screen`). *mentions(text)* returns entity names found in a fact sentence
    (NuNER, background tier); None skips mention links.
    """
    counts = {"assertions": 0, "dropped": 0, "new": 0, "supported": 0, "links": 0,
              "superseded": 0, "reconcile_calls": 0, "reconcile_batches": 0}
    written = set()
    rows = [a for a in (record or {}).get("assertions") or []]
    counts["assertions"] = len(rows)
    when = getattr(turn, "timestamp", None) or datetime.now(timezone.utc)
    prepared = []
    for assertion in rows:
        claim, flags, drop = post_rules(assertion, getattr(turn, "raw_text", "") or "")
        if drop:
            counts["dropped"] += 1
            continue
        prepared.append((assertion, claim, flags))
    if not prepared:
        return counts
    vectors = embedder.encode([c["text"].strip() for _, c, _ in prepared], convert_to_tensor=False)
    for (assertion, claim, flags), vector in zip(prepared, vectors):
        body = claim["text"].strip()
        sha = digest(body)
        existing = (db.query(CodexFact)
                    .filter(CodexFact.text_sha256 == sha, CodexFact.speaker == claim["speaker"],
                            CodexFact.valid_until.is_(None))
                    .order_by(CodexFact.mentioned_at).first())
        replaced = []
        if existing is None and ask is not None:
            pool = candidates(db, claim, vector, exclude=written)
            needs = screen(claim, pool, ask_many=ask_many)
            counts["reconcile_batches"] += ask_many is not None and len(pool) > 1
            for earlier, confirm in zip(pool, needs):
                if not confirm:
                    continue
                counts["reconcile_calls"] += 1
                verdict = reconcile(claim, earlier, ask=ask, when=when,
                                    passages=passages(db, earlier, assertion, claim, when, turn))
                if verdict == "same":
                    existing = earlier
                    break
                if verdict == "replaces":
                    replaced.append(earlier)
        if existing is not None:
            source = _source_row(existing.id, turn, assertion)
            if db.get(CodexFactSource, (existing.id, source.source_batch, source.window_sha256)) is None:
                db.add(source)
                db.flush()
                existing.support_count = db.execute(text(
                    "SELECT count(DISTINCT source_batch) FROM codex_fact_sources WHERE fact_id = :f"),
                    {"f": existing.id}).scalar()
                counts["supported"] += 1
                _journal(db, existing, "fact_supported", turn.batch_id)
            continue
        scores = (assertion.get("verification") or {}).get("scores") or {}
        fact = CodexFact(
            text=body, text_sha256=sha, speaker=claim["speaker"], modality=claim["modality"],
            subject=claim.get("subject", ""), relation=claim.get("relation", ""),
            object=claim.get("object", ""), embedding=list(map(float, vector)),
            confidence=scores.get("entailment"), mentioned_at=when, valid_from=when, flags=flags)
        db.add(fact)
        db.flush()
        db.add(_source_row(fact.id, turn, assertion))
        written.add(fact.id)
        counts["new"] += 1
        counts["links"] += _link(db, fact, claim, embedder=embedder, mentions=mentions)
        _journal(db, fact, "fact_added", turn.batch_id)
        for old in replaced:
            # Kept, never deleted: valid_until + superseded_by; undo = restore_fact.
            old.valid_until, old.superseded_by = when, fact.id
            _journal(db, old, "fact_superseded", turn.batch_id, superseded_by=str(fact.id))
            counts["superseded"] += 1
    db.flush()
    return counts


def _link(db, fact, claim, *, embedder, mentions) -> int:
    targets = []
    for role in ("subject", "object"):
        name = claim.get(role) or ""
        who = party(name, claim["speaker"])
        if who:
            targets.append((role, _party_entity(db, who, embedder=embedder)))
        elif linkable(name):
            targets.append((role, resolve_entity(db, name, embedder=embedder)))
    if mentions is not None:
        named = {canonical(claim.get("subject", "")), canonical(claim.get("object", ""))}
        for name in mentions(fact.text):
            if linkable(name) and canonical(name) not in named and canonical(name) not in PARTIES:
                targets.append(("mention", resolve_entity(db, name, embedder=embedder)))
    seen = set()
    for role, entity in targets:
        if entity is None or (entity.id, role) in seen:
            continue
        seen.add((entity.id, role))
        db.add(CodexFactEntity(fact_id=fact.id, entity_id=entity.id, role=role))
    db.flush()
    return len(seen)


def _journal(db, fact, event_type: str, batch, **extra) -> set:
    """codex_events is per entity: one row per entity the fact links to.
    Returns those entity ids."""
    entity_ids = {r[0] for r in db.execute(text(
        "SELECT DISTINCT entity_id FROM codex_fact_entities WHERE fact_id = :f"), {"f": fact.id})}
    for entity_id in entity_ids:
        db.add(CodexEvent(entity_id=entity_id, event_type=event_type, batch_source=batch,
                          payload={"fact_id": str(fact.id), "text_sha256": fact.text_sha256, **extra}))
    return entity_ids


# ── Deletion and forget (C10 contract: expire + journal, never hard-delete) ──

def source_plan(db, removed_batches) -> tuple[list, list]:
    """Current facts with a source among *removed_batches*: (sole, surviving) ids.
    Sole = every source is being removed; surviving = support remains elsewhere."""
    if not removed_batches:
        return [], []
    rows = db.execute(text("""
        SELECT f.id, array_agg(DISTINCT s.source_batch) AS batches
        FROM codex_facts f JOIN codex_fact_sources s ON s.fact_id = f.id
        WHERE f.valid_until IS NULL
          AND f.id IN (SELECT fact_id FROM codex_fact_sources WHERE source_batch = ANY(:b))
        GROUP BY f.id"""), {"b": list(removed_batches)}).fetchall()
    removed = set(removed_batches)
    sole = [r.id for r in rows if set(r.batches) <= removed]
    surviving = [r.id for r in rows if not set(r.batches) <= removed]
    return sole, surviving


def expire_fact(db, fact, batch, reason: str, **extra) -> set:
    """Retire a fact (valid_until now), journaled with *reason*; never deleted."""
    fact.valid_until = datetime.now(timezone.utc)
    return _journal(db, fact, "fact_expired", batch, reason=reason, **extra)


def apply_source_plan(db, sole, surviving, removed_batches, event_batch, *,
                      reason: str = "source_deleted") -> set:
    """Remove deleted support. The quotes of removed turns are deleted (they are
    the deleted text); a fact left with no source is expired and journaled; a
    corroborated fact keeps going with its support recounted. Returns touched
    entity ids."""
    touched = set()
    for fact_id in sole:
        touched |= expire_fact(db, db.get(CodexFact, fact_id), event_batch, reason)
    if removed_batches:
        db.query(CodexFactSource).filter(CodexFactSource.source_batch.in_(list(removed_batches))).delete(
            synchronize_session=False)
    db.flush()
    for fact_id in surviving:
        fact = db.get(CodexFact, fact_id)
        fact.support_count = db.execute(text(
            "SELECT count(DISTINCT source_batch) FROM codex_fact_sources WHERE fact_id = :f"),
            {"f": fact_id}).scalar() or 1
        touched |= _journal(db, fact, "fact_source_removed", event_batch, reason=reason)
    return touched


def live_fact_links(db, entity_id, excluding=()) -> int:
    """Current facts linked to an entity, ignoring *excluding* fact ids."""
    return db.execute(text("""
        SELECT count(*) FROM codex_fact_entities fe JOIN codex_facts f ON f.id = fe.fact_id
        WHERE fe.entity_id = :e AND f.valid_until IS NULL AND NOT (f.id = ANY(:x))"""),
        {"e": entity_id, "x": list(excluding)}).scalar() or 0



# ── Entity view (CODEX_V4 D11.1: an entity's page is its facts) ──

def _fact_out(fact, role=None) -> dict:
    return {"id": str(fact.id), "text": fact.text, "role": role, "speaker": fact.speaker,
            "modality": fact.modality, "support": fact.support_count,
            "mentioned_at": fact.mentioned_at.isoformat() if fact.mentioned_at else None,
            "until": fact.valid_until.isoformat() if fact.valid_until else None,
            "superseded_by": str(fact.superseded_by) if fact.superseded_by else None}


def has_source():
    """A fact still backed by at least one source turn. Deletion and /forget
    remove the deleted turns' sources (C10), so a fact without one carries
    deleted text and is never shown, as current OR as history."""
    return db_exists().where(CodexFactSource.fact_id == CodexFact.id)


def entity_facts(db, entity_id, *, current: bool = True, limit: int = 30) -> list:
    """Facts linked to an entity: current ones by support then recency, or the
    replaced history (facts another fact superseded) newest first. Expiry by
    deletion is not history (T4: deletion is not evolution)."""
    query = (db.query(CodexFact, CodexFactEntity.role)
             .join(CodexFactEntity, CodexFactEntity.fact_id == CodexFact.id)
             .filter(CodexFactEntity.entity_id == entity_id, has_source()))
    if current:
        query = query.filter(CodexFact.valid_until.is_(None)).order_by(
            CodexFact.support_count.desc(), CodexFact.mentioned_at.desc())
    else:
        query = query.filter(CodexFact.valid_until.is_not(None),
                             CodexFact.superseded_by.is_not(None)).order_by(
            CodexFact.valid_until.desc())
    seen, out = set(), []
    for fact, role in query.limit(limit * 3):
        if fact.id in seen:
            continue
        seen.add(fact.id)
        out.append(_fact_out(fact, role))
        if len(out) >= limit:
            break
    return out



def restore_fact(db, fact_id, batch=None) -> bool:
    """D12 undo for an automatic replacement: the fact is current again."""
    fact = db.get(CodexFact, fact_id)
    if fact is None or fact.valid_until is None:
        return False
    replaced_by = fact.superseded_by
    fact.valid_until, fact.superseded_by = None, None
    _journal(db, fact, "fact_restored", batch or uuid.uuid4(),
             was_superseded_by=str(replaced_by) if replaced_by else None)
    return True
