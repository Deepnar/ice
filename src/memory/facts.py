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


def write_turn_facts(db, turn, record: dict, *, embedder,
                     decide: Optional[Callable] = None, mentions: Optional[Callable] = None) -> dict:
    """Write one turn's verified v4 assertions as facts. Caller commits.

    *decide(new_fact, candidates)* is D6's same / replaces / both_true call; None
    (until CODEX_V4 §5.2 qualifies it) means exact-text duplicates merge and
    everything else is a new fact. *mentions(text)* returns entity names found
    in a fact sentence (NuNER, background tier); None skips mention links.
    """
    counts = {"assertions": 0, "dropped": 0, "new": 0, "supported": 0, "links": 0}
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
        if existing is None and decide is not None:
            existing = decide(claim, db=db)
        if existing is not None:
            source = _source_row(existing.id, turn, assertion)
            if db.get(CodexFactSource, (existing.id, source.source_batch, source.window_sha256)) is None:
                db.add(source)
                db.flush()
                existing.support_count = db.execute(text(
                    "SELECT count(DISTINCT source_batch) FROM codex_fact_sources WHERE fact_id = :f"),
                    {"f": existing.id}).scalar()
                counts["supported"] += 1
                _journal(db, existing, "fact_supported", turn)
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
        counts["new"] += 1
        counts["links"] += _link(db, fact, claim, embedder=embedder, mentions=mentions)
        _journal(db, fact, "fact_added", turn)
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


def _journal(db, fact, event_type: str, turn) -> None:
    """codex_events is per entity: one row per entity the fact links to."""
    entity_ids = [r[0] for r in db.execute(text(
        "SELECT DISTINCT entity_id FROM codex_fact_entities WHERE fact_id = :f"), {"f": fact.id})]
    for entity_id in entity_ids:
        db.add(CodexEvent(entity_id=entity_id, event_type=event_type, batch_source=turn.batch_id,
                          payload={"fact_id": str(fact.id), "text_sha256": fact.text_sha256}))
