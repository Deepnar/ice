"""CODEX_V4 post-flight stage: one turn's verified claims become codex facts.

The only codex writer since CODEX_V4 (D3); inert, with a WARNING, unless
`summary_assertion_method == "v4"`. A turn whose summary was generated already
carries verified v4 claims, so this stage
re-uses them (no second model call); short turns and documents, which get no
summary, get their own claim pass here. Self-idempotent like every chained
stage: a retry after a partial chain failure completes only what is missing.
"""

import uuid
from datetime import datetime, timezone

import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import facts
from src.memory.embedder import get_embedder
from src.memory.models import EpisodicMemory, IdempotencyKey
from src.memory.summary_support import (
    METHOD_V4, assertion_text, current_method, generate_claims, original_units, source_windows,
    verify_assertions,
)
from src.workers.idempotency import job_key

logger = structlog.get_logger("ice.workers.fact_extractor")


def _summary_record(turn):
    record = (turn.representation_verification or {}).get("summary") or {}
    return record if record.get("method") == METHOD_V4 else None


def _claim_record(turn, model_used: str):
    """The claim pass for a turn with no v4 summary record (short turns, documents)."""
    from src.workers.post_flight import _summary_llm_call
    from src.workers.bg_client_factory import get_bg_model_name

    windows = source_windows(original_units([turn]))
    kind = "document" if turn.is_document else "chat"
    model = get_bg_model_name()

    def call(piece, cap, schema, abstract):
        return _summary_llm_call("", "", model, [], windows=piece, max_tokens=cap,
                                 schema=schema, abstract=abstract, source_kind=kind)

    parsed, failures = generate_claims(call, windows, floor=settings.turn_summary_max_tokens)
    claims = parsed["summary"]
    if not claims:
        return {"assertions": [], "dropped": parsed.get("dropped", []), "failures": failures}
    return verify_assertions(turn.raw_text, assertion_text(claims), claims, windows,
                             dropped=parsed.get("dropped", []))


def extract_facts(batch_id, model_used: str = ""):
    log = logger.bind(batch_id=str(batch_id))
    if current_method() != METHOD_V4:
        log.warning("facts_stage_skipped", reason="summary_assertion_method is not v4")
        return
    key = job_key("facts", batch_id)
    db = SessionLocal()
    try:
        if db.query(IdempotencyKey).filter_by(key=key).first():
            log.info("facts_already_extracted")
            return
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(str(batch_id))).first()
        if turn is None:
            raise RuntimeError(f"turn for batch {batch_id} not visible yet")
        if turn.is_private:
            log.info("facts_private_turn_skipped")
            return
        record = _summary_record(turn)
        reused = record is not None
        if record is None:
            record = _claim_record(turn, model_used)
        counts = facts.write_turn_facts(db, turn, record, embedder=get_embedder(),
                                        ask=_ask, mentions=_mentions)
        words = len((turn.raw_text or "").split())
        if counts["new"] + counts["supported"] == 0 and words > 50:
            # Silent-fallback rule (TRAPS 11): say every time a real turn gives nothing.
            log.warning("facts_none_written", words=words, reused_summary=reused,
                        assertions=counts["assertions"], dropped=counts["dropped"])
        db.add(IdempotencyKey(key=key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("facts_written", reused_summary=reused, **counts)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _mentions(sentence: str):
    from src.retrieval.ner_utils import extract_entities
    return extract_entities(sentence, get_embedder(), tier="background")


def _ask(earlier: str, newer: str) -> dict:
    """D6's two questions on the background model (or codex_fact_reconcile_model)."""
    import json
    from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name, json_schema
    from src.workers.completion_text import complete_text
    model = settings.codex_fact_reconcile_model or get_bg_model_name()
    completion = get_bg_client().chat.completions.create(
        model=model, temperature=0.0, max_tokens=60, timeout=bg_timeout(60),
        messages=[{"role": "system", "content": facts.RECONCILE_SYSTEM},
                  {"role": "user", "content": json.dumps({"earlier": earlier, "newer": newer},
                                                         ensure_ascii=False)}],
        response_format=json_schema("fact_relation", facts.RECONCILE_SCHEMA))
    return json.loads(complete_text(completion))
