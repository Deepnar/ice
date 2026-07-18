"""E8 — bounded, constraint-aware decision extraction (spec D7).

A6 stance throughout: deterministic cues gate everything — **no cue, no LLM
call** (an always-on pass would burn the bg model on every coding turn and
hallucinate rationale). Cue kinds map to ``decision_type``:

  constraint — "don't touch" / "must stay" family (the do-not-touch payoff:
               ice_context surfaces these FIRST when a task mentions their
               files);
  incident   — an error→fixed arc (both halves required);
  decision   — decision verbs / "instead of" / "rather than".

One bounded JSON completion (temperature 0) extracts
``{decision, rationale, alternatives_rejected[], files_affected[], type}``.
Dedupe/supersession against ACTIVE decisions of the same project by embedding
similarity (D1-tier rules, thresholds in settings):

  ≥ duplicate_threshold + same type + same files  → silent skip;
  ≥ conflict_threshold + overlapping files        → auto-supersede (Tier 1 —
      old row gets valid_until + superseded_by; supersession is first-class,
      no event journal);
  ≥ conflict_threshold, no file overlap           → new row + review-queue
      proposal (Tier 2, item_type='decision_supersession').

Runs in the gpu-lane ``decision_extract`` runtime job (commit messages, via
the reconciler) and as a direct call inside the post_flight chain for
project-attached turns.
"""
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import text

from src.memory.models import Decision, ReviewQueue

logger = structlog.get_logger("ice.workers.decision_extractor")

_USE_DEFAULT = object()

# ── Deterministic cues (D7; empirical deferral: widen from the
#    decision_cue_fired log if real decisions visibly slip through) ─────────
CONSTRAINT_CUES = (
    "don't touch", "do not touch", "dont touch", "never change", "never touch",
    "must stay", "must not change", "must never", "do not edit", "don't edit",
    "keep it as is", "keep as is", "leave it alone", "hands off",
    "do not modify", "don't modify", "never modify", "off limits",
)
DECISION_CUES = (
    "decided to", "decided against", "we decided", "i decided", "decision:",
    "going with", "we'll go with", "going to use", "chose ", "chosen ",
    "opted for", "opted to", "settled on", "instead of", "rather than",
    "switched to", "switching to", "will use", "we'll use", "let's use",
    "standardize on", "standardise on",
)
_ERROR_CUES = ("error", "exception", "traceback", "failed", "failing",
               "broken", "bug", "crash")
_FIX_CUES = ("fixed", "resolved", "solved", "the fix", "turned out",
             "root cause", "now works", "works now", "was because")


def decision_cue(text_content: str) -> Optional[str]:
    """Which cue family fires, or None. Priority: constraint > incident >
    decision (constraint wording often also contains decision verbs).
    Code blocks are stripped — temporal words inside code are content."""
    if not text_content:
        return None
    stripped = re.sub(r"```.*?```", " ", text_content, flags=re.S)
    low = stripped.lower()
    if any(cue in low for cue in CONSTRAINT_CUES):
        return "constraint"
    if any(c in low for c in _ERROR_CUES) and any(c in low for c in _FIX_CUES):
        return "incident"
    if any(cue in low for cue in DECISION_CUES):
        return "decision"
    return None


def _extract_llm(llm, text_content: str, cue: str) -> Optional[dict]:
    prompt = (
        "Extract the single most important engineering decision, constraint, "
        "or incident-learning from this text. Reply with ONE JSON object:\n"
        '{"decision": "<one-sentence statement>", '
        '"rationale": "<why, from the text; empty if not stated>", '
        '"alternatives_rejected": ["<rejected option>", ...], '
        '"files_affected": ["<file path mentioned>", ...], '
        f'"type": "{cue}"}}\n'
        "Rules: use ONLY what the text says — never invent rationale or "
        "files; files_affected lists file/module paths actually named; "
        'if there is no real decision in the text, reply {"decision": ""}.\n\n'
        f"Text:\n{text_content[:2500]}"
    )
    parsed = llm(prompt, 300)
    if not isinstance(parsed, dict):
        return None
    decision = str(parsed.get("decision", "")).strip()
    if not decision:
        return None
    alts = parsed.get("alternatives_rejected")
    files = parsed.get("files_affected")
    dtype = str(parsed.get("type", cue)).strip().lower()
    return {
        "decision": decision[:500],
        "rationale": str(parsed.get("rationale", "") or "").strip()[:1000],
        "alternatives_rejected": [str(a)[:200] for a in alts[:6]]
        if isinstance(alts, list) else [],
        "files_affected": [str(f)[:300] for f in files[:10]]
        if isinstance(files, list) else [],
        "type": dtype if dtype in ("decision", "constraint", "incident") else cue,
    }


def _embed(decision_text: str):
    # The shared extraction embedder (one process, one model — G13).
    from src.workers.codex_extractor import embedder
    vec = embedder.encode(decision_text, convert_to_tensor=False)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def run_decision_extraction(db, project_id=None, text_content: str = "",
                            source_batch=None, origin: str = "",
                            llm=_USE_DEFAULT) -> dict:
    """The job callable. *llm* defaults to the bg model's JSON decider
    (maintenance_agent.make_llm_decider — same bounded contract); pass a stub
    in tests, or None to no-op the LLM stage entirely."""
    from src.api.config import settings

    if not project_id or not text_content:
        return {"status": "noop"}
    cue = decision_cue(text_content)
    if cue is None:
        return {"status": "no_cue"}
    logger.info("decision_cue_fired", cue=cue, origin=origin,
                project_id=str(project_id))
    if llm is _USE_DEFAULT:
        from src.workers.maintenance_agent import make_llm_decider
        llm = make_llm_decider()
    if llm is None:
        return {"status": "no_llm"}
    extracted = _extract_llm(llm, text_content, cue)
    if extracted is None:
        return {"status": "nothing_extracted", "cue": cue}

    pid = uuid.UUID(str(project_id))
    embedding = _embed(extracted["decision"])

    # Dedupe/supersession against ACTIVE decisions of this project.
    sim_rows = db.execute(text("""
        SELECT id, decision_type, files_affected,
               1 - (embedding <=> CAST(:emb AS vector)) AS sim
        FROM decisions
        WHERE project_id = :pid AND valid_until IS NULL
          AND embedding IS NOT NULL
        ORDER BY sim DESC
        LIMIT 5
    """), {"emb": str(embedding), "pid": pid}).fetchall()

    new_files = set(extracted["files_affected"])
    for row in sim_rows:
        if row.sim is None or row.sim < settings.decision_conflict_threshold:
            break
        old_files = set(row.files_affected or [])
        same_files = old_files == new_files
        overlap = bool(old_files & new_files) or (not old_files and not new_files)
        if row.sim >= settings.decision_duplicate_threshold and \
                row.decision_type == extracted["type"] and same_files:
            logger.info("decision_duplicate_skipped", sim=round(row.sim, 3),
                        existing=str(row.id))
            return {"status": "duplicate", "existing": str(row.id)}
        new_row = _insert(db, pid, extracted, embedding, source_batch)
        if overlap:
            # Tier 1: explicit supersession — bi-temporal close-out.
            db.execute(text("""
                UPDATE decisions SET valid_until = :now, superseded_by = :new
                WHERE id = :old AND valid_until IS NULL
            """), {"now": datetime.now(timezone.utc), "new": new_row.id,
                   "old": row.id})
            db.commit()
            logger.info("decision_superseded", old=str(row.id),
                        new=str(new_row.id), sim=round(row.sim, 3))
            return {"status": "superseded", "id": str(new_row.id),
                    "superseded": str(row.id)}
        # Tier 2: similar but different files — the user decides.
        db.add(ReviewQueue(item_type="decision_supersession", item_content={
            "new_id": str(new_row.id), "old_id": str(row.id),
            "new_decision": extracted["decision"],
            "similarity": round(float(row.sim), 3),
            "suggested_action": "review: possibly supersedes an active "
                                "decision touching different files"}))
        db.commit()
        logger.info("decision_conflict_queued", old=str(row.id),
                    new=str(new_row.id))
        return {"status": "conflict_queued", "id": str(new_row.id)}

    new_row = _insert(db, pid, extracted, embedding, source_batch)
    db.commit()
    logger.info("decision_recorded", id=str(new_row.id),
                type=extracted["type"], origin=origin)
    return {"status": "recorded", "id": str(new_row.id), "type": extracted["type"]}


def _insert(db, pid, extracted: dict, embedding, source_batch) -> Decision:
    row = Decision(
        project_id=pid,
        decision=extracted["decision"],
        rationale=extracted["rationale"] or None,
        alternatives_rejected=extracted["alternatives_rejected"],
        files_affected=extracted["files_affected"],
        decision_type=extracted["type"],
        source_batch=uuid.UUID(str(source_batch)) if source_batch else None,
        embedding=embedding,
        valid_from=datetime.now(timezone.utc),
    )
    db.add(row)
    db.flush()
    return row
