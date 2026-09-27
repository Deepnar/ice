"""Unpromoted question-first factual proof over a complete original source.

Freeze equivalent typed and direct hypotheses before source reading. A source
value must support both; model enums or literal quotes alone are not proof.
"""

import json
from dataclasses import dataclass

import httpx
import structlog

from src.api.config import settings
from src.memory.support import score_pairs, verify_support
from src.memory.tokens import count_messages
from src.workers.bg_client_factory import bg_timeout

logger = structlog.get_logger("ice.api.source_proof")

FRAME_INSTRUCTIONS = """Convert the complete latest user request into two equivalent factual declarative frames. You have no external source. Each frame must contain exactly one literal {{answer}} placeholder for the requested unknown VALUE. Preserve every requested subject, event, relationship, time, status and part. Retain the requested answer CATEGORY outside the placeholder: a port, color, digestive issue, city, person/role, causal factor or list is not an unspecified thing. Do not answer, guess, drop a qualifier, add an unrequested person's name or weaken an existing result into a future plan.
frame: a subject-attribute-value formulation containing the category and all question qualifiers.
assertion_frame: independently asserts the SAME proposition with those conditions as a main predicate, not just a background relative clause. Retain the category in this form too.
Use natural grammatical sentences: keep the category next to its ordinary predicate, never append a second noun phrase such as "{{answer}} a port". A requested unknown value, even when supplied elsewhere in the request, appears only in the placeholder, never copied into a condition. Conditions that actually identify the requested event/time must remain. A supplied value in the request still needs the placeholder. A who question permits a name or role unless specifically asking for a name. New generation and requests to reproduce whole omitted material cannot be proven by one factual frame: both frames are null. reason is informational, never evidence.
Example request: Which glaze did Mira use for the earlier kiln test?
frame: The glaze Mira used for the earlier kiln test was {{answer}}.
assertion_frame: Mira used the glaze {{answer}} for the earlier kiln test.
Return JSON only with frame, assertion_frame and reason."""
FRAME_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": ["string", "null"]}
                   for key in ("frame", "assertion_frame", "reason")},
    "required": ["frame", "assertion_frame", "reason"],
    "additionalProperties": False,
}
FILL_INSTRUCTIONS = """Fill the one {{answer}} slot shared by the two fixed equivalent frames using only the COMPLETE SOURCE. Neither frame may be rewritten or weakened. Give the most specific explicit source value of the requested category, not a vague superclass, a different attribute or a new story. Every subject/event/time/relationship/status and requested part in both frames must be established. A recording date alone is not an event date; explicit relative time may be resolved from that date. A dated statement may identify when the speaker shared it. Related topics, current values for old values, future collaboration for existing music, unresolved antecedents, partial lists and questions alone cannot supply missing material. If any required information is missing or uncertain, answer and evidence_quote are both null. Otherwise answer is the shortest source-grounded value and evidence_quote an unchanged contiguous SOURCE unit establishing the WHOLE proposition. Retain original speaker/date labels and preceding antecedents needed for attribution and time, not just the answer span. Prefer the complete supporting attributed paragraph. Do not include a question as support for a condition that is absent from assertions; a question mentioning a date or body part does not establish that date or part. Return JSON only. Never cite a frame as evidence."""
FILL_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": ["string", "null"]}
                   for key in ("answer", "evidence_quote")},
    "required": ["answer", "evidence_quote"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SourceProof:
    status: str
    evidence_quote: str | None = None
    reason: str | None = None


class ProofError(ValueError):
    pass


def _call(instructions, schema, payload):
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    tokens = count_messages(messages)
    output_tokens = settings.memory_source_proof_output_tokens
    context_tokens = settings.memory_source_gate_context_tokens
    if (tokens > settings.memory_source_gate_input_tokens
            or tokens + output_tokens > context_tokens // 2
            or len((instructions + messages[-1]["content"]).encode("utf-8"))
            > context_tokens // 2):
        raise ProofError("complete_input_exceeds_proof_bound")
    response = httpx.post(settings.ollama_base_url.rstrip("/") + "/api/chat", json={
        "model": settings.memory_source_gate_model, "messages": messages,
        "stream": False, "think": True, "format": schema, "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": context_tokens,
                    "num_predict": output_tokens},
    }, timeout=bg_timeout(output_tokens))
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("done") is not True or data.get("done_reason") != "stop":
        raise ProofError("incomplete_native_response")
    actual_tokens = data.get("prompt_eval_count")
    if (not isinstance(actual_tokens, int) or isinstance(actual_tokens, bool)
            or actual_tokens <= 0 or actual_tokens + output_tokens >= context_tokens):
        raise ProofError("provider_input_bound_unconfirmed")
    value = json.loads((data.get("message") or {}).get("content"))
    if (not isinstance(value, dict) or set(value) != set(schema["required"])
            or any(v is not None and not isinstance(v, str) for v in value.values())):
        raise ProofError("invalid_shape")
    logger.info("memory_source_proof_call", model=settings.memory_source_gate_model,
                task="frame" if schema is FRAME_SCHEMA else "fill",
                estimated_input_tokens=tokens, actual_input_tokens=actual_tokens,
                output_tokens=data.get("eval_count"))
    return value


def prove_source_fact(question, source):
    """Unknown/unsupported proof preserves B2 and never modifies stored sources.

    Source must be a complete original unit, not a generated graph assertion.
    Current-prompt use necessarily includes its supplied facts in the request;
    only external source context is withheld from the first frame call.
    """
    try:
        if not isinstance(question, str) or not question.strip() or not isinstance(source, str) or not source.strip():
            raise ProofError("missing_question_or_source")
        value = _call(FRAME_INSTRUCTIONS, FRAME_SCHEMA, {"latest_user_request": question})
        frames = (value["frame"], value["assertion_frame"])
        if all(f is None for f in frames):
            return SourceProof("not_supplied", reason="unframeable_request")
        if any(not isinstance(f, str) or f.count("{{answer}}") != 1 for f in frames):
            raise ProofError("invalid_frame")
        value = _call(FILL_INSTRUCTIONS, FILL_SCHEMA, {
            "frame": frames[0], "assertion_frame": frames[1], "source": source})
        answer, quote = value["answer"], value["evidence_quote"]
        if answer is None and quote is None:
            return SourceProof("not_supplied", reason="missing_source_value")
        if (not isinstance(answer, str) or not answer.strip() or "{{answer}}" in answer
                or not isinstance(quote, str) or not quote.strip() or quote not in source):
            raise ProofError("invalid_source_value_or_quote")
        claims = [frame.replace("{{answer}}", answer) for frame in frames]
        # Preserve original leading attribution/clock/antecedents, while the
        # citation endpoint excludes later material (including appended tasks).
        # The complete original remains a separate bound-checked premise.
        cited_context = source[:source.index(quote) + len(quote)]
        pairs = [(premise, claim) for premise in (source, cited_context) for claim in claims]
        scores = score_pairs(pairs)
        if len(scores) != len(pairs):
            raise ProofError("invalid_verifier_result")
        # The shared verifier validates probabilities and retains its exact
        # policy threshold. Reuse the batch instead of transferring twice.
        verdicts = [verify_support(premise, claim, scorer=lambda _pairs, sc=score: [sc])
                    for (premise, claim), score in zip(pairs, scores)]
        if any(v.status == "unknown" for v in verdicts):
            logger.warning("memory_source_proof_unknown", reason="source_support_uncertain",
                           model=settings.memory_source_gate_model)
            return SourceProof("unknown", reason="source_support_uncertain")
        supported = all(v.status == "supported" for v in verdicts)
        logger.info("memory_source_proof", supported=supported,
                    statuses=[v.status for v in verdicts])
        return SourceProof("supported" if supported else "not_supplied",
                           quote if supported else None,
                           None if supported else "source_not_entailed_in_both_forms")
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ProofError) else type(exc).__name__
        logger.warning("memory_source_proof_unknown", reason=reason,
                       model=settings.memory_source_gate_model)
        return SourceProof("unknown", reason=reason)
