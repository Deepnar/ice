"""Source-linked summary assertions within the pinned verifier's input contract.

Windows preserve originals and speaker boundaries. Their coverage manifest is
not a claim that a summary retained every fact; readers still have the source.
"""

import json
from dataclasses import asdict

import structlog

from src.api.config import settings
from src.memory.source import digest, source_units

logger = structlog.get_logger("ice.memory.summary_support")
METHOD = "source_window_assertions_v1"


def policy():
    return {"method": METHOD, "model": settings.source_support_model,
            "revision": settings.source_support_revision,
            "pair_tokens": settings.source_support_max_tokens,
            "threshold": settings.source_support_threshold,
            "window_fraction": 0.5, "window_reserve_tokens": 32,
            "overlap_fraction": 0.25}


def original_units(turns):
    units = []
    for index, turn in enumerate(turns):
        identifier = str(getattr(turn, "id", None) or f"turn-{index}")
        for ordinal, unit in enumerate(source_units(turn)):
            if unit.role == "unknown":
                raise ValueError("source_roles_unknown")
            if unit.text.strip():
                units.append({"id": f"{identifier}:{ordinal}", "role": unit.role,
                              "text": unit.text, "start": unit.start,
                              "recorded_at": str(getattr(turn, "timestamp", None) or "unknown"),
                              "timestamp_provenance": getattr(turn, "ts_provenance", "unknown")})
    return units


def source_windows(units, *, tokenizer=None):
    from src.memory.support import support_tokenizer
    tokenizer = tokenizer or support_tokenizer()
    limit = settings.source_support_max_tokens // 2 - 32
    if limit < 24:
        raise ValueError("source_window_capacity_unusable")
    windows = []
    for unit in units:
        text = unit["text"]
        offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True, verbose=False)["offset_mapping"]
        offsets = [(a, b) for a, b in offsets if b > a]
        if not offsets:
            continue
        first = 0
        while first < len(offsets):
            # The exact tokenizer checks role/provenance overhead too.
            last = min(len(offsets), first + limit)
            while last > first:
                start = 0 if first == 0 else offsets[first][0]
                end = len(text) if last == len(offsets) else offsets[last][0]
                excerpt = text[start:end]
                premise = (f"The {unit['role']} said: {json.dumps(excerpt, ensure_ascii=False)}")
                if len(tokenizer(premise, truncation=False)["input_ids"]) <= limit:
                    break
                last -= 1
            if last == first:
                raise ValueError("source_window_capacity_unusable")
            window = {"id": f"w{len(windows)}", "unit": unit["id"], "role": unit["role"],
                      "start": unit["start"] + start, "end": unit["start"] + end,
                      "unit_sha256": digest(text), "text_sha256": digest(excerpt),
                      "premise": premise, "recorded_at": unit.get("recorded_at", "unknown"),
                      "timestamp_provenance": unit.get("timestamp_provenance", "unknown")}
            windows.append(window)
            if last == len(offsets):
                break
            overlap = min(last - first - 1, max(1, (last - first) // 4))
            first = last - overlap
    if not windows:
        raise ValueError("missing_source_windows")
    return windows


def window_prompt(windows):
    return "\n\n".join(
        f"[{w['id']}; source unit {w['unit']}; original offsets {w['start']}:{w['end']}; "
        f"recorded {w['recorded_at']} ({w['timestamp_provenance']})]\n{w['premise']}"
        for w in windows)


def windows_current(units, windows):
    """Validate actual original spans before approving a model's citations."""
    known = {u["id"]: u for u in units}
    covered = {key: [] for key in known}
    try:
        for window in windows:
            unit = known[window["unit"]]
            start, end = window["start"] - unit["start"], window["end"] - unit["start"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(unit["text"])
                    or window["role"] != unit["role"]
                    or window["unit_sha256"] != digest(unit["text"])
                    or window["text_sha256"] != digest(unit["text"][start:end])
                    or window["premise"] != f"The {unit['role']} said: {json.dumps(unit['text'][start:end], ensure_ascii=False)}"):
                return False
            covered[unit["id"]].append((start, end))
        for identifier, spans in covered.items():
            cursor = 0
            for start, end in sorted(spans):
                if start > cursor:
                    return False
                cursor = max(cursor, end)
            if cursor != len(known[identifier]["text"]):
                return False
        return bool(known)
    except (KeyError, TypeError, ValueError):
        return False


def assertion_instruction(*, abstract=False):
    example = {"summary": [{"text": "One complete source-supported assertion.", "sources": ["w0"]}]}
    if abstract:
        example["abstract"] = [{"text": "One short source-supported assertion.", "sources": ["w0"]}]
    return (
        "\nReturn only JSON in this exact structure: " + json.dumps(example) + ". Each assertion must have exactly "
        '{"text": "one complete supported assertion", "sources": ["w0", "w1"]}. '
        "Use the actual original window IDs. Cite every window needed to preserve "
        "speaker, referents, conditions, uncertainty, negation and later corrections. "
        "Cite the smallest sufficient set, usually one or two windows. Do not cite "
        "all repeated background windows for one general statement. Omit repeated "
        "boilerplate and descriptions of the discussion process. Never mention window "
        "IDs, segments, offsets, or citation labels inside text. Windows are excerpts "
        "of messages, not separate turns. Text states the supported fact and speaker "
        "directly, for example: 'The user approved port 8391 for Atlas.' The abstract also "
        "requires its own sources list; use [] if no short supported abstract is possible. "
        "Use 'The user said' or 'The assistant said' for attribution. Do not invent "
        "action verbs such as confirmed, approved, suggested, or repeated unless "
        "the original explicitly establishes that action. "
        "An assistant proposal is not a user decision. Do not assert current truth "
        "from an older mention. Keep numbers and code values with their assignments. "
        "An assertion that needs unavailable context must be omitted. Do not put "
        "uncited text or key-term lists outside these assertions.\n"
    )


def parse_assertions(generated, windows, *, abstract=False):
    body = generated.strip()
    if body.startswith("```") and body.endswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(body)
    fields = {"summary", "abstract"} if abstract else {"summary"}
    if not isinstance(parsed, dict) or "summary" not in parsed or not set(parsed) <= fields:
        raise ValueError("invalid_summary_assertion_fields")
    if abstract:
        parsed.setdefault("abstract", [])
    known = {w["id"] for w in windows}
    for kind, rows in parsed.items():
        if not isinstance(rows, list) or (kind == "abstract" and len(rows) > 1):
            if kind == "abstract":
                logger.warning("summary_abstract_unresolved", reason="invalid_optional_abstract_shape")
                parsed[kind] = []
                continue
            raise ValueError("invalid_summary_assertions")
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"text", "sources"}
                    or not isinstance(row["text"], str) or not row["text"].strip()
                    or not isinstance(row["sources"], list) or not row["sources"]
                    or any(not isinstance(s, str) or s not in known for s in row["sources"])
                    or len(set(row["sources"])) != len(row["sources"])):
                if kind == "abstract":
                    logger.warning("summary_abstract_unresolved", reason="missing_or_invalid_citation")
                    parsed[kind] = []
                    break
                raise ValueError("unresolved_summary_citation")
    return parsed


def assertion_text(assertions):
    return "\n".join(row["text"].strip() for row in assertions)


class SummaryCandidate(str):
    """Keep citations attached while existing coverage/retry code handles text."""
    def __new__(cls, parsed, windows):
        result = super().__new__(cls, assertion_text(parsed["summary"]))
        result.assertions = parsed
        result.windows = windows
        return result


class SummaryResult(tuple):
    """Preserve the existing three-value caller interface, with optional proof."""
    def __new__(cls, summary, coverage, abstract, proof=None):
        result = super().__new__(cls, (summary, coverage, abstract))
        result.proof = proof
        return result


class SourceNotePrompt(str):
    """Keep instructions in the system role for both real note providers."""
    def __new__(cls, instruction, source):
        result = super().__new__(cls, instruction + "\n\nORIGINAL SOURCE GROUP:\n" + source)
        result.instruction = instruction
        result.source = source
        return result


def note_messages(prompt):
    if isinstance(prompt, SourceNotePrompt):
        return [{"role": "system", "content": prompt.instruction},
                {"role": "user", "content": prompt.source}]
    return [{"role": "system", "content": "You are a precise summarisation engine."},
            {"role": "user", "content": prompt}]


def verify_assertions(source, candidate, assertions, windows, *, verifier=None):
    from src.memory.support import SupportInputError, score_pairs, support_tokenizer, verify_support
    window_map = {w["id"]: w for w in windows}
    if (assertion_text(assertions) != candidate
            or len(window_map) != len(windows)
            or any(s not in window_map for a in assertions for s in a["sources"])):
        raise ValueError("summary_proof_candidate_mismatch")
    premises = []
    for assertion in assertions:
        cited = [window_map[s] for s in assertion["sources"]]
        # Verify a complete assertion against all cited evidence TOGETHER.
        # Long combined premises become unknown under the existing scorer.
        premises.append("\n".join(w["premise"] for w in cited))
    lengths, scored, scoring_error = {}, {}, None
    if verifier is None:
        try:
            tokenizer = support_tokenizer()
            lengths = {i: len(tokenizer(p, a["text"].strip(), truncation=False)["input_ids"])
                       for i, (p, a) in enumerate(zip(premises, assertions))}
            eligible = [i for i in lengths if lengths[i] <= settings.source_support_max_tokens]
            values = score_pairs([(premises[i], assertions[i]["text"].strip()) for i in eligible])
            if len(values) != len(eligible):
                raise ValueError("incomplete NLI result batch")
            scored = dict(zip(eligible, values))
        except Exception as exc:
            from src.workers.runtime import JobYielded
            if isinstance(exc, JobYielded):
                raise
            scoring_error = exc
    records = []
    for i, (assertion, premise) in enumerate(zip(assertions, premises)):
        if verifier is not None:
            verdict = asdict(verifier(premise, assertion["text"].strip()))
        else:
            def scorer(_pairs, index=i):
                if scoring_error is not None:
                    raise scoring_error
                if index not in scored:
                    raise SupportInputError("complete cited source/claim exceeds NLI token bound")
                return [scored[index]]
            verdict = asdict(verify_support(premise, assertion["text"].strip(), scorer=scorer))
        records.append({"text": assertion["text"].strip(), "sources": assertion["sources"],
                        "premise": premise, "pair_tokens": lengths.get(i), "verification": verdict})
    statuses = [r["verification"]["status"] for r in records]
    status = ("contradicted" if "contradicted" in statuses else
              "supported" if records and all(s == "supported" for s in statuses) else "unknown")
    scores = min((r["verification"]["scores"] for r in records),
                 key=lambda s: s.get("entailment", -1), default={})
    reason = ("all_assertions_supported" if status == "supported" else
              next((r["verification"]["reason"] for r in records
                    if r["verification"]["status"] == status), "missing_assertions"))
    result = {"status": status, "scores": scores, "reason": reason,
            "source_sha256": digest(source), "claim_sha256": digest(candidate),
            "verifier": f"{settings.source_support_model}@{settings.source_support_revision}",
            "method": METHOD, "policy": policy(), "assertions": records,
            "source_windows": [{k: v for k, v in w.items() if k != "premise"} for w in windows],
            "cited_windows": sorted({s for r in records for s in r["sources"]}),
            "completeness": "originals_retained_not_inferred_from_entailment"}
    result["proof_sha256"] = proof_digest(result)
    return result


def proof_digest(record):
    return digest(json.dumps({k: record.get(k) for k in
        ("policy", "assertions", "source_windows", "cited_windows")},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def proof_current(record, source, candidate):
    """Readers need no tokenizer/model call to recheck frozen assertion receipts."""
    from src.memory.support import supported_current
    if (record.get("method") != METHOD or record.get("policy") != policy()
            or record.get("proof_sha256") != proof_digest(record)):
        return False
    rows = record.get("assertions")
    if not isinstance(rows, list) or not rows:
        return False
    try:
        return (assertion_text(rows) == candidate
                and all(supported_current(row["verification"], row["premise"], row["text"])
                        for row in rows))
    except (KeyError, TypeError):
        return False
