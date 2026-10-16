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
METHOD_V2 = "source_window_assertions_v2"
# v3 (docs/specs/SUMMARY_CLAIMS_V3.md): quoted pointers, speaker, 1-2 sources,
# per-claim acceptance. Selected by settings.summary_assertion_method.
METHOD_V3 = "source_window_assertions_v3"
# v4 (docs/specs/CODEX_V4.md D4): a v3 claim plus modality and the source's own
# subject / relation / object — every supported claim is also a codex fact.
METHOD_V4 = "source_window_facts_v4"
# Methods whose records keep only supported claims (per-claim acceptance).
PER_CLAIM_METHODS = (METHOD_V3, METHOD_V4)
METHOD = METHOD_V2  # legacy name, kept for readers of frozen records
SPEAKERS = ("user", "assistant", "document")
CLAIM_FIELDS = ("sources", "quote", "speaker", "text")
FACT_FIELDS = ("sources", "quote", "speaker", "modality", "subject", "relation", "object", "text")
# ~85-90% reliable whatever the wording (CODEX_V4 §5.1); the sentence carries the act.
MODALITIES = ("asserted", "shared", "suggested", "hypothetical", "asked")
_METHODS = {"v2": METHOD_V2, "v3": METHOD_V3, "v4": METHOD_V4}


def current_method():
    return _METHODS[settings.summary_assertion_method]


def policy(method=None):
    method = method or current_method()
    result = {"method": method, "model": settings.source_support_model,
              "revision": settings.source_support_revision,
              "pair_tokens": settings.source_support_max_tokens,
              "threshold": settings.source_support_threshold,
              "window_fraction": 0.5, "window_reserve_tokens": 32,
              "overlap_fraction": 0.25, "premise_composition": "exact_cited_span_union_v1"}
    if method in PER_CLAIM_METHODS:
        result.update(claim_shape="sources_quote_speaker_text_v1", max_sources=2,
                      acceptance="per_claim_supported_only")
    if method == METHOD_V4:
        result.update(claim_shape="sources_quote_speaker_modality_triple_text_v1",
                      context_header="unit_opening_300_v1")
    return result


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


def piece_windows(windows, budget=None):
    """Group windows, in order, into pieces of at most *budget* prompt tokens.

    One call per piece (SUMMARY_CLAIMS_V3 S5): an oversized turn or group is
    summarised in parts instead of being cut by the server or refused. A
    single window always forms a piece, so this cannot loop or drop text.
    """
    from src.memory import tokens
    budget = budget or settings.summary_piece_tokens
    pieces, current, used = [], [], 0
    for window in windows:
        size = tokens.count(window_prompt([window])) + 2
        if current and used + size > budget:
            pieces.append(current)
            current, used = [], 0
        current.append(window)
        used += size
    if current:
        pieces.append(current)
    return pieces


def output_cap(input_tokens, floor):
    """Output stop that scales with the input (SUMMARY_CLAIMS_V3 S6).

    A cap is a safety stop, not a length control: below it the prompt decides
    the length. *floor* is the job's legacy cap, so small inputs keep it.
    """
    import math
    scaled = math.ceil(settings.summary_output_ratio * max(0, input_tokens))
    return int(min(settings.summary_output_ceiling, max(floor, scaled)))


def generate_claims(call, windows, *, floor, abstract=False):
    """Claims for *windows*, one model call per piece, merged in order.

    ``call(piece, max_tokens, schema, abstract)`` returns the raw generation;
    *schema* is the v3 `format` (None under v2). Only a single-piece source asks
    for an abstract — a piece's overview would claim to describe the whole.
    A failed piece is logged and skipped (per-claim acceptance extends to
    pieces); if every piece fails, the last error propagates so the caller's
    existing fallback runs. Returns ``(parsed, failures)``.
    """
    from src.memory import tokens
    from src.workers.runtime import JobYielded
    pieces = piece_windows(windows)
    want_abstract = abstract and len(pieces) == 1
    merged = {"summary": [], "dropped": []}
    if want_abstract:
        merged["abstract"] = []
    failures, last_error = [], None
    for index, piece in enumerate(pieces):
        if current_method() == METHOD_V4:
            piece = _with_opening(piece, windows)
        cap = output_cap(tokens.count(window_prompt(piece)), floor)
        schema = claim_schema(piece, abstract=want_abstract) if current_method() in PER_CLAIM_METHODS else None
        generated = None
        try:
            generated = call(piece, cap, schema, want_abstract)
            if not isinstance(generated, str) or not generated.lstrip().startswith(("{", "```")):
                raise ValueError("unstructured_generation")
            parsed = parse_assertions(generated, piece, abstract=want_abstract)
        except JobYielded:
            raise
        except Exception as exc:                                    # noqa: BLE001
            # Keep the failed text for the caller's diagnostic record.
            exc.generated_candidate = generated if isinstance(generated, str) else None
            last_error = exc
            failures.append({"piece": index, "error": type(exc).__name__,
                             "finish_reason": getattr(exc, "finish_reason", None),
                             "max_tokens": cap})
            if len(pieces) > 1:
                # A single piece's failure is the caller's to report, once.
                logger.warning("summary_piece_failed", piece=index, pieces=len(pieces),
                               error=f"{type(exc).__name__}: {str(exc)[:120]}", max_tokens=cap)
            continue
        merged["summary"].extend(parsed["summary"])
        merged["dropped"].extend(parsed.get("dropped", []))
        if want_abstract:
            merged["abstract"] = parsed.get("abstract", [])
    if len(failures) == len(pieces):
        raise last_error
    if len(pieces) > 1:
        logger.info("summary_pieces", pieces=len(pieces), failed=len(failures),
                    claims=len(merged["summary"]))
    return merged, failures


_OPENING_CHARS = 300


def _with_opening(piece, windows):
    """CODEX_V4 D4: a piece starting inside a long message gets that message's
    opening as context. A paste's framing ("here is my spec:") is otherwise
    invisible to every later piece. The copy keeps stored windows untouched,
    and the NLI premise never includes it."""
    first = piece[0]
    unit_first = next(w for w in windows if w["unit"] == first["unit"])
    if unit_first is first or unit_first["start"] >= first["start"]:
        return piece
    excerpt = json.loads(unit_first["premise"][len(f"The {unit_first['role']} said: "):])
    opening = " ".join(excerpt[:_OPENING_CHARS].split())
    return [dict(first, continues=opening)] + list(piece[1:])


def window_prompt(windows):
    header = ""
    if windows and windows[0].get("continues"):
        header = (f"[context, not a source: this part continues a {windows[0]['role']} "
                  f"message that began: \"{windows[0]['continues']}…\"]\n\n")
    return header + "\n\n".join(
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
    if current_method() == METHOD_V4:
        return _assertion_instruction_v4(abstract=abstract)
    if current_method() == METHOD_V3:
        return _assertion_instruction_v3(abstract=abstract)
    return _assertion_instruction_v2(abstract=abstract)


def _assertion_instruction_v3(*, abstract=False):
    example = {"summary": [{"sources": ["w0"], "quote": "exact words copied from w0",
                            "speaker": "user", "text": "One complete claim."}]}
    if abstract:
        example["abstract"] = [{"sources": ["w0"], "quote": "exact words copied from w0",
                                "speaker": "user", "text": "One short overview claim."}]
    return (
        "\nReturn JSON in this structure: " + json.dumps(example) + ". For each "
        "memory fact, in this order: (1) sources: the one window (at most two) that "
        "states it; (2) quote: a short exact copy, up to about 25 words, of the words "
        "in that window that state it - copy, do not paraphrase; (3) speaker: who "
        "said those words (the role shown on that window); (4) text: the fact as one "
        "complete sentence that the quote supports, naming the speaker ('The user "
        "said...', 'The assistant said...'). One fact per entry. Keep names, numbers, "
        "dates and code values with what they belong to. An assistant proposal is not "
        "a user decision; a question is not a statement. Do not assert current truth "
        "from an older mention. Skip pleasantries, repeated background and "
        "descriptions of the discussion process. Never put window IDs inside text. "
        + ("The abstract is at most one entry giving a one-sentence overview, with its "
           "own source and quote; use [] if none is supported. " if abstract else "")
        + "\n"
    )


def _assertion_instruction_v4(*, abstract=False):
    """Locked 2026-10-16 after four measured wordings (CODEX_V4 §5.1, instruction
    "m5"). Model-coupled to gemma4:e4b: changing the background model means
    re-qualifying this text, not editing a value."""
    example = {"summary": [{"sources": ["w0"], "quote": "exact words copied from w0",
                            "speaker": "user", "modality": "asserted", "subject": "user",
                            "relation": "works at", "object": "Acme",
                            "text": "The user said they work at Acme."}]}
    if abstract:
        example["abstract"] = [{"sources": ["w0"], "quote": "exact words copied from w0",
                                "speaker": "user", "modality": "asserted", "subject": "",
                                "relation": "", "object": "", "text": "One short overview claim."}]
    return (
        "\nReturn JSON in this structure: " + json.dumps(example) + ". For each "
        "memory fact, in this order: (1) sources: the one window (at most two) that "
        "states it; (2) quote: a short exact copy, up to about 25 words, of the words "
        "in that window that state it - copy, do not paraphrase; (3) speaker: who "
        "said those words (the role shown on that window); (4) modality - what the speaker "
        "DOES with it in those words: 'asserted' only if the speaker presents it as true or as "
        "their own decided plan, in their own voice; 'shared' if the words are material the "
        "speaker pastes, quotes or forwards into the conversation - a document, specification, "
        "design notes, tutorial, article, log, code or data, even one they wrote themselves "
        "elsewhere - rather than saying it in this message; the assistant's own explanations, "
        "analysis and answers are never 'shared'; 'suggested' if the speaker "
        "offers it as an option, idea or recommendation; 'hypothetical' if it is a "
        "possibility, a condition or not decided; 'asked' if the speaker asks about it. "
        "When unsure between 'asserted' and another, do not choose 'asserted'; "
        "(5) subject: who or what the fact is about, named "
        "as in the source - write 'user' when the user speaks about themselves and "
        "'assistant' when the assistant speaks about itself; (6) relation: the source's "
        "own short verb phrase linking subject and object; (7) object: the other thing or "
        "the value; use empty strings for subject, relation and object when the fact has "
        "no such form; (8) text: the fact as one complete sentence that the quote "
        "supports, naming the speaker and what they did ('The user said...', 'The assistant "
        "suggested...', 'The user shared a design document stating...'), understandable on "
        "its own without the conversation: name the person, story, project or document the "
        "fact belongs to when the source names it. One fact per entry. For pasted data or "
        "long lists of records, state facts about the collection, not one per record; for pasted "
        "logs, measurements or program output, state one fact about that run (what ran and its "
        "key result), not one per line. Keep figurative language figurative: write that the "
        "speaker described something as like or as the equivalent of something, never state a "
        "metaphor as a literal event. Keep "
        "names, numbers, dates and code values with what they belong to. An assistant "
        "proposal is not a user decision; a question is not a statement; pasted material is "
        "not the speaker's own statement. Do not assert current truth from an older mention. "
        "Skip pleasantries, repeated background and descriptions of the discussion process. "
        "Never put window IDs inside text."
        + (" The abstract is at most one entry giving a one-sentence overview, with its "
           "own source and quote; use [] if none is supported." if abstract else "")
        + "\n"
    )


def claim_schema(windows, *, abstract=False):
    """Ollama `format` for v3/v4: window ids as an enum, 1-2 sources, pointer first.

    The speaker enum offers only the roles present in these windows: offered all
    three, a document piece lost 9/9 and 12/12 claims to speaker_mismatch (the
    model wrote "user"; CODEX_V4 §5.1)."""
    ids = [w["id"] for w in windows]
    roles = [r for r in SPEAKERS if any(w["role"] == r for w in windows)]
    props = {"sources": {"type": "array", "items": {"type": "string", "enum": ids},
                         "minItems": 1, "maxItems": 2},
             "quote": {"type": "string", "minLength": 1},
             "speaker": {"type": "string", "enum": roles},
             "text": {"type": "string", "minLength": 1}}
    fields = list(CLAIM_FIELDS)
    if current_method() == METHOD_V4:
        props.update(modality={"type": "string", "enum": list(MODALITIES)},
                     subject={"type": "string"}, relation={"type": "string"},
                     object={"type": "string"})
        fields = list(FACT_FIELDS)
    row = {"type": "object", "additionalProperties": False, "required": fields,
           "properties": {k: props[k] for k in fields}}
    props = {"summary": {"type": "array", "items": row}}
    if abstract:
        props["abstract"] = {"type": "array", "items": row, "maxItems": 1}
    return {"type": "object", "additionalProperties": False, "required": list(props),
            "properties": props}


def _assertion_instruction_v2(*, abstract=False):
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
    if current_method() in PER_CLAIM_METHODS:
        return _parse_v3(parsed, windows, abstract=abstract)
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


_MARKUP = str.maketrans({"*": " ", "`": " ", "#": " ", "\u2018": "'", "\u2019": "'",
                         "\u201c": '"', "\u201d": '"'})


def _collapse(text):
    """Compare words, not formatting. Measured 2026-10-15 (C, gemma4:e4b): of
    12 failed quotes, 10 were the exact words with Markdown markers dropped
    (`**Phase 2 is done.**` quoted as `Phase 2 is done.`, backticked names
    quoted bare). Word order and spelling must still match exactly."""
    return " ".join(text.translate(_MARKUP).split())


def _claim_problem(row, by_id):
    """Mechanical v3/v4 checks for one claim; None when it may go to the verifier."""
    if current_method() == METHOD_V4:
        if (not isinstance(row, dict) or set(row) != set(FACT_FIELDS)
                or row.get("modality") not in MODALITIES
                or not all(isinstance(row[k], str) for k in ("subject", "relation", "object"))):
            return "invalid_claim_shape"
        row = {k: row[k] for k in CLAIM_FIELDS}
    if (not isinstance(row, dict) or set(row) != {"sources", "quote", "speaker", "text"}
            or not all(isinstance(row[k], str) and row[k].strip() for k in ("quote", "speaker", "text"))
            or not isinstance(row["sources"], list)):
        return "invalid_claim_shape"
    sources = row["sources"]
    if (not 1 <= len(sources) <= 2 or len(set(sources)) != len(sources)
            or any(not isinstance(x, str) or x not in by_id for x in sources)):
        return "unresolved_source"
    quote = _collapse(row["quote"])
    try:
        premise, intervals = cited_premise(list(by_id.values()), sources)
    except ValueError:
        return "unresolved_source"
    roles = []
    for part, interval in zip(premise.split("\n"), intervals):
        text = json.loads(part[len(f"The {interval['role']} said: "):])
        if quote in _collapse(text):
            roles.append(interval["role"])
    if not roles:
        return "quote_not_in_source"
    if row["speaker"] not in roles:
        return "speaker_mismatch"
    return None


def _parse_v3(parsed, windows, *, abstract=False):
    """A bad claim is dropped with its reason; it never rejects its siblings."""
    by_id = {w["id"]: w for w in windows}
    if abstract:
        parsed.setdefault("abstract", [])
    dropped = []
    for kind in list(parsed):
        rows = parsed[kind]
        if not isinstance(rows, list) or (kind == "abstract" and len(rows) > 1):
            if kind == "abstract":
                logger.warning("summary_abstract_unresolved", reason="invalid_optional_abstract_shape")
                parsed[kind] = []
                continue
            raise ValueError("invalid_summary_assertions")
        kept = []
        for row in rows:
            problem = _claim_problem(row, by_id)
            if problem is None:
                kept.append(row)
            else:
                dropped.append({"kind": kind, "reason": problem,
                                "text": row.get("text") if isinstance(row, dict) else None})
        parsed[kind] = kept
    if dropped:
        logger.warning("summary_claims_dropped", stage="mechanical", count=len(dropped),
                       reasons=sorted({d["reason"] for d in dropped}))
    parsed["dropped"] = dropped
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


def cited_premise(windows, source_ids):
    """Resolve every cited original byte once, keeping source units separate."""
    wanted = set(source_ids)
    if len(wanted) != len(source_ids):
        raise ValueError("duplicate_cited_window")
    units = {}
    for w in windows:
        if w["id"] not in wanted:
            continue
        prefix = f"The {w['role']} said: "
        if not w["premise"].startswith(prefix):
            raise ValueError("cited_window_role_mismatch")
        text = json.loads(w["premise"][len(prefix):])
        if (not isinstance(text, str) or type(w["start"]) is not int
                or type(w["end"]) is not int or not 0 <= w["start"] < w["end"]
                or len(text) != w["end"] - w["start"] or digest(text) != w["text_sha256"]):
            raise ValueError("cited_window_span_mismatch")
        units.setdefault(w["unit"], []).append((w, text))
        wanted.remove(w["id"])
    if wanted or not units:
        raise ValueError("unresolved_cited_window")
    parts, intervals = [], []
    for unit, spans in units.items():
        spans.sort(key=lambda item: (item[0]["start"], item[0]["end"]))
        anchor = spans[0][0]
        identity = (anchor["role"], anchor["unit_sha256"], anchor.get("recorded_at"),
                    anchor.get("timestamp_provenance"))
        groups = []
        for w, text in spans:
            if (w["role"], w["unit_sha256"], w.get("recorded_at"),
                    w.get("timestamp_provenance")) != identity:
                raise ValueError("cited_source_unit_mismatch")
            if groups and w["start"] <= groups[-1][1]:
                start, end, body = groups[-1]
                overlap = min(end, w["end"]) - w["start"]
                if body[w["start"] - start:w["start"] - start + overlap] != text[:overlap]:
                    raise ValueError("cited_window_overlap_mismatch")
                if w["end"] > end:
                    groups[-1] = (start, w["end"], body + text[end - w["start"]:])
            else:
                groups.append((w["start"], w["end"], text))
        for start, end, body in groups:
            parts.append(f"The {anchor['role']} said: {json.dumps(body, ensure_ascii=False)}")
            intervals.append(dict(unit=unit, role=anchor["role"], start=start, end=end,
                                  unit_sha256=anchor["unit_sha256"], text_sha256=digest(body)))
    return "\n".join(parts), intervals


def verify_assertions(source, candidate, assertions, windows, *, verifier=None, dropped=()):
    from src.memory.support import SupportInputError, score_pairs, support_tokenizer, verify_support
    window_map = {w["id"]: w for w in windows}
    if (assertion_text(assertions) != candidate
            or len(window_map) != len(windows)
            or any(s not in window_map for a in assertions for s in a["sources"])):
        raise ValueError("summary_proof_candidate_mismatch")
    resolved = [cited_premise(windows, a["sources"]) for a in assertions]
    premises = [premise for premise, _ in resolved]
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
                        "premise": premise, "cited_spans": resolved[i][1],
                        "pair_tokens": lengths.get(i), "verification": verdict,
                        **{k: assertion[k] for k in FACT_FIELDS[1:-1] if k in assertion}})
    method = current_method()
    unsupported = []
    if method in PER_CLAIM_METHODS:
        # Per-claim acceptance: keep exactly the supported claims; the stored
        # text is rebuilt from them, so the reader's digest check binds to it.
        unsupported = [{"kind": "summary", "reason": r["verification"]["status"],
                        "text": r["text"], "scores": r["verification"]["scores"]}
                       for r in records if r["verification"]["status"] != "supported"]
        records = [r for r in records if r["verification"]["status"] == "supported"]
        if unsupported:
            logger.warning("summary_claims_dropped", stage="verification", count=len(unsupported),
                           kept=len(records))
        candidate = assertion_text(records)
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
            "method": method, "policy": policy(method), "assertions": records,
            "source_windows": [{k: v for k, v in w.items() if k != "premise"} for w in windows],
            "cited_windows": sorted({s for r in records for s in r["sources"]}),
            "completeness": "originals_retained_not_inferred_from_entailment"}
    result["proof_sha256"] = proof_digest(result)
    if method in PER_CLAIM_METHODS:
        result["dropped"] = list(dropped) + unsupported  # outside the digest: audit only
    return result


def proof_digest(record):
    return digest(json.dumps({k: record.get(k) for k in
        ("policy", "assertions", "source_windows", "cited_windows")},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def proof_current(record, source, candidate):
    """Readers need no tokenizer/model call to recheck frozen assertion receipts."""
    from src.memory.support import supported_current
    # Existing supported proofs scored every cited byte with the same model,
    # threshold and window contract. Preserve those exact legacy approvals.
    if record.get("method") == "source_window_assertions_v1":
        expected_policy = {k: v for k, v in policy(METHOD_V2).items() if k != "premise_composition"}
        expected_policy["method"] = "source_window_assertions_v1"
    elif record.get("method") in (METHOD_V2, *PER_CLAIM_METHODS):
        # A record is judged under its own method, whichever one writes today.
        expected_policy = policy(record["method"])
    else:
        return False
    if (record.get("policy") != expected_policy
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
