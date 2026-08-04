"""C12 D6: is this blob a DOCUMENT or a TRANSCRIPT of a conversation?

The two have different destinations — a document becomes sections of a
document conversation, a transcript goes through F14's amnesia slicer and
becomes a ghost conversation with real user/assistant roles — so the question
has to be answered before ingestion, and answering it wrongly wastes the whole
run.

**An explicit choice always wins.** The UI (Track F), the CLI flag and the
REST/MCP field all pass `kind` straight through. This module only runs when
nobody said.

**And then it asks the model, not the punctuation.** The rule being replaced is
`post_flight.py`'s `raw_text.count("Assistant:") >= 3`, which is wrong in both
directions: a transcript whose speakers are "Me:"/"Bot:" is filed as a
document, and a 2,001-word prose answer that quotes the word "Assistant:" three
times is filed as a document too. Counting a label is a bet on how the text was
formatted; per CLAUDE.md's standing rule, no decision may depend on how a thing
is written. A model reading a sample judges CONTENT, which is the actual
question.

⚠ G28 acceptance applies to THIS module, not only to what it replaced: the same
conversation, reformatted (labels stripped, renamed, cased differently), must
come back with the same answer. `tests/test_documents.py` check 14 is that
test in miniature and G28's probe set inherits it.
"""

import structlog

logger = structlog.get_logger("ice.ingestion.documents.kind")

DOCUMENT = "document"
TRANSCRIPT = "transcript"
KINDS = (DOCUMENT, TRANSCRIPT)

SAMPLE_CHARS = 1500      # per window; head + middle + tail are sent

_SYSTEM = "You classify text. Answer with exactly one word."
_PROMPT = (
    "Below are three excerpts (beginning, middle, end) of one text.\n\n"
    "Decide what the text IS:\n"
    "- TRANSCRIPT — a back-and-forth dialogue between a person and an AI "
    "assistant (or between people). Turns alternate: someone asks or says "
    "something, someone else responds. Speaker labels may be present, "
    "inconsistent, or missing entirely — judge the STRUCTURE OF THE EXCHANGE, "
    "not the labels.\n"
    "- DOCUMENT — anything written as a standalone artifact: an article, "
    "report, specification, manual, notes, code, a data table, a book chapter. "
    "One authorial voice, no exchange. A document may quote a conversation and "
    "is still a document.\n\n"
    "Answer with exactly one word: TRANSCRIPT or DOCUMENT.\n\n")


def _sample(text: str) -> str:
    """Head + middle + tail. A transcript's shape shows up anywhere in it, but
    a document's front matter can look conversational (a preface, an epigraph),
    so one window from the top is not enough evidence."""
    if len(text) <= SAMPLE_CHARS * 3:
        return text
    middle_start = (len(text) - SAMPLE_CHARS) // 2
    return (
        "--- beginning ---\n" + text[:SAMPLE_CHARS] +
        "\n\n--- middle ---\n" + text[middle_start:middle_start + SAMPLE_CHARS] +
        "\n\n--- end ---\n" + text[-SAMPLE_CHARS:])


def _default_llm(prompt: str, system: str) -> str:
    from src.workers.bg_client_factory import (
        bg_timeout,
        get_bg_client,
        get_bg_model_name,
        json_schema,
    )
    client = get_bg_client()
    completion = client.chat.completions.create(
        model=get_bg_model_name(),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=16, timeout=bg_timeout(16),
        # A genuinely closed two-value set, so an enum is safe here: there is
        # no third honest answer to force the model away from.
        # A bare top-level enum, NOT an object wrapper: the caller matches on
        # the word, and {"kind": "..."} does not fit the token budget this
        # call has always used — wrapping it truncated every answer to empty
        # and the DOCUMENT default then silently swallowed real transcripts.
        response_format=json_schema("blob_kind", {
            "type": "string", "enum": ["DOCUMENT", "TRANSCRIPT"]}))
    return completion.choices[0].message.content or ""


def detect_blob_kind(text: str, *, explicit: str = None, llm=None) -> str:
    """DOCUMENT or TRANSCRIPT. `explicit` short-circuits everything.

    On any model failure the answer is DOCUMENT, and that asymmetry is
    deliberate: a document mis-run through the amnesia slicer is *mangled* —
    the model invents turns and roles that were never there, and the invention
    is stored as memory. A transcript ingested as a document keeps its text
    intact and retrievable; it only loses the role structure. One failure mode
    corrupts, the other under-reads.
    """
    if explicit:
        if explicit not in KINDS:
            raise ValueError(f"unknown document kind '{explicit}' — one of {KINDS}")
        return explicit
    if not text or not text.strip():
        return DOCUMENT

    call = llm or _default_llm
    try:
        answer = (call(_PROMPT + _sample(text), _SYSTEM) or "").strip().lower()
    except Exception as exc:
        logger.warning("blob_kind_detection_failed", error=str(exc),
                       defaulting_to=DOCUMENT)
        return DOCUMENT

    # Substring, because a small model will say "TRANSCRIPT." or "It is a
    # document." however firmly the prompt asked for one word.
    has_transcript = "transcript" in answer
    has_document = "document" in answer
    if has_transcript and not has_document:
        return TRANSCRIPT
    if has_document and not has_transcript:
        return DOCUMENT
    logger.warning("blob_kind_unparsed", answer=answer[:80], defaulting_to=DOCUMENT)
    return DOCUMENT
