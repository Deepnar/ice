"""C1 — real per-turn density signal + the raw-vs-summary decision.

Replaces post_flight's crude ``is_lossless`` heuristic (``` → True, >500 words
→ True, ≥3 capitalized words → True — which fired on nearly every substantive
turn) and the backwards summarisation branch that crushed the *densest* long
turns into 2–3 sentences while leaving short banter raw.

Three pure pieces (no DB, no LLM — smoke-testable, G22):

  * ``extract_key_terms`` — the MUST-PRESERVE vocabulary of a turn: named
    entities (shared MicroNER), figures (numbers/dates/versions/money), and
    identifiers (code-ish names). Used to ground the summary prompt (the same
    ground-then-generate pattern as Codex A2 and clustering v5) and to
    *measure* a summary afterwards.

  * ``compute_entropy`` — facts-per-token ∈ [0,1], stored in
    ``episodic_memory.entropy_score`` (the column existed since v2 but was
    written as None forever). Dense turns compress badly; diffuse turns
    compress well.

  * ``decide_representation`` + ``summary_coverage`` — the decision matrix.

STORE BOTH, CHOOSE AT READ TIME (user design decision, 2026-07): a turn's raw
text and its grounded summary are BOTH stored — the storage layer never
permanently forces one representation. ``inject_raw`` is only the *default
hint*; the retrieval layer picks per query (intent preference, keyword
protection, budget degradation — see orchestrator._choose_representation).
The same turn legitimately deserves raw for an exact-fact question and
summary under budget pressure — a write-time decision can't know which.

The coverage gate (roadmap: "a summary that drops the exact fact the user
will ask for is worse than raw"): every summary is *measured* against the
must-preserve terms (one retry naming dropped terms), and the score is stored
in ``episodic_memory.summary_coverage`` — read-time never prefers, and budget
never degrades to, a summary that failed it. Retrievability is a gate, not a
hope. (Search-side is safe either way — BM25/vector index raw_text
regardless; coverage protects what the *answering model sees*.)

Matrix (want_summary = generate & store; inject_raw = the default hint):
  document        → raw hint, no summary (C2 will chunk; a 300-token summary
                    of a 5k-word paste is useless — and this path owns the
                    old clobber bug: document raw now survives assignment)
  short (≤350w)   → raw hint, no summary (savings trivial — skip the LLM call)
  code (long)     → raw hint + summary stored ("what the code does" — the
                    degraded budget form; code itself can't be lossless-summarised)
  creative (long) → raw hint (narrative continuity) + summary stored
  long diffuse    → summary stored; coverage gate sets the hint
  long dense      → summary stored; coverage gate sets the hint (more
                    must-terms → harder to pass → raw hint wins more often)
"""

import re

# ── Thresholds (module constants; G9 sweeps tunables into settings later) ──
RAW_KEEP_MAX_WORDS = 350        # at/below: raw only, no summary generated
LONG_TURN_CHUNK_WORDS = 600     # C3: turns longer than this get chunk-level embeddings
                                # (document_chunker), not just is_document pastes
DENSITY_LOSSLESS_THRESHOLD = 0.35  # entropy at/above ⇒ lossless_flag (codex-extraction gate);
                                   # deliberately generous — codex was historically starved
SUMMARY_COVERAGE_THRESHOLD = 0.7   # must-term fraction a summary must retain to be trusted
MAX_MUST_TERMS = 25

# No trailing \b: after a non-word unit char like '%' a word boundary can
# never match, which silently stripped units ("42%" captured as bare "42").
_NUM_RE = re.compile(
    r"\b\d+(?:[.,:]\d+)*\s?(?:%|km|kg|mb|gb|ms|min|h|am|pm|usd|eur|inr|\$|€|₹|[kmb]\b|s\b)?",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(
    r"\b(?:[a-z0-9]+_[a-z0-9_]+"          # snake_case
    r"|[a-z]+[A-Z][A-Za-z0-9]+"           # camelCase / PascalCase tail
    r"|[A-Za-z_][\w-]*\.[a-z]{1,4}"       # dotted.paths / file.ext
    r"|[A-Z]{2,}[0-9]*)\b"                # ACRONYMS / codes
)


def extract_key_terms(text: str, embedder, max_chars: int = 2500) -> dict:
    """MUST-PRESERVE terms of a turn. NER via the shared MicroNER (pass the
    worker's already-loaded embedder — never instantiate a new one, G13)."""
    from src.retrieval.ner_utils import extract_entities

    # A9b: the BACKGROUND tier. Precision is the whole game here — these terms
    # are injected into the summariser prompt as "must appear verbatim" AND are
    # what `summary_coverage` scores against, so a junk term makes ICE demand
    # noise and then grade itself on having preserved it. Measured 2026-08-03:
    # the micro-NER saturated the MAX_MUST_TERMS cap on EVERY turn (median 25
    # of 25), meaning the list was simply the first 25 spans it emitted —
    # fragments like "gaining conciousness" and "deeply rooted" included.
    entities = list(dict.fromkeys(
        extract_entities(text or "", embedder, max_chars=max_chars,
                         tier="background")
    ))[:MAX_MUST_TERMS]

    body = (text or "")[:max_chars * 2]
    figures = []
    for m in _NUM_RE.finditer(body):
        tok = m.group(0).strip()
        # bare tiny integers ("a 2 of them") are noise, not facts
        if len(tok) > 1 and tok not in figures:
            figures.append(tok)
    identifiers = []
    for m in _IDENT_RE.finditer(body):
        tok = m.group(0)
        if tok not in identifiers and tok.lower() not in {e.lower() for e in entities}:
            identifiers.append(tok)
    return {
        "entities": entities,
        "figures": figures[:15],
        "identifiers": identifiers[:15],
    }


def compute_entropy(text: str, key_terms: dict, has_code: bool) -> float:
    """Facts-per-token ∈ [0,1]. Weighted blend of entity density, figure/
    identifier density, code presence, and lexical diversity."""
    words = (text or "").split()
    n = max(1, len(words))
    per100 = n / 100.0

    entity_density = min(1.0, (len(key_terms["entities"]) / per100) / 6.0)
    figure_density = min(1.0, ((len(key_terms["figures"]) + len(key_terms["identifiers"])) / per100) / 6.0)
    # type-token ratio over a fixed window (raw TTR shrinks with length)
    window = [w.lower() for w in words[:300]]
    diversity = len(set(window)) / max(1, len(window))

    score = (
        0.45 * entity_density +
        0.25 * figure_density +
        0.20 * (1.0 if has_code else 0.0) +
        0.10 * diversity
    )
    return round(min(1.0, max(0.0, score)), 4)


def decide_representation(word_count: int, entropy: float, has_code: bool,
                          is_creative: bool, is_document: bool) -> dict:
    """The storage-side matrix. ``want_summary`` = generate & store the
    grounded summary (both representations kept — read time chooses).
    ``inject_raw`` = the *default hint* only. ``summary_decides=True`` means
    the measured coverage sets the hint (summary-by-default only when it
    demonstrably preserved the key terms)."""
    if is_document:
        # Whole-doc raw until C2 chops at ingestion; a 300-token summary of a
        # 5,000-word paste is useless (and could blow the bg model's window).
        # This path also owns the old clobber bug: document raw-injection now
        # survives to the final assignment.
        return {"inject_raw": True, "want_summary": False,
                "summary_decides": False, "reason": "document"}
    if word_count <= RAW_KEEP_MAX_WORDS:
        return {"inject_raw": True, "want_summary": False,
                "summary_decides": False, "reason": "short"}
    if has_code:
        # Code can't be summarised losslessly → raw hint; the summary
        # ("what the code does") is stored as the degraded budget form.
        return {"inject_raw": True, "want_summary": True,
                "summary_decides": False, "reason": "code"}
    if is_creative:
        # Narrative continuity wants exact wording → raw hint; summary stored
        # for classifier context / budget degradation / future C3/C4 layers.
        return {"inject_raw": True, "want_summary": True,
                "summary_decides": False, "reason": "creative_continuity"}
    reason = "dense_long" if entropy >= DENSITY_LOSSLESS_THRESHOLD else "diffuse_long"
    return {"inject_raw": True, "want_summary": True,
            "summary_decides": True, "reason": reason}


def must_terms(key_terms: dict) -> list:
    seen, out = set(), []
    for term in (key_terms["entities"] + key_terms["figures"] + key_terms["identifiers"]):
        low = term.lower()
        if low not in seen:
            seen.add(low)
            out.append(term)
    return out[:MAX_MUST_TERMS]


def summary_coverage(summary: str, key_terms: dict) -> float:
    """Fraction of must-terms present in the summary (case-insensitive).
    No must-terms → vacuously 1.0."""
    terms = must_terms(key_terms)
    if not terms:
        return 1.0
    low = (summary or "").lower()
    hit = sum(1 for t in terms if t.lower() in low)
    return round(hit / len(terms), 4)
