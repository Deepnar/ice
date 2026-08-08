"""G9 behaviour freeze — every knob that moved into settings kept its value.

The point of G9 is that tuning stops requiring a code edit. The risk of G9 is
that a value changes *while* it moves, silently, and every measurement taken
afterwards describes a system nobody chose.

So this suite does not check the numbers against a list somebody typed. It
reads each literal **out of the git blob at the base commit** and compares it
to what `settings` serves today. Retyping the numbers here would mirror any
mistake made in the move; deriving them cannot (TRAPS #5 — a two-sided
assertion, or it proves nothing).

Each row is (setting_name, path_at_base, line_no_in_base_blob, regex). The line
number is stable because the base blob is immutable. The regex must match on
that exact line, and its group(1) is the old literal — if the anchor drifts,
the row fails loudly rather than silently checking nothing.

Run:  uv run pytest tests/test_settings_freeze.py -q
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.config import settings  # noqa: E402

# The commit G9 started from — `ae3abe8 Record cluster ②`. Every literal below
# is read from this tree, not from memory of it.
BASE = "ae3abe82d275ffb0a8d1ccd59057f2dbc8779aa1"

ORCH = "src/retrieval/orchestrator.py"
DECAY = "src/workers/decay.py"
CODEX_DECAY = "src/workers/codex_decay.py"
PROC_DECAY = "src/workers/procedural_decay.py"
COMPACTION = "src/workers/compaction.py"
IMPORTER = "src/ingestion/importer.py"
DENSITY = "src/workers/turn_density.py"
CHUNKING = "src/memory/chunking.py"
CONV_SUM = "src/workers/conversation_summary.py"
REFLECTION = "src/workers/reflection.py"
BATCH_SUM = "src/workers/batch_summarizer.py"
EXTRACTOR = "src/workers/codex_extractor.py"
TEMPLATES = "src/classifier/templates.py"

# (setting, file, line in the base blob, regex whose group(1) is the literal)
FROZEN = [
    # ── module constants ────────────────────────────────────────────────
    ("retrieval_episodic_recency_boost", ORCH, 73, r"EPISODIC_RECENCY_BOOST = ([\d.]+)"),
    ("retrieval_episodic_recency_tau_days", ORCH, 74, r"EPISODIC_RECENCY_TAU_DAYS = ([\d.]+)"),
    ("retrieval_wide_net_budget_fraction", ORCH, 78, r"WIDE_NET_BUDGET_FRACTION = ([\d.]+)"),
    ("retrieval_wide_net_budget_floor", ORCH, 79, r"WIDE_NET_BUDGET_FLOOR = (\d+)"),
    ("retrieval_bonus_bookmarked", ORCH, 126, r"BONUS_BOOKMARKED = ([\d.]+)"),
    ("retrieval_bonus_recent_top_10pct", ORCH, 127, r"BONUS_RECENT_TOP_10PCT = ([\d.]+)"),
    ("retrieval_bonus_recent_top_30pct", ORCH, 128, r"BONUS_RECENT_TOP_30PCT = ([\d.]+)"),
    ("retrieval_bonus_long_narrative", ORCH, 129, r"BONUS_LONG_NARRATIVE = ([\d.]+)"),
    ("retrieval_bonus_substantial", ORCH, 130, r"BONUS_SUBSTANTIAL = ([\d.]+)"),
    ("retrieval_penalty_short", ORCH, 131, r"PENALTY_SHORT = (-[\d.]+)"),
    ("retrieval_bonus_keyword_match", ORCH, 132, r"BONUS_KEYWORD_MATCH = ([\d.]+)"),
    ("retrieval_max_total_bonus_multiplier", ORCH, 133, r"MAX_TOTAL_BONUS_MULTIPLIER = ([\d.]+)"),
    ("retrieval_meta_downweight_factor", ORCH, 138, r"META_DOWNWEIGHT_FACTOR = ([\d.]+)"),

    # ── __init__ instance attributes ────────────────────────────────────
    ("codex_relation_top_k", ORCH, 169, r"RELATION_TOP_K = (\d+)"),
    ("codex_relation_sim_floor", ORCH, 170, r"RELATION_SIM_FLOOR = ([\d.]+)"),
    ("codex_relation_overlap_boost", ORCH, 171, r"RELATION_OVERLAP_BOOST = ([\d.]+)"),
    ("codex_expansion_max_terms", ORCH, 172, r"EXPANSION_MAX_TERMS = (\d+)"),
    ("codex_enum_edge_limit", ORCH, 173, r"ENUM_EDGE_LIMIT = (\d+)"),
    ("codex_enum_entity_limit", ORCH, 174, r"ENUM_ENTITY_LIMIT = (\d+)"),
    ("codex_max_depth", ORCH, 181, r"CODEX_MAX_DEPTH = (\d+)"),
    ("codex_direct_trust_floor", ORCH, 182, r"CODEX_DIRECT_TRUST_FLOOR = ([\d.]+)"),
    ("codex_deep_strength_floor", ORCH, 183, r"CODEX_DEEP_STRENGTH_FLOOR = ([\d.]+)"),
    ("codex_reinforce_increment", ORCH, 184, r"CODEX_REINFORCE_INCREMENT = ([\d.]+)"),
    ("codex_strength_cap", ORCH, 185, r"CODEX_STRENGTH_CAP = ([\d.]+)"),
    ("codex_promote_strength", ORCH, 186, r"CODEX_PROMOTE_STRENGTH = ([\d.]+)"),
    ("codex_promote_min_confidence", ORCH, 187, r"CODEX_PROMOTE_MIN_CONFIDENCE = ([\d.]+)"),
    ("codex_recency_boost", ORCH, 191, r"CODEX_RECENCY_BOOST = ([\d.]+)"),
    ("codex_recency_tau_days", ORCH, 192, r"CODEX_RECENCY_TAU_DAYS = ([\d.]+)"),

    # ── literals that were buried inside function bodies ────────────────
    ("retrieval_long_narrative_words", ORCH, 281, r"word_count > (\d+)"),
    ("retrieval_substantial_words", ORCH, 283, r"word_count > (\d+)"),
    ("retrieval_short_words", ORCH, 285, r"word_count < (\d+)"),
    ("retrieval_min_total_bonus", ORCH, 299, r"bonus = max\((-[\d.]+),"),
    ("retrieval_recency_min_turns", ORCH, 311, r"if total <= (\d+):"),
    ("retrieval_recent_top_pct", ORCH, 318, r"recency_pct < ([\d.]+)"),
    ("retrieval_recent_mid_pct", ORCH, 320, r"recency_pct < ([\d.]+)"),
    ("retrieval_cluster_top_k", ORCH, 212, r"top_k=(\d+)"),
    ("retrieval_cluster_candidate_multiplier", ORCH, 230, r'"limit": top_k \* (\d+)'),
    ("codex_entity_match_threshold", ORCH, 1166, r"threshold: float = ([\d.]+)"),
    ("codex_entity_payload_match_limit", ORCH, 1442, r"\.limit\((\d+)\)\.all\(\)"),
    ("codex_entity_edge_limit", ORCH, 1493, r"\.limit\((\d+)\)\.all\(\):"),
    ("retrieval_rrf_k", ORCH, 2223, r"k: int = (\d+)"),
    ("retrieval_max_per_conversation", ORCH, 2372, r"max_per_conversation=(\d+)"),
    ("retrieval_bm25_candidate_limit", ORCH, 920, r"LIMIT (\d+)"),
    ("retrieval_vector_candidate_limit", ORCH, 1038, r'else (\d+),'),
    ("retrieval_vector_candidate_limit_evolution", ORCH, 1038, r'"cand_limit": (\d+) if'),
    ("retrieval_chunk_candidate_limit", ORCH, 1127, r"LIMIT (\d+)"),
    ("retrieval_wide_net_candidate_limit", ORCH, 2541, r"LIMIT (\d+)"),
    ("retrieval_procedural_limit", ORCH, 1904, r"LIMIT (\d+)"),
    ("retrieval_batch_summary_limit", ORCH, 1980, r"LIMIT (\d+)"),
    ("retrieval_conversation_summary_limit", ORCH, 2012, r"LIMIT (\d+)"),

    # ── budget ladder scalars ───────────────────────────────────────────
    ("context_total_budget_fallback", ORCH, 345, r"TOTAL_CONTEXT_BUDGET = ([\d_]+)"),
    ("context_overhead_reserve", ORCH, 346, r"OVERHEAD_RESERVE = ([\d_]+)"),

    # ── memory dynamics ─────────────────────────────────────────────────
    # The DAILY targets, read off the exponent expressions they used to be
    # buried inside; the per-cycle rates are derived from these now.
    ("decay_cycles_per_day", DECAY, 21, r"CYCLES_PER_DAY = ([\d.]+)"),
    ("decay_daily_unaccessed", DECAY, 22, r"= ([\d.]+) \*\* \(1\.0 / CYCLES_PER_DAY\)"),
    ("decay_daily_accessed", DECAY, 23, r"= ([\d.]+) \*\* \(1\.0 / CYCLES_PER_DAY\)"),
    ("decay_daily_creative", DECAY, 26, r"= ([\d.]+) \*\* \(1\.0 / CYCLES_PER_DAY\)"),
    ("decay_strengthen_amount", DECAY, 27, r"STRENGTHEN_AMOUNT = ([\d.]+)"),
    ("decay_archive_threshold", DECAY, 28, r"ARCHIVE_THRESHOLD = ([\d.]+)"),
    ("decay_cold_threshold", DECAY, 29, r"COLD_THRESHOLD = ([\d.]+)"),
    ("decay_creative_floor", IMPORTER, 59, r"CREATIVE_FLOOR = ([\d.]+)"),
    ("codex_decay_cycles_per_day", CODEX_DECAY, 15, r"CYCLES_PER_DAY = ([\d.]+)"),
    ("codex_decay_daily", CODEX_DECAY, 16, r"= ([\d.]+) \*\* \(1\.0 / CYCLES_PER_DAY\)"),
    ("codex_demotion_threshold", CODEX_DECAY, 17, r"DEMOTION_THRESHOLD = ([\d.]+)"),
    ("codex_expiry_threshold", CODEX_DECAY, 18, r"EXPIRY_THRESHOLD = ([\d.]+)"),
    ("procedural_stale_days", PROC_DECAY, 11, r"STALE_DAYS = (\d+)"),
    ("procedural_min_reinforcement", PROC_DECAY, 12, r"MIN_REINFORCEMENT = (\d+)"),
    ("compaction_event_threshold", COMPACTION, 12, r"EVENT_THRESHOLD = (\d+)"),

    # ── write path ──────────────────────────────────────────────────────
    ("turn_raw_keep_max_words", DENSITY, 55, r"RAW_KEEP_MAX_WORDS = (\d+)"),
    ("turn_long_turn_chunk_words", DENSITY, 56, r"LONG_TURN_CHUNK_WORDS = (\d+)"),
    ("turn_density_lossless_threshold", DENSITY, 58, r"DENSITY_LOSSLESS_THRESHOLD = ([\d.]+)"),
    ("turn_summary_coverage_threshold", DENSITY, 60, r"SUMMARY_COVERAGE_THRESHOLD = ([\d.]+)"),
    ("turn_max_must_terms", DENSITY, 61, r"MAX_MUST_TERMS = (\d+)"),
    ("chunk_tokens", CHUNKING, 21, r"CHUNK_TOKENS = (\d+)"),
    ("chunk_overlap_words", CHUNKING, 22, r"OVERLAP_WORDS = (\d+)"),
    ("conversation_summary_max_words", CONV_SUM, 36, r"SUMMARY_MAX_WORDS = (\d+)"),
    ("conversation_summary_chunk_words", CONV_SUM, 37, r"CHUNK_WORDS = (\d+)"),
    ("conversation_summary_per_turn_words", CONV_SUM, 38, r"PER_TURN_WORDS = (\d+)"),
    ("reflection_enrich_limit", REFLECTION, 299, r"ENRICH_LIMIT = (\d+)"),
    ("reflection_enrich_refresh_days", REFLECTION, 300, r"ENRICH_REFRESH_DAYS = (\d+)"),
    ("batch_summary_age_days", BATCH_SUM, 22, r"BATCH_SUMMARY_AGE_DAYS = (\d+)"),
    ("codex_conf_grounded", EXTRACTOR, 308, r"CONF_GROUNDED = ([\d.]+)"),
    ("codex_conf_ungrounded", EXTRACTOR, 309, r"CONF_UNGROUNDED = ([\d.]+)"),
    ("codex_conf_rejected", EXTRACTOR, 310, r"CONF_REJECTED = ([\d.]+)"),
    ("classifier_context_turns", TEMPLATES, 98, r"CONTEXT_TURNS = (\d+)"),
    ("classifier_context_max_words", TEMPLATES, 99, r"CONTEXT_MAX_WORDS = (\d+)"),
    ("classifier_turn_word_cap", TEMPLATES, 100, r"TURN_WORD_CAP = (\d+)"),
]

_blob_cache: dict[str, list[str]] = {}


def _base_lines(path: str) -> list[str]:
    if path not in _blob_cache:
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "show", f"{BASE}:{path}"],
                             cwd=repo, capture_output=True, text=True, check=True)
        _blob_cache[path] = out.stdout.splitlines()
    return _blob_cache[path]


@pytest.mark.parametrize("name,path,lineno,pattern",
                         FROZEN, ids=[r[0] for r in FROZEN])
def test_default_matches_base_commit(name, path, lineno, pattern):
    line = _base_lines(path)[lineno - 1]
    m = re.search(pattern, line)
    assert m, (f"anchor drifted: {path}:{lineno} in {BASE[:7]} does not match "
               f"{pattern!r}\n  line: {line!r}")
    old = float(m.group(1))
    new = float(getattr(settings, name))
    assert new == old, (f"{name} changed while moving into settings: "
                        f"{old} at {BASE[:7]} -> {new} now")


def test_every_frozen_setting_exists():
    """A typo'd setting name would otherwise make its row vanish silently."""
    missing = [n for n, *_ in FROZEN if not hasattr(settings, n)]
    assert not missing, f"declared frozen but absent from Settings: {missing}"


def _base_literal(path: str, assignment: str):
    """ast.literal_eval the literal assigned by *assignment* in the base blob.

    Transcribing the weight table into this file by hand would be a
    hand-authored probe of my own edit (TRAPS #13) — it would agree with
    whatever I typed. Lifting the source text cannot.

    Scans from the opening delimiter to its match so the extracted text is a
    complete literal; `#` comments inside it are fine, ast strips them.
    """
    import ast
    src = "\n".join(_base_lines(path))
    i = src.index(assignment) + len(assignment)
    while src[i] in " \n":
        i += 1
    opener = src[i]
    closer = {"{": "}", "[": "]", "(": ")"}[opener]
    depth, j = 0, i
    while True:
        if src[j] == opener:
            depth += 1
        elif src[j] == closer:
            depth -= 1
            if depth == 0:
                break
        j += 1
    return ast.literal_eval(src[i:j + 1])


def test_leg_base_weights_match_base_commit():
    old = {k: float(v) for k, v in _base_literal(ORCH, "base_weights =").items()}
    new = {k: float(v) for k, v in settings.retrieval_leg_base_weights.items()}
    assert new == old, f"base leg weights changed: {old} -> {new}"


def test_leg_profiles_match_base_commit():
    """Every intent's row must still hold the weights it held at the base commit.

    The old table keyed one override off a SET of intents; the settings table
    is flat (one row per intent), so the comparison expands the sets.
    """
    rows = _base_literal(ORCH, "PROFILES =")
    expanded = {}
    for intents, override in rows:
        for intent in intents:
            expanded[intent] = {k: float(v) for k, v in override.items()}
    new = {k: {kk: float(vv) for kk, vv in v.items()}
           for k, v in settings.retrieval_leg_profiles.items()}
    assert new == expanded, (
        f"leg profiles diverged from {BASE[:7]}\n"
        f"  only in base: {set(expanded) - set(new)}\n"
        f"  only in settings: {set(new) - set(expanded)}\n"
        f"  differing: {[k for k in set(new) & set(expanded) if new[k] != expanded[k]]}")


def test_topic_overrides_match_base_commit():
    """The two cumulative topic bumps, read off the base blob's if-statements."""
    lines = _base_lines(ORCH)
    src = "\n".join(lines)
    creative = re.search(r'blend_weights\["codex"\] = blend_weights\.get\("codex", [\d.]+\) \+ ([\d.]+)', src)
    software = re.search(r'blend_weights\["procedural"\] = blend_weights\.get\("procedural", [\d.]+\) \+ ([\d.]+)', src)
    assert creative and software, "topic-override anchors drifted in the base blob"
    got = settings.retrieval_leg_topic_overrides
    assert float(got["Creative_&_Media"]["codex"]) == float(creative.group(1))
    assert float(got["Software_&_Tech"]["procedural"]) == float(software.group(1))


def test_removed_constants_are_gone():
    """The old names must not survive as a second copy of the value.

    Two copies of one number is exactly how the ablation harness's
    `recency_boost` flag came to do nothing (ROADMAP G19): the subclass zeroed
    its own imported copy while the parent scored from another.
    """
    import src.retrieval.orchestrator as orch
    stale = [n for n in (
        "EPISODIC_RECENCY_BOOST", "EPISODIC_RECENCY_TAU_DAYS",
        "WIDE_NET_BUDGET_FRACTION", "WIDE_NET_BUDGET_FLOOR",
        "BONUS_BOOKMARKED", "BONUS_RECENT_TOP_10PCT", "BONUS_RECENT_TOP_30PCT",
        "BONUS_LONG_NARRATIVE", "BONUS_SUBSTANTIAL", "PENALTY_SHORT",
        "BONUS_KEYWORD_MATCH", "MAX_TOTAL_BONUS_MULTIPLIER",
        "META_DOWNWEIGHT_FACTOR",
    ) if hasattr(orch, n)]
    assert not stale, f"module constant survived the move: {stale}"


def test_budget_ladders_match_base_commit():
    """The growth-cap and recent-fraction brackets, lifted off the base blob.

    Both ladders were if/elif chains; the numbers are read back out of those
    chains rather than retyped here, for the reason in _base_literal.
    """
    src = "\n".join(_base_lines(ORCH))

    cap = [[int(a.replace("_", "")), int(b.replace("_", "")), int(c)]
           for a, b, c in re.findall(
               r"turn_count < ([\d_]+):\s*\n\s*growth_cap = ([\d_]+) \+ "
               r"(?:turn_count|\(turn_count - [\d_]+\)) \* ([\d_]+)", src)]
    assert len(cap) == 3, f"growth-cap anchors drifted: found {len(cap)} brackets"
    got = [[int(x) for x in row] for row in settings.context_growth_cap_ladder]
    assert got == cap, f"growth-cap ladder changed: {cap} -> {got}"

    frac = [[int(a), float(b)] for a, b in re.findall(
        r"turn_count < (\d+):\s*\n\s*base = ([\d.]+)", src)]
    assert len(frac) == 4, f"recent-fraction anchors drifted: found {len(frac)}"
    got_f = [[int(r[0]), float(r[1])] for r in settings.context_recent_fraction_ladder]
    assert got_f == frac, f"recent-fraction ladder changed: {frac} -> {got_f}"

    dens = [[int(a), -float(b)] for a, b in re.findall(
        r"avg_tokens_per_turn > (\d+):\s*\n\s*base -= ([\d.]+)", src)]
    assert len(dens) == 3, f"density anchors drifted: found {len(dens)}"
    got_d = [[int(r[0]), float(r[1])] for r in settings.context_recent_density_ladder]
    assert got_d == dens, f"density ladder changed: {dens} -> {got_d}"


def test_recent_fraction_groups_apply_once_each():
    """A group must not double-count when two of its labels are active.

    The pre-G9 code tested set INTERSECTION and applied its delta once; a
    label->delta map would have applied it per matching label, which is a
    behaviour change disguised as a data-structure change.
    """
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator as H

    class C:
        def __init__(self, i, t):
            self.intent_tags, self.topic_tags = i, t

    o = object.__new__(H)
    one = H._compute_recent_fraction(o, 100, 0, C(["Factual_Retrieval"], []))
    two = H._compute_recent_fraction(
        o, 100, 0, C(["Factual_Retrieval", "Troubleshooting"], []))
    assert one == two, ("two labels from one group moved the fraction "
                        f"({one} -> {two}); the delta is being applied twice")
