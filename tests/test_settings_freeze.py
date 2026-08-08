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
