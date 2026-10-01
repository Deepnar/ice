"""Admission facts recorded by source reviewers, never guessed from overlap."""

KNOWLEDGE_SCOPES = {"private_history", "assistant_history", "public_knowledge", "mixed"}
TASK_TYPES = {"episodic_lookup", "entity_relation", "summary_synthesis", "temporal",
              "knowledge_update", "procedural", "abstention", "negative", "multi_hop"}


def blank_review() -> dict:
    return {"reviewed_through_turn": None, "recent_only_answerable": None,
            "knowledge_scope": None, "task_types": []}


def validate_review(record: dict, cutoff: int) -> None:
    """A reason + old source ID alone cannot certify long-term answerability."""
    if (type(record.get("reviewed_through_turn")) is not int
            or record["reviewed_through_turn"] != cutoff):
        raise ValueError("valid label must record review through the exact cutoff")
    if record.get("recent_only_answerable") is not False:
        raise ValueError("long-term label must confirm recent history alone cannot answer")
    if record.get("knowledge_scope") not in KNOWLEDGE_SCOPES:
        raise ValueError("valid label must declare private/assistant/public/mixed knowledge scope")
    tasks = record.get("task_types")
    if (not isinstance(tasks, list) or not tasks or any(t not in TASK_TYPES for t in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError("valid label must declare reviewed semantic task types")


def validate_packet(audit: dict, catalog: dict, inputs: list, gap: int, *, native=False) -> int:
    """Catalog-only checks must run before expensive replay, not after it."""
    kind = "native_checkpoint_source_review" if native else "longterm_probe_label_review"
    if audit.get("kind") != kind or audit.get("inputs") != inputs[:4 if native else 5]:
        raise ValueError("review packet belongs to different corpus/catalog inputs")
    if native and audit.get("recent_window_turns") != gap:
        raise ValueError("native review has a different recent-history window")
    seen = set()
    valid = 0
    for row in audit.get("records", []):
        identity = row.get("probe_id")
        if identity not in catalog or identity in seen:
            raise ValueError("review packet has unknown or duplicate probe identity")
        seen.add(identity)
        probe = catalog[identity]
        if (any(row.get(key) != probe.get(key) for key in ("question", "expected_answer", "conversation"))
                or row.get("cutoff_turn") != probe["split_turn"]
                or (not native and row.get("gold_turns") != probe["gold_turns"])):
            raise ValueError("review question/source/cutoff differs from frozen catalog")
        verdict = row.get("answer_verdict" if native else "verdict")
        if verdict not in {"valid", "invalid", "uncertain", None}:
            raise ValueError("review packet has unknown verdict")
        if verdict != "valid":
            continue
        if not row.get("reason", "").strip():
            raise ValueError("valid label needs a concrete review reason")
        validate_review(row, probe["split_turn"])
        gold = row.get("reviewed_gold_turns") if native else row.get("gold_turns")
        if (not isinstance(gold, list) or not gold or any(type(t) is not int for t in gold)
                or len(gold) != len(set(gold)) or min(gold) < 1
                or max(gold) > probe["split_turn"] - gap):
            raise ValueError("valid source labels must lie outside the recent-history window")
        valid += 1
    return valid
