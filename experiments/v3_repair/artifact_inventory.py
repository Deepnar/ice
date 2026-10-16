"""Isolated research catalogs: physical live rows and row-payload revisions."""

import json
import os

from sqlalchemy import select

from experiments.v3_repair.prepare_inputs import digest

POLICY = "ice-v3-stored-row-catalog-1"
CONTENT = {
    "episodic_memory": ("raw_text", "source_spans", "summary_text", "abstract_text", "timestamp", "ts_provenance"),
    "cold_storage": ("raw_text", "source_spans", "summary_text", "abstract_text", "timestamp", "ts_provenance"),
    "episodic_chunks": ("turn_id", "chunk_index", "chunk_text"),
    "cold_chunks": ("turn_id", "chunk_index", "chunk_text"),
    "batch_summaries": ("conversation_id", "summary_text", "source_manifest"),
    "batch_notes": ("text", "mode", "recorded_range", "source_ids", "batch_ids"),
    "conversation_summaries": ("summary_text", "source_manifest", "covers_through", "covers_turns"),
    "conversation_notes": ("text", "mode", "recorded_range", "source_ids", "batch_ids"),
    "codex_entities": ("canonical_name", "aliases", "entity_type", "description", "properties", "context_payload"),
    "codex_edges": ("source_id", "target_id", "relation", "negated", "source_batch", "observed_batches",
                    "valid_from", "valid_until", "learned_at", "unlearned_at"),
    "codex_claims": ("source_batch", "episodic_id", "conversation_id", "raw_sha256", "start", "end",
                     "role", "sentence", "sentence_sha256", "text"),
    # CODEX_V4: verified fact sentences, their sources and entity links.
    "codex_facts": ("text", "speaker", "modality", "subject", "relation", "object", "negated",
                    "mentioned_at", "valid_from", "valid_until", "superseded_by"),
    "codex_fact_sources": ("source_batch", "episodic_id", "conversation_id", "window", "quote",
                           "recorded_at"),
    "codex_fact_entities": ("entity_id", "role"),
    "procedural_memory": ("pattern_name", "pattern_description", "topic_tags", "trigger_conditions", "source_batch_ids"),
    "memory_slots": ("slot_name", "content", "scope_tier", "project_id", "conversation_id"),
}


def key(values):
    return str(values[0]) if len(values) == 1 else json.dumps(
        [str(v) if not isinstance(v, int) else v for v in values], separators=(",", ":"))


def capture(db):
    from src.memory.models import Base

    database = db.get_bind().engine.url.database
    if database == "ice_db" or not database or os.environ.get("ICE_TEST_DATABASE") != database:
        raise ValueError("artifact inventories require the isolated research store")
    rows, schemas = [], {}
    with db.no_autoflush:
        for name, fields in sorted(CONTENT.items()):
            table = Base.metadata.tables[name]
            columns = [c for c in table.columns if c.name != "embedding"]
            if not set(fields) <= {c.name for c in columns}:
                raise ValueError("artifact inventory payload schema changed: " + name)
            primary = list(table.primary_key.columns)
            schemas[name] = dict(primary=[c.name for c in primary], state=[c.name for c in columns],
                                 content=list(fields))
            for row in db.execute(select(*columns).order_by(*primary)).mappings():
                identifier = key([row[c.name] for c in primary])
                logical = (["original", identifier] if name in {"episodic_memory", "cold_storage"}
                    else ["chunk", identifier] if name in {"episodic_chunks", "cold_chunks"}
                    else [name, identifier])
                state = json.loads(json.dumps(dict(row), default=str, ensure_ascii=False))
                rows.append(dict(table=name, key=identifier, logical_identity=logical,
                    stored_content_sha256=digest({f: state[f] for f in fields}),
                    stored_state_sha256=digest(state)))
    value = dict(version="v3", policy=POLICY, schema=schemas, rows=rows,
                 eligible_inventory="unobserved", lifetime_write_events="unobserved")
    value["catalog_sha256"] = digest(value)
    return value


def validate(value):
    if (not isinstance(value, dict) or value.get("version") != "v3" or value.get("policy") != POLICY
            or set(value.get("schema", {})) != set(CONTENT)
            or value.get("eligible_inventory") != "unobserved"
            or value.get("lifetime_write_events") != "unobserved"
            or value.get("catalog_sha256") != digest({k: v for k, v in value.items() if k != "catalog_sha256"})):
        raise ValueError("stored artifact inventory identity differs")
    observed = {}
    for name, schema in value["schema"].items():
        if (schema.get("content") != list(CONTENT[name]) or not schema.get("primary")
                or not set(schema["primary"]) <= set(schema.get("state", []))
                or not set(CONTENT[name]) <= set(schema.get("state", []))):
            raise ValueError("stored artifact inventory field policy differs")
    for row in value["rows"]:
        ref = (row["table"], row["key"])
        if ref[0] not in CONTENT or not isinstance(ref[1], str) or not ref[1] or ref in observed:
            raise ValueError("stored artifact inventory key malformed/duplicate")
        logical = (["original", ref[1]] if ref[0] in {"episodic_memory", "cold_storage"}
            else ["chunk", ref[1]] if ref[0] in {"episodic_chunks", "cold_chunks"}
            else list(ref))
        if row.get("logical_identity") != logical:
            raise ValueError("stored artifact logical identity differs")
        for label in ("stored_content_sha256", "stored_state_sha256"):
            h = row[label]
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("stored artifact revision hash malformed")
        observed[ref] = row
    return observed
