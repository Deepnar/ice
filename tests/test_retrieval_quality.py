"""G30: real-encoder retrieval ranking among competing synthetic memories.

This is a small regression fixture for the v3 reader, not a benchmark or an
answer-quality estimate. Run only in a disposable database:
    uv run python tests/support/disposable_database.py tests/test_retrieval_quality.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.engine import make_url

from src.api.config import settings

if (not os.environ.get("ICE_TEST_DATABASE")
        or make_url(settings.database_url).database != os.environ["ICE_TEST_DATABASE"]
        or not os.environ["ICE_TEST_DATABASE"].startswith("ice_test_")):
    raise SystemExit("Use tests/support/disposable_database.py")

from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.embedder import get_embedder
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


# Several facts share a relation and vocabulary with a gold fact. Names and
# values are invented so the test cannot be answered from the model's prior.
MEMORIES = [
    ("atlas_port", "Atlas inventory sync listens on port 8391."),
    ("boreal_port", "Boreal inventory sync listens on port 8000."),
    ("atlas_backup_port", "Atlas backup dashboard listens on port 7722."),
    ("mira_tool", "Mira stopped using Slack for team chat and switched to Matrix."),
    ("sam_tool", "Sam stopped using Teams for team chat and switched to Signal."),
    ("mira_theme", "Mira prefers a dark theme in her editor."),
    ("atlas_database", "Atlas changed its database from SQLite to PostgreSQL on June 4, 2026."),
    ("boreal_database", "Boreal changed its database from MySQL to PostgreSQL on May 2, 2026."),
    ("atlas_language", "Atlas uses Python for its ingestion workers."),
    ("atlas_deploy", "Before deploying Atlas, run migration checks and save a database backup."),
    ("boreal_deploy", "Before deploying Boreal, renew its TLS certificate."),
    ("atlas_region", "Atlas production is hosted in the north Europe region."),
    ("atlas_auth", "Atlas authentication is implemented in src/security/auth.py."),
    ("boreal_auth", "Boreal authentication is implemented in src/auth.py."),
    ("atlas_logging", "Atlas request logging is implemented in src/logging.py."),
    ("tara_food", "Tara avoids sesame when choosing lunch."),
    ("nora_food", "Nora avoids peanuts when choosing lunch."),
    ("tara_drink", "Tara usually orders iced tea in the afternoon."),
    ("kyoto_day", "The Kyoto museum visit is planned for April 12."),
    ("osaka_day", "The Osaka museum visit is planned for April 15."),
    ("kyoto_hotel", "The Kyoto hotel is beside the central station."),
    ("orion_budget", "The approved Orion quarterly budget is 72,000 credits."),
    ("helix_budget", "The approved Helix quarterly budget is 18,000 credits."),
    ("orion_owner", "Nora owns the Orion budget spreadsheet."),
    ("sable_outage", "The Sable outage was caused by an expired TLS certificate."),
    ("boreal_outage", "The Boreal outage was caused by a failed database migration."),
    ("sable_fix", "Sable recovered after the certificate was renewed."),
    ("nora_review", "Nora promised to finish the Orion design review by Thursday."),
    ("mira_review", "Mira scheduled the Helix design review for Friday."),
    ("orion_merge", "Before merging Orion schema changes, run the migration test and back up the database."),
]

QUERIES = [
    ("Which port does Atlas inventory sync use?", "atlas_port"),
    ("ok so what did Mira leave for team messages?", "mira_tool"),
    ("When did Atlas switch its database to PostgreSQL?", "atlas_database"),
    ("How should we get ready to roll Atlas out safely?", "atlas_deploy"),
    ("Where in the code do we check Atlas logins?", "atlas_auth"),
    ("What does Tara need left out of her lunch?", "tara_food"),
    ("On what day is the Kyoto museum visit planned?", "kyoto_day"),
    ("How much can Orion spend this quarter?", "orion_budget"),
    ("What broke Sable's uptime?", "sable_outage"),
    ("When did Nora promise to finish the Orion design review?", "nora_review"),
    ("What should be done before Orion schema changes are merged?", "orion_merge"),
    ("What port does the Atlas backup dashboard listen on?", "atlas_backup_port"),
]

MULTI_QUERIES = [
    ("Which port does Atlas inventory sync use, and when did Atlas switch databases?",
     ("atlas_port", "atlas_database")),
    ("What caused the Sable outage, and what restored service?",
     ("sable_outage", "sable_fix")),
    ("What does Tara avoid for lunch, and what does she drink in the afternoon?",
     ("tara_food", "tara_drink")),
]


def vector(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def main():
    assert settings.retrieval_rerank_enabled, "qualification requires the v3 default reader"
    db = SessionLocal()
    try:
        encoder = get_embedder()
        embeddings = encoder.encode([text for _, text in MEMORIES],
                                    convert_to_tensor=False, show_progress_bar=False)
        conversations = [Conversation(memory_scope_type="auto") for _ in range(6)]
        query_conversation = Conversation(memory_scope_type="auto")
        db.add_all(conversations + [query_conversation])
        db.flush()
        now = datetime.now(timezone.utc)
        rows = {}
        for index, ((key, statement), embedding) in enumerate(zip(MEMORIES, embeddings)):
            row = EpisodicMemory(
                conversation_id=conversations[index // 5].id,
                batch_id=uuid.uuid4(), timestamp=now - timedelta(hours=index),
                topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
                context_reliance="Long_Term_Memory", raw_text=f"User: {statement}",
                embedding=vector(embedding), decay_score=1.0,
                inject_raw=True, lossless_flag=True,
                idempotency_key=f"g30-quality-{uuid.uuid4()}")
            db.add(row)
            rows[key] = row
        db.commit()

        reader = HybridRetrievalOrchestrator(db, encoder)
        reader.max_retrieval_tokens = 220
        def retrieve_for(question):
            classification = ClassificationResult(
                topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
                context_reliance="Long_Term_Memory", raw_probs=[0.0] * 27,
                max_confidence=1.0, prompt=question, p_ltm=1.0,
                head_confidences={"topic": 1.0, "intent": 1.0})
            embedding = vector(encoder.encode(question, convert_to_tensor=False,
                                              show_progress_bar=False))
            return reader.retrieve(classification, str(query_conversation.id),
                                   embedding, scope={}, defer_exposure=True)

        misses = []
        first_ranked = 0
        for question, gold_key in QUERIES:
            fragments = retrieve_for(question)
            rank = next((i for i, fragment in enumerate(fragments, 1)
                         if fragment.source_batch_id == str(rows[gold_key].id)), None)
            print(f"{gold_key}: rank={rank}, final_fragments={len(fragments)}")
            first_ranked += rank == 1
            if rank is None or rank > 5:
                misses.append((gold_key, rank))
        assert not misses, f"gold source absent from final top five: {misses}"
        assert first_ranked >= 10, f"only {first_ranked}/{len(QUERIES)} gold sources ranked first"
        for question, gold_keys in MULTI_QUERIES:
            fragments = retrieve_for(question)
            selected = {fragment.source_batch_id for fragment in fragments}
            absent = [key for key in gold_keys if str(rows[key].id) not in selected]
            print(f"{'+'.join(gold_keys)}: missing={absent}, final_fragments={len(fragments)}")
            if absent:
                misses.append((gold_keys, absent))
        assert not misses, f"required sources absent from final evidence: {misses}"
        print(f"G30 v3 retrieval quality fixture: {first_ranked}/{len(QUERIES)} "
              f"single-source golds first, {len(MULTI_QUERIES)} two-source controls passed")
    finally:
        db.close()


if __name__ == "__main__":
    main()
