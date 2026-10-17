"""Actual E4B/pinned NLI/entity writer/reader controls on disposable originals."""

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest

from scripts.z1.run_meta import file_digest, run_meta
from src.api.db import SessionLocal
from src.memory import entity_notes
from src.memory.models import CodexEntity, CodexEvent, Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.workers import reflection
from test_entity_notes import renderer


def put(db, name):
    cid, bid = uuid.uuid4(), uuid.uuid4()
    db.add(Conversation(id=cid)); db.flush()
    user = f'For {name}, I approved port 7813. I explicitly refused port 9123.'
    assistant = 'Noted your approval and refusal.'
    turn = EpisodicMemory(conversation_id=cid, batch_id=bid, context_reliance='Long_Term_Memory',
        raw_text=f'User: {user}\n\nAssistant: {assistant}', source_spans=chat_provenance(user, assistant),
        idempotency_key=str(uuid.uuid4()))
    entity = CodexEntity(canonical_name=name)
    db.add_all([turn, entity]); db.flush()
    db.add(CodexEvent(entity_id=entity.id, event_type='edge_added', batch_source=bid)); db.commit()
    return turn, entity


def check():
    assert os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_')
    with SessionLocal() as db:
        turn, positive = put(db, 'Atlas')
        raw = turn.raw_text
        from src.workers import conversation_summary
        original = conversation_summary._default_llm
        def capture(*args, **kwargs):
            generated = original(*args, **kwargs)
            print(json.dumps(dict(stage='positive_generation', generated=generated)))
            return generated
        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(conversation_summary, '_default_llm', capture)
            success = reflection._enrich_codex_entities(db)
        db.commit()
        print(json.dumps(dict(stage='positive_review', receipt=success,
                              review=positive.properties.get(entity_notes.REVIEW_KEY)), indent=2))
        assert success['supported'] == 1 and success['errors'] == 0, success
        assert '7813' in positive.description and entity_notes.current_note_support(db, positive)
        assert '[Source-supported account]' in renderer(db, positive)
        assert db.get(EpisodicMemory, turn.id).raw_text == raw
        other_turn, negative = put(db, 'Boreal')
        # The writer must refuse a complete but false candidate even when the
        # generator returned successfully. Use the real pinned NLI here.
        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(reflection, 'generate_note', lambda entity, turns:
                entity_notes.generate_note(entity, turns, llm=lambda *a, **k: json.dumps(dict(
                    summary=[dict(text='The user approved Boreal port 9123.', sources=['w0'])]))))
            failure = reflection._enrich_codex_entities(db)
        db.commit()
        assert failure['supported'] == 0 and negative.description == ''
        assert db.get(EpisodicMemory, other_turn.id).raw_text == other_turn.raw_text
        return dict(positive=success, opposing=failure,
                    positive_proof=positive.properties[entity_notes.PROOF_KEY],
                    opposing_review=negative.properties[entity_notes.REVIEW_KEY],
                    originals_unchanged=True, actual_models='E4B positive generation and pinned NLI both controls',
                    limit='synthetic mechanism controls; no independent corpus precision or answer benefit')


if __name__ == '__main__':
    meta = run_meta(script=__file__, args={}, inputs=[file_digest(p) for p in
        [__file__, 'src/memory/entity_notes.py', 'src/workers/reflection.py', 'src/retrieval/orchestrator.py']],
        settings_keys=['background_model_name', 'source_support_model', 'source_support_revision', 'source_support_device'],
        extra={'version': 'v3', 'scope': 'synthetic complete entity sources, actual writer/verifier/SQL/reader', 'cloud_calls': 0})
    print(json.dumps(dict(meta=meta, result=check()), indent=2))
