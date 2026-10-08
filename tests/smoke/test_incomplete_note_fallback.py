"""v3 incomplete notes retain complete indexed sources, never generated prefixes."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_assembler import conversation_summary_block
from src.memory.models import Conversation, ConversationNote, ConversationSummary, EpisodicMemory
from src.memory.source import chat_provenance
from src.memory.summary_snapshot import snapshot_matches, source_snapshot
from src.memory.support import verify_support
from src.workers import bg_client_factory, conversation_summary as worker
from src.workers.completion_text import IncompleteCompletion


class Encoder:
    def encode(self, _text, **_kwargs):
        return np.array([1.] + [0.] * 1023)


def test_real_writer_indexes_originals_and_persists_length_reason(monkeypatch):
    cid = uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        for i in range(2):
            user = f'Aster value {i} remains unchanged. The user refused deployment.'
            db.add(EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=f'User: {user}\n\nAssistant: Keep deployment disabled.',
                                  source_spans=chat_provenance(user, 'Keep deployment disabled.'),
                                  timestamp=datetime.now(timezone.utc) + timedelta(minutes=i), ts_provenance='original',
                                  context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4())))
        db.commit()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(usage=NS(prompt_tokens=400, completion_tokens=900, total_tokens=1300),
                  choices=[NS(finish_reason='length', message=NS(content='The user approved deployment.'))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'local-background')
    from src.model_registry import registry, runtime_probe
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *_a: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *_a: 32768)
    monkeypatch.setattr(worker, 'estimate_recent_window_tokens', lambda _n: 0)
    with SessionLocal() as db:
        stats = worker.run_conversation_summaries(db, embedder=Encoder(), conversation_ids=[cid])
        assert stats['incomplete_note_groups'] == 1 and stats['source_fallback_groups'] == 1
        row = db.query(ConversationSummary).filter_by(conversation_id=cid).one()
        notes = db.query(ConversationNote).filter_by(conversation_id=cid).all()
        assert len(notes) == 2 and row.covers_turns == 2
        assert snapshot_matches(row.source_manifest, row.summary_text, source_snapshot(db, cid))
        for part in row.source_manifest['parts']:
            assert part['mode'] == 'source' and part['verification'] is None
            assert part['support_failure'] == 'incomplete_generation'
            assert part['generation_failure']['finish_reason'] == 'length'
            assert part['generation_failure']['completion_tokens'] == 900
            assert 'The user refused deployment.' in part['text']
            assert 'The user approved deployment.' not in part['text']
        block = conversation_summary_block(db, str(cid), 2, 1000, 0, max_tokens=2000)
        assert block and 'The user refused deployment.' in block
        assert 'The user approved deployment.' not in block
        # The complete original checkpoint is reused without the failed model
        # prefix being retried or treated as corroboration.
        again = worker.run_conversation_summaries(db, embedder=Encoder(), conversation_ids=[cid])
        assert again['incomplete_note_groups'] == 0 and len(calls) == 1
    assert calls[0]['max_tokens'] == settings.conversation_summary_max_tokens == 900


@pytest.mark.parametrize('reason', [None, 'content_filter', 'tool_calls'])
def test_nonlength_failures_remain_failures(reason):
    def llm(*_a, **_kw):
        raise IncompleteCompletion('incomplete controlled output', finish_reason=reason)
    with pytest.raises(IncompleteCompletion):
        worker._source_note([NS(id='one',timestamp=None,ts_provenance='unknown')], 'Original source.',
                            False, llm, verify_support)
