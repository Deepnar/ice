"""v3 source claims through storage, SQL visibility and real retrieval packing."""
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.claims import store_claims
from src.memory.models import CodexClaim, ColdStorage, Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

assert (os.environ.get('ICE_TEST_DATABASE','').startswith('ice_test_') and
        make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE'])


class Encoder:
    max_seq_length = 8192
    def tokenizer(self, text, **kwargs):
        return {'input_ids': list(range(len(text.split())))}
    def encode(self, text, **kwargs):
        return np.array([1.0]+[0.0]*1023)


def verified(source, claim):
    return verify_support(source,claim,scorer=lambda pairs:[dict(entailment=.99,neutral=.005,contradiction=.005)])


def test_migration_roundtrip():
    cfg=Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url',settings.database_url.replace('%','%%'))
    command.stamp(cfg,'e2b7c4d91a08')
    command.downgrade(cfg,'d9a1f4b72c60')
    command.upgrade(cfg,'e2b7c4d91a08')


def test_independent_search_and_source_visibility(monkeypatch):
    monkeypatch.setattr(settings,'codex_sentence_claims',True)
    with SessionLocal() as db:
        cid,batch=uuid.uuid4(),uuid.uuid4()
        db.add(Conversation(id=cid));db.flush()
        user='We chose PostgreSQL. It replaced SQLite.'
        assistant='You could consider Redis.'
        row=EpisodicMemory(conversation_id=cid,batch_id=batch,
            raw_text=f'User: {user}\n\nAssistant: {assistant}',
            source_spans=chat_provenance(user,assistant),
            context_reliance='Long_Term_Memory',idempotency_key=str(uuid.uuid4()))
        db.add(row);db.flush()
        claims=store_claims(db,row,['We chose PostgreSQL.',assistant],encoder=Encoder(),verifier=verified)
        store_claims(db,row,['We chose PostgreSQL.'],encoder=Encoder(),verifier=verified)
        assert db.query(CodexClaim).filter_by(source_batch=batch).count()==2
        o=HybridRetrievalOrchestrator(db,None)
        # No entity/triple is created or matched: both search channels stand alone.
        lexical=o._codex_claims('PostgreSQL',None)
        assert len(lexical)==1 and lexical[0].origin_batch_ids==(str(batch),)
        assert 'speaker: user' in lexical[0].text and 'We chose PostgreSQL.' in lexical[0].text
        semantic=o._codex_claims('database decision',Encoder().encode('anything'))
        assert len(semantic)==2
        packed=o._enforce_token_budget(semantic,max_tokens=sum(f.token_count for f in semantic),relevance_order=True)
        assert {f.origin_batch_ids for f in packed}=={(str(batch),)}
        assert len(packed)==2
        assert not o._codex_claims('PostgreSQL',None,{'conversation_ids':[]})
        assert not o._codex_claims('PostgreSQL',None,{'batch_ids':[]})
        assert not o._codex_claims('PostgreSQL',None,{'exclude_conversation_ids':[str(cid)]})
        assert not o._codex_claims('PostgreSQL',None,{'incognito':True})
        row.is_private=True;db.flush()
        assert not o._codex_claims('PostgreSQL',None,{'conversation_id':str(cid)})
        row.is_private=False
        # Archive the source without deleting its Codex evidence.
        db.add(ColdStorage(id=row.id,raw_text=row.raw_text,source_spans=row.source_spans,
            timestamp=row.timestamp,ts_provenance=row.ts_provenance,
            conversation_id=row.conversation_id,batch_id=row.batch_id,is_private=False))
        db.flush()
        db.delete(row);db.flush()
        assert len(o._codex_claims('PostgreSQL',None,{'conversation_id':str(cid)}))==1
        assert not o._codex_claims('PostgreSQL',None,{'exclude_cluster_ids':[str(uuid.uuid4())]})
        row=db.get(ColdStorage,row.id)
        row.raw_text+=' A later correction.';db.flush()
        assert not o._codex_claims('PostgreSQL',None)
        db.delete(row);db.flush()
        assert not o._codex_claims('PostgreSQL',None)
        # The conversation cascade physically removes excerpts, including orphans.
        db.delete(db.get(Conversation,cid));db.flush()
        assert db.query(CodexClaim).filter_by(source_batch=batch).count()==0
        db.rollback()


def test_unknown_verification_preserves_full_context():
    with SessionLocal() as db:
        cid,batch=uuid.uuid4(),uuid.uuid4();db.add(Conversation(id=cid));db.flush()
        user='Only if approved: use Redis. Approval was refused.'
        row=EpisodicMemory(conversation_id=cid,batch_id=batch,
            raw_text=f'User: {user}\n\nAssistant: ',source_spans=chat_provenance(user,''),
            context_reliance='Long_Term_Memory',idempotency_key=str(uuid.uuid4()))
        db.add(row);db.flush()
        def unsupported(source,claim):
            return verify_support(source,claim,scorer=lambda pairs:[dict(entailment=.01,neutral=.98,contradiction=.01)])
        store_claims(db,row,['use Redis.'],encoder=Encoder(),verifier=unsupported)
        fragments=HybridRetrievalOrchestrator(db,None)._codex_claims('Redis',None)
        assert len(fragments)==1 and user in fragments[0].text
        db.rollback()


@pytest.mark.parametrize("subject,expected_edges", [("...",0),("port configuration",1)])
def test_extractor_keeps_source_claims_even_when_graph_names_fail(monkeypatch,subject,expected_edges):
    import json
    from src.workers import codex_extractor as cx
    from src.memory.models import IdempotencyKey
    from src.workers.idempotency import job_key
    monkeypatch.setattr(settings,'codex_sentence_claims',True)
    monkeypatch.setattr(settings,'codex_extraction_chunk_adaptive',False)
    monkeypatch.setattr(cx,'embedder',Encoder())
    monkeypatch.setattr(cx,'known_relations',lambda:[])
    monkeypatch.setattr(cx,'canonical_relation',lambda relation,**kwargs:relation)
    monkeypatch.setattr(cx,'extract_entities',lambda *args,**kwargs:[])
    monkeypatch.setattr(cx,'make_llm_reconciler',lambda:None)
    # Invalid endpoints are deliberately rejected by the real handle_triplet.
    statements=['The port is 8.','You could choose port 9.']
    def create(**kwargs):
        assert kwargs['model']==settings.codex_extraction_model
        prompt=kwargs['messages'][-1]['content']
        sentence=next(s for s in statements if s in prompt)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps({'facts':[dict(subject=subject,relation='uses',object='9',source_sentence=sentence)]})))])
    monkeypatch.setattr(cx,'bg_client',SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    import src.memory.support as support
    monkeypatch.setattr(support,'verify_support',verified)
    with SessionLocal() as db:
        cid,batch=uuid.uuid4(),uuid.uuid4();db.add(Conversation(id=cid));db.flush()
        db.add(EpisodicMemory(conversation_id=cid,batch_id=batch,
            raw_text=f'User: {statements[0]}\n\nAssistant: {statements[1]}',
            source_spans=chat_provenance(*statements),context_reliance='Long_Term_Memory',
            idempotency_key=str(uuid.uuid4())))
        db.commit()
    cx.extract_codex(str(batch),model_used='foreground-must-not-override-specialist')
    with SessionLocal() as db:
        claims=db.query(CodexClaim).filter_by(source_batch=batch).all()
        assert len(claims)==2 and {c.role for c in claims}=={'user','assistant'}
        from src.memory.models import CodexEdge
        assert db.query(CodexEdge).filter_by(source_batch=batch).count()==expected_edges
        from src.memory.models import CodexClaimLink
        assert db.query(CodexClaimLink).count()==2*expected_edges
        assert db.query(IdempotencyKey).filter_by(key=job_key('codex',batch)).count()==1
        o=HybridRetrievalOrchestrator(db,None)
        assert len(o._codex_claims('port',None,{'conversation_id':str(cid)}))==2
        if expected_edges:
            from src.memory.models import CodexEntity
            edge=db.query(CodexEdge).filter_by(source_batch=batch).one()
            src,tgt=db.get(CodexEntity,edge.source_id),db.get(CodexEntity,edge.target_id)
            line=o._fact_line(src,edge,tgt)
            assert all(sentence in line for sentence in statements)
            assert 'speaker: user' in line and 'speaker: assistant' in line
            assert '--uses-->' not in line
            src.context_payload='POISONED cached relationship assertion'
            texts,rendered=[],[]
            o._render_codex_entity(src,0,[edge],[],None,texts,rendered)
            assert 'POISONED' not in '\n'.join(texts)
            assert {e.id for e in rendered}=={edge.id}
        from src.services.conversations import apply_forget
        turn=db.query(EpisodicMemory).filter_by(batch_id=batch).one()
        apply_forget(db,{'turns':[{'id':str(turn.id)}]})
        db.flush()
        assert db.query(CodexClaim).filter_by(source_batch=batch).count()==0
        db.commit()
