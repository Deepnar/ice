"""v3 support verdicts through migration, post-flight and all turn readers."""
import os
import uuid
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal, engine
from src.api.prompt_assembler import get_recent_turns
from src.memory.models import Conversation, EpisodicMemory
from src.memory.representation import verify_representations
from src.memory.source import chat_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services.retrieval_svc import recent_turns

assert (os.environ.get('ICE_TEST_DATABASE','').startswith('ice_test_') and
        make_url(settings.database_url).database==os.environ['ICE_TEST_DATABASE'])


def test_migration_roundtrip():
    cfg=Config('alembic.ini')
    cfg.set_main_option('sqlalchemy.url',settings.database_url.replace('%','%%'))
    command.stamp(cfg,'f3c8d5e02b19')
    command.downgrade(cfg,'e2b7c4d91a08')
    assert 'representation_verification' not in {c['name'] for c in inspect(engine).get_columns('episodic_memory')}
    command.upgrade(cfg,'f3c8d5e02b19')


@pytest.mark.parametrize('supported',[True,False])
def test_postflight_verdict_changes_substitution_in_actual_readers(monkeypatch,supported):
    from src.workers import post_flight as pf
    cid,batch=uuid.uuid4(),uuid.uuid4()
    user='Atlas keeps Redis persistence disabled.'
    assistant='The setting stays disabled.'
    summary='Atlas has Redis persistence disabled.' if supported else 'Atlas enabled Redis persistence.'
    with SessionLocal() as db:
        db.add(Conversation(id=cid));db.flush()
        db.add(EpisodicMemory(conversation_id=cid,batch_id=batch,
            raw_text=f'User: {user}\n\nAssistant: {assistant}',
            source_spans=chat_provenance(user,assistant),context_reliance='Long_Term_Memory',
            idempotency_key=str(uuid.uuid4())))
        db.commit()
    monkeypatch.setattr(pf,'extract_key_terms',lambda *args:{'entities':[],'figures':[],'identifiers':[]})
    monkeypatch.setattr(pf,'compute_entropy',lambda *args:0.1)
    monkeypatch.setattr(pf,'decide_representation',lambda **kwargs:dict(inject_raw=False,
        want_summary=True,summary_decides=True,reason='controlled_summary_candidate'))
    monkeypatch.setattr(pf,'generate_summary',lambda *args,**kwargs:(summary,1.0,None))
    for name in ('extract_codex','extract_procedural','run_chunk_turn'):
        monkeypatch.setattr(pf,name,lambda *args,**kwargs:None)
    calls=[]
    def verifier(source,candidate):
        calls.append((source,candidate))
        return verify_support(source,candidate,scorer=lambda pairs:[dict(
            entailment=.99 if supported else .005,neutral=.005,
            contradiction=.005 if supported else .99)])
    monkeypatch.setattr(pf,'verify_representations',lambda row,s,a:
        verify_representations(row,s,a,verifier=verifier))
    pf.evaluate_turn(str(batch),user,assistant,str(cid),'controlled-model')
    assert len(calls)==1 and 'The user said:' in calls[0][0] and 'The assistant said:' in calls[0][0]
    with SessionLocal() as db:
        row=db.query(EpisodicMemory).filter_by(batch_id=batch).one()
        assert row.inject_raw is not supported
        assert row.representation_verification['summary']['status']==('supported' if supported else 'contradicted')
        recent=recent_turns(db,conversation_id=str(cid))[0]['text']
        prompt=' '.join(m['content'] for m in get_recent_turns(db,str(cid),max_tokens=4000))
        orch=HybridRetrievalOrchestrator(db,None)
        # Actual lexical SQL must SELECT the source metadata and verdict too.
        classification=SimpleNamespace(prompt='Redis',intent_tags=[],topic_tags=[])
        hits=orch._bm25_episodic(classification,None,str(cid))
        assert hits
        selected=' '.join(h.text for h in hits)
        if supported:
            assert summary in recent and summary in prompt and summary in selected
        else:
            assert all(summary not in text for text in (recent,prompt,selected))
            assert all('disabled' in text for text in (recent,prompt,selected))
        db.delete(row);db.flush();db.delete(db.get(Conversation,cid));db.commit()


def test_bookmark_reader_preserves_late_correction_and_refuses_unverified_summary():
    from src.api.prompt_assembler import bookmarked_turn_texts
    with SessionLocal() as db:
        cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
        raw='Earlier discussion. '*600+' Final correction: persistence must remain disabled.'
        row=EpisodicMemory(conversation_id=cid,batch_id=uuid.uuid4(),raw_text=raw,
            summary_text='Enable persistence.',summary_coverage=1.0,inject_raw=False,
            is_bookmarked=True,context_reliance='Long_Term_Memory',idempotency_key=str(uuid.uuid4()))
        db.add(row);db.flush()
        rendered=bookmarked_turn_texts(db,cid)
        assert len(rendered)==1 and rendered[0].endswith('Final correction: persistence must remain disabled.')
        assert 'Enable persistence.' not in rendered[0]
        assert bookmarked_turn_texts(db,uuid.uuid4())==[]
        db.rollback()


def test_chat_route_refuses_required_overflow_before_graph_usage(monkeypatch):
    """Actual v3 route and SQL reads reach the real assembler's refusal."""
    import asyncio
    import json
    from fastapi import BackgroundTasks
    from src.api import main
    from src.classifier.schemas import ClassificationResult

    cid = uuid.uuid4()
    class Request:
        headers = {'X-ICE-Conversation-ID': str(cid)}
        async def json(self):
            return {'model': 'controlled-model', 'messages': [
                {'role': 'user', 'content': 'Explain the complete record. ' * 400}]}
    monkeypatch.setattr(main, 'core', None)
    monkeypatch.setattr(main, 'classifier', SimpleNamespace(classify=lambda *a, **k:
        ClassificationResult([], [], 'Standalone', [], .99)))
    monkeypatch.setattr(main, 'get_fallback_model', lambda: 'controlled-model')
    monkeypatch.setattr(main, 'get_model_context_window', lambda *a: 256)
    monkeypatch.setattr(main, 'serving_window', lambda *a: 256)
    monkeypatch.setattr(main, 'log_window_truth', lambda *a: None)
    monkeypatch.setattr(main, 'decide_memory_retrieval', lambda *a, **k:
        SimpleNamespace(retrieve=False, breakdown={}))
    def unexpected_access(*args, **kwargs):
        raise AssertionError('No evidence was delivered; usage must not be recorded')
    monkeypatch.setattr(main, 'record_graph_access', unexpected_access)
    with SessionLocal() as db:
        try:
            response = asyncio.run(main.chat_completions(Request(), BackgroundTasks(), db))
            assert response.status_code == 400
            assert json.loads(response.body)['error']['code'] == 'context_length_exceeded'
        finally:
            db.query(Conversation).filter_by(id=cid).delete()
            db.commit()


def test_project_constraints_are_live_scoped_and_separate_from_session_status():
    from datetime import datetime, timezone
    from src.memory.models import Decision, Project
    from src.services.projects import chat_constraints, render_session_start

    pid, other = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        try:
            for ident in (pid, other):
                db.add(Project(id=ident, name=str(ident), slug=str(ident), roots=[]))
            db.flush()
            db.add_all([
                Decision(project_id=pid, decision='Keep the schema.', decision_type='constraint'),
                Decision(project_id=pid, decision='Old restriction.', decision_type='constraint',
                         valid_until=datetime.now(timezone.utc)),
                Decision(project_id=other, decision='Foreign restriction.', decision_type='constraint'),
                Decision(project_id=pid, decision='Ordinary decision.', decision_type='decision')])
            db.flush()
            assert chat_constraints(db, pid) == '- Keep the schema.'
            db.add(Decision(project_id=pid, decision='Keep the API.', decision_type='constraint'))
            db.flush()
            assert 'Keep the API.' in chat_constraints(db, pid)
            data = dict(project='Atlas', branch='main', constraints=['Keep the schema.'])
            assert 'Keep the schema.' in render_session_start(data)
            assert 'Keep the schema.' not in render_session_start(data, include_constraints=False)
        finally:
            db.rollback()
