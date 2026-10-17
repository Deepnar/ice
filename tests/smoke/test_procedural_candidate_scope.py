"""Eligible procedures compete before the candidate cap, through real SQL."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, ProceduralMemory, Project
from src.memory.source import chat_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


def exercise_procedural_scope(monkeypatch, case, *, expected=True, actual=False):
    if not os.getenv('ICE_TEST_DATABASE'):
        raise RuntimeError('requires disposable PostgreSQL')
    query = 'When you review my patches, how do I want the review organized?'
    workflow = 'The user prefers patch reviews to list failing tests before proposed changes.'
    classifier = NS(prompt=query, context_reliance='Long_Term_Memory',
        topic_tags=['Software_&_Tech'], intent_tags=['Factual_Retrieval'],
        head_confidences={'topic': 1., 'intent': 1.}, max_confidence=1.)
    db = SessionLocal()
    projects, conversations, batches, pattern_ids = [], [], [], []
    try:
        for _ in range(2):
            p = Project(name=str(uuid.uuid4()), slug=str(uuid.uuid4()), roots=[])
            db.add(p); db.flush(); projects.append(p.id)
            c = Conversation(project_id=p.id)
            db.add(c); db.flush(); conversations.append(c.id)
            b = uuid.uuid4(); batches.append(b)
            db.add(EpisodicMemory(conversation_id=c.id, batch_id=b, session_id=uuid.uuid4(),
                timestamp=datetime.now(timezone.utc), context_reliance='Long_Term_Memory',
                raw_text=f'User: {workflow}\n\nAssistant: Noted.',
                source_spans=chat_provenance(workflow, 'Noted.'), idempotency_key=str(uuid.uuid4())))
        scope = None
        high_project, low_project = None, None
        high_sources, low_sources = [batches[0]], [batches[0]]
        high_trigger = {}
        high_description = 'This higher-cosine procedure is ineligible for this request.'
        low_description = workflow
        high_active, high_confidence = True, .8
        high_observed = low_observed = datetime.now(timezone.utc)
        expected_sources = [batches[0]]
        if case == 'project':
            scope = {'project_id': str(projects[0])}
            high_project, low_project = projects[1], projects[0]
        elif case == 'allowed':
            scope = {'batch_ids': [str(batches[0])]}
            high_sources = [batches[1]]
        elif case == 'excluded':
            scope = {'exclude_conversation_ids': [str(conversations[1])]}
            high_sources = [batches[1]]
        elif case in ('topic', 'intent'):
            high_trigger = {'topic_tags': ['Creative_Writing']} if case == 'topic' else {'intent_tags': ['Casual_Banter']}
        elif case == 'project_exemption':
            scope = {'project_id': str(projects[0]), 'batch_ids': [str(batches[0])]}
            high_project = low_project = projects[0]
            high_sources = [batches[1]]
            expected_sources = high_sources
            high_description, low_description = workflow, high_description
        elif case == 'partial_exclusion':
            scope = {'exclude_conversation_ids': [str(conversations[1])]}
            high_sources = [batches[0], batches[1]]
            expected_sources = high_sources
            high_description, low_description = workflow, high_description
        elif case == 'empty_allow':
            scope = {'batch_ids': []}
        elif case == 'inactive':
            high_active = False
        elif case == 'confidence':
            high_confidence = .1
        elif case == 'source_clock':
            from src.retrieval.timescope import TimeScope, to_scope_dict
            scope = {'timescope': to_scope_dict(TimeScope('range',
                low_observed - timedelta(days=1), low_observed + timedelta(days=1)))}
            high_observed += timedelta(days=10)
        else:
            raise ValueError(case)
        for description, embedding, pid, sources, trigger, active, confidence, observed in (
            (high_description, [1., 0.] + [0.] * 1022, high_project, high_sources,
             high_trigger, high_active, high_confidence, high_observed),
            (low_description, [.8, .6] + [0.] * 1022, low_project, low_sources,
             {'_equivalence_candidates': []}, True, .8, low_observed),
        ):
            p = ProceduralMemory(pattern_name=description[:80], pattern_description=description,
                embedding=embedding, project_id=pid, source_batch_ids=sources, trigger_conditions=trigger,
                reinforcement_count=3, confidence_score=confidence, is_active=active,
                first_observed=observed, last_observed=observed)
            db.add(p); db.flush(); pattern_ids.append(p.id)
        db.commit()
        monkeypatch.setattr(settings, 'retrieval_procedural_limit', 1)
        monkeypatch.setattr(settings, 'retrieval_rerank_enabled', True)
        monkeypatch.setattr(settings, 'retrieval_rerank_min_score', None)
        import src.retrieval.reranker as reranker
        calls = []
        real_score = reranker.score_pairs
        def score(pairs):
            calls.extend(pairs)
            return real_score(pairs) if actual else [5.] * len(pairs)
        monkeypatch.setattr(reranker, 'score_pairs', score)
        o = HybridRetrievalOrchestrator(db, None)
        o.max_retrieval_tokens = 100
        for name in ('_bm25_episodic', '_vector_episodic', '_facts_leg',
                     '_batch_summary_lookup', '_cold_lookup', '_relevant_cluster_ids'):
            monkeypatch.setattr(o, name, lambda *_a, **_kw: [])
        final = o.retrieve(classifier, str(conversations[0]), [1., 0.] + [0.] * 1022,
                           scope=scope, defer_exposure=True)
        assert bool(final) is expected
        if expected:
            # A source-clock query can also carry the ordinary timeline note.
            # The procedural candidate cap does not cap every output family.
            procedures = [f for f in final if f.source_type == 'procedural']
            assert len(procedures) == 1 and procedures[0].text.endswith(workflow)
            assert procedures[0].text.startswith('[Unverified stored procedure]')
            assert procedures[0].origin_batch_ids == tuple(str(b) for b in expected_sources)
            assert sum(f.token_count for f in final) <= o.max_retrieval_tokens
            assert len(calls) == 1  # cap is still one eligible candidate
        return dict(case=case, eligible_in_final=bool(final), limit=1,
                    scored=len(calls), actual_reranker=actual)
    finally:
        db.rollback()
        db.query(ProceduralMemory).filter(ProceduralMemory.id.in_(pattern_ids)).delete(synchronize_session=False)
        db.query(EpisodicMemory).filter(EpisodicMemory.batch_id.in_(batches)).delete(synchronize_session=False)
        db.query(Conversation).filter(Conversation.id.in_(conversations)).delete(synchronize_session=False)
        db.query(Project).filter(Project.id.in_(projects)).delete(synchronize_session=False)
        db.commit(); db.close()


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
@pytest.mark.parametrize('case', ['project', 'allowed', 'excluded', 'topic', 'intent'])
def test_eligibility_precedes_limit(monkeypatch, case):
    exercise_procedural_scope(monkeypatch, case)


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
@pytest.mark.parametrize('case', ['project_exemption', 'partial_exclusion', 'empty_allow',
                                'inactive', 'confidence', 'source_clock'])
def test_existing_authority_guards_survive(monkeypatch, case):
    exercise_procedural_scope(monkeypatch, case, expected=case != 'empty_allow')
