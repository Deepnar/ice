"""Semantic identity controls; equivalence does not establish habit truth."""

import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.memory import pattern_support as support
from src.memory.source import chat_provenance, digest


def probability(status):
    return dict(entailment=.99 if status == 'supported' else .005,
                neutral=.99 if status == 'unknown' else .005,
                contradiction=.99 if status == 'contradicted' else .005)


def test_identity_never_calls_model():
    def forbidden(_pairs):
        raise AssertionError('exact identity needs no model')
    receipt = support.equivalence_receipts('Same.', ['Same.'], scorer=forbidden)[0]
    assert receipt['status'] == 'equivalent' and receipt['directions'] == []
    assert receipt['proposed_sha256'] == receipt['existing_sha256'] == digest('Same.')


@pytest.mark.parametrize('directions,status', [
    (['supported', 'supported'], 'equivalent'),
    (['supported', 'unknown'], 'unknown'),
    (['unknown', 'supported'], 'unknown'),
    (['contradicted', 'unknown'], 'different'),
    (['unknown', 'unknown'], 'unknown'),
])
def test_both_complete_directions_required(directions, status):
    calls = []
    def scorer(pairs):
        calls.append(pairs)
        return [probability(s) for s in directions]
    record = support.equivalence_receipts('Proposal.', ['Existing.'], scorer=scorer)[0]
    assert calls == [[('Proposal.', 'Existing.'), ('Existing.', 'Proposal.')]]
    assert record['status'] == status and record['policy_threshold'] == settings.source_support_threshold
    assert record['directions'][0]['source_sha256'] == digest('Proposal.')
    assert record['directions'][1]['source_sha256'] == digest('Existing.')


@pytest.mark.parametrize('scores', [[], [{}] * 2,
    [dict(entailment=float('nan'), neutral=0, contradiction=0)] * 2,
    [dict(entailment=.99, neutral=.99, contradiction=.99)] * 2])
def test_bad_scores_cannot_approve_merge(scores):
    assert support.equivalence_receipts('New.', ['Old.'], scorer=lambda _: scores)[0]['status'] == 'unknown'


def test_error_does_not_erase_exact_identity_and_yield_propagates():
    def broken(_):
        raise RuntimeError('unavailable')
    records = support.equivalence_receipts('New.', ['Old.', 'New.'], scorer=broken)
    assert [r['status'] for r in records] == ['unknown', 'equivalent']
    from src.workers.runtime import JobYielded
    def yielding(_):
        raise JobYielded()
    with pytest.raises(JobYielded):
        support.equivalence_receipts('New.', ['Old.'], scorer=yielding)


def test_overlong_candidate_does_not_hide_valid_next(monkeypatch):
    monkeypatch.setattr(support, 'support_tokenizer', lambda: lambda p, h, **_: {
        'input_ids': list(range(settings.source_support_max_tokens + 1))
        if 'long' in (p, h) else [1, 2]})
    calls = []
    def scorer(pairs, **_):
        calls.append(pairs)
        return [probability('supported')] * len(pairs)
    monkeypatch.setattr(support, 'score_pairs', scorer)
    records = support.equivalence_receipts('New.', ['long', 'Equivalent.'])
    assert [r['status'] for r in records] == ['unknown', 'equivalent']
    assert calls == [[('New.', 'Equivalent.'), ('Equivalent.', 'New.')]]


OLD = 'The user prefers patch reviews to list failing tests before proposed changes.'
SAME = 'When reviewing patches, the user wants failing tests stated first and proposed changes afterward.'
OPPOSITE = 'The user prefers patch reviews to list proposed changes before failing tests.'
NARROW = 'For Rust patches only, the user wants failing tests listed before proposed changes.'


def exercise_pattern_writer(monkeypatch, case, *, actual=False):
    """Real SQL/writer; controlled proposal. Actual mode uses pinned NLI/encoder."""
    from src.api.db import SessionLocal
    from src.memory.models import Conversation, EpisodicMemory, IdempotencyKey, ProceduralMemory, Project
    from src.workers import procedural_extractor as worker
    if not os.getenv('ICE_TEST_DATABASE'):
        raise RuntimeError('requires disposable database')
    db = SessionLocal()
    project = Project(name=f'pattern-{uuid.uuid4()}', slug=f'pattern-{uuid.uuid4()}', roots=[])
    db.add(project)
    db.flush()
    conversation = Conversation(project_id=project.id)
    db.add(conversation)
    db.flush()
    keys, calls = [], []
    source_ids = []
    def session(description, count=3, session_id=None):
        sid = session_id or uuid.uuid4()
        rows = []
        for _ in range(count):
            batch = uuid.uuid4()
            row = EpisodicMemory(conversation_id=conversation.id, session_id=sid, batch_id=batch,
                raw_text=f'User: {description}\n\nAssistant: Noted.',
                source_spans=chat_provenance(description, 'Noted.'),
                context_reliance='Long_Term_Memory', timestamp=datetime.now(timezone.utc),
                ts_provenance='original', idempotency_key=str(uuid.uuid4()))
            db.add(row)
            rows.append(row)
            source_ids.append(batch)
        db.flush()
        return sid, rows
    try:
        stored = [OPPOSITE, SAME] if case == 'nonnearest' else [OLD]
        proposed = OLD if case == 'nonnearest' else {'same': SAME, 'opposite': OPPOSITE, 'narrow': NARROW}[case]
        if not actual:
            monkeypatch.setattr(worker, 'encode_pattern', lambda text: [1.0] + [0.0] * 1023)
            def proof(p, old):
                def scorer(pairs):
                    return [probability('supported' if {p, s} <= {OLD, SAME} or p == s else 'unknown')
                            for s in old for _ in range(2) if p != s]
                return support.equivalence_receipts(p, old, scorer=scorer)
            monkeypatch.setattr(worker, 'equivalence_receipts', proof)
        patterns = []
        for index, description in enumerate(stored):
            _, rows = session(description)
            pattern = ProceduralMemory(id=uuid.UUID(int=index + 1), pattern_name=description[:80],
                pattern_description=description, project_id=project.id, reinforcement_count=1,
                is_active=False, confidence_score=.3, source_batch_ids=[r.batch_id for r in rows],
                embedding=worker.encode_pattern(description))
            db.add(pattern)
            patterns.append(pattern)
        new_session, rows = session(proposed)
        db.commit()
        proposed_sources = {r.batch_id for r in rows}
        def create(**kwargs):
            calls.append(kwargs)
            return NS(choices=[NS(finish_reason='stop', message=NS(
                content=f'PATTERN: {proposed} | EVIDENCE: 1, 2'))])
        monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=create))))
        monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'controlled-proposal')
        monkeypatch.setattr('src.model_registry.registry.get_model_context_window', lambda _: 32768)
        monkeypatch.setattr('src.model_registry.runtime_probe.serving_window', lambda *_: 32768)
        worker.extract_procedural(str(rows[-1].batch_id))
        keys.append(worker.job_key('procedural', f'{new_session}:0'))
        db.expire_all()
        all_patterns = db.query(ProceduralMemory).filter_by(project_id=project.id).all()
        equivalent = case in ('same', 'nonnearest')
        expected = patterns[-1] if case == 'nonnearest' else patterns[0]
        assert len(all_patterns) == len(stored) + (0 if equivalent else 1)
        assert expected.reinforcement_count == (2 if equivalent else 1)
        assert bool(proposed_sources & set(expected.source_batch_ids)) is equivalent
        if equivalent:
            proof = expected.trigger_conditions['_last_equivalence']
            assert proof['status'] == 'equivalent' and proof['proposed_sha256'] == digest(proposed)
            if case == 'nonnearest':
                checks = expected.trigger_conditions['_equivalence_candidates']
                assert checks[0]['candidate_id'] == str(patterns[0].id)
                assert checks[0]['proof']['status'] != 'equivalent'
                assert checks[1]['candidate_id'] == str(expected.id)
            observed, sources = expected.last_observed, set(expected.source_batch_ids)
            # A growing sitting and an unchanged bucket must not strengthen it.
            _, more = session(proposed, count=2, session_id=new_session)
            db.commit()
            worker.extract_procedural(str(more[-1].batch_id))
            keys.append(worker.job_key('procedural', f'{new_session}:1'))
            db.expire_all()
            assert expected.reinforcement_count == 2 and expected.last_observed == observed
            assert set(expected.source_batch_ids) == sources
            worker.extract_procedural(str(more[-1].batch_id))
            assert len(calls) == 2
        else:
            new = next(p for p in all_patterns if p.pattern_description == proposed)
            assert new.trigger_conditions['_equivalence_candidates'][0]['proof']['status'] != 'equivalent'
            assert set(new.source_batch_ids) <= proposed_sources
            worker.extract_procedural(str(rows[-1].batch_id))
            assert len(calls) == 1
        return dict(case=case, patterns=len(all_patterns), reinforcement=expected.reinforcement_count,
                    proposal_calls=len(calls), actual_encoder_and_nli=actual,
                    opposite_sources_preserved_separately=not equivalent)
    finally:
        db.rollback()
        db.query(ProceduralMemory).filter_by(project_id=project.id).delete(synchronize_session=False)
        db.query(IdempotencyKey).filter(IdempotencyKey.key.in_(keys)).delete(synchronize_session=False)
        db.query(EpisodicMemory).filter(EpisodicMemory.batch_id.in_(source_ids)).delete(synchronize_session=False)
        db.query(Conversation).filter_by(id=conversation.id).delete(synchronize_session=False)
        db.query(Project).filter_by(id=project.id).delete(synchronize_session=False)
        db.commit()
        db.close()


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
@pytest.mark.parametrize('case', ['same', 'opposite', 'narrow', 'nonnearest'])
def test_writer_requires_semantic_identity(monkeypatch, case):
    exercise_pattern_writer(monkeypatch, case)
