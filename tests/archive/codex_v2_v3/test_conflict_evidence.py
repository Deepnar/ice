"""v3 conflict mechanics with synthetic SQL fixtures and controlled decisions."""

import uuid
from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity, CodexEvent, ReviewQueue
from src.services import review
from src.services.errors import ValidationError
from src.workers import codex_extractor as cx
from src.workers import maintenance_agent as ma


@pytest.fixture
def graph():
    with SessionLocal() as db:
        a, b = [CodexEntity(id=uuid.uuid4(), canonical_name=f"control_{uuid.uuid4().hex}")
                for _ in range(2)]
        db.add_all([a, b])
        db.flush()
        yield db, a, b
        db.rollback()


@pytest.mark.parametrize("old,new", [("buys", "sells"), ("teaches", "learns_from"),
                                      ("parent_of", "child_of")])
def test_converses_are_separate_but_not_expiry_evidence(graph, monkeypatch, old, new):
    db, a, b = graph
    edge = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation=old)
    db.add(edge)
    db.flush()
    assert cx._is_inverse_pair(old, new)
    assert cx.check_conflict(db, a.id, new, b.id, "Both relationships hold.") is None
    assert edge.valid_until is None
    monkeypatch.setattr(cx, "get_or_create_entity", lambda db, name, **kw:
                        a if name == a.canonical_name else b)
    monkeypatch.setattr(cx, "_regenerate_context_payload", lambda *args: None)
    cx.handle_triplet(db, a.canonical_name, new, b.canonical_name, uuid.uuid4(),
                      turn_text="Both relationships hold.")
    db.flush()
    assert edge.valid_until is None
    live = db.query(CodexEdge).filter_by(source_id=a.id, target_id=b.id,
                                       valid_until=None).all()
    assert {e.relation for e in live} == {old, new}


def test_opposition_requires_source_decision(graph):
    db, a, b = graph
    edge = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="friend")
    db.add(edge)
    db.flush()
    conflict = cx.check_conflict(db, a.id, "enemy", b.id, "A calls B an enemy.")
    assert conflict
    # Missing source cannot expire, even if a supplied callable would say so.
    assert cx.reconcile_conflict(db, conflict, a, "enemy", b, uuid.uuid4(),
                                 None, lambda _: "expire_old")
    assert edge.valid_until is None
    assert cx.reconcile_conflict(db, conflict, a, "enemy", b, uuid.uuid4(),
                                 "Both are true in different contexts.", lambda _: "keep_both")
    assert edge.valid_until is None


def test_background_candidates_propose_once_without_expiring(graph, monkeypatch):
    db, a, b = graph
    old = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="friend")
    new = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="enemy")
    db.add_all([old, new])
    db.flush()
    # Keep this fixture transaction local even through the production helper.
    monkeypatch.setattr(db, "commit", db.flush)
    items = [i for i in ma._detect_contradictions(db, 20)
             if i.payload["source_id"] == str(a.id)]
    assert len(items) == 1 and items[0].tier == 2
    counters = {"applications": 0, "proposals": 0, "llm_decisions": 0}
    assert ma._process(db, items[0], None, uuid.uuid4(), counters) == "proposed"
    assert counters["proposals"] == 1 and counters["applications"] == 0
    assert ma._apply_contradiction(db, items[0], uuid.uuid4()) == "noop"
    assert old.valid_until is None and new.valid_until is None
    assert db.query(ReviewQueue).filter_by(item_type="codex_contradiction").count() == 1


@pytest.mark.parametrize("keep_count", [0, 1, 2])
@pytest.mark.parametrize("kind", ["codex_contradiction", "codex_reconciliation"])
def test_review_requires_explicit_choice_and_applies_it(graph, monkeypatch, keep_count, kind):
    db, a, b = graph
    edges = [CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id,
                       relation=rel) for rel in ("friend", "enemy")]
    db.add_all(edges)
    db.flush()
    pair = [str(e.id) for e in edges]
    content = {"edge_ids": pair} if kind == "codex_contradiction" else {
        "old_edge_id": pair[0], "new": {"subject": a.canonical_name,
        "relation": "enemy", "object": b.canonical_name}}
    item = ReviewQueue(item_type=kind, item_content=content)
    db.add(item)
    db.flush()
    monkeypatch.setattr(db, "commit", db.flush)
    monkeypatch.setattr(cx, "_regenerate_context_payload", lambda *args: None)
    with pytest.raises(ValidationError):
        review.approve(db, str(item.id))
    assert item.status == "pending"
    with pytest.raises(ValidationError):
        review.approve(db, str(item.id), keep_edge_ids=[str(uuid.uuid4())])
    review.approve(db, str(item.id), keep_edge_ids=pair[:keep_count])
    assert item.status == "approved"
    assert sum(e.valid_until is None for e in edges) == keep_count
    assert db.query(CodexEvent).filter_by(entity_id=a.id,
                                         event_type="edge_expired").count() == 2 - keep_count


@pytest.mark.parametrize("content,finish,expected", [
    ("expire_old", "stop", "expire_old"),
    ("do not expire_old", "stop", "review"),
    ("keep_both or expire_old", "stop", "review"),
    ("expire_old", "length", "review"),
    ("keep_both", "stop", "keep_both"),
])
def test_reconciler_exact_complete_output(monkeypatch, content, finish, expected):
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)])
    monkeypatch.setattr(cx, "bg_client", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    source = "Earlier context. " * 60 + "The final correction must be included."
    ctx = dict(subject="Atlas", relation="enemy", object="Beacon", old_relation="friend",
               old_object="Beacon", turn=source)
    assert cx.make_llm_reconciler()(ctx) == expected
    assert source in captured[0]["messages"][1]["content"]
    captured.clear()
    monkeypatch.setattr(settings, "codex_reconcile_input_tokens", 1)
    assert cx.make_llm_reconciler()(ctx) == "review"
    assert not captured


@pytest.mark.parametrize('relation,negative', [('age',False), ('uses',True), ('lives_in',False),
                                             ('invented_relation',False)])
def test_all_writers_preserve_unresolved_sources(graph, monkeypatch, relation, negative):
    db, a, b = graph
    other = CodexEntity(canonical_name=f'other_{uuid.uuid4().hex}')
    db.add(other); db.flush()
    old = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id,
                    relation=relation, negated=False)
    db.add(old); db.flush()
    nodes = {n.canonical_name:n for n in (a,b,other)}
    monkeypatch.setattr(cx,'get_or_create_entity',lambda db,name,**kw:nodes[name])
    target = b if negative else other
    calls = []
    new = cx.handle_triplet(db,a.canonical_name,relation,target.canonical_name,uuid.uuid4(),
        turn_text='A hypothetical alternative, not a correction.',negated=negative,
        reconciler=lambda ctx:calls.append(ctx) or 'expire_old')
    assert old.valid_until is None and new is not None and new.id != old.id
    assert not calls
    repeated = cx.handle_triplet(db,a.canonical_name,relation,target.canonical_name,
                                 new.source_batch,negated=negative)
    assert repeated.id == new.id and new.valid_until is None
    assert new.strength == 1.0
    if relation in cx.PROPERTY_RELATIONS:
        assert a.properties[relation] == sorted([b.canonical_name,other.canonical_name])


def _claim(db, cid, body, when, role='user', provenance='original'):
    from src.memory.models import CodexClaim, EpisodicMemory
    from src.memory.source import chat_provenance, digest, source_units
    user, assistant = (body,'') if role == 'user' else ('',body)
    raw=f'User: {user}\n\nAssistant: {assistant}'
    row=EpisodicMemory(conversation_id=cid,batch_id=uuid.uuid4(),raw_text=raw,
        timestamp=when,ts_provenance=provenance,source_spans=chat_provenance(user,assistant),
        context_reliance='Long_Term_Memory',idempotency_key=str(uuid.uuid4()))
    db.add(row);db.flush()
    unit=next(u for u in source_units(row) if u.role==role)
    c=CodexClaim(source_batch=row.batch_id,episodic_id=row.id,conversation_id=cid,
        raw_sha256=digest(raw),start=unit.start,end=unit.end,role=role,
        sentence=body,sentence_sha256=digest(body),text=body,verification={})
    db.add(c);db.flush()
    return row,c


@pytest.mark.parametrize('case,expected_calls', [('correction',1),('older_import',0),
    ('other_speaker',0),('unknown_time',0),('other_conversation',0),('edited',0),('same_time',0)])
def test_source_authority_and_chronology_reach_writer(graph,monkeypatch,case,expected_calls):
    from datetime import datetime,timedelta,timezone
    from src.memory.models import CodexClaimLink,Conversation
    db,a,b=graph
    cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
    now=datetime(2026,9,1,tzinfo=timezone.utc)
    oldrow,oldclaim=_claim(db,cid,'Atlas uses Beacon.',now)
    edge=CodexEdge(source_batch=oldrow.batch_id,source_id=a.id,target_id=b.id,relation='uses')
    db.add(edge);db.flush();db.add(CodexClaimLink(claim_id=oldclaim.id,edge_id=edge.id));db.flush()
    if case=='other_conversation':
        cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
    delta=-1 if case=='older_import' else 0 if case=='same_time' else 1
    newrow,newclaim=_claim(db,cid,'Atlas no longer uses Beacon.',now+timedelta(days=delta),
        role='assistant' if case=='other_speaker' else 'user',
        provenance='unknown' if case=='unknown_time' else 'original')
    if case=='edited':oldrow.raw_text+=' Correction outside the saved excerpt.';db.flush()
    nodes={n.canonical_name:n for n in (a,b)}
    monkeypatch.setattr(cx,'get_or_create_entity',lambda db,name,**kw:nodes[name])
    calls=[]
    def decide(ctx):
        calls.append(ctx)
        assert ctx['old_source']['text']=='Atlas uses Beacon.'
        assert ctx['new_source']['text']=='Atlas no longer uses Beacon.'
        assert ctx['old_source']['role']==ctx['new_source']['role']=='user'
        return 'expire_old'
    new=cx.handle_triplet(db,a.canonical_name,'uses',b.canonical_name,newrow.batch_id,
        negated=True,source_claims=[newclaim],reconciler=decide)
    assert new is not None and new.negated
    assert len(calls)==expected_calls
    assert (edge.valid_until is not None)==bool(expected_calls)


def test_all_candidates_and_rejected_replacement_roll_back_expiries(graph,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from src.memory.models import CodexClaimLink,Conversation
    db,a,b=graph
    cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
    now=datetime(2026,9,1,tzinfo=timezone.utc)
    targets=[b]+[CodexEntity(canonical_name=f'alt_{uuid.uuid4().hex}') for _ in range(2)]
    db.add_all(targets[1:]);db.flush()
    edges=[]
    for target in targets[:2]:
        row,c=_claim(db,cid,'Atlas uses '+target.canonical_name+'.',now)
        edge=CodexEdge(source_batch=row.batch_id,source_id=a.id,target_id=target.id,relation='uses')
        db.add(edge);db.flush();db.add(CodexClaimLink(claim_id=c.id,edge_id=edge.id));edges.append(edge)
    row,c=_claim(db,cid,'Atlas uses the alternative.',now+timedelta(days=1));db.flush()
    nodes={n.canonical_name:n for n in [a]+targets}
    monkeypatch.setattr(cx,'get_or_create_entity',lambda db,name,**kw:nodes[name])
    calls=[]
    def decide(ctx):
        calls.append(ctx)
        return 'expire_old' if len(calls)==1 else 'reject_new'
    result=cx.handle_triplet(db,a.canonical_name,'uses',targets[2].canonical_name,row.batch_id,
                             reconciler=decide,source_claims=[c])
    assert result is None and len(calls)==2
    assert all(e.valid_until is None for e in edges)


def test_maintenance_cannot_bypass_source_boundary(graph,monkeypatch):
    db,a,b=graph
    edges=[CodexEdge(source_batch=uuid.uuid4(),source_id=a.id,target_id=b.id,
                     relation='uses',negated=neg) for neg in (False,True)]
    db.add_all(edges);db.flush()
    content={'old_edge_id':str(edges[0].id),'old_relation':'uses','old_object':b.canonical_name,
             'new':{'subject':a.canonical_name,'relation':'uses','object':b.canonical_name,'negated':True}}
    row=ReviewQueue(item_type='codex_reconciliation',item_content=content)
    db.add(row);db.flush()
    item=ma.WorkItem('reconciliation_leftover',1,{'review_id':str(row.id),'content':content})
    calls=[]
    assert ma._decide_reconciliation(db,item,lambda *args:calls.append(args) or {'decision':'expire_old'})=='unsure'
    assert calls==[]
    assert ma._find_edge_by_names(db,a.canonical_name,'uses',b.canonical_name,True).id==edges[1].id
    monkeypatch.setattr(db,'commit',db.flush)
    assert ma._apply_reconciliation(db,item,'expire_old',uuid.uuid4())=='still_unsure'
    assert all(e.valid_until is None for e in edges)


def test_maintenance_uses_complete_sources_and_applies_qualified_decision(graph,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from src.memory.models import CodexClaimLink,Conversation
    db,a,b=graph
    cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
    now=datetime(2026,9,1,tzinfo=timezone.utc)
    edges=[]
    for days,neg,body in [(0,False,'Atlas uses Beacon.'),(1,True,'Atlas no longer uses Beacon.')]:
        row,claim=_claim(db,cid,body,now+timedelta(days=days))
        edge=CodexEdge(source_batch=row.batch_id,source_id=a.id,target_id=b.id,relation='uses',negated=neg)
        db.add(edge);db.flush();db.add(CodexClaimLink(claim_id=claim.id,edge_id=edge.id));edges.append(edge)
    db.flush()
    content={'old_edge_id':str(edges[0].id),'old_relation':'uses','old_object':b.canonical_name,
             'new_batch_id':str(edges[1].source_batch),
             'new':{'subject':a.canonical_name,'relation':'uses','object':b.canonical_name,'negated':True}}
    review_row=ReviewQueue(item_type='codex_reconciliation',item_content=content)
    db.add(review_row);db.flush()
    item=ma.WorkItem('reconciliation_leftover',1,{'review_id':str(review_row.id),'content':content})
    prompts=[]
    def decide(prompt,max_tokens=200):
        prompts.append(prompt)
        return {'decision':'expire_old'}
    verdict=ma._decide_reconciliation(db,item,decide)
    assert verdict=='expire_old' and len(prompts)==1
    assert 'Atlas uses Beacon.' in prompts[0] and 'Atlas no longer uses Beacon.' in prompts[0]
    assert 'role' in prompts[0] and 'recorded_at' in prompts[0]
    monkeypatch.setattr(db,'commit',db.flush)
    assert ma._apply_reconciliation(db,item,verdict,uuid.uuid4())=='applied'
    assert edges[0].valid_until is not None and edges[1].valid_until is None
    assert 'NOT' in a.context_payload and review_row.status=='resolved'
