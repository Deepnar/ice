"""Bounded synthetic v3 qualification of the actual two-source reconciler.

Run from repo root with uv run python experiments/v3_repair/qualify_reconciliation.py.
Not an answer-quality evaluation or a statistical accuracy estimate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.workers.codex_extractor import make_llm_reconciler
from src.workers.bg_client_factory import get_bg_model_name

CASES = [
    ('replacement','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite.','Atlas has replaced SQLite with PostgreSQL. SQLite is no longer used.',{'expire_old'}),
    ('addition','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite for local tests.','Atlas also uses PostgreSQL for production.',{'keep_both'}),
    ('proposal','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite.','We might replace SQLite with PostgreSQL, but have not decided.',{'keep_both','reject_new'}),
    ('negation','uses','SQLite','uses','SQLite',True,
     'Atlas uses SQLite.','Atlas no longer uses SQLite. We removed it yesterday.',{'expire_old'}),
    ('quoted_denial','uses','SQLite','uses','SQLite',True,
     'Atlas uses SQLite.','Someone claimed "Atlas no longer uses SQLite". That claim is false; Atlas still uses it.',{'keep_both','reject_new'}),
    ('two_locations','lives_in','Paris','lives_in','Rome',False,
     'Atlas lives in Paris during winter.','Atlas lives in Rome during summer and Paris during winter.',{'keep_both'}),
    ('move','lives_in','Paris','lives_in','Rome',False,
     'Atlas lives in Paris.','Atlas moved to Rome permanently and no longer lives in Paris.',{'expire_old'}),
    ('multiple_properties','color','blue','color','red',False,
     'The Atlas logo includes blue.','The Atlas logo includes both blue and red.',{'keep_both'}),
    ('old_event','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite now.','Historical note: Atlas used PostgreSQL in 2020, before switching to SQLite.',{'keep_both','reject_new'}),
    ('conditional_denied','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite.','If the migration were approved Atlas would use PostgreSQL. Approval was denied.',{'keep_both','reject_new'}),
    ('style_correction','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite.','ok so atlas runs postgres now, sqlite got removed. by postgres i mean PostgreSQL.',{'expire_old'}),
    ('reported_suggestion','uses','SQLite','uses','PostgreSQL',False,
     'Atlas uses SQLite.','The assistant suggested PostgreSQL. I rejected that suggestion; we still use SQLite.',{'keep_both','reject_new'}),
]

def database_decision(ctx):
    import os
    import uuid
    from datetime import datetime
    from sqlalchemy.engine import make_url
    from src.api.config import settings
    from src.api.db import SessionLocal
    from src.memory.models import CodexClaim, CodexClaimLink, Conversation, EpisodicMemory
    from src.memory.source import chat_provenance, digest, source_units
    from src.workers.codex_extractor import handle_triplet
    assert (os.environ.get('ICE_TEST_DATABASE','').startswith('ice_test_') and
            make_url(settings.database_url).database==os.environ['ICE_TEST_DATABASE'])
    with SessionLocal() as db:
        cid=uuid.uuid4();db.add(Conversation(id=cid));db.flush()
        def source(evidence):
            body=evidence['text'];raw=f'User: {body}\n\nAssistant: '
            row=EpisodicMemory(conversation_id=cid,batch_id=uuid.uuid4(),raw_text=raw,
                source_spans=chat_provenance(body,''),ts_provenance='original',
                timestamp=datetime.fromisoformat(evidence['recorded_at']),
                context_reliance='Long_Term_Memory',idempotency_key=str(uuid.uuid4()))
            db.add(row);db.flush();unit=next(u for u in source_units(row) if u.role=='user')
            claim=CodexClaim(source_batch=row.batch_id,episodic_id=row.id,conversation_id=cid,
                raw_sha256=digest(raw),start=unit.start,end=unit.end,role='user',
                sentence=body,sentence_sha256=digest(body),text=body,verification={})
            db.add(claim);db.flush();return row,claim
        oldrow,oldclaim=source(ctx['old_source'])
        old=handle_triplet(db,ctx['subject'],ctx['old_relation'],ctx['old_object'],oldrow.batch_id)
        db.add(CodexClaimLink(claim_id=oldclaim.id,edge_id=old.id));db.flush()
        newrow,newclaim=source(ctx['new_source'])
        new=handle_triplet(db,ctx['subject'],ctx['relation'],ctx['object'],newrow.batch_id,
            negated=ctx['negated'],source_claims=[newclaim],reconciler=make_llm_reconciler())
        outcome='expire_old' if old.valid_until is not None else 'reject_new' if new is None else 'keep_both'
        db.rollback()
        return outcome


def main():
    reconcile=make_llm_reconciler()
    rows=[]
    for name,oldrel,oldobj,rel,obj,neg,old,new,expected in CASES:
        ctx=dict(subject='Atlas',
            old_relation=oldrel,old_object=oldobj,relation=rel,object=obj,
            old_negated=False,negated=neg,turn=new,
            old_source=dict(role='user',recorded_at='2026-09-01T12:00:00Z',text=old),
            new_source=dict(role='user',recorded_at='2026-09-02T12:00:00Z',text=new))
        try:
            decision=database_decision(ctx) if '--database' in sys.argv else reconcile(ctx)
            row=dict(name=name,context=ctx,expected=sorted(expected),decision=decision,
                     passed=decision in expected)
        except Exception as exc:
            row=dict(name=name,error_type=type(exc).__name__,passed=False)
        rows.append(row)
        print(name,row.get('decision',row.get('error_type')),row['passed'],flush=True)
    output=Path('experiments/v3_repair/results/reconciliation_'+('database' if '--database' in sys.argv else 'two_sources')+'.json')
    output.write_text(json.dumps(dict(version='v3',model=get_bg_model_name(),
        limitation='Synthetic qualification only; source eligibility tested separately in SQL.',
        passed=sum(r['passed'] for r in rows),total=len(rows),rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
