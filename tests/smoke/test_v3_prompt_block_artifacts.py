"""v3 final prompt receipts bind only selected native stored rows and spans."""
import hashlib
import json
import uuid
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
from transformers import AutoTokenizer

from src.api import memory_preparation as preparation, prompt_assembler as assembler, prompt_budget as budget
from src.classifier.schemas import ClassificationResult
from src.memory import tokens
from scripts.z1.historical_clock import source_datetime

import test_v3_retrieved_artifact_ids as helpers

store = helpers.store

@pytest.fixture(autouse=True)
def clock_and_tokenizer(monkeypatch):
    monkeypatch.setattr(tokens,'_tokenizer',AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-Embedding-0.6B',local_files_only=True))
    monkeypatch.setattr(tokens,'_load_attempted',True)
    monkeypatch.setattr(assembler,'datetime',source_datetime(helpers.NOW))

def hash_text(value):return hashlib.sha256(value.encode()).hexdigest()

def assert_final_receipts(value):
    for name,block in value.artifact_blocks.items():
        if name=='recent_turns':
            for item in block['items']:
                selected=[value.messages[index] for index in item['message_indices']]
                assert len(selected)==item['message_count']
                assert hash_text(json.dumps(selected,sort_keys=True,ensure_ascii=False))==item['rendered_messages_sha256']
        else:
            message=value.messages[block['message_index']]['content']
            assert hash_text(message[block['char_start']:block['char_end']])==block['rendered_text_sha256']

@pytest.mark.parametrize('allowance',[0,40,2000])
def test_recent_metadata_tracks_only_fitted_pairs_in_final_order(store,allowance):
    db=store
    row=helpers.source(db,prompt='First answer is copper. '+ 'Unrelated prose. '*150)
    newer=helpers.source(db,prompt='Second answer is silver.',cid=row.conversation_id)
    newer.timestamp=row.timestamp+timedelta(minutes=1);db.flush()
    expected=assembler.get_recent_turns(db,str(row.conversation_id),max_tokens=allowance)
    receipts=[]
    actual=assembler.get_recent_turns(db,str(row.conversation_id),max_tokens=allowance,artifact_receipts=receipts)
    assert expected==actual
    offset=0
    for item in receipts:
        pair=actual[offset:offset+item['message_count']];offset+=item['message_count']
        assert hash_text(json.dumps(pair,sort_keys=True,ensure_ascii=False))==item['rendered_messages_sha256']
    assert offset==len(actual)
    assert [key for item in receipts for table,key in item['stored_artifact_ids']] == (
        [str(row.id),str(newer.id)] if allowance==2000 else [str(newer.id)] if allowance==40 else [])

def test_slots_respect_exact_scope_and_same_text_different_rows():
    pid,cid=uuid.uuid4(),uuid.uuid4()
    slots=[NS(id=uuid.uuid4(),slot_name='preferences',content='Same content.',is_active=True,
        scope_tier=tier,project_id=project,conversation_id=conv) for tier,project,conv in [
            ('global',None,None),('global',None,None),('project',pid,None),
            ('project',uuid.uuid4(),None),('conversation',None,cid),('conversation',None,uuid.uuid4())]]
    arguments=dict(memory_slots=slots,retrieved_fragments=[],user_message='Current question.',
        conversation_id=str(cid),scope={'project_id':str(pid)})
    expected=assembler.assemble_prompt(**arguments)
    meta={};actual=assembler.assemble_prompt(**arguments,block_artifacts=meta)
    assert expected==actual
    assert [item['stored_artifact_ids'] for item in meta['slots']['items']]==[
        (('memory_slots',str(slots[i].id)),) for i in (0,1,2,4)]
    assert meta['slots']['items'][0]['rendered_text_sha256']==meta['slots']['items'][1]['rendered_text_sha256']

def test_bookmark_ids_are_parallel_native_rows_not_text_guesses(store):
    db=store
    first=helpers.source(db);second=helpers.source(db,cid=first.conversation_id)
    first.is_bookmarked=second.is_bookmarked=True;db.flush()
    meta=[]
    expected=assembler.bookmarked_turn_texts(db,str(first.conversation_id))
    actual=assembler.bookmarked_turn_texts(db,str(first.conversation_id),artifact_receipts=meta)
    assert expected==actual and len(meta)==2
    assert {item['stored_artifact_ids'] for item in meta}=={(('episodic_memory',str(row.id)),) for row in (first,second)}
    for item,text in zip(meta,actual):assert item['rendered_text_sha256']==hash_text(text)
    value=budget.assemble_budgeted_prompt(serving_window=8192,generation_reserve=300,
        memory_slots=[],retrieved_fragments=[],user_message='Current question.',
        bookmarked_texts=actual,bookmarked_artifacts=meta)
    assert value.artifact_blocks['bookmarks']['items']==meta
    assert_final_receipts(value)

@pytest.mark.parametrize('span',['missing','indexed'])
def test_conversation_fold_ids_follow_native_options_or_compatibility_root(store,span):
    db=store
    # Native fixture snapshot construction, not a new model-generated summary.
    row,root=helpers.notes(db,'conversation',span)
    arguments=(db,str(row.conversation_id),1,1000,0)
    expected=assembler.conversation_summary_block(*arguments,include_options=True,include_source_ids=True)
    identities={}
    actual=assembler.conversation_summary_block(*arguments,include_options=True,include_source_ids=True,
        artifact_options=identities)
    assert expected==actual and actual
    ref=('conversation_summaries',str(row.conversation_id)) if span=='missing' else (
        'conversation_notes',json.dumps([str(row.conversation_id),1],separators=(',',':')))
    assert identities=={text:[ref] for text,_ in actual}

def test_budget_metadata_is_only_last_iteration_after_replacement_and_eviction():
    texts=['A long whole note. '*100,'Short whole note.']
    slot=NS(id=uuid.uuid4(),slot_name='preferences',content='Large slot. '*200,is_active=True,scope_tier='global')
    arguments=dict(memory_slots=[slot],retrieved_fragments=[],user_message='Current question.',
        bookmarked_texts=['Pinned text. '*100],conversation_summary_text=texts[0],
        conversation_summary_options=texts)
    options={text:[('conversation_notes',json.dumps(['fixture',i]))] for i,text in enumerate(texts,1)}
    for window in (8192,700,500):
        actual=budget.assemble_budgeted_prompt(serving_window=window,generation_reserve=200,
            **arguments,conversation_summary_artifact_options=options)
        assert actual.messages[-1] == dict(role='user',content='Current question.')
        assert actual.ledger.fits()
        assert_final_receipts(actual)
        assert all(name not in actual.artifact_blocks for name in actual.removed)
        if 'conversation_summary' in actual.artifact_blocks:
            selected=texts[0] if texts[0] in actual.messages[0]['content'] else texts[1]
            assert actual.artifact_blocks['conversation_summary']['items']==[dict(stored_artifact_ids=tuple(options[selected]))]

def test_shared_preparation_preserves_prompt_and_forwards_selected_bookmarks(store):
    db=store
    row=helpers.source(db);row.is_bookmarked=True;db.flush()
    metadata=[];bookmarks=assembler.bookmarked_turn_texts(db,str(row.conversation_id),artifact_receipts=metadata)
    arguments=dict(db=db,classifier=NS(embedder=helpers.Encoder()),conversation_id=row.conversation_id,
        user_message='What did I say?',scope={'conversation_id':str(row.conversation_id)},turn_count=1,
        total_tokens=100,total_budget=2048,serving_window=8192,base_retrieve=False,
        memory_slots=[],bookmarked_texts=bookmarks)
    c=ClassificationResult([],[],'Zero_Shot',[0.]*27,1.,prompt=arguments['user_message'],p_ltm=0.)
    value=preparation.prepare_memory_context(**arguments,classification=c,bookmarked_artifacts=metadata)
    assert value.retrieve is False and not value.fragments
    assert value.prepared.ledger.fits()
    assert value.prepared.artifact_blocks['bookmarks']['items']==metadata
    assert value.prepared.artifact_blocks['recent_turns']['items'][0]['stored_artifact_ids']==(('episodic_memory',str(row.id)),)
    assert_final_receipts(value.prepared)

def test_actual_multi_note_options_replace_or_evict_with_exact_ids(store):
    from src.memory.models import ConversationNote,ConversationSummary
    from src.memory.summary_snapshot import source_snapshot,compose_parts,bind_snapshot
    db=store
    first=helpers.source(db,prompt='Copper account. '+ 'Historical context. '*60)
    second=helpers.source(db,prompt='Silver account. '+ 'More historical context. '*60,cid=first.conversation_id)
    rows={str(row.id):row for row in (first,second)}
    snapshot=source_snapshot(db,first.conversation_id)
    parts=[dict(source_ids=[item['id']],mode='source',text=rows[item['id']].raw_text,
                recorded_range='unknown') for item in snapshot]
    body=compose_parts(parts)
    root=ConversationSummary(conversation_id=first.conversation_id,summary_text=body,
        source_manifest=bind_snapshot(snapshot,body,parts=parts),covers_turns=2,embedding=helpers.VEC)
    db.add(root);db.flush()
    for i,part in enumerate(parts,1):
        row=rows[part['source_ids'][0]]
        db.add(ConversationNote(conversation_id=first.conversation_id,ordinal=i,text=part['text'],
            mode='source',recorded_range='unknown',source_ids=part['source_ids'],
            batch_ids=[str(row.batch_id)],embedding=helpers.VEC))
    db.flush()
    identities={}
    options=assembler.conversation_summary_block(db,str(first.conversation_id),2,10000,0,
        max_tokens=2000,include_options=True,artifact_options=identities)
    assert len(options)==2
    kwargs=dict(memory_slots=[],retrieved_fragments=[],user_message='Current question.',
                conversation_summary_text=options[0],conversation_summary_options=options)
    single=budget.assemble_budgeted_prompt(serving_window=10000,generation_reserve=200,
        **{**kwargs,'conversation_summary_text':options[1]})
    for window in (10000,single.ledger.total()+200,400):
        after=budget.assemble_budgeted_prompt(serving_window=window,generation_reserve=200,
            **kwargs,conversation_summary_artifact_options=identities)
        if window != 400:
            assert after.ledger.fits()
        assert_final_receipts(after)
        if window==single.ledger.total()+200:
            assert options[1] in after.messages[0]['content'] and options[0] not in after.messages[0]['content']
            assert after.artifact_blocks['conversation_summary']['items']==[dict(stored_artifact_ids=tuple(identities[options[1]]))]
        if window==400:assert 'conversation_summary' not in after.artifact_blocks

def test_shared_retrieving_preparation_keeps_native_messages_and_visible_fragment_ids(store,monkeypatch):
    db=store
    row=helpers.source(db);row.is_bookmarked=True;db.flush()
    monkeypatch.setattr(helpers.candidate,'extract_entities',lambda *_a,**_k:[])
    receipts=[]
    bookmarks=assembler.bookmarked_turn_texts(db,str(row.conversation_id),artifact_receipts=receipts)
    arguments=dict(db=db,classifier=NS(embedder=helpers.Encoder()),conversation_id=row.conversation_id,
        user_message='What did I say?',scope={'conversation_id':str(row.conversation_id)},turn_count=1,
        total_tokens=100,total_budget=2048,serving_window=8192,base_retrieve=True,
        memory_slots=[],bookmarked_texts=bookmarks)
    c=ClassificationResult([],['Factual_Retrieval'],'Long_Term_Memory',[0.]*27,1.,
        prompt=arguments['user_message'],p_ltm=1.,head_confidences={'topic':1.,'intent':1.})
    after=preparation.prepare_memory_context(**arguments,classification=c,bookmarked_artifacts=receipts)
    assert after.retrieve and after.fragments and after.prepared.ledger.fits()
    assert after.prepared.artifact_blocks['bookmarks']['items']==receipts
    assert_final_receipts(after.prepared)
