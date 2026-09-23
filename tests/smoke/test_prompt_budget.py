"""v3 real assembled blocks, not hypothetical ledger-only accounting."""
from types import SimpleNamespace

from src.api.prompt_assembler import assemble_prompt
from src.api.prompt_budget import assemble_budgeted_prompt
from src.memory.tokens import count_messages
from src.retrieval.orchestrator import ContextFragment


def inputs():
    return dict(memory_slots=[SimpleNamespace(is_active=True,content='Preference '*50,
        scope_tier='global',slot_name='preferences')],
        retrieved_fragments=[ContextFragment('The evidence says port 8391.','episodic',1.0,8)],
        user_message='Which port?',session_start_text='Project status '*250,
        conversation_summary_text='Conversation overview '*50,
        bookmarked_texts=['Pinned statement '*100],constraints_text='Do not change the database.')


def test_assembler_reports_exact_optional_blocks_and_envelopes():
    costs={}
    messages=assemble_prompt(**inputs(),block_tokens=costs)
    assert sum(costs.values())==count_messages(messages)
    for name in ('slots','session_start','bookmarks','conversation_summary','constraints','evidence','user_message'):
        assert costs[name]>0
    # The generated acknowledgement belongs to evidence, not recent history.
    assert costs['recent_turns']==0


def test_static_eviction_preserves_evidence_and_constraints():
    args=inputs();costs={};initial=assemble_prompt(**args,block_tokens=costs)
    budget=sum(costs.values())-costs['session_start']-costs['slots']+10
    result=assemble_budgeted_prompt(**args,serving_window=budget+100,generation_reserve=100)
    assert result.removed==['session_start','slots']
    assert result.ledger.fits()
    assert [entry['block'] for entry in result.ledger.evictions] == result.removed
    assert result.ledger.total()==count_messages(result.messages)
    rendered=' '.join(m['content'] for m in result.messages)
    assert '8391' in rendered and 'Do not change the database.' in rendered
    assert 'Project status' not in rendered and 'Preference' not in rendered
    assert result.item_counts==dict(slots=0,bookmarks=1)


def test_required_overflow_is_explicit_and_never_truncates_question():
    args=inputs();args['user_message']='Long question '*400
    result=assemble_budgeted_prompt(**args,serving_window=250,generation_reserve=100)
    assert not result.ledger.fits() and result.ledger.pressure=='extreme'
    assert result.messages[-1]['content']==args['user_message']
    assert 'Do not change the database.' in result.messages[0]['content']
    assert result.item_counts==dict(slots=0,bookmarks=0)


def test_unknown_window_does_not_invent_a_capacity():
    args=inputs()
    result=assemble_budgeted_prompt(**args,serving_window=0,generation_reserve=100)
    assert not result.removed and result.ledger.serving_window==0
    assert result.ledger.total()==count_messages(result.messages)


def test_margin_applies_to_remaining_tokens_when_planning_drops():
    from src.api.context_ledger import ContextLedger
    ledger=ContextLedger(serving_window=1200,generation_reserve=100,safety_margin=2)
    ledger.add('system_prompt',100);ledger.add('slots',100);ledger.add('bookmarks',100);ledger.add('evidence',400)
    assert ledger.evict_plan()==['slots','bookmarks']


def test_answer_reserve_is_not_silently_halved_for_small_windows():
    result = assemble_budgeted_prompt(**inputs(), serving_window=256,
                                     generation_reserve=256)
    assert result.ledger.available() == 0
    assert not result.ledger.fits()


def test_tight_window_keeps_best_complete_note_instead_of_dropping_block():
    best = '[Original-source segment 2; supported evidence] Current port 9010.'
    older = '[Original-source segment 1; supported evidence] Old port 8391.'
    whole = older + '\n\n' + best
    args = dict(memory_slots=[], retrieved_fragments=[], user_message='Which port?',
                conversation_summary_text=whole)
    one = assemble_prompt(**{**args, 'conversation_summary_text': best})
    result = assemble_budgeted_prompt(**args,
        conversation_summary_options=[whole, best],
        serving_window=count_messages(one) + 100, generation_reserve=100)
    rendered = ' '.join(message['content'] for message in result.messages)
    assert result.ledger.fits()
    assert 'Current port 9010.' in rendered and 'Old port 8391.' not in rendered
    assert 'conversation_summary' not in result.removed
    assert any(item['block'] == 'conversation_summary_part'
               for item in result.ledger.evictions)
