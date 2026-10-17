"""Track T (T1–T4) behavioral test: date-grounding, TimeScope detection,
time-scoped retrieval, timeline builder / evolution surfacing, cold-storage
second chance, decay freeze fix.

Covers specs/T_temporal.md §5 checks 1–30 (20–24 are the T4 timeline/diff/
journaling slice).

Runs against the live Postgres (docker up). Inserts its own uniquely-marked
rows and deletes them afterwards — never truncates (the dev DB holds real
data). No LLM needed (the track is fully deterministic); the embedder is a
stub returning a constant vector, which makes our marker rows perfect-match
top hits.

Run: uv run python tests/support/disposable_database.py tests/test_timescope.py
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.config import settings

# Temporal mechanics are isolated from semantic ranking, qualified separately
# by tests/test_reranker_quality.py through both production retrieval paths.
settings.retrieval_rerank_enabled = False
# This suite invokes whole-store decay. Never run it on the working store.
from sqlalchemy.engine import make_url
if (not os.environ.get("ICE_TEST_DATABASE")
        or make_url(settings.database_url).database != os.environ["ICE_TEST_DATABASE"]
        or not os.environ["ICE_TEST_DATABASE"].startswith("ice_test_")):
    raise SystemExit("Use: uv run python tests/support/disposable_database.py tests/test_timescope.py")

from src.api.db import SessionLocal
from src.api.memory_decision import decide_memory_retrieval
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    CodexEntity,
    CodexFact,
    CodexFactEntity,
    CodexFactSource,
    ColdStorage,
    Conversation,
    EpisodicMemory,
)
from src.retrieval.evolution import entity_diff, log_description_update
from src.services import graph as graph_svc
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import (
    CURRENT,
    TimeScope,
    detect_timescope,
    from_scope,
    to_scope_dict,
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)  # detector anchor (spec dates)
EMB = [0.05] * 1024
MARK = f"zephyrglass{uuid.uuid4().hex[:6]}"   # unique lexical marker per run
D = timedelta(days=1)


class StubEmbedder:
    def encode(self, texts, convert_to_tensor=False, **kw):
        single = isinstance(texts, str)
        if convert_to_tensor:  # the NER path wants a tensor with .device
            import torch
            return torch.tensor(EMB if single else [EMB for _ in texts])
        return list(EMB) if single else [list(EMB) for _ in texts]


def clf(prompt="", topics=None, intents=None, p_ltm=0.0):
    return ClassificationResult(
        topic_tags=topics or ["Software_&_Tech"],
        intent_tags=intents or ["Factual_Retrieval"],
        context_reliance="Long_Term_Memory",
        raw_probs=[0.0] * 25,
        max_confidence=1.0,
        prompt=prompt,
        p_ltm=p_ltm,
    )


# ═══════════════════════════════════════════════════════════════════════════
print("── Detector (pure; checks 1–11) ──")

ts = detect_timescope("what did I think about ice in march 2025?", now=NOW)
check("1  'in march 2025' → as_of", ts.mode == "as_of")
check("1  window ≈ [2025-02-15, 2025-04-15)",
      ts.t0.date() == datetime(2025, 2, 15).date()
      and ts.t1.date() == datetime(2025, 4, 15).date())

ts = detect_timescope("what was I working on two years ago?", now=NOW)
center = ts.t0 + (ts.t1 - ts.t0) / 2
half_days = (ts.t1 - ts.t0).total_seconds() / 86400.0 / 2
check("2  'two years ago' → as_of", ts.mode == "as_of")
check("2  centered 2024-07-10, halfwidth ≈ 110d",
      abs((center - datetime(2024, 7, 10, tzinfo=timezone.utc)).days) <= 1
      and 100 <= half_days <= 120)

ts = detect_timescope("my ideas from the first half of 2025", now=NOW, p_ltm=0.8)
check("3  'first half of 2025' → range [~2024-12-11, ~2025-07-22)",
      ts.mode == "range"
      and ts.t0.date() == datetime(2024, 12, 11).date()
      and ts.t1.date() == datetime(2025, 7, 22).date())

ts = detect_timescope("what have I written since january?", now=NOW)
check("4  'since january' → range [jan−pad, now)",
      ts.mode == "range"
      and ts.t0.date() == datetime(2025, 12, 18).date()
      and ts.t1 == NOW)

ts = detect_timescope("how did the saga 2 ending evolve?", now=NOW)
check("5  evolution cue, no window", ts.mode == "evolution" and ts.t0 is None and ts.t1 is None)

ts = detect_timescope("how did my thinking change during 2025?", now=NOW)
check("6  evolution WITH window", ts.mode == "evolution" and ts.t0 is not None and ts.t1 is not None)

ts = detect_timescope("I was in Paris last summer.", now=NOW, p_ltm=0.1)
ts2 = detect_timescope("I was in Paris last summer. What did I write about it?", now=NOW, p_ltm=0.1)
check("7  statement → current; +recall shape → fires",
      ts.mode == "current" and ts2.mode != "current")

ts = detect_timescope("remind me next month", now=NOW)
check("8  future guard → current", ts.mode == "current")

ts = detect_timescope("look at this: ```python\n# in 2024 we did X\n``` fix it?", now=NOW)
check("9  code guard → current", ts.mode == "current")

ts = detect_timescope("what happened in december?", now=NOW)
check("10 future-this-year month → last year's",
      ts.mode == "as_of" and ts.t0.year == 2025 and (ts.t0 + D * 20).month == 12)

settings.timescope_enabled = False
ts = detect_timescope("what did I think about ice in march 2025?", now=NOW)
settings.timescope_enabled = True
check("11 timescope_enabled=False → always current", ts.mode == "current")

sd = to_scope_dict(TimeScope(mode="as_of", t0=NOW - D * 30, t1=NOW - D * 20))
rt = from_scope({"timescope": sd})
check("—  to_scope_dict/from_scope round-trip",
      rt.mode == "as_of" and rt.t0 == NOW - D * 30 and rt.t1 == NOW - D * 20)
check("—  from_scope tolerant of junk",
      from_scope({"timescope": {"mode": "as_of"}}) == CURRENT
      and from_scope({"timescope": "garbage"}) == CURRENT
      and from_scope(None) == CURRENT)

# ═══════════════════════════════════════════════════════════════════════════
print("── Legs & scoring (live DB; checks 12–19) ──")

db = SessionLocal()
orch = HybridRetrievalOrchestrator(db, StubEmbedder())
real_now = datetime.now(timezone.utc)

conv = Conversation(memory_scope_type="auto")
conv2 = Conversation(memory_scope_type="auto")   # era-stratification sandbox (check 22)
db.add_all([conv, conv2])
db.commit()

created_episodic = []
created_cold = []
created_codex_entities = []


def mk_fact(sentence, turn, links, when, until=None, superseded_by=None):
    """A verified CODEX_V4 fact sourced from *turn* (the writer is tested elsewhere)."""
    f = CodexFact(text=sentence, text_sha256=uuid.uuid4().hex, speaker="user",
                  modality="asserted", embedding=EMB, mentioned_at=when, valid_from=when,
                  valid_until=until, superseded_by=superseded_by)
    db.add(f)
    db.flush()
    db.add(CodexFactSource(fact_id=f.id, source_batch=turn.batch_id,
                           window_sha256=uuid.uuid4().hex, episodic_id=turn.id,
                           conversation_id=turn.conversation_id, window={},
                           quote=sentence[:40], recorded_at=when, verification={}))
    for ent, role in links:
        db.add(CodexFactEntity(fact_id=f.id, entity_id=ent.id, role=role))
    db.flush()
    return f


def mk_turn(ts_, text_, decay=1.0, archived=False, summary=None, coverage=None,
            abstract=None, is_private=False, access=0, conversation=None):
    t = EpisodicMemory(
        conversation_id=(conversation or conv).id, batch_id=uuid.uuid4(), timestamp=ts_,
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_text=text_,
        summary_text=summary, summary_coverage=coverage, abstract_text=abstract,
        embedding=EMB, decay_score=decay, is_archived=archived,
        is_private=is_private, access_count=access, inject_raw=True,
        idempotency_key=f"test-t3-{uuid.uuid4()}",
    )
    if summary or abstract:
        from src.memory.source import single_provenance
        from src.memory.representation import verify_representations
        from src.memory.support import verify_support
        t.source_spans = single_provenance(t.raw_text, "user")
        t.representation_verification = verify_representations(t, summary, abstract,
            verifier=lambda source, claim: verify_support(source, claim,
                scorer=lambda pairs: [dict(entailment=.99, neutral=.005, contradiction=.005)]))
    db.add(t)
    db.commit()
    created_episodic.append(t.id)
    return t


try:
    # window for the leg tests: march 2025 ± ~2 weeks
    W0 = datetime(2025, 2, 15, tzinfo=timezone.utc)
    W1 = datetime(2025, 4, 15, tzinfo=timezone.utc)
    AS_OF = TimeScope(mode="as_of", t0=W0, t1=W1)
    RANGE = TimeScope(mode="range", t0=W0, t1=W1)

    t_old = mk_turn(datetime(2024, 6, 15, tzinfo=timezone.utc),
                    f"User: {MARK} report from june 2024\n\nAssistant: noted")
    t_in = mk_turn(datetime(2025, 3, 10, tzinfo=timezone.utc),
                   f"User: {MARK} report from march 2025\n\nAssistant: noted",
                   summary=f"{MARK} report from march 2025", coverage=0.9,
                   abstract=f"{MARK} report from march 2025")
    t_now = mk_turn(real_now - timedelta(hours=1),
                    f"User: {MARK} report from right now\n\nAssistant: noted")

    c = clf(prompt=f"what did the {MARK} report say?")

    # 12: window predicate on vector + BM25
    orch._active_timescope = AS_OF
    vec = orch._vector_episodic(EMB, c, None, None)
    bm = orch._bm25_episodic(c, None, None)
    def _mine(frags):
        ids = {str(t_old.id): "old", str(t_in.id): "in", str(t_now.id): "now"}
        return {ids[f.source_batch_id] for f in frags if f.source_batch_id in ids}
    check("12 vector: as_of retrieves ONLY the in-window row", _mine(vec) == {"in"})
    check("12 bm25:   as_of retrieves ONLY the in-window row", _mine(bm) == {"in"})

    # 13: v3 warm archival changes storage priority, not source reachability.
    t_arch = mk_turn(datetime(2025, 3, 20, tzinfo=timezone.utc),
                     f"User: {MARK} archived thought\n\nAssistant: ok",
                     decay=0.08, archived=True)
    orch._active_timescope = AS_OF
    vec_asof = orch._vector_episodic(EMB, c, None, None)
    orch._active_timescope = CURRENT
    vec_cur = orch._vector_episodic(EMB, c, None, None)
    has = lambda frags, turn: any(f.source_batch_id == str(turn.id) for f in frags)
    check("13 archived row retrieved under as_of", has(vec_asof, t_arch))
    check("13 archived row retrieved under current", has(vec_cur, t_arch))

    # 14: as_of recency re-anchors to window center; range is flat
    W0b = datetime(2025, 3, 1, tzinfo=timezone.utc)
    W1b = datetime(2025, 4, 1, tzinfo=timezone.utc)
    t_near = mk_turn(datetime(2025, 3, 15, tzinfo=timezone.utc),
                     f"User: {MARK} near center\n\nAssistant: ok")
    t_far = mk_turn(datetime(2025, 3, 2, tzinfo=timezone.utc),
                    f"User: {MARK} far from center\n\nAssistant: ok")
    orch._active_timescope = TimeScope(mode="as_of", t0=W0b, t1=W1b)
    frs = {f.source_batch_id: f.score for f in orch._vector_episodic(EMB, c, None, None)}
    check("14 as_of: nearer-to-center row outscores",
          frs.get(str(t_near.id), 0) > frs.get(str(t_far.id), 1e9))
    orch._active_timescope = TimeScope(mode="range", t0=W0b, t1=W1b)
    frs = {f.source_batch_id: f.score for f in orch._vector_episodic(EMB, c, None, None)}
    check("14 range: flat (equal scores)",
          abs(frs.get(str(t_near.id), 0) - frs.get(str(t_far.id), 1)) < 1e-9)

    # 15: date stamps under current mode; degrade/abstract carry the stamp
    orch._active_timescope = CURRENT
    frags = orch._vector_episodic(EMB, c, None, None)
    fin = next((f for f in frags if f.source_batch_id == str(t_in.id)), None)
    check("15 fragment text carries source date and time",
          fin is not None and fin.text.startswith("[source recorded: 2025-03-10T00:00:00Z] "))
    check("15 degrade_text carries the same stamp",
          fin is not None and fin.degrade_text is not None
          and fin.degrade_text.startswith("[source recorded: 2025-03-10T00:00:00Z] ")
          and fin.abstract_text.startswith("[source recorded: 2025-03-10T00:00:00Z] "))

    # 16/17: codex valid_at + dated fact line
    eA = CodexEntity(canonical_name=f"{MARK}-alpha", aliases=[], tags=["tool"],
                     context_payload=f"{MARK}-alpha payload")
    eB = CodexEntity(canonical_name=f"{MARK}-beta", aliases=[], tags=["tool"],
                     context_payload=f"{MARK}-beta payload")
    eC = CodexEntity(canonical_name=f"{MARK}-gamma", aliases=[], tags=["tool"],
                     context_payload=f"{MARK}-gamma payload")
    db.add_all([eA, eB, eC])
    db.commit()
    created_codex_entities += [eA.id, eB.id, eC.id]
    old_source = mk_turn(real_now - D * 100,
                         f"User: {MARK}-alpha used {MARK}-beta.\n\nAssistant: noted")
    new_source = mk_turn(real_now - D * 2,
                         f"User: {MARK}-alpha now uses {MARK}-gamma.\n\nAssistant: noted")
    # A fact replaced yesterday (stated ~100 days ago) and its successor,
    # stated 2 days ago — after the as-of window below.
    f_new = mk_fact(f"The user said {MARK}-alpha now uses {MARK}-gamma.", new_source,
                    [(eA, "subject"), (eC, "object")], real_now - D * 2)
    f_old = mk_fact(f"The user said {MARK}-alpha used {MARK}-beta.", old_source,
                    [(eA, "subject"), (eB, "object")], real_now - D * 100,
                    until=real_now - D * 1, superseded_by=f_new.id)
    db.commit()

    def fact_ids(tscope):
        orch._active_timescope = tscope
        return {i for f in orch._facts_leg(f"{MARK}-alpha uses", EMB, None)
                for i in f.origin_fact_ids}

    cur_ids = fact_ids(CURRENT)
    week_ago = TimeScope(mode="as_of", t0=real_now - D * 14, t1=real_now - D * 7)
    asof_ids = fact_ids(week_ago)
    check("16 replaced fact absent under current, present under as_of(last week)",
          str(f_old.id) not in cur_ids and str(f_old.id) in asof_ids)
    check("16 fact stated after the window absent under as_of, present under current",
          str(f_new.id) not in asof_ids and str(f_new.id) in cur_ids)

    orch._active_timescope = CURRENT
    rendered = orch._render_fact(f_new, current=True, earlier=f_old)
    since = (real_now - D * 2).astimezone(timezone.utc).strftime("%Y-%m-%d")
    check("17 fact line carries its source date and the replaced predecessor",
          rendered.startswith(f"[Fact; user, asserted; {since}; current]")
          and f"(earlier: The user said {MARK}-alpha used {MARK}-beta." in rendered)

    # 18: memory decision bump
    md_base = decide_memory_retrieval(clf(p_ltm=0.1), 1, 100.0, settings)
    md_ts = decide_memory_retrieval(clf(p_ltm=0.1), 1, 100.0, settings,
                                    timescope_mode="as_of")
    check("18 p_ltm=0.1 alone → no retrieval; +timescope → retrieve",
          md_base.retrieve is False and md_ts.retrieve is True)
    check("18 breakdown carries timescope", md_ts.breakdown.get("timescope") == "as_of")

    # 19: wide net honors explicit windows; archival does not hide source evidence.
    scope_ts = {"timescope": to_scope_dict(AS_OF)}
    wide = orch._wide_net_fallback(c, EMB, None, scope_ts)
    check("19 wide net: in-window + archived rows in, out-of-window out",
          has(wide, t_in) and has(wide, t_arch)
          and not has(wide, t_old) and not has(wide, t_now))
    # Current retrieval has more recent peers in the same conversation. The
    # ordinary diversification cap may omit this low-ranked source from the
    # final list; that is not a SQL visibility failure. Observe candidacy.
    wide_candidates = []
    original_rows_to_fragments = orch._rows_to_fragments
    def capture_rows(*args, **kwargs):
        fragments = original_rows_to_fragments(*args, **kwargs)
        wide_candidates.extend(fragments)
        return fragments
    orch._rows_to_fragments = capture_rows
    try:
        orch._wide_net_fallback(c, EMB, None, {})
    finally:
        orch._rows_to_fragments = original_rows_to_fragments
    check("19 wide net current: archived source enters candidates",
          has(wide_candidates, t_arch))

    # ablation seam / kill switch on the orchestrator gate
    orch.timescope_allowed = False
    check("—  ablation seam: timescope=False forces CURRENT",
          orch._resolve_timescope(scope_ts) == CURRENT)
    orch.timescope_allowed = True
    settings.timescope_enabled = False
    check("—  kill switch forces CURRENT on the read path",
          orch._resolve_timescope(scope_ts) == CURRENT)
    settings.timescope_enabled = True

    # ═══════════════════════════════════════════════════════════════════════
    print("── Timeline / evolution (checks 20–24) ──")

    # Fixture (spec check 20, CODEX_V4): entity "saga" with a replacement
    # chain — fact A (replaced by B, D6) → B (current) — plus "lone", whose only
    # fact was /forget-expired (not replaced), and "deleted", whose only fact
    # lost its source to deletion (C10: not history either).
    TZ = timezone.utc
    e_saga = CodexEntity(canonical_name=f"{MARK}-saga", tags=["idea"],
                         context_payload=f"{MARK}-saga payload")
    e_mv = CodexEntity(canonical_name=f"{MARK}-multiverse", tags=["idea"],
                       context_payload=f"{MARK}-multiverse payload")
    e_mercy = CodexEntity(canonical_name=f"{MARK}-mercy", tags=["idea"],
                          context_payload=f"{MARK}-mercy payload")
    e_lone = CodexEntity(canonical_name=f"{MARK}-lone", tags=["idea"],
                         context_payload=f"{MARK}-lone payload")
    e_del = CodexEntity(canonical_name=f"{MARK}-deleted", tags=["idea"],
                        context_payload=f"{MARK}-deleted payload")
    db.add_all([e_saga, e_mv, e_mercy, e_lone, e_del])
    db.commit()
    created_codex_entities += [e_saga.id, e_mv.id, e_mercy.id, e_lone.id, e_del.id]
    saga_id = e_saga.id   # plain-value capture

    T_A0 = datetime(2025, 11, 5, tzinfo=TZ)
    T_A1 = datetime(2026, 2, 10, tzinfo=TZ)
    original_old = mk_turn(T_A0,
        f"User: I planned the {MARK}-saga as {MARK}-multiverse.\n\nAssistant: noted")
    original_new = mk_turn(T_A1,
        f"User: I used to plan the {MARK}-saga as {MARK}-multiverse, "
        f"but now plan it as {MARK}-mercy.\n\nAssistant: noted")
    fact_B = mk_fact(f"The user said they now plan the {MARK}-saga as {MARK}-mercy.",
                     original_new, [(e_saga, "subject"), (e_mercy, "object")], T_A1)
    fact_A = mk_fact(f"The user said they planned the {MARK}-saga as {MARK}-multiverse.",
                     original_old, [(e_saga, "subject"), (e_mv, "object")], T_A0,
                     until=T_A1, superseded_by=fact_B.id)
    lone_turn = mk_turn(datetime(2025, 1, 1, tzinfo=TZ), f"User: {MARK}-lone note\n\nAssistant: ok")
    mk_fact(f"The user said {MARK}-lone was a draft.", lone_turn, [(e_lone, "subject")],
            datetime(2025, 1, 1, tzinfo=TZ), until=datetime(2025, 6, 1, tzinfo=TZ))
    del_turn = mk_turn(datetime(2025, 1, 1, tzinfo=TZ), f"User: {MARK}-deleted note\n\nAssistant: ok")
    fact_E = mk_fact(f"The user said {MARK}-deleted was a draft.", del_turn, [(e_del, "subject")],
                     datetime(2025, 1, 1, tzinfo=TZ), until=datetime(2025, 6, 1, tzinfo=TZ))
    db.query(CodexFactSource).filter_by(fact_id=fact_E.id).delete()
    db.commit()

    # 20: timeline = the replaced facts, each naming its successor
    tl = graph_svc.entity_timeline(db, str(saga_id))
    check("20 timeline: replaced A names its successor B",
          tl["has_history"] and [f["text"] for f in tl["timeline"]] == [fact_A.text]
          and tl["timeline"][0]["superseded_by"] == str(fact_B.id))
    check("20 history: saga True; forget-expired False; source-deleted False",
          tl["has_history"]
          and not graph_svc.entity_timeline(db, str(e_lone.id))["has_history"]
          and not graph_svc.entity_timeline(db, str(e_del.id))["has_history"])
    check("20 no-history entity yields no timeline",
          graph_svc.entity_timeline(db, str(e_lone.id))["timeline"] is None)
    # (21 — the edge timeline LEG under current mode — was deleted with
    #  CODEX_V4: a current fact carries its replaced predecessor inline, which
    #  check 17 pins.)

    # 22: era-stratified episodic sampling under evolution mode (own
    # conversation so the candidate set is exactly the fixture)
    era_starts = [datetime(2023, 3, 1, tzinfo=TZ), datetime(2024, 3, 1, tzinfo=TZ),
                  datetime(2025, 3, 1, tzinfo=TZ), real_now - D * 10]
    for i, base_t in enumerate(era_starts, 1):
        for j in range(5):
            mk_turn(base_t + timedelta(days=j),
                    f"User: {MARK} era{i} note {j}\n\nAssistant: ok",
                    conversation=conv2)
    orch._active_timescope = TimeScope(mode="evolution")
    ev_frags = orch._vector_episodic(EMB, c, None, str(conv2.id))
    orch._active_timescope = CURRENT
    joined = " ".join(f.text for f in ev_frags)
    check("22 evolution: every era represented in stratified output",
          all(f"{MARK} era{k}" in joined for k in (1, 2, 3, 4)))
    check("22 stratifier selects (≤ buckets × per_era of 20 candidates)",
          len(ev_frags) <= settings.evolution_era_buckets * settings.evolution_per_era)

    # 23: entity_diff over a window bracketing the supersession
    dd = entity_diff(db, e_saga,
                     datetime(2026, 1, 1, tzinfo=TZ), datetime(2026, 3, 1, tzinfo=TZ))
    check("23 diff: A in expired, naming what replaced it",
          any(x["fact"] == fact_A.text and x["replaced_by"] == fact_B.text
              for x in dd["expired"]))
    check("23 diff: B in added; A (stated before the window) not added",
          any(x["fact"] == fact_B.text for x in dd["added"])
          and all(x["fact"] != fact_A.text for x in dd["added"]))
    check("23 diff: a source-deleted fact appears nowhere",
          all(f"{MARK}-deleted" not in x["fact"] for x in
              entity_diff(db, e_del, datetime(2000, 1, 1, tzinfo=TZ),
                          datetime(2100, 1, 1, tzinfo=TZ))["expired"]))

    # 24: D13 journaling helper (the write sites call exactly this)
    e_saga_fresh = db.get(CodexEntity, saga_id)
    log_description_update(db, e_saga_fresh, e_saga_fresh.description,
                           f"{MARK} new enriched note body", "reflection_enrichment")
    db.commit()
    ev_row = db.execute(text(
        "SELECT payload FROM codex_events "
        "WHERE event_type = 'description_updated' AND entity_id = :e"),
        {"e": str(saga_id)}).fetchone()
    check("24 description_updated event carries old/new snippets + source",
          ev_row is not None
          and ev_row.payload["new"].startswith(f"{MARK} new enriched")
          and ev_row.payload["source"] == "reflection_enrichment"
          and "old" in ev_row.payload)

    # ═══════════════════════════════════════════════════════════════════════
    print("── Cold & decay (checks 25–29) ──")

    # 25: migration columns present
    cols = {r.column_name for r in db.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'cold_storage'")).fetchall()}
    check("25 cold_storage has conversation_id/is_private/batch_id",
          {"conversation_id", "is_private", "batch_id"} <= cols)

    # 26: v3 cold leg surfaces queried originals with or without a date.
    cold_row = ColdStorage(
        id=uuid.uuid4(), raw_text=f"User: {MARK} frozen idea\n\nAssistant: ok",
        summary_text=f"{MARK} frozen summary", topic_tags=["Software_&_Tech"],
        embedding=EMB,
        timestamp=datetime(2025, 3, 12, tzinfo=timezone.utc),
        conversation_id=conv.id, is_private=False, batch_id=uuid.uuid4(),
    )
    legacy_cold = ColdStorage(
        id=uuid.uuid4(), raw_text=f"User: {MARK} legacy frozen\n\nAssistant: ok",
        summary_text=None, topic_tags=[],
        timestamp=datetime(2025, 3, 13, tzinfo=timezone.utc),
        conversation_id=None, is_private=False, batch_id=None,
    )
    db.add_all([cold_row, legacy_cold])
    db.commit()
    created_cold += [cold_row.id, legacy_cold.id]

    orch._active_timescope = AS_OF
    orch._cold_hits = {}
    cold_frags = orch._cold_lookup({MARK}, None)
    got_ids = {f.source_batch_id for f in cold_frags}
    check("26 cold row surfaced under as_of (dated, keyword-matched)",
          str(cold_row.id) in got_ids
          and next(f for f in cold_frags
                   if f.source_batch_id == str(cold_row.id)).text.startswith("[recorded; provenance unknown: 2025-03-12T00:00:00Z] "))
    orch._active_timescope = CURRENT
    current_cold = orch._cold_lookup({MARK}, None)
    check("26 cold leg surfaces queried originals under current",
          str(cold_row.id) in {f.source_batch_id for f in current_cold})

    # 27: resurrection of budget survivors; legacy row cite-only
    # (capture plain values first — the resurrection commit expires the ORM
    # instances, and the cold row's own row gets deleted)
    cold_id, cold_ts = cold_row.id, cold_row.timestamp
    legacy_id = legacy_cold.id
    orch._active_timescope = AS_OF
    orch._resurrect_cold_hits(cold_frags)   # both frags "survived the budget"
    res = db.get(EpisodicMemory, cold_id)
    created_episodic.append(cold_id)        # cleanup either way
    check("27 resurrected: original id+timestamp, probation decay, embedding, live",
          res is not None
          and res.timestamp == cold_ts
          and abs(res.decay_score - settings.timescope_probation_score) < 1e-9
          and res.embedding is not None and res.is_archived is False
          and res.inject_raw is True and res.access_count == 1)
    check("27 cold row deleted after resurrection",
          db.get(ColdStorage, cold_id) is None)
    check("27 legacy (NULL-conversation) row cite-only: still cold, not episodic",
          db.get(ColdStorage, legacy_id) is not None
          and db.get(EpisodicMemory, legacy_id) is None)

    # 28: decay freeze fix + un-archive clause (module-level, live table)
    from src.workers.decay import apply_decay
    t_frozen = mk_turn(real_now - D * 30, f"User: {MARK} frozen archived\n\nAssistant: ok",
                       decay=0.08, archived=True)
    t_recover = mk_turn(real_now - D * 30, f"User: {MARK} recovering archived\n\nAssistant: ok",
                        decay=0.15, archived=True)
    apply_decay(cycles=1)
    db.expire_all()
    t_frozen = db.get(EpisodicMemory, t_frozen.id)
    t_recover = db.get(EpisodicMemory, t_recover.id)
    check("28 archived row keeps decaying (freeze fixed)", t_frozen.decay_score < 0.08)
    check("28 recovered row un-archived by the new clause",
          t_recover.is_archived is False and t_recover.decay_score >= 0.1)

    # 28b: cold move carries the new columns
    t_cold_bound = mk_turn(real_now - D * 30, f"User: {MARK} nearly cold\n\nAssistant: ok",
                           decay=0.049, archived=True, is_private=True)
    cold_bound_id = t_cold_bound.id   # capture before apply_decay deletes the row
    apply_decay(cycles=1)
    db.expire_all()
    moved = db.get(ColdStorage, cold_bound_id)
    created_cold.append(cold_bound_id)
    check("28 cold move carries conversation_id/is_private/batch_id",
          moved is not None and moved.conversation_id == conv.id
          and moved.is_private is True and moved.batch_id is not None)

    # 29: honest emptiness
    empty_win = TimeScope(mode="as_of",
                          t0=datetime(1999, 1, 1, tzinfo=timezone.utc),
                          t1=datetime(1999, 3, 1, tzinfo=timezone.utc))
    orch._active_timescope = empty_win
    noted = orch._append_empty_window_note([], EMB, None)
    check("29 empty window → single [Memory note] naming the window",
          len(noted) == 1 and noted[0].text.startswith(
              "[Memory note] No episodic source records were selected for this query between 1999-01-01 and 1999-03-01")
          and noted[0].source_type == "episodic")
    check("29 note names nearest eras (probe found rows)",
          "Nearby source records are dated" in noted[0].text)

    # ═══════════════════════════════════════════════════════════════════════
    print("── Regression (check 30) ──")

    # 30a: recency formula numerically identical for past rows (center=now)
    row = db.execute(text("""
        SELECT (1 + 0.25 * EXP(-GREATEST(EXTRACT(EPOCH FROM (NOW() - timestamp)), 0) / 86400.0 / 30.0)) AS old_f,
               (1 + 0.25 * EXP(-ABS(EXTRACT(EPOCH FROM (timestamp - NOW()))) / 86400.0 / 30.0)) AS new_f
        FROM episodic_memory WHERE id = :id
    """), {"id": str(t_in.id)}).fetchone()
    check("30 recency formula: old ≡ new at center=now", abs(row.old_f - row.new_f) < 1e-12)

    # 30b: v3 current sources remain searchable after archival or probation;
    # source representation/date and unrelated temporal predicates remain.
    orch._active_timescope = CURRENT
    t_low = mk_turn(datetime(2025, 5, 5, tzinfo=timezone.utc),
                    f"User: {MARK} low decay\n\nAssistant: ok", decay=0.15)
    cur = orch._vector_episodic(EMB, c, None, None)
    check("30 current: low-decay source remains searchable", has(cur, t_low))
    check("30 current: archived source remains searchable", has(cur, t_frozen))
    fin = next((f for f in cur if f.source_batch_id == str(t_in.id)), None)
    check("30 current: fragment text = old text modulo date prefix",
          fin is not None and fin.text == f"[source recorded: 2025-03-10T00:00:00Z] User: {MARK} report from march 2025\n\nAssistant: noted")

finally:
    # Resilient cleanup: marker-keyed bulk deletes (catches rows created by
    # side effects like resurrection/cold-move), each group committed on its
    # own so one failure can't strand the rest. Never truncates.
    db.rollback()
    for label, stmt, params in [
        ("cold", text("DELETE FROM cold_storage WHERE raw_text LIKE :m"), {"m": f"%{MARK}%"}),
        ("episodic", text("DELETE FROM episodic_memory WHERE raw_text LIKE :m OR conversation_id = :c"),
         {"m": f"%{MARK}%", "c": conv.id}),
        ("codex_events", text("DELETE FROM codex_events WHERE entity_id IN "
                              "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m)"),
         {"m": f"{MARK}%"}),
        ("codex_facts", text("DELETE FROM codex_facts WHERE id IN (SELECT fact_id FROM "
                             "codex_fact_entities WHERE entity_id IN "
                             "(SELECT id FROM codex_entities WHERE canonical_name LIKE :m))"),
         {"m": f"{MARK}%"}),
        ("codex_entities", text("DELETE FROM codex_entities WHERE canonical_name LIKE :m"),
         {"m": f"{MARK}%"}),
        ("conversation", text("DELETE FROM conversations WHERE id IN (:c, :c2)"),
         {"c": conv.id, "c2": conv2.id}),
    ]:
        try:
            db.execute(stmt, params)
            db.commit()
        except Exception as exc:
            db.rollback()
            print(f"  cleanup({label}) failed: {exc}")
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
