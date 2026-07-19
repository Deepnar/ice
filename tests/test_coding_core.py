"""E-coding-core behavioral test — specs/E_coding_core.md §5 checks 1–12.

Runs against the live Postgres (docker up). Copies tests/fixtures/mini_repo
to a scratch dir, git-inits it, and registers it as a throwaway project;
inserts its own uniquely-marked rows and deletes them in `finally` — NEVER
truncates (the dev DB holds real data). The LLM is always a stub, and the
decision embedder is monkeypatched to fixed vectors (deterministic
similarity). No model loads anywhere — the visibility checks exercise the
source-filter primitives directly (the conversational path cannot even name
a namespaced code canonical, so the primitives ARE the invariant).

Run: uv run python tests/test_coding_core.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import (
    CodexEdge,
    CodexEntity,
    Conversation,
    Decision,
    EpisodicMemory,
    Project,
    ProjectState,
    ProjectTask,
    ReviewQueue,
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


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, capture_output=True, text=True).stdout.strip()


HEX = uuid.uuid4().hex[:6]
NAME = f"coretest-{HEX}"
SLUG = f"ct{HEX}"                      # lowercase (canonicals are casefolded)
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini_repo")
REPO = os.path.join(tempfile.gettempdir(), f"ice_coding_core_{HEX}")
# letters-only: BM25 strips non-alpha, a digit-bearing marker never matches
MARK = "codemark" + HEX.translate(str.maketrans("0123456789", "abcdefghij"))

V1 = [1.0] + [0.0] * 1023               # fixed decision embeddings
V2 = [0.9, 0.4359] + [0.0] * 1022       # cos(V1,V2) ≈ 0.90 — the conflict band
V3 = [0.0, 0.0, 1.0] + [0.0] * 1021     # (0.85 ≤ sim < 0.95 ⇒ supersede)

shutil.copytree(FIXTURE, REPO)
git(REPO, "init", "-b", "main")
git(REPO, "add", "-A")
git(REPO, "commit", "-m", "initial fixture commit")

db = SessionLocal()
project_id = None
conv_ids: list = []
turn_ids: list = []

import src.workers.decision_extractor as de  # noqa: E402
import src.workers.maintenance_agent as ma  # noqa: E402

_orig_embed = de._embed
_orig_detectors = dict(ma.DETECTORS)

try:
    from src.coding.code_graph import code_graph_batch_id
    from src.coding.reconciler import marker_path, poll_projects, reconcile_project
    from src.services import graph as graph_svc
    from src.services import projects as projects_svc
    from src.services.retrieval_svc import constraints_for_task

    # ── 1. registration bootstrap ────────────────────────────────────────
    print("── check 1: bootstrap of mini_repo ──")
    report = projects_svc.register_project(db, NAME, REPO, slug=SLUG,
                                           install_git_hook=True)
    project_id = uuid.UUID(report["project_id"])
    project = db.query(Project).filter_by(id=project_id).first()
    g = report["graph"]
    check("bootstrap report lists entities/edges/facts",
          g.get("created", 0) >= 17 and g.get("edges", 0) >= 6
          and report["facts"].get("derived", 0) == 6)
    check("resolved vs heuristic tagged (ratio loggable)",
          g.get("resolved", 0) >= 5 and g.get("heuristic", 0) >= 1)
    ents = {e.canonical_name: e for e in
            db.query(CodexEntity).filter_by(project_id=project_id,
                                            source="static_analysis").all()}
    expected = {f"{SLUG}:minipkg.core", f"{SLUG}:minipkg.core.greet",
                f"{SLUG}:minipkg.core.helper", f"{SLUG}:minipkg.core.animal",
                f"{SLUG}:minipkg.core.animal.speak", f"{SLUG}:minipkg.core.dog",
                f"{SLUG}:minipkg.core.dog.speak",
                f"{SLUG}:minipkg.core.dog.tail_wag",
                f"{SLUG}:minipkg.util.shout", f"{SLUG}:minipkg.config.settings"}
    check("qualified canonical names present", expected <= set(ents))
    greet = ents[f"{SLUG}:minipkg.core.greet"]
    check("pointers (file:line) correct",
          greet.properties["file_path"] == "minipkg/core.py"
          and greet.properties["line_start"] == 4
          and greet.properties["line_end"] > 4
          and greet.properties["signature"] == "def greet(name)")
    check("docstring FIRST LINE only",
          greet.properties["docstring_summary"] == "Return a friendly greeting."
          and "never be stored" not in str(greet.properties))
    bodies = [e for e in ents.values()
              if "hello {name}" in str(e.properties) + (e.context_payload or "")
              or "return f" in str(e.properties) + (e.context_payload or "")]
    check("zero source bodies stored", not bodies)
    check("code entities: no embedding, no aliases",
          all(e.embedding is None and not e.aliases for e in ents.values()))
    edges = db.query(CodexEdge).filter_by(
        source_batch=code_graph_batch_id(project_id)).all()
    check("static edges: source tag + deterministic batch",
          len(edges) >= 6 and all(e.source == "static_analysis" for e in edges))
    heuristic = [e for e in edges if (e.extraction_confidence or 1.0) < 1.0]
    check("heuristic call edge tagged via extraction_confidence",
          len(heuristic) == 1)
    check("consent hook installed (post-commit)",
          report["hook"]["installed"] is True
          and os.path.exists(os.path.join(REPO, ".git", "hooks", "post-commit")))
    check("no CodexEvents journaled for static analysis",
          db.execute(text("SELECT count(*) FROM codex_events WHERE entity_id = ANY(:ids)"),
                     {"ids": [e.id for e in ents.values()]}).scalar() == 0)

    # ── 5. ice_where ─────────────────────────────────────────────────────
    print("── check 5: ice_where resolves a fixture symbol ──")
    res = graph_svc.where_symbol(db, "Dog.speak")
    check("where_symbol: code-graph engine + pointer",
          res["resolved"] and res["engine"] == "code_graph"
          and any(m["file"] == "minipkg/core.py" and m["line_start"] == 28
                  for m in res["matches"]))
    res2 = graph_svc.where_symbol(db, "shout")
    check("where_symbol: bare function name",
          res2["resolved"] and any(m["name"] == "minipkg.util.shout"
                                   for m in res2["matches"]))

    # ── 3. decay exemption ───────────────────────────────────────────────
    print("── check 3: decay leaves derived entities untouched ──")
    conv_edge = CodexEdge(
        source_id=list(ents.values())[0].id, target_id=list(ents.values())[1].id,
        relation="references", strength=1.0, source_batch=uuid.uuid4(),
        confidence="active", source="conversation",
        valid_from=datetime.now(timezone.utc))
    db.add(conv_edge)
    db.commit()
    static_before = {e.id: e.strength for e in edges}
    conv_before = conv_edge.strength
    from src.workers.codex_decay import decay_codex_edges
    decay_codex_edges(cycles=4)
    db.expire_all()
    static_after = {e.id: e.strength for e in
                    db.query(CodexEdge).filter_by(
                        source_batch=code_graph_batch_id(project_id)).all()}
    conv_after = db.query(CodexEdge).get(conv_edge.id).strength
    check("static edges: strength unchanged after decay",
          all(abs(static_after[i] - s) < 1e-9 for i, s in static_before.items()))
    check("conversation edge: decayed", conv_after < conv_before)

    # ── 4. source visibility ─────────────────────────────────────────────
    print("── check 4: conversational vs project-scoped visibility ──")
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    orch = HybridRetrievalOrchestrator(db, embedder=None)
    canonical = f"{SLUG}:minipkg.core.greet"
    orch._scope_project_id = None
    check("unscoped exact-match never returns a static entity",
          orch._match_entities_exact([canonical]) == [])
    orch._scope_project_id = project_id
    matched = orch._match_entities_exact([canonical])
    check("project-scoped exact-match resolves the code entity",
          len(matched) == 1 and matched[0].canonical_name == canonical)
    ent_set, batch_set = orch._codex_scope_sets(
        {"project_id": str(project_id), "conversation_ids": []})
    check("_codex_scope_sets: code-graph allowance (entities + batch)",
          ent_set is not None and greet.id in ent_set
          and code_graph_batch_id(project_id) in batch_set)
    texts: list = []
    orch._traverse_graph(matched[0], 0, 1, set(), texts, [],
                         allowed_entity_ids=ent_set, allowed_batch_ids=batch_set)
    check("scoped traversal renders the derived pointer payload",
          any("minipkg/core.py:4" in t for t in texts))
    orch._scope_project_id = None
    check("_entity_visible: derived hidden unscoped, visible in-project",
          not orch._entity_visible(greet)
          and (setattr(orch, "_scope_project_id", project_id)
               or orch._entity_visible(greet)))

    # D11 routing bits
    from src.api.memory_decision import decide_memory_retrieval
    clf = ClassificationResult(
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Zero_Shot", raw_probs=[0.0] * 25,
        max_confidence=0.9, prompt="hello")
    d_no = decide_memory_retrieval(clf, 0, 0.0, settings)
    d_yes = decide_memory_retrieval(clf, 0, 0.0, settings, coding_scope=True)
    check("D11: ltm_bump_coding raises p_need_mem + breakdown key",
          d_yes.p_need_mem > d_no.p_need_mem
          and d_yes.breakdown.get("coding_scope") is True)

    # episodic conversation-list filter (project scope → many conversations)
    conv_a, conv_b, conv_c = Conversation(), Conversation(), Conversation()
    db.add_all([conv_a, conv_b, conv_c])
    db.commit()
    conv_ids += [conv_a.id, conv_b.id, conv_c.id]
    for conv, tag in ((conv_a, "one"), (conv_b, "two"), (conv_c, "three")):
        t = EpisodicMemory(
            conversation_id=conv.id, batch_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            context_reliance="Long_Term_Memory",
            raw_text=f"User: about {MARK} {tag}\n\nAssistant: ok",
            decay_score=1.0, idempotency_key=f"coding-core-{uuid.uuid4()}")
        db.add(t)
        db.commit()
        turn_ids.append(t.id)
    clf_mark = ClassificationResult(
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_probs=[0.0] * 25,
        max_confidence=0.9, prompt=MARK)
    frags = orch._bm25_episodic(
        clf_mark, scope={"conversation_ids": [str(conv_a.id), str(conv_b.id)]})
    check("episodic leg: conversation-list scope (in: a+b, out: c)",
          any(f"{MARK} one" in f.text for f in frags)
          and any(f"{MARK} two" in f.text for f in frags)
          and not any(f"{MARK} three" in f.text for f in frags))

    # ── 6/7. decisions: supersession, constraint-first, incident ────────
    print("── checks 6+7: decisions bi-temporal + constraint-first + incident ──")
    calls = {"n": 0}

    def stub_llm_factory(payload):
        def _stub(prompt, max_tokens=200):
            calls["n"] += 1
            return payload
        return _stub

    de._embed = lambda text_content: V1
    r0 = de.run_decision_extraction(
        db, project_id=project_id, text_content="nothing interesting here",
        llm=stub_llm_factory({}))
    check("no cue ⇒ no LLM call (A6 stance)",
          r0["status"] == "no_cue" and calls["n"] == 0)

    r1 = de.run_decision_extraction(
        db, project_id=project_id,
        text_content="we decided to use postgres for storage",
        llm=stub_llm_factory({
            "decision": f"{MARK}: use postgres for storage",
            "rationale": "relational fits", "alternatives_rejected": ["sqlite"],
            "files_affected": ["minipkg/core.py"], "type": "decision"}))
    check("decision recorded from cue", r1["status"] == "recorded")
    de._embed = lambda text_content: V2   # near-identical → conflict path
    r2 = de.run_decision_extraction(
        db, project_id=project_id,
        text_content="we decided to use sqlite for storage instead of postgres",
        llm=stub_llm_factory({
            "decision": f"{MARK}: use sqlite for storage",
            "rationale": "simpler", "alternatives_rejected": ["postgres"],
            "files_affected": ["minipkg/core.py"], "type": "decision"}))
    old_row = db.query(Decision).filter_by(id=uuid.UUID(r1["id"])).first()
    check("conflicting insert supersedes (valid_until + superseded_by)",
          r2["status"] == "superseded"
          and old_row.valid_until is not None
          and str(old_row.superseded_by) == r2["id"])

    de._embed = lambda text_content: V3
    r3 = de.run_decision_extraction(
        db, project_id=project_id,
        text_content=f"don't touch minipkg/config.py — {MARK} settings order matters",
        llm=stub_llm_factory({
            "decision": f"{MARK}: do not touch minipkg/config.py settings order",
            "rationale": "load order is load-bearing",
            "alternatives_rejected": [],
            "files_affected": ["minipkg/config.py"], "type": "constraint"}))
    check("constraint row recorded", r3["status"] == "recorded")
    cfrags = constraints_for_task(
        db, "please refactor minipkg/config.py default values")
    check("ice_context engine surfaces the constraint FIRST for affected file",
          cfrags and cfrags[0]["source_type"] == "constraint"
          and MARK in cfrags[0]["text"])
    check("unrelated task pulls no constraint",
          constraints_for_task(db, "write a poem about autumn") == [])

    de._embed = _orig_embed
    de._embed = lambda text_content: [0.0] * 1023 + [1.0]
    r4 = de.run_decision_extraction(
        db, project_id=project_id,
        text_content="hit a KeyError traceback in util; fixed by guarding the "
                     "dict access — root cause was a missing default",
        llm=stub_llm_factory({
            "decision": f"{MARK}: guard dict access in util (KeyError)",
            "rationale": "missing default", "alternatives_rejected": [],
            "files_affected": ["minipkg/util.py"], "type": "incident"}))
    inc = db.query(Decision).filter_by(id=uuid.UUID(r4["id"])).first()
    check("incident row from error→fixed arc",
          r4["status"] == "recorded" and inc.decision_type == "incident")

    # ── 8. E9 facts + hash gate ──────────────────────────────────────────
    print("── check 8: project facts + hash-gated re-derive ──")
    facts = {e.canonical_name: e for e in db.query(CodexEntity).filter_by(
        project_id=project_id, source="derived").all()}
    kinds = {"dependencies", "db_schema", "commands", "config_surface",
             "infrastructure", "data_shapes"}
    check("each parser yields its unit",
          {f"{SLUG}:fact:{k}" for k in kinds} == set(facts))
    deps = facts[f"{SLUG}:fact:dependencies"]
    check("dependencies unit carries declared deps + pointer",
          "requests>=2.31" in str(deps.properties["data"])
          and deps.properties["source_files"] == ["pyproject.toml"])
    check("config unit stores KEY NAMES, never values",
          "API_KEY" in str(facts[f"{SLUG}:fact:config_surface"].properties)
          and "replace-me" not in str(facts[f"{SLUG}:fact:config_surface"].properties))
    infra_hash = facts[f"{SLUG}:fact:infrastructure"].properties["content_hash"]
    deps_hash = deps.properties["content_hash"]
    with open(os.path.join(REPO, "pyproject.toml"), "a") as f:
        f.write('\n[tool.marker]\nx = "httpx>=0.28"\n')
    from src.coding.project_facts import derive_project_facts
    stats = derive_project_facts(db, project, changed_files=["pyproject.toml"])
    db.expire_all()
    facts2 = {e.canonical_name: e for e in db.query(CodexEntity).filter_by(
        project_id=project_id, source="derived").all()}
    check("editing pyproject re-derives deps only (others skipped/unchanged)",
          stats["skipped"] >= 4
          and facts2[f"{SLUG}:fact:dependencies"].properties["content_hash"] != deps_hash
          and facts2[f"{SLUG}:fact:infrastructure"].properties["content_hash"] == infra_hash)

    # ── 2 + 9. incremental reconcile on commit ──────────────────────────
    print("── checks 2+9: commit → reconcile (incremental, task link) ──")
    task_report = projects_svc.task_add(db, SLUG, f"{MARK} implement extra_util",
                                        status="active")
    core_ids_before = {c: ents[c].id for c in expected if ".core" in c}
    with open(os.path.join(REPO, "minipkg", "util.py"), "w") as f:
        f.write('"""Utilities built on core."""\n'
                "from minipkg.core import greet\n\n\n"
                "def shout(name):\n"
                '    """Uppercase greeting."""\n'
                "    return greet(name).upper()\n\n\n"
                "def extra_util(x):\n"
                '    """Fresh function added incrementally."""\n'
                "    return x + 1\n")
    git(REPO, "add", "-A")
    git(REPO, "commit", "-m", "add extra_util, drop mystery/configured")
    new_head = git(REPO, "rev-parse", "HEAD")
    check("post-commit hook wrote the fallback marker (no app running)",
          marker_path(REPO).exists())
    result = reconcile_project(db, project_id=project_id)
    db.expire_all()
    state = db.query(ProjectState).filter_by(project_id=project_id).first()
    check("reconciler advances last_reconciled_commit",
          result["status"] == "reconciled"
          and state.last_reconciled_commit == new_head)
    check("marker consumed by reconcile", not marker_path(REPO).exists())
    task = db.query(ProjectTask).filter_by(
        id=uuid.UUID(task_report["task_id"])).first()
    check("commit linked to the active task",
          new_head in (task.commit_hashes or [])
          and "minipkg/util.py" in (task.files_changed or []))
    ents_after = {e.canonical_name: e for e in
                  db.query(CodexEntity).filter_by(project_id=project_id,
                                                  source="static_analysis").all()}
    check("incremental: unchanged file's entity ids stable",
          all(ents_after[c].id == i for c, i in core_ids_before.items()))
    check("incremental: new symbol present, removed symbols gone",
          f"{SLUG}:minipkg.util.extra_util" in ents_after
          and f"{SLUG}:minipkg.util.mystery" not in ents_after
          and f"{SLUG}:minipkg.util.configured" not in ents_after)

    # poll fallback: HEAD drift after a commit made without reconcile
    with open(os.path.join(REPO, "minipkg", "core.py"), "a") as f:
        f.write("\n\ndef late_addition():\n"
                '    """Added for the poll test."""\n'
                "    return True\n")
    git(REPO, "add", "-A")
    git(REPO, "commit", "-m", "late addition")
    poll_stats = poll_projects(db)
    db.expire_all()
    state = db.query(ProjectState).filter_by(project_id=project_id).first()
    check("poll fallback reconciles HEAD drift",
          poll_stats["reconciled"] >= 1
          and state.last_reconciled_commit == git(REPO, "rev-parse", "HEAD"))

    # ── 10. session-start block ──────────────────────────────────────────
    print("── check 10: session-start block ──")
    projects_svc.set_goal(db, SLUG, f"{MARK} finish the fixture feature")
    block = projects_svc.chat_session_start(db, project_id)
    check("block renders state + constraints + tasks + decisions",
          block is not None
          and f"Project: {NAME}" in block
          and "branch: main" in block
          and f"{MARK}: do not touch minipkg/config.py" in block
          and f"{MARK} implement extra_util" in block
          and f"{MARK}: use sqlite for storage" in block)
    db.expire_all()
    state = db.query(ProjectState).filter_by(project_id=project_id).first()
    first_stamp = state.last_session_at
    projects_svc.chat_session_start(db, project_id)   # same sitting
    db.expire_all()
    state = db.query(ProjectState).filter_by(project_id=project_id).first()
    check("last_session_at advances once per sitting (rev 8)",
          first_stamp is not None and state.last_session_at == first_stamp)

    # ── 11. stale task → agent worklist ─────────────────────────────────
    print("── check 11: stale task → agent worklist item ──")
    stale = db.query(ProjectTask).filter_by(
        id=uuid.UUID(task_report["task_id"])).first()
    stale.updated_at = datetime.now(timezone.utc) - timedelta(days=20)
    db.commit()
    items = ma._detect_stale_work(db, 10)
    mine = [i for i in items if i.payload.get("task_id") == task_report["task_id"]]
    check("stale_work detector finds the 20-day task",
          len(mine) == 1 and mine[0].tier == 2
          and mine[0].payload["days_stale"] >= 19)
    ma.DETECTORS.clear()
    ma.DETECTORS["stale_work"] = lambda db_, cap: mine
    run = ma.run_maintenance_agent(db, llm_decider=None)
    proposal = db.query(ReviewQueue).filter(
        ReviewQueue.item_type == "stale_work",
        ReviewQueue.item_content["task_id"].astext == task_report["task_id"],
    ).first()
    check("agent run proposes it (no LLM needed)",
          run["outcomes"].get("stale_work:proposed") == 1 and proposal is not None)
    ma.DETECTORS.clear()
    ma.DETECTORS.update(_orig_detectors)
    check("pending proposal blocks re-detection (idempotent)",
          not [i for i in ma._detect_stale_work(db, 10)
               if i.payload.get("task_id") == task_report["task_id"]])

    # ── 12. architecture doc render ──────────────────────────────────────
    print("── check 12: render_architecture_doc ──")
    doc = graph_svc.render_architecture_doc(db, SLUG)
    check("all four layers present",
          "## Modules" in doc and "minipkg.core" in doc
          and f"{MARK}: use sqlite for storage" in doc
          and f"{MARK}: do not touch minipkg/config.py" in doc
          and "## Conventions" in doc
          and "## Recent commits" in doc and "late addition" in doc)

finally:
    de._embed = _orig_embed
    ma.DETECTORS.clear()
    ma.DETECTORS.update(_orig_detectors)
    try:
        if project_id is not None:
            db.rollback()
            db.execute(text(
                "DELETE FROM review_queue WHERE item_content->>'project_id' = :p "
                "OR item_content->>'task_id' IN "
                "(SELECT id::text FROM tasks WHERE project_id = :pid)"),
                {"p": str(project_id), "pid": project_id})
            db.execute(text("DELETE FROM decisions WHERE project_id = :pid"),
                       {"pid": project_id})
            db.execute(text("DELETE FROM tasks WHERE project_id = :pid"),
                       {"pid": project_id})
            db.execute(text("DELETE FROM project_state WHERE project_id = :pid"),
                       {"pid": project_id})
            ent_ids = [r.id for r in db.execute(text(
                "SELECT id FROM codex_entities WHERE project_id = :pid"),
                {"pid": project_id}).fetchall()]
            if ent_ids:
                db.execute(text(
                    "DELETE FROM codex_edges WHERE source_id = ANY(:ids) "
                    "OR target_id = ANY(:ids)"), {"ids": ent_ids})
                db.execute(text("DELETE FROM codex_entities WHERE id = ANY(:ids)"),
                           {"ids": ent_ids})
            db.execute(text(
                "UPDATE conversations SET project_id = NULL WHERE project_id = :pid"),
                {"pid": project_id})
            db.execute(text("DELETE FROM projects WHERE id = :pid"),
                       {"pid": project_id})
        if turn_ids:
            db.execute(text("DELETE FROM episodic_memory WHERE id = ANY(:ids)"),
                       {"ids": turn_ids})
        if conv_ids:
            db.execute(text("DELETE FROM conversations WHERE id = ANY(:ids)"),
                       {"ids": conv_ids})
        db.commit()
    finally:
        db.close()
        shutil.rmtree(REPO, ignore_errors=True)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
