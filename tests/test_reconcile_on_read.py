"""E11 behavioral test — reconcile-on-read (working-tree freshness).

Validation list from the E11 roadmap entry (the entry IS the spec): an
uncommitted edit is visible through where_symbol/context_for; the commit
anchor (project_state.last_reconciled_commit) does NOT advance; a second read
with no edits does zero reparse (signature) and a burst is throttled by the
min interval; the clean-tree fast path runs only git status; kill-switch off
⇒ commit-fresh. Plus D3 resolution (explicit arg / slug-prefix / no-hint ⇒
freshen nothing) and the D4 advisory lock (a held lock blocks the reparse).

House pattern: live Postgres, throwaway mini_repo copies registered as
throwaway projects, uniquely-marked rows, cleanup in `finally` — NEVER
truncates. context_for runs over a stubbed classifier (no model loads).

Run: uv run python tests/test_reconcile_on_read.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.models import CodexEntity, Conversation, Project, ProjectState

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


def add_fn(repo, fn_name):
    with open(os.path.join(repo, "minipkg", "util.py"), "a") as f:
        f.write(f"\n\ndef {fn_name}():\n"
                f'    """Uncommitted probe function."""\n'
                f"    return True\n")


HEX = uuid.uuid4().hex[:6]
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini_repo")
NAME_A, SLUG_A = f"rrtest-{HEX}", f"rra{HEX}"
NAME_B, SLUG_B = f"rrtest2-{HEX}", f"rrb{HEX}"
REPO_A = os.path.join(tempfile.gettempdir(), f"ice_rr_a_{HEX}")
REPO_B = os.path.join(tempfile.gettempdir(), f"ice_rr_b_{HEX}")

for repo in (REPO_A, REPO_B):
    shutil.copytree(FIXTURE, repo)
    git(repo, "init", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial fixture commit")

db = SessionLocal()
project_ids: list = []
conv_ids: list = []

import src.coding.code_graph as cg  # noqa: E402
import src.coding.reconciler as rec  # noqa: E402
from src.services import graph as graph_svc  # noqa: E402
from src.services import projects as projects_svc  # noqa: E402
from src.services import retrieval_svc  # noqa: E402

_orig_interval = settings.reconcile_on_read_min_interval_seconds
_orig_enabled = settings.reconcile_on_read
_orig_sync = cg.CodeExtractor.sync_files
_orig_get_classifier = retrieval_svc._get_classifier

sync_calls = {"n": 0}


def counting_sync(self, db_, paths):
    sync_calls["n"] += 1
    return _orig_sync(self, db_, paths)


try:
    settings.reconcile_on_read = True
    settings.reconcile_on_read_min_interval_seconds = 0.0
    rec._freshen_cache.clear()
    cg.CodeExtractor.sync_files = counting_sync

    rep_a = projects_svc.register_project(db, NAME_A, REPO_A, slug=SLUG_A)
    rep_b = projects_svc.register_project(db, NAME_B, REPO_B, slug=SLUG_B)
    project_ids += [uuid.UUID(rep_a["project_id"]), uuid.UUID(rep_b["project_id"])]
    project_a = db.query(Project).filter_by(id=project_ids[0]).first()
    anchor0 = db.query(ProjectState).filter_by(
        project_id=project_ids[0]).first().last_reconciled_commit

    # ── clean-tree fast path ────────────────────────────────────────────
    print("── clean tree: only git status ──")
    sync_calls["n"] = 0
    r = rec.freshen_working_tree(db, project_a)
    check("clean tree ⇒ status 'clean', zero reparse",
          r["status"] == "clean" and sync_calls["n"] == 0)

    # ── uncommitted edit visible through where_symbol ───────────────────
    print("── uncommitted edit → where_symbol ──")
    util = os.path.join(REPO_A, "minipkg", "util.py")
    src_txt = open(util).read()
    src_txt = src_txt.replace(
        "def mystery():\n"
        '    """Calls a bare name defined only in core (heuristic edge)."""\n'
        "    return helper()  # noqa: F821 — deliberate: heuristic resolution fixture\n",
        "def fresh_probe_fn():\n"
        '    """Added without committing."""\n'
        "    return 41\n")
    open(util, "w").write(src_txt)

    res = graph_svc.where_symbol(db, "fresh_probe_fn", project=SLUG_A)
    check("new uncommitted symbol resolves with pointer",
          res["resolved"] and res.get("engine") == "code_graph"
          and any(m["file"] == "minipkg/util.py" and m["project"] == SLUG_A
                  for m in res["matches"]))
    gone = db.query(CodexEntity).filter_by(
        canonical_name=f"{SLUG_A}:minipkg.util.mystery").first()
    check("symbol removed in the working tree is gone from the graph",
          gone is None)
    db.expire_all()
    anchor1 = db.query(ProjectState).filter_by(
        project_id=project_ids[0]).first().last_reconciled_commit
    check("last_reconciled_commit did NOT advance (stays a commit concept)",
          anchor1 == anchor0)

    # ── second read: signature ⇒ zero reparse; interval ⇒ throttled ─────
    print("── throttle: signature + min interval ──")
    sync_calls["n"] = 0
    graph_svc.where_symbol(db, "fresh_probe_fn", project=SLUG_A)
    r = rec.freshen_working_tree(db, project_a)
    check("no further edits ⇒ zero reparse (signature unchanged)",
          sync_calls["n"] == 0 and r["status"] == "unchanged")
    settings.reconcile_on_read_min_interval_seconds = 60.0
    r = rec.freshen_working_tree(db, project_a)
    check("burst of reads inside the min interval ⇒ throttled (no git status)",
          r["status"] == "throttled")
    settings.reconcile_on_read_min_interval_seconds = 0.0

    # ── the commit path re-runs idempotently and still owns the anchor ──
    print("── commit reconcile after a freshen ──")
    git(REPO_A, "add", "-A")
    git(REPO_A, "commit", "-m", "commit the probe edit")
    new_head = git(REPO_A, "rev-parse", "HEAD")
    rec.reconcile_project(db, project_id=project_ids[0])
    db.expire_all()
    state = db.query(ProjectState).filter_by(project_id=project_ids[0]).first()
    probe = db.query(CodexEntity).filter_by(
        canonical_name=f"{SLUG_A}:minipkg.util.fresh_probe_fn").first()
    check("commit reconcile advances the anchor over the same files",
          state.last_reconciled_commit == new_head and probe is not None)
    r = rec.freshen_working_tree(db, project_a)
    check("just-committed tree ⇒ clean fast path", r["status"] == "clean")

    # ── kill switch ⇒ commit-fresh ──────────────────────────────────────
    print("── kill switch ──")
    settings.reconcile_on_read = False
    add_fn(REPO_A, "killswitch_fn")
    r = rec.freshen_working_tree(db, project_a)
    res = graph_svc.where_symbol(db, "killswitch_fn", project=SLUG_A)
    check("reconcile_on_read=False ⇒ disabled + commit-fresh reads",
          r["status"] == "disabled" and not res["resolved"])
    settings.reconcile_on_read = True

    # ── D3 resolution order ─────────────────────────────────────────────
    print("── D3: project resolution for the freshen ──")
    res = graph_svc.where_symbol(db, "killswitch_fn")
    check("bare symbol + several projects ⇒ freshen nothing (commit-fresh)",
          not res["resolved"])
    res = graph_svc.where_symbol(db, "killswitch_fn", project=SLUG_A)
    check("explicit project arg freshens that project",
          res["resolved"] and any(m["project"] == SLUG_A for m in res["matches"]))
    add_fn(REPO_A, "prefix_probe_fn")
    res = graph_svc.where_symbol(db, f"{SLUG_A}:minipkg.util.prefix_probe_fn")
    check("slug: prefix on the symbol resolves the project",
          res["resolved"] and any(m["project"] == SLUG_A for m in res["matches"]))
    res = graph_svc.where_symbol(db, "shout", project=f"nonexistent-{HEX}")
    check("unknown explicit project ⇒ lookup still answers (freshen skipped)",
          res["resolved"])

    # ── D4: advisory lock serializes reparse cross-session ──────────────
    print("── D4: advisory lock ──")
    add_fn(REPO_A, "lock_probe_fn")
    holder = SessionLocal()
    holder.execute(text("SELECT pg_advisory_xact_lock(hashtext(:p))"),
                   {"p": str(project_ids[0])})
    outcome = {}

    def blocked_freshen():
        db2 = SessionLocal()
        try:
            proj2 = db2.query(Project).filter_by(id=project_ids[0]).first()
            outcome["result"] = rec.freshen_working_tree(db2, proj2)
        finally:
            db2.close()

    t = threading.Thread(target=blocked_freshen, daemon=True)
    t.start()
    t.join(timeout=1.5)
    still_blocked = t.is_alive()
    holder.rollback()          # release the lock
    holder.close()
    t.join(timeout=30)
    check("freshen blocks while another session holds the project lock, "
          "then completes",
          still_blocked and not t.is_alive()
          and outcome.get("result", {}).get("status") == "freshened")

    # ── context_for freshens under a project-attached conversation ──────
    print("── context_for: scope enrichment + freshen ──")

    class StubEmbedder:
        def encode(self, texts, convert_to_tensor=False, **kwargs):
            import torch
            if isinstance(texts, (list, tuple)):
                vecs = torch.zeros((len(texts), 1024))
                return vecs if convert_to_tensor else vecs.numpy()
            vec = [1.0] + [0.0] * 1023
            return torch.tensor(vec) if convert_to_tensor else vec

    class StubClassifier:
        embedder = StubEmbedder()

        def classify(self, text_, conversation_id=None):
            return ClassificationResult(
                topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
                context_reliance="Long_Term_Memory", raw_probs=[0.0] * 25,
                max_confidence=0.9, prompt=text_)

    retrieval_svc._get_classifier = lambda: StubClassifier()
    conv = Conversation(memory_scope_type="auto", project_id=project_ids[0])
    db.add(conv)
    db.commit()
    conv_ids.append(conv.id)
    add_fn(REPO_A, "ctx_probe_fn")
    out = retrieval_svc.context_for(
        db, "where is ctx_probe_fn defined",
        scope={"conversation_id": str(conv.id)})
    ent = db.query(CodexEntity).filter_by(
        canonical_name=f"{SLUG_A}:minipkg.util.ctx_probe_fn").first()
    check("context_for under an attached conversation freshens the project",
          isinstance(out.get("fragments"), list) and ent is not None)
    db.expire_all()
    check("context_for freshen left the anchor untouched",
          db.query(ProjectState).filter_by(project_id=project_ids[0])
          .first().last_reconciled_commit == new_head)

finally:
    settings.reconcile_on_read = _orig_enabled
    settings.reconcile_on_read_min_interval_seconds = _orig_interval
    cg.CodeExtractor.sync_files = _orig_sync
    retrieval_svc._get_classifier = _orig_get_classifier
    rec._freshen_cache.clear()
    try:
        db.rollback()
        if conv_ids:
            db.execute(text("DELETE FROM conversations WHERE id = ANY(:ids)"),
                       {"ids": conv_ids})
        for pid in project_ids:
            db.execute(text("DELETE FROM decisions WHERE project_id = :pid"),
                       {"pid": pid})
            db.execute(text("DELETE FROM tasks WHERE project_id = :pid"),
                       {"pid": pid})
            db.execute(text("DELETE FROM project_state WHERE project_id = :pid"),
                       {"pid": pid})
            ent_ids = [r.id for r in db.execute(text(
                "SELECT id FROM codex_entities WHERE project_id = :pid"),
                {"pid": pid}).fetchall()]
            if ent_ids:
                db.execute(text(
                    "DELETE FROM codex_edges WHERE source_id = ANY(:ids) "
                    "OR target_id = ANY(:ids)"), {"ids": ent_ids})
                db.execute(text("DELETE FROM codex_entities WHERE id = ANY(:ids)"),
                           {"ids": ent_ids})
            db.execute(text("DELETE FROM projects WHERE id = :pid"), {"pid": pid})
        db.commit()
    finally:
        db.close()
        shutil.rmtree(REPO_A, ignore_errors=True)
        shutil.rmtree(REPO_B, ignore_errors=True)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
