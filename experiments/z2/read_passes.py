"""Z2 S9: read passes on the replay's checkpoint stores (spec D10).

For every checkpoint that has Z2 questions, the store saved at that session end is
restored into the replay's scratch database and every question is read under each
condition. Questions never write: each read runs inside seed_v3.probe_observation
(a rolled-back, read-only transaction), as the tuning reads did.

Decay (D10): the default overlay is first re-validated on this checkpoint — the
event log must reproduce the store exactly, or the checkpoint is refused and
recorded. `full_default` reads the store as the system ran it (live decay); every
other ICE condition reads with the neutral overlay applied (decay reported as its
own contrast: full_default vs full).

Conditions today — the ones needing no production switch (CONDITIONS holds the exact list):
  full_default   full ICE, live decay (the system as it runs)
  full           full ICE, neutral decay (the ablation reference)
  no_codex       full ICE without the codex facts leg (group 3 removed)
  warm_vector    ICE's own vector leg with its gate, reranker and representation policy
  recent_only    no retrieved memory (the recent window only)
  bare_vector    whole role-tagged original messages before the recent window, cosine
                 order, packed to full ICE's own retrieval budget for that question
                 (frozen development choice, PROVENANCE 2026-10-14); same envelope
  full_b/warm_b/bare_b<N>  D11b budget curves: the same three systems at fixed memory
                 budgets of 2k, 4k and 8k tokens (questions only)
  full_guard_every_word    D10 summary-guard pair: the old guard on a stratified ~80
The remaining D10 ablation groups join CONDITIONS when their read switches are decided.

Each (question, condition) is one durable unit file: units/<condition>/<sha>.json
holding the exact prompt messages the answerer will see, the token ledger, the
selected fragments and the gate record. Done units are skipped on resume.

    uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
        --replay logs/z2-run-main/s7-B --asks <z2-asks.json> --out logs/z2-run-main/s9-B
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-read-passes-1"
BUDGETS = (2_000, 4_000, 8_000)          # D11b memory budgets, beside each system's own default
GUARD_SAMPLE = 80                        # D10 summary-guard pair: 60-100 questions, stratified
# name: overlay (decay), arm (seed_v3 prompt arm) or bare, optional fixed memory budget,
# settings overridden for the read, and the asks it applies to (None = every ask).
CONDITIONS = {
    "full_default": dict(overlay="default", arm="full"),
    "full": dict(overlay="neutral", arm="full"),
    "no_codex": dict(overlay="neutral", arm="no_codex"),
    "warm_vector": dict(overlay="neutral", arm="vector_only"),
    "recent_only": dict(overlay="neutral", arm="recent_only"),
    "bare_vector": dict(overlay="none", bare=True),
    **{f"full_b{b}": dict(overlay="default", arm="full", budget=b, subset="questions") for b in BUDGETS},
    **{f"warm_b{b}": dict(overlay="neutral", arm="vector_only", budget=b, subset="questions") for b in BUDGETS},
    **{f"bare_b{b}": dict(overlay="none", bare=True, budget=b, subset="questions") for b in BUDGETS},
    "full_guard_every_word": dict(overlay="default", arm="full", subset="guard",
                                  settings={"representation_guard_terms": "every_word"}),
}

# D10 ablation groups in the frozen additive order, each switched off with an existing
# production knob (maintainer decision 2026-10-19): ConfigurableOrchestrator flags,
# settings, or a harness patch around the read. Group 10 (decay) is not a switch: it is
# full_default vs full (live vs neutral overlay). Group 11's entity merges are undone on
# the restored copy, so conditions with it off read in a third phase ("unmerged").
GROUPS = (
    ("bm25", dict(flags={"bm25": False})),
    ("rerank", dict(settings={"retrieval_rerank_enabled": False})),
    ("codex", dict(flags={"codex": False})),
    ("summaries", dict(settings={"turn_summaries_in_reads": False})),
    ("notes", dict(flags={"batch_summary": False}, harness={"no_fold"})),
    ("procedural", dict(flags={"procedural": False})),
    ("gate", dict(settings={"memory_source_gate_enabled": False, "memory_source_rescue_enabled": False},
                  harness={"always_retrieve"})),
    ("clusters", dict(flags={"cluster_restrict": False})),
    ("temporal", dict(flags={"timescope": False}, settings={"timescope_enabled": False})),
    ("maintenance", dict(harness={"no_slots", "unmerged"})),
)


def switched_off(names):
    """A full-ICE condition (neutral decay) with the named groups switched off."""
    flags, settings, harness = {}, {}, set()
    for name, switch in GROUPS:
        if name in names:
            flags.update(switch.get("flags", {}))
            settings.update(switch.get("settings", {}))
            harness |= switch.get("harness", set())
    return dict(overlay="neutral", arm="full", flags=flags, settings=settings, harness=sorted(harness),
                groups_off=sorted(names))


ORDER = [name for name, _ in GROUPS]
CONDITIONS.update({f"loo_{name}": switched_off({name}) for name in ORDER})          # leave one out
CONDITIONS.update({f"add_{k}_{ORDER[k - 1]}": switched_off(set(ORDER[k:]))          # groups 1..k on
                   for k in range(1, len(ORDER))})


def phase_of(spec):
    if spec.get("bare"):
        return "bare"
    return "unmerged" if "unmerged" in spec.get("harness", ()) else spec["overlay"]


def guard_sample(asks, n=GUARD_SAMPLE):
    """The summary-guard pair's questions: per type in proportion (at least one each),
    deterministic by hash, so the sample spreads over checkpoints and types."""
    questions = [a for a in asks if a["kind"] == "question"]
    by_type = {}
    for a in questions:
        by_type.setdefault(a.get("type"), []).append(a)
    picked = set()
    for kind, group in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        take = max(1, round(n * len(group) / max(len(questions), 1)))
        ranked = sorted(group, key=lambda a: hashlib.sha256(f"{POLICY}|guard|{a['id']}".encode()).hexdigest())
        picked |= {a["id"] for a in ranked[:take]}
    return picked


def applies(spec, ask, guard):
    subset = spec.get("subset")
    return (subset is None or (subset == "questions" and ask["kind"] == "question")
            or (subset == "guard" and ask["id"] in guard))
CONTEXT_HEADER = "=== RETRIEVED CONTEXT ==="           # prompt_assembler's evidence pair
CONTEXT_ACK = "Understood — I have the background context. What would you like to know?"


def unit_path(out, condition, ask_id):
    return Path(out) / "units" / condition / (hashlib.sha256(ask_id.encode()).hexdigest()[:24] + ".json")


def write_unit(path, unit):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(unit, ensure_ascii=False, default=str))
    tmp.replace(path)


def build_asks(split_dir, occurrences_path, controls_path=None):
    """The Z2 asks of one history: occurrences of Z2 families (never the sealed FINAL
    ones — read through split.load_z2) plus the no-memory controls."""
    from experiments.z2.split import load_z2
    z2 = load_z2(Path(split_dir) / "z2-families.json")
    keep = {f["id"] for f in z2["families"]}
    asks = []
    for o in json.loads(Path(occurrences_path).read_text()):
        if o["family"] in keep:
            asks.append(dict(id=f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}",
                             checkpoint_turn=o["checkpoint_turn"], question=o["question"], kind="question",
                             type=o["type"]))
    for c in z2.get("controls", []) + (json.loads(Path(controls_path).read_text()) if controls_path else []):
        asks.append(dict(id=c["id"], checkpoint_turn=c["checkpoint_turn"], question=c.get("question") or c["prompt"],
                         kind="control", type=c.get("kind")))
    return asks


def pending(asks, out, conditions=CONDITIONS):
    """{checkpoint: [ask, ...]} with at least one applicable condition unit still missing."""
    guard, todo = guard_sample(asks), {}
    for a in asks:
        if any(applies(spec, a, guard) and not unit_path(out, c, a["id"]).exists()
               for c, spec in conditions.items()):
            todo.setdefault(a["checkpoint_turn"], []).append(a)
    return dict(sorted(todo.items()))


def checkpoint_times(events_path):
    """{checkpoint turn: the user's return time} from the replay's event log."""
    from datetime import datetime
    out = {}
    for line in Path(events_path).read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "checkpoint":
            out[int(e["store"].split("-")[1].split(".")[0])] = datetime.fromisoformat(str(e["at"]))
    return out


def bare_messages(recent_stage, chosen, question):
    """recent_only's envelope with bare vector's chunks in ICE's evidence-pair position."""
    messages = copy.deepcopy(recent_stage["prompt_messages"])
    assert messages[-1]["role"] == "user"
    block = "\n\n".join(chunk.render() for chunk, _ in chosen)
    evidence = [{"role": "user", "content": f"{CONTEXT_HEADER}\n{block}"},
                {"role": "assistant", "content": CONTEXT_ACK}] if chosen else []
    return messages[:-1] + evidence + messages[-1:]


class Reader:
    """Loads the models once; reads one ask under one ICE condition."""

    def __init__(self):
        from src.api.config import settings
        from src.classifier.classifier import PyTorchClassifier
        from src.memory.embedder import get_embedder
        self.embedder = get_embedder()
        self.classifier = PyTorchClassifier(model_path=settings.classifier_model_path,
                                            schema_path=settings.label_schema_path)

    def read(self, db, cid, ask, arm, now, *, fixed_budget=None, overrides=None, flags=None, harness=()):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from scripts.z1 import production_parity as pp, seed_v3
        from src.api import memory_preparation as mp
        from src.api.config import settings
        from src.retrieval import orchestrator as orchestrator_module
        from src.retrieval.configurable_orchestrator import ConfigurableOrchestrator, SUPPORTED_FLAGS
        HybridRetrievalOrchestrator = orchestrator_module.HybridRetrievalOrchestrator
        budget = {}
        real = HybridRetrievalOrchestrator.retrieve
        real_budget = HybridRetrievalOrchestrator.set_budget_from_turn_count

        def retrieve(self_, *args, **kwargs):
            result = real(self_, *args, **kwargs)
            budget["max_retrieval_tokens"] = getattr(self_, "max_retrieval_tokens", None)
            return result

        def sized(self_, *args, **kwargs):
            result = real_budget(self_, *args, **kwargs)
            self_.max_retrieval_tokens = fixed_budget        # D11b: the memory budget, set exactly
            return result

        with ExitStack() as stack:
            stack.enter_context(seed_v3.probe_observation(db))
            stack.enter_context(patch.object(HybridRetrievalOrchestrator, "retrieve", retrieve))
            if fixed_budget:
                stack.enter_context(patch.object(HybridRetrievalOrchestrator, "set_budget_from_turn_count", sized))
            for key, value in (overrides or {}).items():
                if not hasattr(settings, key):
                    raise ValueError(f"unknown setting {key}: a switch that reaches nothing")
                stack.enter_context(patch.object(settings, key, value))
            if flags:
                unknown = set(flags) - SUPPORTED_FLAGS
                if unknown:
                    raise ValueError(f"unsupported orchestrator flags {sorted(unknown)}")

                class Flagged(ConfigurableOrchestrator):
                    def __init__(self, db_, embedder):
                        super().__init__(db_, embedder, overrides=dict(flags))
                # seed_v3 patches the source-time clock in the orchestrator class's module;
                # the inherited retrieval code reads its clock from the orchestrator module.
                Flagged.__module__ = orchestrator_module.__name__
                stack.enter_context(patch.object(orchestrator_module, "HybridRetrievalOrchestrator", Flagged))
            if "no_fold" in harness:          # group 5: the conversation fold is not shown
                stack.enter_context(patch.object(mp, "conversation_summary_block", lambda *_a, **_k: []))
            if "always_retrieve" in harness:  # group 7: the learned decision does not gate memory
                real_decide = pp.decide_memory_retrieval
                stack.enter_context(patch.object(pp, "decide_memory_retrieval", lambda *a, **k: SimpleNamespace(
                    retrieve=True, p_need_mem=real_decide(*a, **k).p_need_mem)))
            if "no_slots" in harness:         # group 11: reflection's slots are not shown
                real_prepare = pp.prepare
                stack.enter_context(patch.object(pp, "prepare", lambda *a, **k: real_prepare(
                    *a, **dict(k, memory_slots=[]))))
            pre = pp.build(db, ask["question"], cid, self.classifier, self.embedder, source_time=now)
            current = copy.deepcopy(pre)
            memory, generated, ranked, budgeted, notes = seed_v3.prepare_with_trace(
                db, current, self.classifier, arm=arm, source_time=now)
            stage = seed_v3.stage_record(current, memory, generated, ranked, budgeted, notes, include_prompt=True)
            # The budget ICE would give this question if it retrieved (memory_preparation
            # sets Long_Term_Memory before sizing it), so bare vector has one to match
            # even when ICE's gate skips memory; checked against the real one when not.
            would = HybridRetrievalOrchestrator(db, self.embedder)
            classification = copy.deepcopy(pre.classification)
            classification.context_reliance = "Long_Term_Memory"
            would.set_budget_from_turn_count(pre.turn_count, total_tokens=pre.total_tokens,
                                             classification=classification, total_budget=pre.total_budget)
        stage["retrieval_budget"] = budget.get("max_retrieval_tokens")
        stage["retrieval_budget_if_retrieved"] = would.max_retrieval_tokens
        stage["read_overrides"] = dict(fixed_budget=fixed_budget, settings=overrides or {}, flags=flags or {},
                                       harness=sorted(harness))
        if stage["retrieval_budget"] is not None and stage["retrieval_budget"] != would.max_retrieval_tokens:
            raise ValueError(f"budget replica {would.max_retrieval_tokens} != ICE's {stage['retrieval_budget']}")
        return stage


def unmerge_all(db):
    """Group 11 off: undo every applied entity merge on the restored copy, newest first
    (codex_ops.unmerge_entities reverses one exactly from its journal). The copy is
    restored afresh for the next checkpoint. Returns how many were undone."""
    from sqlalchemy import text
    from src.workers.codex_ops import unmerge_entities
    pending = [r[0] for r in db.execute(text(
        "SELECT id FROM codex_events WHERE event_type = 'entity_merged' ORDER BY timestamp DESC"))]
    undone = 0
    while pending:
        # Merges sharing a timestamp have no recorded order; a nested merge can only be
        # undone after the one on top of it, so retry the skipped until nothing moves.
        left = []
        for event in pending:
            result = unmerge_entities(db, event)
            if result.get("status") == "skipped":
                left.append((event, result))
            else:
                undone += 1
        if len(left) == len(pending):
            raise RuntimeError(f"{len(left)} entity merges could not be undone: {[r for _, r in left][:3]}")
        pending = [event for event, _ in left]
    return undone


def annotate_turns(db, stage):
    """Each shown fragment's source turns (episodic rows; facts via their sources), so
    S11's semantic credit can tell which fragments cite a gold turn without the store."""
    from sqlalchemy import text
    selected = stage.get("selected") or []
    rows = {f["source_row"] for f in selected if f.get("source_row")}
    facts = {i for f in selected for i in f.get("origin_facts") or []}
    turn_of = {}
    if rows:
        turn_of.update({str(i): int(str(k).rsplit("-", 1)[-1]) for i, k in db.execute(text(
            "SELECT id, idempotency_key FROM episodic_memory WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": list(rows)})})
    fact_turns = {}
    if facts:
        for fid, key in db.execute(text(
                "SELECT s.fact_id, e.idempotency_key FROM codex_fact_sources s JOIN episodic_memory e "
                "ON e.id = s.episodic_id WHERE s.fact_id = ANY(CAST(:ids AS uuid[]))"), {"ids": list(facts)}):
            fact_turns.setdefault(str(fid), set()).add(int(str(key).rsplit("-", 1)[-1]))
    for f in selected:
        turns = set()
        if f.get("source_row") in turn_of:
            turns.add(turn_of[f["source_row"]])
        for i in f.get("origin_facts") or []:
            turns |= fact_turns.get(i, set())
        f["source_turns"] = sorted(turns)
    return stage


def future_sources(db, stage, checkpoint):
    """Selected episodic rows from turns after the checkpoint (must be none)."""
    from sqlalchemy import text
    rows = [f["source_row"] for f in stage.get("selected") or [] if f.get("source_row")]
    if not rows:
        return []
    keys = db.execute(text("SELECT id, idempotency_key FROM episodic_memory WHERE id = ANY(CAST(:ids AS uuid[]))"),
                      {"ids": rows}).all()
    return [str(i) for i, k in keys if int(str(k).rsplit("-", 1)[-1]) > checkpoint]


def run(history, replay, asks, out, *, conditions=CONDITIONS, limit_checkpoints=None):
    """Child-side (DATABASE_URL already points at the replay database)."""
    from scripts.z1 import snapshot
    from sqlalchemy import text
    from src.api.db import SessionLocal
    from experiments.z2 import decay_overlay as O
    from experiments.v3_repair.bare_vector import Index, chunk_turns, select
    from experiments.z2.ledger import load_history
    from src.memory import tokens
    replay, out = Path(replay), Path(out)
    times = checkpoint_times(replay / "events.jsonl")
    todo = pending(asks, out, conditions)
    if limit_checkpoints:
        todo = dict(list(todo.items())[:limit_checkpoints])
    reader = Reader()
    originals = load_history(history)
    encode = lambda texts: reader.embedder.encode(texts, convert_to_tensor=False, batch_size=8,
                                                  show_progress_bar=False)
    index = Index(encode)
    failures = []
    for cp, group in todo.items():
        started = time.time()
        if cp not in times:
            failures.append(dict(checkpoint=cp, reason="no checkpoint store in the replay"))
            continue
        with patch.object(snapshot, "SNAPDIR", replay / "checkpoints"):
            if snapshot.restore(f"cp-{cp}"):
                failures.append(dict(checkpoint=cp, reason="restore failed"))
                continue
        events = O.load_events(replay / "events.jsonl", f"cp-{cp}")
        with SessionLocal() as db:
            episodic, cold, facts = O.read_store(db)
            check = O.validate(O.trajectory(events, episodic, cold, facts, {}), episodic, cold, facts)
            cid = str(db.execute(text("SELECT id FROM conversations")).scalar_one())
        phases = ("default", "neutral", "unmerged")
        if check["mismatches"]:
            # The live store can still be read; nothing that needs the overlay can.
            failures.append(dict(checkpoint=cp, reason="default decay overlay does not reproduce the store; "
                                 "neutral-decay conditions refused here", detail=check))
            phases = ("default",)
        now = times[cp]
        guard = guard_sample(asks)
        full_budget = {}
        for phase in phases:
            if phase == "neutral":
                with SessionLocal() as db:
                    target = O.trajectory(events, episodic, cold, facts, O.PROFILES["neutral"])
                    applied = O.apply(db, target, episodic, cold)
                if applied["refused"]:
                    failures.append(dict(checkpoint=cp, reason="neutral overlay refused rows", detail=applied))
            if phase == "unmerged":
                if not any(phase_of(spec) == "unmerged" and applies(spec, a, guard)
                           and not unit_path(out, n, a["id"]).exists()
                           for n, spec in conditions.items() for a in group):
                    continue
                with SessionLocal() as db:
                    undone = unmerge_all(db)
                print(json.dumps(dict(checkpoint=cp, entity_merges_undone=undone)), flush=True)
            for name, spec in conditions.items():
                if phase_of(spec) != phase:
                    continue
                for ask in group:
                    if not applies(spec, ask, guard):
                        continue
                    path = unit_path(out, name, ask["id"])
                    if path.exists():
                        if name == "full_default":
                            full_budget[ask["id"]] = json.loads(path.read_text())["stage"].get(
                                "retrieval_budget_if_retrieved")
                        continue
                    with SessionLocal() as db:
                        stage = reader.read(db, cid, ask, spec["arm"], now, fixed_budget=spec.get("budget"),
                                            overrides=spec.get("settings"), flags=spec.get("flags"),
                                            harness=set(spec.get("harness", ())) - {"unmerged"})
                        future = future_sources(db, stage, cp)
                        annotate_turns(db, stage)
                    if future:
                        raise ValueError(f"{ask['id']}/{name}: future evidence {future}")
                    if name == "full_default":          # the budget does not depend on decay
                        full_budget[ask["id"]] = stage.get("retrieval_budget_if_retrieved")
                    write_unit(path, dict(policy=POLICY, history=history, condition=name, ask=ask,
                                          checkpoint=cp, question_time=now, decay_overlay=spec["overlay"],
                                          stage=stage))
        if "neutral" in phases:                     # bare vector needs recent_only's envelope
            for name, spec in conditions.items():
                if not spec.get("bare"):
                    continue
                for ask in group:
                    path = unit_path(out, name, ask["id"])
                    if not applies(spec, ask, guard) or path.exists():
                        continue
                    recent = json.loads(unit_path(out, "recent_only", ask["id"]).read_text())["stage"]
                    recent_from = cp - sum(1 for m in recent["prompt_messages"][:-1] if m["role"] == "assistant") + 1
                    chunks = chunk_turns([r for r in originals if r["turn_number"] < recent_from], None)
                    budget = spec.get("budget") or full_budget.get(ask["id"])
                    if not budget:
                        raise ValueError(f"{ask['id']}: full ICE recorded no retrieval budget to match")
                    chosen = select(index, ask["question"], chunks, budget, None, tokens.count)
                    write_unit(path, dict(policy=POLICY, history=history, condition=name, ask=ask,
                                          checkpoint=cp, question_time=now, decay_overlay="none",
                                          stage=dict(prompt_messages=bare_messages(recent, chosen, ask["question"]),
                                                     retrieval_budget=budget,
                                                     selected=[dict(turn=c.turn, role=c.role, score=sc,
                                                                    tokens=tokens.count(c.render()))
                                                               for c, sc in chosen],
                                                     selected_tokens=sum(tokens.count(c.render()) for c, _ in chosen))))
        print(json.dumps(dict(checkpoint=cp, asks=len(group), seconds=round(time.time() - started))), flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=1, default=str))
    # Spec §5 item 6 on every run: what each leave-one-out switch changed vs full.
    (out / "switch-report.json").write_text(json.dumps(switch_report(out, asks), indent=1))
    return failures


def switch_report(out, asks):
    """Spec §5 item 6: per leave-one-out condition, what differs from `full` across the
    asks — selected tokens per retrieval leg, prompt block tokens, the retrieve decision.
    A switch that changes nothing, or changes a part it does not own, shows up here."""
    import collections
    report = {}
    for name in [n for n in CONDITIONS if n.startswith("loo_")]:
        legs, blocks, decisions, compared, identical = collections.Counter(), collections.Counter(), 0, 0, 0
        for ask in asks:
            a, b = unit_path(out, "full", ask["id"]), unit_path(out, name, ask["id"])
            if not (a.exists() and b.exists()):
                continue
            full, off = json.loads(a.read_text())["stage"], json.loads(b.read_text())["stage"]
            compared += 1
            identical += full["prompt_messages"] == off["prompt_messages"]
            for st, sign in ((full, -1), (off, 1)):
                for f in st.get("selected") or []:
                    legs[f.get("leg") or f.get("type")] += sign * (f.get("tokens") or 0)
                for block, n in ((st.get("context_ledger") or {}).get("blocks") or {}).items():
                    blocks[block] += sign * n
            decisions += (full.get("gate") or {}).get("final_retrieve") != (off.get("gate") or {}).get("final_retrieve")
        report[name] = dict(asks=compared, identical_prompts=identical, retrieve_decisions_changed=decisions,
                            leg_token_delta={k: v for k, v in legs.items() if v},
                            block_token_delta={k: v for k, v in blocks.items() if v})
    return report


def progress(out, asks_path, conditions=CONDITIONS):
    """(done, total) units for the runner's meter, from files only."""
    if not Path(asks_path).exists():
        return 0, None
    asks = json.loads(Path(asks_path).read_text())
    guard = guard_sample(asks)
    wanted = [(c, a) for a in asks for c, spec in conditions.items() if applies(spec, a, guard)]
    return sum(1 for c, a in wanted if unit_path(out, c, a["id"]).exists()), len(wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--asks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-checkpoints", type=int)
    parser.add_argument("--switch-report", action="store_true", help="print §5 item 6 and exit")
    args = parser.parse_args()
    if args.switch_report:
        print(json.dumps(switch_report(args.out, json.loads(args.asks.read_text())), indent=1))
        return 0
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("read-pass output is private: keep it under logs/")
    if os.environ.get("ICE_TEST_DATABASE") != args.database:
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        url = make_url(live.database_url).set(database=args.database).render_as_string(hide_password=False)
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database))
    asks = json.loads(args.asks.read_text())
    failures = run(args.history, args.replay, asks, args.out, limit_checkpoints=args.limit_checkpoints)
    print(json.dumps(dict(failures=len(failures))))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
