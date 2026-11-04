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
    # FINAL (Z3 D4): no memory system — the newest original messages packed into a
    # LONG_CONTEXT_PROMPT_TOKENS prompt, newest first.
    "long_context": dict(overlay="none", bare=True, long=True),
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


def build_final_asks(split_dir, occurrences_path):
    """FINAL's asks of a Z2 history: occurrences of the sealed FINAL families only (opened
    through split.load_final, which checks the seal) plus the same no-memory controls. Families
    listed in <split_dir>/../answer-key-defects.json are left out (Z3 2026-11-04: never-said
    questions the conversation does answer, found once the check read whole messages)."""
    from experiments.z2.split import load_final, load_z2
    defects = Path(split_dir).parent / "answer-key-defects.json"
    dropped = set(json.loads(defects.read_text())["drop"]) if defects.exists() else set()
    keep = {f["id"] for f in load_final(split_dir)["families"]} - dropped
    asks = [dict(id=f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}", checkpoint_turn=o["checkpoint_turn"],
                 question=o["question"], kind="question", type=o["type"])
            for o in json.loads(Path(occurrences_path).read_text()) if o["family"] in keep]
    for c in load_z2(Path(split_dir) / "z2-families.json").get("controls", []):
        asks.append(dict(id=c["id"], checkpoint_turn=c["checkpoint_turn"], question=c.get("question") or c["prompt"],
                         kind="control", type=c.get("kind")))
    return asks


# Parts not in FINAL's parts test: clusters (audit A12, below) and what Z3 switched off by default
# (turn summaries in reads, R4; the fold and batch summaries, procedures, D9).
PARTS_OUT = ("clusters", "summaries", "notes", "procedural")

# FINAL condition sets (FINAL_experiments.md §2.1; Z3 D4, D5, D10). "z2" = everything Z2 read.
PRESETS = {
    # the main experiment: ICE frozen, as it runs, against the baselines (maintainer, 2026-10-28) —
    # `full` (decay neutral) and ICE's own vector leg alone belong to S9's parts test only
    "final_main": ["full_default", "recent_only", "bare_vector", "long_context"],
    # Clusters are not in FINAL's parts test. Reason (audit A12, 2026-10-15, commit 22d8f1d):
    # automatic scoping to the top clusters, measured on meaningful clusters (38, largest 44%),
    # scoped out 19 of 108 gold turns across 67 reviewed questions, so it ships OFF
    # (`retrieval_auto_cluster_scope=False`; an explicit user scope still applies). The parts test's
    # cluster switch therefore changes no read (Z2 F18: 0 of 208 prompts on B). Reported as
    # "automatic scoping off by default after audit A12 — not tested" (Z3 D14 item 5).
    # D9 rule 4 (Z3, 2026-11-04): a part the freeze switches off leaves the parts test — off by
    # default, its leave-one-out and build-up step would read exactly what the step before does.
    "final_ablation": ["full", "recent_only"] + [f"loo_{n}" for n in ORDER if n not in PARTS_OUT],
    # warm_vector is the build-up's step 0 (ICE's vector leg alone; Z2 F13 measured +10.4 for the
    # first step from it) — the pre-registration's "warm vector appears only inside S9's parts test"
    "final_buildup": ["full", "recent_only", "warm_vector"] + [f"add_{k}_{ORDER[k - 1]}" for k in range(1, len(ORDER))
                                                if ORDER[k - 1] not in PARTS_OUT],
    # Decay (Z2 F18): live (`full_default`) vs neutral (`full`) on the longest histories, split by
    # the evidence's age and by update questions — S9 has both through its parts test; S4 (353 days)
    # adds `full` here.
    "final_decay": ["full"],
    # LoCoMo (an extra table, prereg r2 §4): every history fits the long-context window; recent_only
    # is the envelope the memory systems' prompts reuse
    # LoCoMo: its whole conversations fit 256K, so long_context is the field's "full context"; ICE
    # also at a 2K-token memory budget, the size the published memory systems hand the model
    # (their top 10 memories), so the comparison is not won by reading most of a short chat
    "final_locomo": ["full_default", "full_b2000", "recent_only", "bare_vector", "long_context"],
}


# Z2 findings retests (experiments/z2/results/RETESTS.md): read only when named, never part of
# "z2", so a finished Z2 read stage stays finished. Questions only; full ICE as it runs plus one
# change each. Read into a fresh --out (never Z2's own units folder).
RETEST_CONDITIONS = {
    **{f"retest_cand{n}": dict(overlay="default", arm="full", subset="questions",
                               settings={"retrieval_rerank_candidates": n}) for n in (64, 128, 256, 512)},
    **{f"retest_cand{n}_keep": dict(overlay="default", arm="full", subset="questions",
                                    settings={"retrieval_rerank_candidates": n,
                                              "retrieval_rerank_keep_unranked": True}) for n in (64, 128)},
    "retest_cand256_batch16": dict(overlay="default", arm="full", subset="questions",
                                   settings={"retrieval_rerank_candidates": 256, "retrieval_rerank_batch_size": 16}),
    "retest_conversation_order": dict(overlay="default", arm="full", subset="questions",
                                      settings={"retrieval_evidence_order": "conversation"}),
    "retest_no_summaries": dict(overlay="default", arm="full", subset="questions",
                                settings={"turn_summaries_in_reads": False}),
    "retest_policy_text": dict(overlay="default", arm="full", subset="questions",
                               settings={"retrieval_rerank_representation": "policy"}),
    "retest_originals_first": dict(overlay="default", arm="full", subset="questions",
                                   settings={"retrieval_rerank_representation": "policy",
                                             "representation_summary_first_intents": False}),
    **{f"retest_original_margin{m}": dict(overlay="default", arm="full", subset="questions",
                                          settings={"retrieval_rerank_original_margin": float(m)}) for m in (1, 2)},
}
PRESETS["retest_summaries"] = ["retest_cand64", "retest_no_summaries", "retest_original_margin1", "retest_original_margin2",
                               "retest_policy_text", "retest_originals_first"]
# Z2 R2 follow-up (2026-10-29): at 256 candidates 90.4% of B's cited turns reach the prompt; of the rest
# 5.4% are never found by any leg (vague questions) and 4.2% are ranked but over the budget. Variants:
# wider legs for the never-found, and the summary settings' coverage cost at 256.
_WIDE = {"retrieval_rerank_candidates": 512, "retrieval_bm25_candidate_limit": 300,
         "retrieval_vector_candidate_limit": 300, "codex_fact_candidate_limit": 192}
_ORIGINALS = {"retrieval_rerank_representation": "policy", "representation_summary_first_intents": False}
RETEST_CONDITIONS.update({
    "retest_c256_wide": dict(overlay="default", arm="full", subset="questions", settings=dict(_WIDE)),
    "retest_c256_originals_first": dict(overlay="default", arm="full", subset="questions",
                                        settings={"retrieval_rerank_candidates": 256, **_ORIGINALS}),
    "retest_c256_policy": dict(overlay="default", arm="full", subset="questions",
                               settings={"retrieval_rerank_candidates": 256, "retrieval_rerank_representation": "policy"}),
    "retest_c256_wide_originals": dict(overlay="default", arm="full", subset="questions",
                                       settings={**_WIDE, **_ORIGINALS}),
})
# At the chosen 256 candidates (2026-10-30): the order and summary comparisons the answers decide.
_C256 = {"retrieval_rerank_candidates": 256}
RETEST_CONDITIONS.update({
    "retest_c256_order": dict(overlay="default", arm="full", subset="questions",
                              settings={**_C256, "retrieval_evidence_order": "conversation"}),
    "retest_c256_no_summaries": dict(overlay="default", arm="full", subset="questions",
                                     settings={**_C256, "turn_summaries_in_reads": False}),
})
PRESETS["retest_c256_answered"] = ["retest_cand256", "retest_c256_order", "retest_c256_originals_first",
                                   "retest_c256_policy", "retest_c256_wide_originals", "retest_c256_no_summaries"]
# The reranker's inputs (Z2 F22, 2026-10-31), at the 256 default: the user's recent turns before the question,
# the topical instruction, and both.
RETEST_CONDITIONS.update({
    "retest_c256_rq_context": dict(overlay="default", arm="full", subset="questions",
                                   settings={**_C256, "retrieval_rerank_query_context": "recent"}),
    "retest_c256_rq_topical": dict(overlay="default", arm="full", subset="questions",
                                   settings={**_C256, "retrieval_rerank_instruction": "topical"}),
    "retest_c256_rq_both": dict(overlay="default", arm="full", subset="questions",
                                settings={**_C256, "retrieval_rerank_query_context": "recent",
                                          "retrieval_rerank_instruction": "topical"}),
})
PRESETS["retest_candidates"] = [k for k in RETEST_CONDITIONS if k.startswith("retest_cand")]

# R7 (RETESTS): the frozen candidate — full ICE as it runs, plus the values Z3 judging run 2 chose
# for the switches built from Z2's findings, and nothing else (maintainer, 2026-11-03: R7 sets
# existing switches only; anything new it shows is reported, never fixed before FINAL). The values
# live in R7_SETTINGS (tracked), written once run 2 is graded; until then r7_frozen does not exist
# and R7 waits. Every ask — questions and no-memory controls — beside no memory and plain search.
R7_SETTINGS = Path(os.environ.get("ICE_R7_SETTINGS") or ROOT / "experiments/z2/results/r7_settings.json")  # env: a trial
R7_SWITCHES = frozenset({
    "retrieval_rerank_candidates", "retrieval_rerank_keep_unranked", "retrieval_evidence_order",
    "retrieval_rerank_original_margin", "turn_summaries_in_reads", "retrieval_rerank_representation",
    "representation_summary_first_intents", "memory_source_rescue_mode", "memory_source_rescue_min_score",
    "retrieval_rerank_query_context", "retrieval_rerank_instruction"})


# D9 parts R7 may switch off, by the same switches Z2's ablations measured them off with (group 5:
# the fold and the batch-summary leg; group 6: procedures) — production gets its own off switch
# at the freeze, checked equal by query (maintainer, 2026-11-04).
R7_GROUPS_OFF = frozenset({"notes", "procedural"})


def r7_condition(path=R7_SETTINGS):
    """The frozen candidate's read spec, or None before its settings are chosen. A setting outside
    R7_SWITCHES, or a part outside R7_GROUPS_OFF, is refused: R7 chooses among existing switches,
    it does not try new ones."""
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    chosen, off = data["settings"], set(data.get("off", ()))
    extra = sorted(set(chosen) - R7_SWITCHES) + sorted(off - R7_GROUPS_OFF)
    if extra:
        raise ValueError(f"R7 sets existing switches only; not one: {extra}")
    spec = dict(overlay="default", arm="full", settings=dict(chosen))
    if off:
        parts = switched_off(off)
        spec.update(flags=parts["flags"], harness=parts["harness"], groups_off=parts["groups_off"])
    return spec


def r7_conditions():
    """{name: spec} of R7's three conditions, read fresh (a waiting runner sees the settings appear)."""
    frozen = r7_condition()
    return None if frozen is None else {"r7_frozen": frozen, "recent_only": CONDITIONS["recent_only"],
                                        "bare_vector": CONDITIONS["bare_vector"]}


if r7_condition():
    RETEST_CONDITIONS["r7_frozen"] = r7_condition()
PRESETS["r7"] = ["r7_frozen", "recent_only", "bare_vector"]
BUDGET_SOURCES = ("full_default", "r7_frozen")    # plain search's budget is matched to ICE's own


def select_conditions(spec):
    """'z2', a preset name, presets joined with '+', or comma-separated condition names."""
    if not spec or spec == "z2":
        return dict(CONDITIONS)
    names = []
    for part in spec.split("+"):
        names += PRESETS[part] if part in PRESETS else part.split(",")
    known = {**CONDITIONS, **RETEST_CONDITIONS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    return {n: known[n] for n in dict.fromkeys(names)}


def pending(asks, out, conditions=CONDITIONS):
    """{checkpoint: [ask, ...]} with at least one applicable condition unit still missing."""
    guard, todo = guard_sample(asks), {}
    for a in asks:
        if any(applies(spec, a, guard) and not unit_path(out, c, a["id"]).exists()
               for c, spec in conditions.items()):
            todo.setdefault(a["checkpoint_turn"], []).append(a)
    return dict(sorted(todo.items()))


def complete_checkpoints(out, asks_path, conditions=CONDITIONS):
    """Checkpoints whose every expected unit (each applicable condition × each ask) exists — the ones a
    streaming cloud runner may answer (units are written atomically, a checkpoint at a time)."""
    asks = json.loads(Path(asks_path).read_text())
    guard, need = guard_sample(asks), {}
    for a in asks:
        paths = [unit_path(out, c, a["id"]) for c, spec in conditions.items() if applies(spec, a, guard)]
        need.setdefault(str(a["checkpoint_turn"]), []).extend(paths)
    return {cp for cp, paths in need.items() if paths and all(p.exists() for p in paths)}


def checkpoint_times(events_path):
    """{checkpoint turn: the user's return time} from the replay's event log."""
    from datetime import datetime
    out = {}
    for line in Path(events_path).read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "checkpoint":
            out[int(e["store"].split("-")[1].split(".")[0])] = datetime.fromisoformat(str(e["at"]))
    return out


# 256K (maintainer, 2026-10-28): a window many providers offer (a quarter of the answerer's 1M);
# was 37,500 — the largest prompt ICE's budget produced in Z2 — while the answerer was local.
LONG_CONTEXT_PROMPT_TOKENS = 256_000


def long_context_chunks(chunks, budget, count):
    """Newest-first original messages that fit *budget* tokens, returned oldest first."""
    picked, used = [], 0
    for chunk in sorted(chunks, key=lambda c: (c.turn, c.role != "user"), reverse=True):
        cost = count(chunk.render())
        if used + cost > budget:
            break
        picked.append((chunk, None))
        used += cost
    return list(reversed(picked)), used


def bare_messages(recent_stage, chosen, question):
    """recent_only's envelope with bare vector's chunks in ICE's evidence-pair position."""
    messages = copy.deepcopy(recent_stage["prompt_messages"])
    assert messages[-1]["role"] == "user"
    block = "\n\n".join(chunk.render() for chunk, _ in chosen)
    evidence = [{"role": "user", "content": f"{CONTEXT_HEADER}\n{block}"},
                {"role": "assistant", "content": CONTEXT_ACK}] if chosen else []
    return messages[:-1] + evidence + messages[-1:]


def time_order_messages(recent_stage, chosen, question):
    """Long context and full history (maintainer, 2026-10-28): the conversation in time order —
    recent_only's system message, the older history as one block, recent_only's recent turns,
    the question. Every question at a checkpoint then shares the long start, which the provider
    serves from its cache (bare vector keeps ICE's evidence position: bare_messages)."""
    messages = copy.deepcopy(recent_stage["prompt_messages"])
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    block = "\n\n".join(chunk.render() for chunk, _ in chosen)
    evidence = [{"role": "user", "content": f"{CONTEXT_HEADER}\n{block}"},
                {"role": "assistant", "content": CONTEXT_ACK}] if chosen else []
    return messages[:1] + evidence + messages[1:]


def shared_long_context(plans, count, shift=0):
    """{ask id: chosen chunks} for one checkpoint's questions: each starts at the same chunk — the
    latest of their own newest-first fills — so each still fits its budget and all share a start.
    *plans* maps ask id -> (its older chunks oldest first, its room); older chunks of one history
    are prefixes of each other, so a position names the same chunk in every list."""
    starts = []
    for chunks, room in plans.values():
        fill, _ = long_context_chunks(chunks, max(0, room), count)
        starts.append(len(chunks) - len(fill))
    start = max(starts, default=0) + shift
    return {ask: [(c, None) for c in chunks[start:]] for ask, (chunks, _) in plans.items()}


def long_context_stages(history, checkpoint, recents, originals, count):
    """{ask id: long_context stage} for one checkpoint's questions, from each one's recent_only
    stage: time order, one shared start, one cache session per checkpoint."""
    from experiments.v3_repair.bare_vector import chunk_turns
    plans = {}
    for ask_id, recent in recents.items():
        recent_from = checkpoint - sum(1 for m in recent["prompt_messages"][:-1] if m["role"] == "assistant") + 1
        chunks = sorted(chunk_turns([r for r in originals if r["turn_number"] < recent_from], None),
                        key=lambda c: (c.turn, c.role != "user"))
        plans[ask_id] = (chunks, LONG_CONTEXT_PROMPT_TOKENS - sum(count(m["content"]) for m in recent["prompt_messages"]))
    # The chunks' own counts leave out the block's header, its reply and the joins (B, 2026-10-28:
    # 256,274 tokens at most): the shared start moves on a chunk at a time until every prompt fits.
    shift = 0
    while True:
        shared = shared_long_context(plans, count, shift)
        stages = {ask_id: dict(prompt_messages=time_order_messages(recents[ask_id], chosen, None),
                               long_context_budget=plans[ask_id][1],
                               selected_tokens=sum(count(c.render()) for c, _ in chosen),
                               cache_session=f"ice-final-long-{history}-{checkpoint}",
                               selected=[dict(turn=c.turn, role=c.role, source_turns=[c.turn]) for c, _ in chosen])
                  for ask_id, chosen in shared.items()}
        for s in stages.values():                       # counted once here, so the report need not
            s["prompt_tokens"] = sum(count(m["content"]) for m in s["prompt_messages"])   # re-tokenize ~256K a unit
        if all(s["prompt_tokens"] <= LONG_CONTEXT_PROMPT_TOKENS or not s["selected"] for s in stages.values()):
            return stages
        shift += 1


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


def encode_backing_off(embedder, texts, batch_size=8):
    """Bare-vector chunk embeddings, batch 8; a batch that runs out of GPU memory
    (whole-message chunks: a few very long messages together) is retried at half the
    size, down to one text. Same vectors either way, up to float rounding."""
    import torch
    out, size, i, small_until = [], batch_size, 0, 0
    while i < len(texts):
        batch = texts[i:i + size]
        try:
            out.extend(embedder.encode(batch, convert_to_tensor=False, batch_size=size,
                                       show_progress_bar=False))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if size == 1:
                raise
            small_until = max(small_until, i + len(batch))     # keep small past the failed batch
            size = max(1, size // 2)
            print(f"[bare_vector] GPU out of memory — retrying at batch {size}", flush=True)
            continue
        i += len(batch)
        if i >= small_until:
            size = batch_size
    return out


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
    index = Index(lambda texts: encode_backing_off(reader.embedder, texts))
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
                        if name in BUDGET_SOURCES:
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
                    if name in BUDGET_SOURCES:          # the budget does not depend on decay
                        full_budget[ask["id"]] = stage.get("retrieval_budget_if_retrieved")
                    write_unit(path, dict(policy=POLICY, history=history, condition=name, ask=ask,
                                          checkpoint=cp, question_time=now, decay_overlay=spec["overlay"],
                                          stage=stage))
        if "neutral" in phases:                     # bare vector needs recent_only's envelope
            for name, spec in conditions.items():
                if not spec.get("bare"):
                    continue
                if spec.get("long"):                    # every applicable question sets the shared start
                    recents = {ask["id"]: json.loads(unit_path(out, "recent_only", ask["id"]).read_text())["stage"]
                               for ask in group if applies(spec, ask, guard)}
                    built = long_context_stages(history, cp, recents, originals, tokens.count)
                    for ask in group:
                        path = unit_path(out, name, ask["id"])
                        if ask["id"] in built and not path.exists():
                            write_unit(path, dict(policy=POLICY, history=history, condition=name, ask=ask,
                                                  checkpoint=cp, question_time=now, decay_overlay="none",
                                                  stage=built[ask["id"]]))
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
                                                     # source_turns and text: context precision judges
                                                     # plain search's pieces as it judges ICE's (Z2 F27)
                                                     selected=[dict(turn=c.turn, role=c.role, score=sc,
                                                                    tokens=tokens.count(c.render()),
                                                                    source_turns=[c.turn], text=c.render())
                                                               for c, sc in chosen],
                                                     selected_tokens=sum(tokens.count(c.render()) for c, _ in chosen))))
        print(json.dumps(dict(checkpoint=cp, asks=len(group), seconds=round(time.time() - started))), flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)       # a read with nothing to do writes an empty record
    (Path(out) / "failures.json").write_text(json.dumps(failures, indent=1, default=str))
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
    parser.add_argument("--conditions", default="z2", help="z2 | final_main | final_ablation | final_buildup | a+b | names")
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
    failures = run(args.history, args.replay, asks, args.out, conditions=select_conditions(args.conditions),
                   limit_checkpoints=args.limit_checkpoints)
    print(json.dumps(dict(failures=len(failures))))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
