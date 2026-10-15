# Handoff — ICE v3, 2026-10-15 (b) (Claude Code)

**State, not a queue.** ROADMAP owns work; the v2 paper remains frozen.

## Told → did

**Previous handoff (2026-10-15) told:** do the pre-Z2 short list first — G27 (wrong
background model), G12 (timeouts), G4 + G32 (explicit Ollama control), G68 check on
C — then build Z2 in spec order (S0 → S12).

**This session was told:** the pre-Z2 list and Z2 prep; mid-session the maintainer
added a design audit ("things that look correct but are badly designed") and then
**re-ordered the work: non-codex fixes → full codex rework → Z2.** Z2 build did not
start, by that decision. **Did (all local commits; push freeze):**

- **G27** `3a12359` — `background_model_name` defaults to `gemma4:e4b`.
- **G32/G4/G12** `d9e3707` — background calls over native `/api/chat`, context sized
  to the counted input, server-side cuts fail loudly (both Ollama doors cut an
  oversized input silently on 0.32.13 and the model invented the cut fact; TRAPS 130).
- **Summary claims v3** `692033c` ([spec](specs/SUMMARY_CLAIMS_V3.md)) — quoted
  `{sources, quote, speaker, text}` claims, enforced schema, per-claim acceptance,
  pieces, output cap 1.3 × input; fold no longer freezes on an oversized turn.
  Qualified: reader-usable pieces 9 → 28 of 31.
- **Audit fixes:** A4 reranker best-window `6255e2a`; A10 procedures = answer
  preferences + topic triggers `cd47c52`; A11 bounded memory-pressure term `f200699`;
  A12 cluster bonus floor + automatic cluster scoping off `22d8f1d`.
- **Stale tests repaired:** settings-freeze row retired with proof `ae13d3e`;
  `test_turn_density` 32/32 `f69a81d`; `test_documents` 52/53 `2cfda93`.
- **Docs:** roadmap order + A6/A7, provenance, architecture, inventory, MODELS,
  TRAPS 130–135.

## Current measured state (v3, development data; not a score of record)

The machinery mostly did not reach prompts before these fixes: on the 2026-10-14
reads only 6/920 selected turns arrived as summaries; the graph leg produced 42
candidates over 67 questions; procedures were one generic note. Summaries and
procedures are fixed but **not re-measured end to end** (A6 — counted in Z2 S0's
slice). **The codex is broken by design, not by a bug:** grounding hides 70–89% of
live edges (confidence 0.35 < floor 0.5); a "Source excerpt" renders the whole
message (median 3,488 chars for a 52-char sentence), 34% duplicating a turn already
in the prompt; reworded re-observations become new edges (2 strengthened / 1,356),
so changes never meet (1 expired); `I`/`you` facts are dropped; extraction splits a
big turn at every heading (~130 chunks, 12–16 min per C turn). Numbers:
[PROVENANCE 2026-10-15](PROVENANCE.md).

## Position and next work

**Next session: the codex rework — remove and rewrite, not patch** (maintainer).
Target design: a relation is a term (not only familial); the triple carries validity
(valid from / until), strength and confidence; the attached source sentence(s) are
what the answering model sees — close facts, not walls of text. It absorbs G68 / A8
(extraction chunking), A1 (512-token check bound on graph checks), A3, A9, B1–B6.
**Read first:** [codex prior-art review](research/2026-10-15-codex-prior-art.md) — Graphiti and
Mem0 source read: both show the answering model a short, self-contained fact sentence and find
duplicates/changes by entity + fact similarity. Proposal: a v3 summary claim (`{sources, quote,
speaker, text}`, quote-checked and NLI-verified) IS the fact; one write pass serves summaries and
facts; stay on Postgres; port ideas, not the libraries (graph server, cloud-class models, no
quote verification). LangGraph is agent orchestration, not memory.
Then **Z2**, unchanged spec, with two notes in ROADMAP CURRENT POSITION: A6 (count
what reaches prompts on S0's 50-pair slice) and A7 (report the budget ICE actually
used — ~28% of the stated one, by design).

Decided, do not re-derive: order fixes → codex rework → Z2; summary storage is merged
on the turn row (option a); every Ollama-facing call is native (OpenAI-compatible
door only for cloud, dedicated vLLM, or `background_transport=openai`); automatic
cluster scoping stays off unless recall is re-measured.

Z2 spec gained **D4b** (maintainer): no-memory control prompts at each checkpoint
(general knowledge, self-contained tasks, small talk, recent-window follow-ups),
outside the D4 quotas — retrieve rate, memory tokens injected, answer quality vs
recent-only; share decided at the Z2 build. CLAUDE.md: shared-mode line corrected (`5eef60e`)
and TRAPS is read by titles, not in full (`a231e40`), both maintainer-approved. Z2's
USER-REQUIRED items are unchanged.

Decided: commit `cd47c52`'s message (quotes a stored procedure note) stays as is — maintainer,
2026-10-15; the docstring copy was neutralised in `f875ad3`. All session commits are unpushed.

## Validation and propagation

Quick group 636 passed; db group 292/292 (disposable database, 16 min) at close; standalone summary/clustering/scoping/decision suites pass.
`test_documents` keeps 1 failing check (codex entities from stub triplets without a
source sentence) — left to the codex rework. Known costs: v3 summaries ~2.2×
background time; long-passage rerank ~1.3 s each. Alembic head unchanged. Private
evidence under `logs/` (gitignored). Not pushed (experiment-phase freeze).
