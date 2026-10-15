# CODEX_V4 — the codex as verified facts (remove and rewrite)

**Version: v3** (the system on `main`; "v4" names the codex design, not a system version).
**Status: APPROVED by the maintainer 2026-10-15 (c)**, with D11 defaults and D12.
Assumes decided specs: `SUMMARY_CLAIMS_V3.md` (quoted claims, pieces, per-claim
acceptance), `G32_ollama_transport.md` (native background calls),
`C10_C11_deletion_chat_control.md` (deletion expires + journals, never hard-deletes graph
rows), `T_temporal.md` (bi-temporal validity, `valid_at(T)`), `G50_entity_consolidation.md`
(`merge_key`, maintenance-agent merge tiers). Supersedes for conversation text:
`G34_relations.md`, `G45_open_relation_vocabulary.md`, `G51_containment_linking.md`,
`G63_extractor_decision.md` (they stay as history; the code they govern is removed).
Research input: [2026-10-15 prior-art review](../research/2026-10-15-codex-prior-art.md)
and §8 below.

## 0. Why (measured, PROVENANCE 2026-10-15, development store unless stated)

| Problem today | Measured | Audit id |
|---|---|---|
| NER-token grounding marks most edges 0.35 < read floor 0.5 → never read | 70–89% of live edges hidden | A9 |
| The model is shown the **whole source message** | median 3,488 chars for a 52-char sentence; 30/89 duplicate a turn already in the prompt | B1–B4 |
| A reworded re-observation is a new edge; conflicts need the same subject+relation | 1,356 added / 2 strengthened / 1 expired | B5–B6 |
| `I`/`you` facts refused at write (after paying for NLI + LLM review) | 168 refused writes | A3 |
| Extraction splits at every Markdown heading | ~130 chunks, 12–16 min per large C turn | A8 / G68 |
| Relation check reads a paragraph through a 512-token NLI | 7% of pairs over the bound | A1 |

Not a bug list — the design centres on the triple and the raw message; every system
that works (Graphiti/Zep, Mem0, Hindsight, SodaMem) centres on a **verified,
self-contained fact sentence** and uses structure only to find duplicates and changes.

## 1. Decisions

**D1 — The unit is a fact.** A fact is one self-contained sentence written by the
background model, with the exact quote it rests on, who said it, when, and whether it is
still current. The **sentence is what the answering model sees**; the quote stays attached
for provenance ("where did I say that"). Rendered:
`[Fact; user said; 2025-01-15; current] The user said they moved to Lisbon in March.`

**D2 — Relationships (how they change).** *Today:* an edge `entity --relation--> entity`;
the relation is forced through a 197-word list + seed + embedding merge at 0.90, and it is
the **key** for dedupe and conflict (same subject + relation). *New:* a fact still connects
`subject entity → object entity` (or `subject → value`, e.g. an age or a version), and the
relation is the source's own short term ("works at", "stopped using", "is writing") — no
vocabulary, no canonicaliser, no gaps ledger. The relation's jobs are display, filtering
and one-hop navigation ("everything about Lisbon" = facts linked to that entity). It is
**not** the identity key: duplicates and changes are found by shared entities plus
fact-sentence similarity and decided by one small model call that reads both sentences
and both quotes (D6). Converses (`parent_of`/`child_of`) are therefore never merged by a
cosine (TRAPS 26). Evidence for dropping the list: PROVENANCE 2026-08-24 isolated the
relation vocabulary as the cause of false facts — vocabulary-only arm 22% correct vs 60%
without it (−38 pts, CI clears zero); given a list, the model reaches for a listed word
when none fits (`University of Melbourne --employs--> IBM`, reversed). G45 opened the list
but the canonicaliser still snaps new words onto known ones at cosine 0.90; this removes
both. Cost accepted: relation words are inconsistent ("works at" / "is employed by"), which
only weakens filtering by relation; grouping them is G69's job, at read time, if Z2 asks. Fact↔fact links: `superseded_by` (a replacement), merged duplicates
(more support), and "both true" (kept apart).

**D3 — One write pass for summaries and facts.** The SUMMARY_CLAIMS_V3 call (pieces ≤ 4,096
tokens, enforced schema, mechanical quote + speaker check, per-claim NLI at 0.95) is
extended; every supported claim becomes a fact. It runs on **every non-private turn**,
including short turns and documents, which get no summary today
(`turn_density.py:146-155`); summary text is still stored only where `want_summary` says
so. Fixes A8 (no heading cuts — pieces pack windows) and A1 (premise = 1–2 windows of
≤ 224 tokens, always inside the 512 bound). The separate NuExtract3 extractor, grounding,
relation canonicalisation, sentence-claim paragraphs, relation-support NLI and the LLM
author review are deleted.

**D4 — Claim shape (extends S1 of SUMMARY_CLAIMS_V3).**
`{"sources": [..], "quote": "..", "speaker": "user|assistant|document",
"modality": "asserted|suggested|hypothetical|asked", "subject": "..", "relation": "..",
"object": "..", "text": ".."}` — pointer first, sentence last. `subject`/`relation`/`object`
may be empty strings when the claim has no clean two-place form. `modality` follows the
Z2 ledger's `asserted|suggested|hypothetical` plus `asked`. The qualification (§5.1)
decides whether the 4B model holds this shape; fallback is the bare v3 claim plus local
NER links (D5).

**D5 — Entities.** Links per fact: subject, object, plus NuNER mentions in the fact text
(`ner_utils.extract_entities`, background tier). Pronouns resolve by speaker before any
lookup: `I/me/my/mine/myself/we/our` → the **user** entity, `you/your/yours` → the
**assistant** entity of that source unit's speaker (fixes A3). Resolution online is
deterministic only: (1) normalised exact canonical name / alias; (2) `merge_key`
(G50, `maintenance_agent.py:110`); (3) trigram similarity ≥ 0.9 via Postgres `pg_trgm`
for names of ≥ 6 characters (short names are unstable — Graphiti's entropy gate);
otherwise a new entity. Embedding near-matches (cosine ≥ `agent_dup_cosine_threshold`
0.90, same type) go to the maintenance agent's Tier 2 LLM call, which **decides and
applies** the merge itself (D12) — journaled with every moved fact link so
`unmerge_entities(event_id)` restores it exactly. Cosine alone never authorises identity
(TRAPS 101); it only nominates the pair for the model.

**D6 — Duplicates and changes.** For each new supported fact, candidates = current facts
sharing ≥ 1 resolved entity, plus the top 8 by fact-embedding cosine, same store scope.
Exact `text_sha256` match → duplicate without a model call. Otherwise one batched
background call per turn (all new facts + their candidates, both quotes, conversation
order) returns per pair `same | replaces | both_true`, enforced schema. Rules:
- `same` → add the new source to the existing fact (support +1); no new row.
- `replaces` → only when both are `asserted`, same speaker role, and the new source is
  later in conversation order (`source_order.compare_observations`); old fact gets
  `valid_until` = the new source's recorded time and `superseded_by`. Never deleted.
- anything else, an unknown order, or a failed call → both kept (`both_true`), counted.
  Unknown never expires anything (TRAPS 99).

**D7 — Strength (maintainer, 2026-10-15).** Two counts, two roles:
- `support_count` — distinct source turns that produced the same fact (D6 `same`).
  **Majority share** of the weight.
- `usage` — times the fact reached the **final prompt** (`record_graph_access` at stage
  `prompt_prepared`, `main.py:631`), decayed by the existing daily job (`codex_decay_daily`
  0.99/day). **A real share, not a majority, not negligible: one third.**
- Read-time prior on a fact fragment, applied **after** the reranker (a prior applied
  before it only moves the 64-candidate cut — see §7):
  `bonus = B · (⅔ · min(1, (support_count−1)/3) + ⅓ · min(1, usage/5))`, added to the
  rerank logit. Both terms saturate (TRAPS 135). `B` is set from the measured spread of
  rerank logits on the development questions (§5.3), not guessed.
- It **ranks, never admits**: visibility is decided by quote + NLI only. `confidence`
  (the NLI score) is stored separately and is not part of the bonus.

**D8 — Read path.** One `facts` leg replaces `_codex_graph`, `_codex_claims`,
`_codex_enumeration`, `_relation_facts` and the 3-hop traversal:
- channels: BM25 over fact text (`tsvector`), fact-embedding cosine, entity match
  (pre-flight MicroNER → `_match_entities_by_similarity`, kept) → the matched entities'
  current facts (**one hop**). Internal RRF, then the shared fusion/rerank/budget.
- current facts by default; replaced facts only under a history/as-of timescope. A shown
  current fact that replaced another renders its immediate predecessor in one clause
  (`earlier: …, until 2024-06`), for knowledge-update questions.
- **skip a fact whose every source turn is already in the prompt** (retrieved episodic
  fragment with `covers_entire_source`, or a recent-window turn), checked at final
  assembly (`prompt_budget.py`), not only against candidates (TRAPS 66).
- Deeper traversal returns only if Z2's multi-hop questions show the need.

**D9 — Storage.** Stay on Postgres + pgvector. New tables (§2). `codex_entities` and
`codex_edges` **stay** — the code graph (`source='static_analysis'`) and project facts
(`source='derived'`) live there. Conversation-sourced `codex_edges`, `codex_claims`,
`codex_claim_links`, `codex_relation_gaps` rows are backed up (`pg_dump`, path logged in
CLEANUP) and the three claim/gap tables dropped; conversation edges are deleted from
`codex_edges`. Z2 builds fresh stores, so no conversion of old rows.

**D10 — Deletion and forget keep C10's contract.** A deleted source is removed from a
fact's sources; a fact left with none is **expired** with a `fact_expired` event, reason
`source_deleted` (C10: expire + journal, never hard-delete). `/forget` stays propose →
review → apply.

**D11 — Maintainer decisions (2026-10-15, approved):**
1. *Entity notes* (reflection-written per-entity descriptions): **stop generating them for
   conversation entities**; the entity view (`ice_why`, `ice_graph view`) lists the
   entity's current facts. User-edited descriptions are kept and shown as `[user note]`
   (VISION l.68).
2. *Compaction + `CodexSnapshot`* (write-only, FEATURE_INVENTORY DEAD/INERT #7):
   **removed**, with backup and a CLEANUP entry.

**D12 — No human wall (maintainer, 2026-10-15).** ICE is research: a decision that waits
for a human never acts, so its effect can never be measured. Measured: the development
store holds 571 pending `codex_reconciliation` items (largest store 7,641), none ever
applied. In this item **every decision applies itself**: D6 same/replaces/both_true, D5
entity merges, deletion expiry. Each is journaled in `codex_events` with enough payload to
undo it, and written to `review_queue` with `status='applied'` as an **audit log, not a
gate**. User-started actions stay user-driven (`/forget` proposes from a fuzzy match, so it
still confirms — C10 trap). The codex review item types (`codex_reconciliation`,
`codex_contradiction`, `entity_merge` proposals) are retired; their pending rows are
archived with the backup.

## 1b. Models in the codex — before and after (maintainer request, 2026-10-15)

Every model call the conversation codex makes, per turn (write) and per question (read).
"Stays elsewhere" = the model remains in ICE for a non-codex job.

**Write path (per turn)**

| Step | Before | After |
|---|---|---|
| Fact extraction | **NuExtract3** (5.2 GB), one call per heading chunk — ~130 calls, 12–16 min on a large C turn | **Removed.** Facts come from the claim pass `gemma4:e4b` already runs for summaries (pieces ≤ 4,096 tokens). New cost only for short turns and documents, which get no claims today. NuExtract3 leaves ICE (no other user); `extraction_result.original_source_quote` moves, since `api/source_proof.py` and `api/source_need.py` import it. |
| Grounding gate | **NuNER Zero** over every chunk; a subject/object not in its list → confidence 0.35 → hidden | **Removed** (the A9 gate). NuNER now runs once over each short *fact sentence* and only **adds links**; a miss loses a link, never a fact. Stays elsewhere: clustering entity hints, summary must-keep terms. |
| Relation canonicaliser | **Qwen3-Embedding** compares each relation with ~111 seed + live relations, merges at cosine 0.90 | **Removed.** The relation is the source's words. |
| Sentence-claim verdicts | **DeBERTa NLI**, paragraph → sentence, stored but never gating | **Removed.** |
| Relation support | **DeBERTa NLI**, paragraph vs "subject relation object.", twice (canonical + source relation); 7% over the 512-token bound | **Removed.** Replaced by the claim pass's own NLI check: claim sentence vs its 1–2 cited windows (≤ 224 tokens each, always inside 512). Long turns already pay it for summaries; short turns are new. |
| Author-assertion review | **gemma4:e4b** with thinking, batches of 12 facts, 8K input | **Removed.** Guard moves to `modality` + the §5.1 controls (TRAPS 70/97/116). |
| Conflict reconciler | **gemma4:e4b**, one-word verdict per same-subject+relation pair; 571 left pending for a human | **Replaced** by one batched `gemma4:e4b` call per turn (D6: same / replaces / both_true), only when candidate facts exist; applies itself (D12). |
| Entity embedding | Qwen3-Embedding per new entity | **Stays.** |
| Fact / claim embedding | Qwen3-Embedding per claim paragraph (`codex_claims`) | **Stays**, per fact sentence (shorter input). |
| Entity name matching | exact → alias → `merge_key` | **Same + trigram** (`pg_trgm`, not a model). |

**Background jobs**

| Job | Before | After |
|---|---|---|
| Maintenance: reconciliation, contradiction, pending pile-up detectors | gemma4:e4b decisions → review queue | **Removed** (D6 owns changes at write time). |
| Maintenance: duplicate entities | embedding cosine 0.90 → gemma4:e4b → review queue | **Stays**, now applies itself with undo (D12). |
| Reflection: entity description enrichment | gemma4:e4b per entity | **Removed** for conversation entities (D11.1). |
| Entity notes (quoted claims per entity) | gemma4:e4b + DeBERTa NLI | **Removed** for conversation entities (D11.1); stays elsewhere for summaries. |
| Compaction snapshots | no model | **Removed** (D11.2). |

**Read path (per question)**

| Step | Before | After |
|---|---|---|
| Names in the question | **micro-NER** (our 384-dim head on Qwen3-Embedding); missed names in 48/58 short prompts (2026-09-24, old corpus) → graph not consulted | **Known-name lookup first**: the store's own entity names searched in the question with `pg_trgm` (no model). micro-NER kept as a second source for unseen names. Measured on the 67 development questions before adoption. |
| Question → entity match | Qwen3-Embedding cosine ≥ 0.85, payload ILIKE fallback | Embedding match **stays**; payload ILIKE **removed**. |
| Relation detection / relation fit | Qwen3-Embedding against relation glosses | **Removed** (no relation vocabulary to fit). |
| Fact search | — | **New**: BM25 + fact-embedding cosine (the query is already embedded once for episodic search). |
| Reranker | **Qwen3-Reranker** over whole source messages (best-window scoring, ~1.3 s per over-length passage) | **Stays**, over short fact sentences (no over-length windows for facts). |

Net: four model steps leave the write path (NuExtract3 extraction, NuNER gate, relation
canonicaliser, relation NLI + author review), one batched gemma call is added, and NLI moves
onto the claim pass that already runs.

## 2. Algorithm & data model

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE codex_facts (
  id uuid PRIMARY KEY,
  text text NOT NULL, text_sha256 text NOT NULL,
  speaker text NOT NULL,            -- user|assistant|document
  modality text NOT NULL,           -- asserted|suggested|hypothetical|asked
  subject text, relation text, object text, negated boolean NOT NULL DEFAULT false,
  embedding vector(1024),
  confidence double precision,      -- NLI entailment of the first supporting source
  support_count integer NOT NULL DEFAULT 1,
  usage double precision NOT NULL DEFAULT 0, last_used_at timestamptz,
  mentioned_at timestamptz NOT NULL,    -- first source's recorded time (NOT ingest time)
  valid_from timestamptz NOT NULL, valid_until timestamptz,
  superseded_by uuid REFERENCES codex_facts(id),
  created_at timestamptz NOT NULL DEFAULT now());
-- GIN to_tsvector('english', text); HNSW/ivfflat on embedding as episodic does;
-- partial index (valid_until IS NULL); index on text_sha256.
CREATE TABLE codex_fact_sources (
  fact_id uuid REFERENCES codex_facts(id) ON DELETE CASCADE,
  episodic_id uuid, source_batch uuid NOT NULL, conversation_id uuid,
  window jsonb NOT NULL,            -- unit, start, end, unit_sha256, text_sha256
  quote text NOT NULL, recorded_at timestamptz,
  verification jsonb NOT NULL,      -- method, policy, label, score, pair_tokens
  PRIMARY KEY (fact_id, source_batch, (window->>'text_sha256')));
CREATE TABLE codex_fact_entities (
  fact_id uuid REFERENCES codex_facts(id) ON DELETE CASCADE,
  entity_id uuid REFERENCES codex_entities(id),
  role text NOT NULL,               -- subject|object|mention
  PRIMARY KEY (fact_id, entity_id, role));
```
`codex_events` is kept and gains `fact_added`, `fact_supported`, `fact_superseded`,
`fact_expired` (payload carries `fact_id`; `entity_id` = each linked entity).

**Write pass (per turn, inside `evaluate_turn`, background lane):**
```
windows = source_windows(original_units([turn]))            # existing
claims  = generate_claims(call, windows, ...)                # existing; schema = D4
claims  = mechanical checks (quote, speaker, shape, modality enum)   # existing + modality
claims  = verify_assertions(claims) -> keep supported (>=0.95)       # existing
if want_summary: store summary_text = kept claim texts       # unchanged behaviour
for c in kept: links = resolve(c.subject, c.object, ner(c.text), speaker)
dedupe(kept)  -> same / replaces / both_true                 # D6, one batched call
write facts, sources, entity links, events in one transaction
```
Private turns: no facts (unchanged). Idempotency: `job_key("facts", batch_id)`.

**Read pass:** §1 D8. Fragment fields: `source_type="fact"`, `origin_fact_ids`,
`claim_source_row_ids` (all source episodic ids, for the D8 skip), `conversation_id`.

## 3. Files & integration points (build order; each step = commits + its test)

1. **Move the couplings out of `codex_extractor.py` first** (no behaviour change):
   `embedder` → `src/memory/embedder.py` imports at `post_flight.py:26`,
   `document_chunker.py:31`, `conversation_summary.py:81`, `decision_extractor.py:117`;
   `_KNOWN_TYPES` → `ner_utils`; `_chunk_text` alias → `chunking.chunk_text`.
2. **Migration** (read every autogenerated line — TRAPS 14): §2 tables + `pg_trgm`.
   ORM models in `src/memory/models.py`; register `codex_facts.embedding` in
   `reembed.py` (and check `codex_claims.embedding` was missing — SESSION finding).
3. **Write:** extend `summary_support.claim_schema`/`_assertion_instruction_v3`/
   `_claim_problem` behind `summary_assertion_method="v4"` (v3 stays readable);
   new `src/memory/facts.py` (resolve, dedupe, write); `post_flight.evaluate_turn` runs
   the claim pass for every non-private turn and calls `facts.write_turn_facts`.
   Bookmark/document/import paths reach it through `evaluate_turn` already.
4. **Read:** new `_facts_leg` in `orchestrator.py` (reuse `_codex_claims`' scope SQL and
   `_match_entities_by_similarity`); `leg_weights.LEGS`: `codex` → `facts`, `timeline`
   rebuilt on `superseded_by` chains (`evolution.py`); `prompt_budget.py` D8 skip;
   `usage.record_graph_access` keyed on `origin_fact_ids`; D7 bonus after rerank.
5. **Services/MCP:** `services/conversations.py` (deletion/forget on fact sources),
   `services/review.py` (audit-only for codex items; `forget_request` unchanged), `services/graph.py` (`entity_view` = facts; edit = user note),
   `mcp/server.py` (`ice_graph`, `ice_why`, `ice_context`), `chat_commands.py`
   (`/forget`, `/delete-conversation` manifest), `api/main.py:691` SSE count.
6. **Workers:** `maintenance_agent.py` (keep entity duplicate tiers; drop edge pileup /
   contradiction / reconciliation detectors — D6 owns changes), `codex_decay.py`
   (decay `codex_facts.usage`), `reflection.py` (D11.1), `runtime.JOBS` (`codex_extract`
   → re-run fact pass for a bookmarked turn; `compaction` per D11.2).
7. **Delete** (after 1–6 pass; backup + CLEANUP ledger, `scripts/oneoff/` for moved
   scripts): `codex_extractor.py` text paths, orchestrator codex block
   (`orchestrator.py` ~1134–2346 minus code-graph rendering), `claims.py`,
   `relation_support.py`, `assertion_support.py`, `nuextract_transport.py` (if no other
   user), `codex_inject_watcher.py` (dead), `data/relation_seed.json`, dead `codex_*`
   settings + their freeze rows (retired with proof), conversation rows per D9.
8. **Harnesses:** `scripts/z1/seed_v3.py` (`no_codex` arm → disable `_facts_leg`),
   `experiments/v3_repair/chronological_replay.py` (`graph_exposure`),
   `experiments/v3_repair/artifact_inventory.py` (fact columns for Z2 D9).
9. **Docs:** FEATURE_INVENTORY rows (§3/§7/§8 graph rows replaced), ICE_Architecture §4
   rewritten (+ the partial mentions listed in SESSION), MODELS (NuExtract3 role ends),
   ROADMAP (A1/A3/A8/A9/B1–B6/G68 closed by this item; G63/G69/G76/G60 re-pointed).

## 4. Edge cases & failure modes

- **No claim survives** (all quotes/NLI fail): no facts; counted per turn, WARNING with
  reason when a turn of > 50 words yields zero (silent-fallback rule, TRAPS 11).
- **Claim piece fails** (cap, broken JSON): that piece only; others stand (existing).
- **Pronoun in subject/object with no speaker** (document): `document` speaker → no
  pronoun mapping; the term is kept as text, no entity link.
- **Empty subject/object**: fact stored, linked by NER mentions only; still retrievable
  by BM25/vector.
- **Dedupe call fails / times out**: all pairs `both_true`, counted, WARNING.
- **Same fact twice in one turn** (two windows): `text_sha256` + D6 inside the batch.
- **Replacement chain** A→B→C: each old fact points to its direct successor; the read
  side shows one predecessor only.
- **Out-of-order import** (older turn ingested after a newer one): D6 uses conversation
  order, so an older observation can never replace a newer one.
- **Concurrency**: one gpu-lane job per turn (runtime de-duplicates); entity creation
  keeps the unique `canonical_name` + retry-on-conflict pattern of `get_or_create_entity`.
- **Deleted source** mid-dedupe: the transaction re-reads candidate facts' `valid_until`.
- **Wrong automatic decision** (D12): `unmerge_entities(event_id)` and
  `restore_fact(event_id)` (clears `valid_until`/`superseded_by`, or splits a merged
  source back out) undo from the journal; tested two-sided.
- **Legacy rows**: v3 summary records stay readable under their method; no old edge is
  read by the new leg.
- **Usage under replay/historical clock**: `record_graph_access` must use the replay
  clock (TRAPS 79) so Z2 replays decay like real time.

## 5. Validation checklist

**5.1 Qualification of the write prompt (before building §3 step 3).** ~30 real pieces
(C large/mid/small, B short, dev short turns, 2 documents), native door, GPU otherwise
idle, `gemma4:e4b` with the D4 schema vs v3 claims + local NER links. Report: unparseable
outputs; drops per mechanical check; supported/unknown/contradicted; facts per piece;
`modality` agreement on a maintainer-read sample; subject/object present in `text`;
self-contained rate (a reader can understand the sentence alone); wall time.
**Controls (TRAPS 70/97/116):** hold one proposition fixed and vary the source —
hypothetical, question, other speaker, former state, same names in an unrelated sentence,
reversed direction. Pass: no rise in contradicted claims vs v3, false admissions on the
controls ≤ 1 of ~20, and modality right on ≥ 90% of the sample. Fail → bare v3 claims.

**5.2 Qualification of the dedupe call.** Planted corrections + paraphrases + converses +
"both true" lists (from real dev/B/C sentences, not invented — TRAPS 13) and the real
updates the dev history holds. Report per label precision/recall; pass: `replaces` never
fires on a converse or a list member; `same` precision ≥ 0.95.

**5.3 Strength bound `B`.** Rerank logits of fact candidates on the 67 development
questions: set `B` = median gap between adjacent selected candidates; report how many
selections flip with the bonus on vs off (standing rule: "turning it off changes N").

**5.4 Behavioural tests** (standalone `tests/test_codex_facts.py`, inserts and deletes its
own rows, never truncates, stub LLM): claim → fact written with sources/entities;
`I`/`you` → user/assistant entity; exact duplicate → support 2, one row; `replaces` →
old `valid_until` + `superseded_by`; suggestion never replaces an assertion; older source
never replaces newer; deletion of sole source → `fact_expired`, corroborated fact
survives; private turn → no facts; read leg returns current only, history under as-of;
skip when source turn in prompt (episodic **and** recent window); usage increments only
at `prompt_prepared`, decays; bonus saturates. Two-sided assertions (TRAPS 5).

**5.5 End-to-end on the development history** (fresh store, chronological replay):
storing (facts/turn, % turns with ≥ 1 fact, chars/fact), retrieving (facts reaching
prompts — A6), answering (67 questions vs 61 full / 63 no-codex, inside the ±3 floor);
duplicate rate; changes detected vs the dev history's real updates. Reported in
PROVENANCE as development evidence, not a score of record.

**USER-REQUIRED:** read ~20 sampled facts + their quotes from §5.1 and mark each
right/wrong/not self-contained (~20 min); D11 approved 2026-10-15.

## 6. Look-ahead constraints

- **Z2** (approved spec): D9 audits "graph claims and edges" → becomes facts; fact fields
  must map to the ledger entry (`subject, value, speaker, asserted|suggested|hypothetical,
  citations, valid_from_turn, superseded_by`). D10 ablation groups 3/4/12 → one `facts`
  switch + maintenance switch. D11c keep rule applies to the facts leg.
- **T track** (T3 checkout / T4 log/diff, T5 conditional): validity + `superseded_by` +
  events are the history; do not overwrite fact text in place, ever.
- **Code graph / E-core**: `codex_entities`/`codex_edges` with `source='static_analysis'`
  and `project_facts` untouched; `ice_where` / `render_architecture_doc` unchanged.
- **G23/C17 portability**: new tables export/import generically; `_regen_payload_sweep`
  becomes a no-op for conversation entities.
- **Occurrence time** (Hindsight/SodaMem's `occurred_at`): not in this item; the schema
  must allow adding it later without rewriting facts.

## 7. Traps

- **"Keep the triple, fix the gates"** — the obvious simpler version. Rejected: each gate
  was individually correct; together they hid most facts and showed the rest as raw text.
- **A prior applied before the reranker does nothing.** The cross-encoder replaces the
  RRF score; leg weights and pre-rerank boosts only move the candidate cut. D7's bonus
  goes after rerank, and its effect is measured as flips (standing rule).
- **NLI imports plausible relations** (TRAPS 70) and **copied clauses are not adopted
  assertions** (TRAPS 97): modality + the §5.1 source-varied controls are the guard.
- **Cosine cannot authorise identity or direction** (TRAPS 26, 101): entities merge
  deterministically or offline with review; facts merge only by the D6 call.
- **A review queue nobody reads is a switched-off feature** (D12): never add a
  "propose" step to this item; decide, apply, journal, make it undoable.
- **Usage feedback loop**: retrieved → more usage → retrieved more. Bounded at ⅓ of `B`,
  decayed, and reported in Z2's ablation.
- **Fixing a component can switch on a dormant restriction** (TRAPS 133): when the facts
  leg starts returning hundreds of candidates, re-check the per-conversation cap and the
  leg guarantee on the admitted prompt.
- **A harness that skips the new pass measures the old system** (TRAPS 1, 9): seed/replay
  harnesses call `evaluate_turn`, never a private shortcut.
