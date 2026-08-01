# Brutal Assessment — 2026-07-12 (Fable close-out)

> Written at the user's request in the last Fable 5 session: a no-deference pass over the
> whole project — codebase, roadmap, specs, experiment results, and the rough notes — to
> name every flaw worth naming and propose a solution for each. Grounded in: the full
> [ROADMAP.md](ROADMAP.md), the full [rough_post_paper_work.md](outdated/rough_post_paper_work.md) (since archived),
> [ICE_Architecture.md](ICE_Architecture.md), the Exp1–3 summaries, the T/C7 implementation
> sessions (orchestrator, models, workers, tests read in depth), and the FINAL spec.
> **Nothing here is a work order** — it's a standing critique to be mined; promote items
> into the roadmap deliberately, one at a time, with the usual discussion-first rule.
> Items marked ⚡ are the ones I'd act on soonest.

---

## 0. The one-paragraph verdict

ICE is a genuinely good idea (memory as an earned, structured, decaying store with
time as a first-class axis — nobody local-first does this properly) executed with unusually
good engineering discipline (the spec/roadmap/propagation system is better than most
funded teams'), sitting on top of **one unresolved existential question the experiments
already asked and the project hasn't answered: on 3 of 4 datasets, all of this machinery
beat a plain vector store by 0.01.** Everything below is secondary to how the project
answers that. The answer can be "the machinery exists for capabilities the old probes
never measured — temporal, cross-conversation, evolution, coding memory — and FINAL will
measure them" (this is the current implicit bet, and it's a *reasonable* bet), but then
FINAL must be designed to *kill features that don't cash in*, not to vindicate the system.

---

## 1. Existential / strategic flaws

### 1.1 ⚡ Complexity accretion with no deletion budget
**Flaw:** Exp2's headline is +0.4 driven almost entirely by one dataset where the baseline
collapsed (ICE-Dev: 4.33 vs 1.23); excluding it, 4.26 vs 4.25. The project's response to
"the machinery barely beats a vector store" has been to *add* machinery: Tracks A, T, D,
E, plus 16 specs. P0.1 "sentenced every feature" but the verdict was "keep" for everything
except MERA. One deletion in a system of ~20 subsystems is not pruning; it's ratification.
**Question to sit with:** if FINAL shows codex+timeline+procedural+clusters each worth
+0.02, will anything actually get deleted? History says no.
**Solution:** give FINAL a **pre-registered deletion rule** per feature *before* running it
(e.g. "if the ablated delta of X is below +0.1 on the probe classes X exists for, X is
removed in the following cycle, not retuned"). Write the expected effect size next to each
ablation flag now, while there's no result to anchor on. A feature that survives only
because removing it feels wasteful is dead weight with maintenance costs.

### 1.2 ⚡ Research artifact vs product — the fork is still unpicked
**Flaw:** the roadmap simultaneously drives toward (a) an arXiv paper redo (FINAL), (b) a
packaged consumer app (Track F: custom frontend, settings UI, graph view, hardware
advisor), and (c) a developer-tool wedge (E7 ICE-as-MCP). One person cannot ship all
three; the F-track alone (a full chat frontend) is months of work that duplicates Open
WebUI to mediocrity, and every F item waits behind FINAL anyway.
**Solution:** pick the wedge explicitly. My honest recommendation: **the MCP coding-memory
wedge is the product; the conversational exocortex is the paper.** E7 gives real daily
value to the person who builds it (self-hosting developers are exactly the audience that
tolerates docker+postgres), while a custom chat frontend serves nobody Open WebUI doesn't
already serve. Concretely: keep F2 (review queue), F3 (graph view), F5 (telemetry) as
*panels bolted onto Open WebUI or a minimal single-page app*; kill or indefinitely park F1
as "the big one." Revisit only if ICE-as-MCP gets traction.

### 1.3 The paper describes a system that no longer exists
**Flaw:** the paper is written against v2. Since then: Celery/Redis deleted, decay math
rewritten (cycles-parameterized + freeze fix), retrieval legs reworked (A1–A11, C1–C3,
C8), the LTM override replaced (B2), temporal retrieval added (Track T). The roadmap's
correction block fixes the two *false* claims (NER, corroboration trap) but not the drift:
posting the paper as-is publishes an architecture description that is materially wrong
about the current code.
**Solution:** decide explicitly, in writing: either (a) post the paper as a dated snapshot
with a one-line "the system has since evolved; see repo" disclaimer, or (b) hold it and
publish once with FINAL's numbers against the current system. Do not drift into (a) by
default without the disclaimer — reviewers who read the repo will notice.

### 1.4 The token-efficiency claim is currently only true where the baseline broke
**Flaw:** the rough notes' critic #3 is correct and nobody has refuted it: on datasets
where both systems worked, ICE consumed *more* total tokens (22,411 vs 21,025). "Fewer
but better fragments" is the thesis's central efficiency claim and it is empirically
unsupported outside ICE-Dev. The fix (C16's need-based filling / marginal-relevance
stopping) is specced in the entry but **absent from the execution order** — it's not in
the numbered flow at all, so as sequenced FINAL would re-measure the same fill-to-cap
policy that produced the −0.11 dynamic-budget ablation.
**Solution:** slot C16's "smart half" into the flow explicitly — it belongs with Z1-prep
(it's a tuning-adjacent policy change and the auto-scorer can validate it cheaply). If it
can't be built in time, FINAL's write-up must drop the efficiency claim rather than
lean on ICE-Dev again.

### 1.5 The name
**Flaw:** "ICE" collides with a charged American institution and with "In-Context
Everything"-style acronyms; for a public paper/product it invites jokes and muddles
search. Trivial today, annoying forever after arXiv.
**Solution:** decide pre-arXiv, once. Renaming after publication is much worse than either
keeping it deliberately or changing it now.

---

## 2. Data-safety flaws (the scariest section)

### 2.1 ⚡⚡ The dev database IS the user's real exocortex and has NO backup until phase 7
**Flaw:** G23 (export/backup) sits at flow position 7, after D1/D2 and the whole E-core.
Meanwhile the same postgres instance holds the user's actual accumulated memory, and —
worse — **several standalone tests TRUNCATE tables against that same live DB**
(CLAUDE.md itself documents this). One wrong `uv run python tests/test_*.py`, one bad
alembic downgrade, one docker volume mishap, and years of memory are gone. This is the
single highest-expected-loss item in the entire project and it costs ten lines to hedge.
**Solution (do not wait for G23):** (a) tonight-tier: a cron/systemd-timer `pg_dump -Fc`
to a dated file + copy to a second disk/cloud — 10 lines, zero design; (b) a
`ICE_TEST_DATABASE_URL` env var + a guard line at the top of every truncating test that
refuses to run against the main DB name (one shared helper, mechanical retrofit);
(c) G23 later replaces this with the real machinery. The marker-keyed never-truncate
pattern (test_timescope.py, test_maintenance_runtime.py) should be retrofitted
opportunistically to the older suites.

### 2.2 Imported/ingested content is a memory-poisoning vector nobody has threat-modeled
**Flaw:** F10 (chat-log import), C12 (document ingestion), and E10 (project docs) all feed
third-party text through codex extraction into a store whose contents are later injected
into *system-side* prompt context. Adversarial text in an imported log or a PDF ("ignore
previous instructions", fabricated "facts" about the user) becomes durable, retrievable,
authority-carrying memory. The specs handle decay policy and formats but not trust.
**Solution:** provenance-tiered trust: every memory row already carries source provenance
(G17 formalizes it) — extend it to an origin class (`native` / `imported` / `document`),
render non-native fragments with an origin tag in the assembled context (the model should
know a "fact" came from an imported file), and consider capping imported edges'
`extraction_confidence` below native until corroborated by live conversation. One
paragraph in the F10/C12 specs now saves a nasty surprise later.

### 2.3 Logs are an unmanaged second copy of everything (G25 is under-prioritized)
**Flaw:** raw prompts and responses go to plaintext `logs/*.log` with no rotation policy
tied to memory semantics: C10 deletion won't touch them, incognito turns land in them,
and they never decay. The sovereign-memory story has a hole in it as long as this is
default-on. G25 exists but is unscheduled tail work.
**Solution:** flip the default to redacted-content logging (keep IDs/decisions/telemetry)
*when the F/E product era starts* at the latest; incognito turns should redact regardless,
now — that's a one-condition change in the logging call, not a project.

---

## 3. Architecture & code flaws

### 3.1 ⚡ Pre-flight latency has never been budgeted or measured
**Flaw:** the synchronous path now runs: NER, classifier forward, relation detection
(gloss embeddings), 3-stage entity resolution, trust-gated graph traversal per anchor,
`history_exists` (a JSONB-expression join with **no supporting index** — payload->>'edge_id'
= id::text over codex_events) plus timeline building, BM25, vector + chunk search, cold
lookup, RRF, bonuses, budget, resurrection writes. Nobody knows what p95 looks like; every
track added "a few ms" without a meter. Retrieval quality no one perceives; latency
everyone perceives.
**Solution:** (a) a per-stage timing wrapper (structlog `retrieval_stage_ms`) — trivial,
huge diagnostic value, feeds F5; (b) set a budget (e.g. pre-flight ≤ 300 ms p95 on the dev
box) and check it at Z1; (c) for the new T4 join specifically: an expression index
(`CREATE INDEX ... ON codex_events ((payload->>'edge_id')) WHERE event_type='edge_expired'`)
or a real `edge_id` column on codex_events next time a migration is open (G6's home), and
per-request memoization of `history_exists` per anchor.

### 3.2 The classifier taxonomy is expensive relative to what consumes it
**Flaw:** 25 labels, a training pipeline, weekly fine-tune, DI3, curation UI plans — and
the consumers are: leg-weight profiles (hand-tuned, never validated — the blend table's
numbers have no experiment behind them), MoE routing (measured ≈neutral), procedural
gating (being *removed* by C9), and B2 (which uses only the ctx head's p_ltm scalar).
It is entirely possible that a 4-output head ({needs-memory, temporal, live-info,
complexity} — which is exactly B1's new ctx head) plus topic embeddings would deliver
everything the 22 topic/intent labels currently deliver.
**Solution:** before B1's expensive relabeling, run the cheap falsification at Z1: ablate
the intent-blend (flat base_weights vs profile blend) on the auto-scorer. If flat ≈
blended, B1 should *shrink* the taxonomy rather than re-label all of it — that's a
labeling-cost decision worth making from data, and B1's spec already carries the ctx-head
redesign either way.

### 3.3 Pending-validation debt is concentrating into a cliff
**Flaw:** a dozen shipped items carry "LLM half pends Z1 / live validation" notes (A1
extraction quality, A2 over-rejection, A6 LLM reconciler, A7.3 enrichment of ~1,150
entities, C1 summary quality, B4 promotion run, C5 naming quality…). Each was
individually reasonable; collectively Z1 now fronts for a stack of unproven LLM behavior,
and if several fail there, the fixes will be urgent, interleaved, and late.
**Solution:** run a **mini-Z1 now-ish** (one evening, current stack, shared mode): replay a
slice of ICE-Dev through the live pipeline and eyeball the codex/summary/enrichment
output. Not the tuning gate — just burning down the "does the LLM half work at all" class
early. The E0/E7 session doesn't depend on it, so it can happen any idle evening; the
point is not to discover extraction is broken *after* D1 builds an agent on top of it.

### 3.4 The orchestrator God-class is still growing
**Flaw:** the decomposition note (Track C preamble) says shrink it as items open it; T
added ~200 lines net (timescope/evolution went to modules — good — but leg edits, cold
leg, stratifier, resurrection all landed in the class). It's ~2,200 lines. Every new
session pays reading cost; every change risks the configurable-subclass drift (G19).
**Solution:** E0's retrieval_svc extraction is the natural moment — when it opens the
file, actually carve legs/fusion/budget into modules rather than wrapping the monolith.
And fold the ablation flags into settings-driven parent behavior (G19's recommendation)
before FINAL, or the ablations may silently lie.

### 3.5 Config duplication and drift hazards
**Flaw:** the DB URL lives in both config.py and alembic.ini (documented, still a trap);
tunables split across settings, module constants (G9 pending), and class attributes;
`models/` checkpoints are promoted by editing config. None of this bites until it bites.
**Solution:** G9 (constants→settings) is already queued for Z1-prep — hold that line; have
alembic read the settings URL (env.py one-liner) instead of a second copy.

### 3.6 Cold-start and empty-store behavior was never designed
**Flaw:** a brand-new user (or a fresh install) has an empty store, yet pays the full
pre-flight machinery on every turn, and B2's length prior will eventually vote "retrieve"
into a store with nothing in it. The first weeks of use — exactly when a new user judges
the product — are pure overhead with zero payoff. Nothing in any track addresses this.
**Solution:** an empty/sparse-store fast path (store row-count below N ⇒ skip retrieval
legs, or a cheap prior into B2), plus onboarding surface: F10 import is *the* cold-start
answer ("bring your history") — market it as such in the F-track design conversation.

---

## 4. Evaluation flaws (FINAL-adjacent)

### 4.1 The FINAL flow itself will mutate — planned for, now stated in the roadmap
The entire previous experiment flow — sourcing conversations, numbering/timing them,
probe generation, ground-truth construction, selecting which conversations to test,
the testing process, and how each stage was split into scripts — **will change**: parts
added, parts removed, parts redesigned. The FINAL spec is the starting design, not a
contract. (A line to this effect now lives in the roadmap's FINAL section; this doc is
its long-form justification.) The corollary: don't cargo-cult the old scripts — the
frozen `experiments/*` folders are the record of what WAS done, not the template for
what will be done.

### 4.2 Judge noise vs claimed deltas
**Flaw:** the deltas being argued over (±0.01–0.4 on a 5-point LLM-judged scale) are near
or below plausible judge noise; Exp3's hallucination numbers are explicitly uncorrected.
The FINAL spec's calibrated-judge + ledger auto-scores design is the right correction —
but only if the *reported* claims are limited to what survives confidence intervals.
**Solution:** pre-register per-hypothesis expected effects + CIs; report per-dataset with
uncertainty; let the planted-fact ledger (objective, judge-free) carry the headline where
possible and use the judge only where it must.

### 4.3 The capabilities that justify the machinery still have zero probes — until FINAL
**Flaw:** temporal (T), cross-conversation (C6/H1), evolution narration (T4), procedural
usefulness (C9), coding memory (E) — the differentiating features — have never been
measured; the old probe set structurally couldn't see them. This is *why* 1.1's bet is
still open. FINAL pt.5 plans temporal probes; the others need the same first-class
treatment or the next paper repeats the "we built it but didn't measure it" limitation.
**Solution:** the probe taxonomy must have a named class per differentiating capability,
each with its own deletion rule (1.1). A capability that can't be probed shouldn't ship
as a claim.

---

## 5. Rough-notes sweep — what was actually forgotten (answering the specific ask)

Verdict on the explicit question: **the LSREP-as-import feature is NOT missing** — it is
[F10](ROADMAP.md) ("Conversation import (LSREP as migration tool)", flagship, spec'd in
`specs/F10_F14_import.md` with the decay-policy trio and the shared ingestion engine).
F14 (raw-log extraction v2) carries the unformatted-dump half. Nothing to add there.

Full sweep result — everything else in rough_post_paper_work.md maps to a live roadmap
item (coding vision→Track E; active-ICE→D1; NER/chunking→A1/A2/A9; relation search→A4;
scoping/session→C5/C6/C7; thumbs feedback→B4/F9; OKF→E6; deletion→C10; caching→C13/C14;
the old missing-items table→F13/G14/G15/G16/G17/G19; the limitations section→B2/B3/H1–H5/
C14/F5) — **except two genuinely unacknowledged features**, both from the "Unified
Context Budgeting with User Control" / "User-Defined Scoping" passages, now added to the
roadmap:

1. **User retrieval overrides (force-deep-search + aggressiveness)** → new **F16**. The
   "deep search" button (bypass B2 gating + run the wide net deliberately) and a
   per-conversation retrieval-aggressiveness setting are *backend behaviors*, not just
   settings exposure — F4 didn't cover them and nothing else did.
2. **Selective memory exclusion (scope blacklist)** → new **C6 addendum**. Excluding a
   conversation/cluster from retrieval *without deleting it* (abandoned projects, privacy)
   existed in the notes and fell through: C6 only ever *adds* to scope; C10 only deletes.

Judged present-but-superseded (deliberately not added): "temporary relevance for N
queries" (covered by F6 + C6 @-mention), "cross-project entity relevance" (E1b's
namespacing + cross-links carry it; revisit at E1b if real), ensemble classifiers (B5,
parked correctly), multi-model responses (F12, parked correctly).

---

## 6. If I could only make you do five things

1. **Back up the memory DB tonight and fence the truncating tests** (2.1). Everything
   else on this list is recoverable; this one isn't.
2. **Write the per-feature deletion rules into the FINAL spec before any result exists**
   (1.1 + 4.3). It's the only defense against the system ratifying itself.
3. **Pick the wedge: MCP coding memory is the product, the frontend is a panel, not a
   platform** (1.2). E0→E7 being next is already right — protect that priority when the
   F-track siren starts.
4. **Slot C16's need-based filling before FINAL** (1.4) — the efficiency thesis is
   currently carried by one dataset where the baseline crashed.
5. **Meter the pre-flight path** (3.1). One timing wrapper, one budget number, checked at
   Z1 — before the latency story becomes a user-facing surprise.
