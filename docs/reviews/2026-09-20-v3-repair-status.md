# ICE v3 repair status — 2026-09-20

This is a progress snapshot, not another queue. ROADMAP.md owns outstanding work;
SESSION.md records live findings. Implementation and focused qualification are
separate from an end-to-end answer-quality result. No new v3 LME/vector win or
context-saving percentage is claimed.

## Objective and evaluation decision

Improve supported answers per prompt token. Prefer less context at equal or better
quality; better memory at the same cost also counts. Do not optimize graph size,
summary length, or test count as substitutes for answers.

The user now authorizes exactly two end-stage evaluations: **LME oracle and
semi-LSREP**, after repairs and targeted configuration/integration checks. **No
full LME-S campaign.** Foreground answering and judging should use cloud models;
background models and memory remain local. Provider/model setup is not complete.
Recorded-response LSREP reconstruction must perform the whole ICE pre/post-flight
path; historical responses become visible only at post-flight. Separate answer
probes measure memory benefit. Oracle perfection is an aspiration, not a license
to tune on held-out answers. These runs do not establish broad multi-user or
full-history distractor robustness.

## Implemented during this repair phase

| Area | What changed | What this does not establish |
|---|---|---|
| Extraction | Complete structured-output parsing, retryable failure, atomic multi-chunk writes, short-turn eligibility, specialist model separation | Universal extraction correctness |
| Codex sentences | Original source sentences/paragraphs, writer-supplied speaker boundaries, source hashes, graph links, direct lexical/vector search | Every legacy graph row upgraded; historical data was not reseeded |
| NLI | Selected adversarially trained DeBERTa after the first candidate failed; current source/model/text identities gate compact claims and turn summaries | World truth, completeness, or reliable arbitrary-length verification |
| Retrieval | Qwen3 reranker orders actual alternatives before packing; full PostgreSQL lexical queries preserve numeric, Unicode and late terms | A qualified hard rejection threshold or an answer-quality win |
| Representation | Shared supported summary/abstract selection across turn readers; complete raw fallback; bookmarks use the same rule | Rolling/batch summary semantic quality |
| Graph retention | Retrieval usage is separate from support; rereading cannot corroborate; quiet valid facts do not expire merely from nonuse | Independence of every repeated observation |
| Conflicts | Compare all relevant assertions using both attributed sources; atomic rollback when a replacement is rejected; explicit manual keep sets | Cross-conversation authority and general event-time inference |
| Time | Precise UTC/unknown provenance, source vs learned/valid clocks, dated evidence and current-time anchor | Source recording time equals real-world event time |
| Cold recovery | Linked claim evidence can resolve cold sources; preserve original vectors and known/unknown timestamp provenance | Complete archive metadata/cluster/source parity |
| Context budget | Actual optional-block accounting and reassembly, static context removed before evidence, protected project constraints, explicit required-overflow refusal | Exact tokenization for every future cloud model |
| Background outputs | Foreground model names cannot override summary/procedural background jobs; incomplete outputs cannot advance coverage/checkpoints; runtime yield propagates | All runtime/provider-control defects repaired |
| Summary scope | All-source conversation/batch/cluster eligibility before ranking; exact batch IDs and conversation identity; matching ablation wrapper | Query-selectable segments or completed batch-summary semantic verification |
| Summary freshness | Source/output snapshots catch edits, deletion, backfill and changed verification policy; both readers reject stale checkpoints | Fresh output is necessarily faithful |
| Independent source notes | Original-only generation, NLI-gated note substitution, complete source fallback, cached parts never fed into generation; complete embedding coverage | Compact query-selectable overview, long-source compression or answer-quality improvement |

## Validation so far

Latest regression selection:478 smoke, SQL and settings checks passed. Independent
note generation also passed8/8 actual NLI-plus-writer synthetic controls. The standalone
conversation/slot suite passed28/28. These selections overlap earlier test runs;
do not sum their counts as unique coverage.

Earlier actual-model qualification in this repair phase: the selected NLI
accepted13 supported and rejected17 unsupported controls; the summary/NLI selector
passed12 controls; the two-source reconciler passed12 direct and12 full SQL/writer
controls. Reranker ranking passed15 positive sets, while4/45 negative admissions
at a zero floor prevented activating hard rejection. These small, synthetic
controls informed implementation choices; they are not LME/LSREP answer scores.

## Newly exposed mechanisms and their repairs

- Optional slots/bookmarks/session status were inside the protected system block;
  the advertised static-first eviction could not actually remove them. Fixed.
- Bookmark injection bypassed summary support and clipped late evidence. Fixed.
- Background summary/procedural jobs inherited the foreground model name. Fixed.
- Truncated model output could look like a completed memory write. Fixed at the
  four repaired background text consumers.
- A max-timestamp cursor missed older edits/deletions/imports. Source snapshots
  now detect these; a snapshot is explicitly not an entailment verdict.
- The disposable ORM schema lacked the existing migration-owned slot identity
  index. Mirroring it removed two misleading regression failures; the production
  index was already correct.
- The rolling fold had its own unchecked-summary/raw-prefix reader and output
  word-cut. Complete evidence now reaches its generator, subject to an explicit
  capacity refusal rather than silent loss. Recursive generation is now removed;
  notes use originals and uncertain compression retains source evidence.

## Repair work still required

1. **Summary quality:** query-selectable independent notes and bounded overview;
   qualify long-source compression, batch summaries, and richer graph notes.
   Preserve supported detail and temporal corrections without dumping all history.
2. **Graph semantics:** resolve remaining open-vocabulary relation conflicts,
   event time and cross-conversation speaker authority; check useful connections
   rather than rewarding a denser graph for its own sake.
3. **Lifecycle and scope:** archive/source/cluster metadata parity; deletion of
   multiply supported facts; inspect shared retrieval and MCP path differences.
4. **Procedural memory:** authentic user-source boundaries, valid cited evidence,
   actual cited-source counts, project-safe matching, and reflection's second
   writer. A session's length must not masquerade as corroboration.
5. **Runtime and controls:** native model-server controls, residency/yield behavior,
   remaining asynchronous work, timeout and privacy issues, cloud reader integration.
6. **Instrument correctness:** current production path, classifier history, all-leg
   provenance, meaningful typed metrics, complete answer judging, configuration
   sensitivity and matched-budget comparisons. Fit/tune on development data only.
7. **End-stage confirmation:** only the agreed LME oracle and semi-LSREP runs.

The roadmap also retains product work (UI, graph/review panels, packaging, settings,
branching, search), learned routing/feedback work, and longer-term research. Those
are not equivalent to defects in the current memory repair. Existing unchecked
items often combine implemented pieces with remaining validation; do not turn
this list into a percentage-complete claim or silently check them off.

## Open roadmap records at this snapshot

The following is a mechanical listing of unchecked anchored entries, **not a
fresh assertion that every historical diagnosis/model/number in its title remains
true**. Follow the links for original scope and partial-completion notes. Current
behavior is owned by code and FEATURE_INVENTORY.md.

### Models and extraction

- [A9 Context-aware NER rework — THREE SEPARATE PIECES, three different gates](../ROADMAP.md#a9)
- [A12 Background-model specialisation — the whole background pipeline runs on ONE 26B general model, and it need not](../ROADMAP.md#a12)

### Classification, routing and feedback

- [B3 Learned MoE routing](../ROADMAP.md#b3)
- [B4 Feedback loop + fine-tune promotion (close the broken loop)](../ROADMAP.md#b4)
- [B6 Tree conversation](../ROADMAP.md#b6)

### Context and caching

- [C13 Caching strategy](../ROADMAP.md#c13)
- [C14 KV-cache persistence & cache-aware retrieval](../ROADMAP.md#c14)

### Temporal memory

- [T5 Wire `Temporal_Recall` into Track T — the label exists and Track T never got it](../ROADMAP.md#t5)

### Maintenance

- [D3 Agentic telemetry](../ROADMAP.md#d3)

### Coding memory

- [E10 Docs-for-coding](../ROADMAP.md#e10)

### Product and provider features

- [F1 Custom web frontend foundation](../ROADMAP.md#f1)
- [F2 Review-queue panel](../ROADMAP.md#f2)
- [F3 Graph view of Codex](../ROADMAP.md#f3)
- [F4 Full settings exposure](../ROADMAP.md#f4)
- [F5 Telemetry & forensics layer](../ROADMAP.md#f5)
- [F6 Select-text → add-to-context](../ROADMAP.md#f6)
- [F7 Real-time search integration](../ROADMAP.md#f7)
- [F8 Deep research mode](../ROADMAP.md#f8)
- [F9 Feedback UI](../ROADMAP.md#f9)
- [F11 Cloud API models](../ROADMAP.md#f11)
- [F12 Multi-model responses](../ROADMAP.md#f12)
- [F13 Session replay + conversation branching](../ROADMAP.md#f13)
- [F15 Hardware advisor & model recommender](../ROADMAP.md#f15)
- [F16 User retrieval overrides: force-deep-search + aggressiveness control](../ROADMAP.md#f16)

### Repair and instrument records

- [G2 Background-model client cleanup](../ROADMAP.md#g2)
- [G3 Runtime shared↔dedicated switching](../ROADMAP.md#g3)
- [G4 GPU gating fixes — RESCOPED 2026-07-29 (user): ICE owns its GPU share, and is EXPLICIT with Ollama](../ROADMAP.md#g4)
- [G8 Sticky-state persistence](../ROADMAP.md#g8)
- [G12 Dynamic LLM timeouts](../ROADMAP.md#g12)
- [G15 Null_Noise / Casual_Banter routing](../ROADMAP.md#g15)
- [G17 Audit trail](../ROADMAP.md#g17)
- [G19 Simulation-harness upkeep](../ROADMAP.md#g19)
- [G20 Dead / half-dead code sweep](../ROADMAP.md#g20)
- [G24 Async hygiene in the hot path](../ROADMAP.md#g24)
- [G25 Log privacy](../ROADMAP.md#g25)
- [G27 Shared-mode background model resolves to the WRONG model](../ROADMAP.md#g27)
- [G28 Style invariance — no decision may depend on HOW a thing is written](../ROADMAP.md#g28)
- [G29 The drift audit — logic copy-pasted, then quietly diverged](../ROADMAP.md#g29)
- [G30 Test-suite blind spots — the suite proves things connect, not that they work](../ROADMAP.md#g30)
- [G32 ICE↔Ollama control surface — ICE talks through a door that drops most of it](../ROADMAP.md#g32)
- [G40 No curated probe carries a usable `probe_type`](../ROADMAP.md#g40)
- [G48b Recall could only ever credit the episodic leg — every retrieval number is an episodic score](../ROADMAP.md#g48b)
- [G49 Two signals that land nowhere, and one that will read as a retrieval failure](../ROADMAP.md#g49)
- [G51 The graph is disconnected because entities were never LINKED, not because duplicates were never merged](../ROADMAP.md#g51)
- [G52 The evaluation harnesses do not call retrieval the way production does](../ROADMAP.md#g52)
- [G53 `conv_id` means two different things and `auto` mode silently loses one of them](../ROADMAP.md#g53)
- [G54 The harnesses classify without the conversation, and it moves 65% of the fusion weights](../ROADMAP.md#g54)
- [G55 Three typed metrics cannot measure what they are named after](../ROADMAP.md#g55)
- [G56 The answer judge truncates the answers it is comparing](../ROADMAP.md#g56)
- [G57 The write path, measured for the first time — four findings](../ROADMAP.md#g57)
- [G70 ⚑ WE HAVE NEVER MEASURED WHETHER ANY STORED MEMORY IS EVER READ](../ROADMAP.md#g70)
- [G73 ⚑ THE CONVERSATION FOLD FABRICATES IN 33-67% OF FOLDS, AND NOTHING HAS EVER MEASURED IT](../ROADMAP.md#g73)
- [G74 ⛔ THE CLUSTER-NAMING NUMBERS ARE RETRACTED — the harness did not call what production calls](../ROADMAP.md#g74)
- [G75 The summary trust gate admits invention PREFERENTIALLY — decide what replaces it AFTER the reseed](../ROADMAP.md#g75)
- [G72 ⚑ THE CLEAN BREAK — every pre-2026-08-26 store is dead data, and the reseed starts over](../ROADMAP.md#g72)
- [G71 ⚠ MOST OF THIS ROADMAP'S GRAPH FINDINGS WERE MEASURED ON A SUPERSEDED EXTRACTOR](../ROADMAP.md#g71)
- [G69 The relation vocabulary can be grown from the corpus instead of guessed — the loop is half-built and inert](../ROADMAP.md#g69)
- [G68 Extraction chunks every turn at 550 tokens, and truncates its output at 1,200](../ROADMAP.md#g68)
- [G66 ⚑ NOBODY HAS EVER MEASURED WHETHER BETTER FACTS PRODUCE BETTER ANSWERS](../ROADMAP.md#g66)
- [G63 ⚑ THE MODEL IS DECIDED: NuExtract3 replaces `qwen3:4b-instruct` as the background extractor](../ROADMAP.md#g63)
- [G65 The judge is unstable, and every graph number depends on it](../ROADMAP.md#g65)
- [G67 The maintenance agent has never run, and NuExtract3 may be what makes the graph graph-shaped](../ROADMAP.md#g67)
- [G60 Supersession semantics are unknown for 93% of the graph's relations](../ROADMAP.md#g60)
- [G48 The probe metric can only score ONE leg family, and tuning on it would delete the others](../ROADMAP.md#g48)
- [G76 Attributed claims and source-support verification — keep an assistant hypothesis from becoming an asserted fact](../ROADMAP.md#g76)

### Longer-term research

- [H1 Cross-conversation retrieval evaluation](../ROADMAP.md#h1)
- [H2 Multi-user evaluation](../ROADMAP.md#h2)
- [H3 Year-scale memory studies](../ROADMAP.md#h3)
- [H4 Probe realism](../ROADMAP.md#h4)
- [H5 Fine-tune scheduling on user machines](../ROADMAP.md#h5)

### Final checks and evaluation

- [Z1 Complete live system test](../ROADMAP.md#z1)
- [Z2 The read-the-output pass — one real conversation, judged by eye, BEFORE the experiments](../ROADMAP.md#z2)
- [Z3 FINAL redesign — the criticisms the next experiment must answer](../ROADMAP.md#z3)

## Continuation —2026-09-21, retrieval evidence preservation

V3 retains existing long-turn excerpts beside complete parent candidates until
packing; candidate presence cannot consume a diversity allowance. Removed warm
and cold prefix cuts. The reranker scores fitting pairs despite an oversized
neighbor and retains complete unscored fallbacks.510 smoke/SQL/settings controls
passed; actual local Qwen capacity control4/4. These synthetic mechanism checks
do not measure LME answer quality. Archive metadata parity, batch-summary support
and conversation-note selection remain unfinished.

Batch-note source support/freshness now implemented: NLI-supported compression
or original-source fallback, output-bound source manifests, stale-cache rebuilding,
complete provider bounds and concurrent-edit protection.523 focused checks and
11 standalone coverage checks passed. Cold archive metadata/source parity and
query-selectable compact conversation notes remain open. No new semantic model or
benchmark win is claimed from these controlled writer/reader tests.

Archive collision fix: an older cold copy no longer wins over a newer live
correction/privacy change.313 smoke and actual decay checks passed. This does not
finish archive metadata, scope or aggregate freshness parity.

Cold representation/session/intent/idempotency metadata now survives archival
and restoration.527 regression checks and61 temporal checks passed; source support
scores were controlled in round-trip tests. No new model promoted. Cluster/chunk/
parent links and summary freshness across storage moves remain outstanding.
Retrieval failure logging no longer emits exception strings containing raw memory.

Cold cluster visibility now preserves link membership and primary cluster,
applies exclusions/positive scope/batch lists before ranking, and does not invent
legacy membership or recreate deleted clusters. Archive excerpt/parent links and
aggregate manifest continuity remain open. Retrieval exception-payload privacy
failure and its control are recorded in TRAPS61.
