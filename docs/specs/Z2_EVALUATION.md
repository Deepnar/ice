# Z2 — question machinery, semi-LSREP, background-content quality, LME oracle

**Version: v3.** Decided with the maintainer 2026-10-14/15. **Approved by the maintainer 2026-10-15**; build starts in the next session, after tuning closes. **Z2 is not FINAL.**
FINAL is only the paper experiment that follows Z2 → fix pass → Z3 (roadmap
gates). Z2 builds the evaluation machinery FINAL will reuse, then measures the
frozen, tuned v3 system once. Z2 reruns only for parts still broken after its
first run.

**Controls over:** the Z2 sections of
[V3_POST_RESEED_REPAIR](V3_POST_RESEED_REPAIR.md) (§2.4–2.6 and the §2.5/§5.2
maintainer decisions), which remain the historical record and the source of the
oracle, bare-vector and replay contracts this spec cites. Where they conflict,
this spec wins. [FINAL_experiments](FINAL_experiments.md) §2.4/§2.5 (probe
taxonomy, citation-verified ground truth) are adopted and extended here.

Assumes decided specs: `V3_POST_RESEED_REPAIR.md` (repairs, tuning, frozen
configuration), `T_temporal.md` (timescope modes).

## 1. Decisions

**D1 — What Z2 is.** Seven parts, in this order:

1. **Speed measurement and fixes** for the replay write path (before any long run).
2. **Question machinery** on histories B and C: an evolving fact ledger, new
   questions with checkpoint-correct ("evolving") ground truth, and type quotas.
3. **One full-system semi-LSREP replay** per history, live decay, every checkpoint
   store saved, every strengthening/decay event logged.
4. **Background-content quality audit** of everything the background models wrote
   into the full-system store (full ICE only, no ablation).
5. **Read passes** on the saved checkpoint stores: full ICE, recent-only, ICE
   warm-vector, bare vector, and the read-time additive and leave-one-out
   ablations.
6. **Answers and grading** for every read pass, plus fair retrieval diagnostics.
7. **LongMemEval oracle** (500 instances).

**D2 — Histories.** B (1,119 original pairs, long-range entity/revision memory)
and C (325 pairs, dense long-input/compression), frozen in
`logs/v3-offline-input-plan-2026-10-10-r4/final-originals-{B,C}.json`. Both are
excluded from tuning. C is only the first segment (ICE-1, 325 pairs, ~1.1M tokens)
of the stitched ICE project conversation (1,737 user turns, ~5.9M tokens, real
timestamps); Z2 keeps C as frozen for replay cost, and FINAL uses the full
stitched conversation. Their legacy v2 probes (B: 91 families/638 observations;
C: 37/154) are **replaced**, kept only as a historical reference: they were mostly
single-message lookups with end-state answers.

**D3 — Clock.** B's and C's timestamps are synthetic imports compressed into four
days each, so decay cannot act plausibly on them. Z2 replays on a **declared
clock**: sessions of 25 consecutive pairs; 3 minutes between turns inside a
session; between sessions a deterministic seeded gap of 1, 2 or 3 days. B is then
about 45 sessions over roughly three months, C about 13 sessions over roughly a
month. Real, authentic timestamps are used unchanged wherever a history has them.
The clock table (session → start time) is written before the replay and is a
USER-REQUIRED review.

**D4 — Question machinery (built before Z2, reused by FINAL).**

- *Ledger first, questions second.* Two writer families extract, window by window,
  every durable fact, preference, decision, plan, piece of advice/procedure and
  every change to one (correction, reversal, update), each with exact citations:
  (turn, role, start, end, source sha256). Entries carry `valid_from` and, when
  superseded, `superseded_by` (the turn and entry that replaced it). Agreement
  between the two families marks an entry verified. A disagreement goes to a
  third-family tie-break, then to the review queue.
- *Evolving ground truth: ledger first, v2's forward walk as the audit.* Changes
  are captured when they are extracted (a correction or reversal links to the
  entry it supersedes). Then v2's method runs as an independent check: for each
  family, a verifier reads the new turns between consecutive checkpoints and
  answers "does anything here change this answer? cite it", with v2's
  temporal-anchor guard so a later step cannot swap in a different event. A change
  the walk finds but the ledger missed (or the reverse) goes to the review queue.
- *Ground truth is computed, not written.* A question's answer at checkpoint t is
  derived from the ledger state at t: the entries valid at t, plus the history
  of superseded ones for "what did it used to be" questions. Each occurrence
  stores its reference answer, its required-content list (meaning, not wording,
  as in the 2026-10-13 content contracts) and its citations.
- *Types and quotas* (share of question families per history):

  | Type | Share | What it asks |
  |---|---:|---|
  | Fact recall (user- or assistant-stated) | 15% | a stated fact or decision |
  | Knowledge update / correction | 15% | the *current* value after a change |
  | Temporal | 12% | when, in what order, or as of a date |
  | **Cumulative mega-span** | 10% | reconstruct a whole thread from many entries spread over ≥ 100 turns: "list all the characters and their powers", "re-explain the whole plan", "the full timeline so far" (the shape of the maintainer's handwritten B questions); the answer grows and changes over time |
  | Aggregation across sessions | 8% | counts and narrow lists spread over time |
  | Procedure / advice recall | 10% | steps or advice given earlier |
  | Negative / abstention | 10% | never stated by t: "don't know" is correct |
  | Preference | 5% | stable likes, dislikes, constraints |
  | Multi-hop | 5% | two cited entries combined |
  | Messy-phrasing variants | 10% | an existing family re-asked with typos, lowercase, terse or rambling wording (style invariance) |

- *Fact-based ground truth.* An occurrence's reference is a set of required ledger
  items plus their qualifiers, not a model-written model answer. An answer is
  correct when it carries every required item's meaning (any wording) with its
  qualifiers and contradicts none; cumulative and aggregation questions are also
  scored by item coverage (how many of the required items, and any wrong extras).
- *How questions are asked.* The question writer phrases each question the way the
  user actually writes in that history (casual, lowercase, "do you remember…",
  "can you list down…"), never leaks the answer, and asks something answerable
  from the cited entries alone and unambiguous at its checkpoint. A verifier
  rejects questions that fail any of these.

- *Cutoffs and repeats.* A family is first asked only after its supporting turns
  have left the recent window (40 turns). Every fact-like family is asked at
  least twice (first eligible checkpoint, and again ≥ 150 turns later); 30% of
  families get a third ask after the longest available gap. Update families are
  asked before **and** after the change, so the right answer differs.
- *Checkpoints* are session boundaries on the declared clock (D3). A question
  never sees turns after its checkpoint, and its ground truth never uses them.
- *Volume targets:* B about 300 occurrences, C about 120, after the split (D5).

**D4b — No-memory control prompts (maintainer, 2026-10-15).** Besides the memory
question types, each checkpoint gets prompts that need **no memory at all**:
general knowledge, self-contained tasks, small talk and follow-ups answerable from
the recent window. They are not abstention items (those ask about the history and
expect "don't know"); they test the retrieve-or-not decision (B2 + source-need gate;
audit A11 found base_retrieve True on 518/519 development prompts before its fix)
and whether injected memory costs tokens or harms an answer. Reported per system:
retrieve rate, memory tokens injected, and answer quality against recent-only on the
same prompts. Their share and generation are decided at the Z2 build; they sit
outside the D4 quota table so the memory types keep their shares.

**D5 — Keeping FINAL clean.** Questions are generated over B and C and split **by
question family**: 60% to Z2, 40% sealed for FINAL (hash-recorded, never read,
answered or graded before FINAL). FINAL additionally covers the maintainer's other
v1 conversations with the same machinery. Whatever Z2 measures will be fixed
against, so Z2 families are development material for FINAL purposes.

**D6 — Model roles** (high-limit opencode profiles, maintainer-approved):

| Role | Model |
|---|---|
| Ledger/question writers (two families) | DeepSeek V4.1 Flash; MiMo V2.5/V2.6 Flash |
| Ledger verifier / tie-break | the non-writing family of the two, then LongCat-2.0 |
| Answerer | GPT 6 Luna (frozen profile `opencode-luna6`) |
| Answer judge, background-content judge | Muse Spark 1.3 Contributor (`opencode-muse13`) |
| Second-opinion sample (background audit) | DeepSeek V4.1 Flash on 20% of judged items |
| Second answer judge | **FINAL only**, untuned, reported beside Muse |

Writers never answer or judge. If a profile becomes unavailable, the replacement
is recorded before use; no silent substitution.

**D6a — Every prompt is qualified before use.** Each writer, verifier,
question-phrasing and judge prompt is run on the development history first and
passes measured checks before it touches B or C: citation pass rate, two-family
agreement, question rejection rate, planted-control accuracy for judges (a
deliberately wrong but on-topic answer must not pass), plus a maintainer-reviewed
sample of its outputs. A prompt that fails is revised and re-qualified; the
qualification record is kept with the prompt's hash.

**D7 — Answer output cap.** Raise the answer output reserve from 1,500 to 4,000
tokens for Z2. Evidence: on the tuning answers, one question failed in 6 of 9
arms, and its successes used 1,427–1,492 tokens, about 1,030 of them reasoning.
This is a new frozen answer identity; tuning answers are not mixed with Z2 answers.

**D8 — Replay.** One full-system semi-LSREP replay per history with the frozen
tuned configuration (spec §5.2 decisions): recorded original replies, full ICE
preflight and post-flight per historical prompt, all ten maintenance jobs on the
declared clock, **live default decay** (option A). The replay logs every
strengthening (row, source time, amount, turn or probe) and every decay pass.
Questions retrieve like real use; their exposure is logged and tagged. Every
checkpoint store is saved. Speed fixes (D1.1) come first, measured on a short
slice of B, and never change the stored outputs' identity silently.

**D9 — Background-content quality audit** (full-system store only). Every kind of
artifact the background models write is inventoried from the actual writers
before judging. At minimum: turn summaries and abstracts; conversation folds;
batch summaries; graph claims and edges (relation, direction, polarity,
modality); entity notes and descriptions; cluster names and memberships;
procedural patterns; reconciliation and expiry decisions; session summaries;
compaction snapshots. For each kind, a stratified sample (all items if fewer than
100, else 100 per kind per history) is judged with its cited sources supplied:

- **faithful** — supported by its sources; nothing invented or contradicted;
- **attributed** — the right speaker; an assertion kept separate from a suggestion;
- **current** — not presenting a superseded fact as current (checked against the ledger);
- **complete** — required detail kept (summaries, folds, notes);
- **decision valid** — merges, expiries, supersessions match the ledger.

Graph facts and ledger entries are also matched directly, giving graph precision
and recall against the verified ledger. Each artifact kind also reports how often
it reached a prompt and the graded answers it appeared in. Muse judges; DeepSeek
V4.1 Flash re-judges a 20% sample for agreement; disagreements are listed for the
maintainer. This settles the open quality owners G49, G57, G60, G63, G67, G69,
G73, G74, G75 and G76 at Z2 scale.

**D10 — Read passes and ablations.** On every saved checkpoint store, with that
checkpoint's questions:

- *Controls:* recent-only; ICE warm-vector; bare vector (whole role-tagged
  messages filling the matched memory budget — frozen development choice,
  PROVENANCE 2026-10-14).
- *Ablations:* the twelve groups of V3_POST_RESEED_REPAIR §2.5 as **read-time
  switches**. Additive from bare vector up to full ICE in the frozen order;
  leave-one-out from full ICE, restoring between removals. Semantic maintenance is
  its own condition. All ablation conditions read with the same **neutral decay
  overlay**; decay is reported separately as its own condition and as settings
  sensitivity via overlays (D8 event log). Label: read-time ablation on a fully
  seeded store; indirect write-time effects are not removed.
- Identical assembled prompts reuse one cached answer.

**D11 — Scoring.**

- *Answers:* Muse absolute grades (correct, partial, incorrect, uncertain) against
  the checkpoint's required items, qualifiers and cited sources; the arm is never
  disclosed.
- *LSREP-native scores (v2, reformed):*
  - **1-5 score** per answer, anchored to required items: 5 all items and
    qualifiers, no errors; 4 one minor omission; 3 partial; 2 mostly missing or
    stale; 1 wrong or invented. Separate flags: used a superseded value; invented
    content (replaces v2's hallucination percentage); abstained.
  - **Ranking** of the conditions per question (v2 ranked four answers 1-4):
    blind, order randomised and repeated in two orders, ties allowed, judged
    separately from the absolute score; reported as win rates against bare vector
    and warm vector.
  - **Token efficiency** (v2): TUR = 1-5 score per 1,000 injected tokens, SPF =
    score per retrieved fragment, plus quality at matched budget; full prompts
    counted on both sides. Reported beside quality, never instead of it.
  - **Item coverage** for cumulative and aggregation questions.
- *Retrieval diagnostics:* source-ID credit (existing); verbatim span presence
  (existing); and **semantic credit**: every graph fact, claim or summary in the
  prompt that cites a gold turn is judged for "carries requirement X (meaning,
  not wording)" and for wrong content. This lets the diagnostic see compressed
  memory instead of rewarding only raw text.
- *Reporting:* per history and per question type; never one pooled headline.
  Paired comparisons against bare vector and warm vector; bootstrap intervals over
  **question families**, not occurrences; repeats reported as retention
  transitions. Incomplete answers are ungraded, never counted as wrong.

**D11a — Difficulty by construction (maintainer, 2026-10-15).** Tuning's
development set was near ceiling (full ICE 61/67, ICE vector-only 63/67), so it
could not show what ICE's machinery adds. Z2 must be hard enough to separate
systems, and the difficulty must come from how questions are built, never from
selecting questions a competitor happens to fail:

- weighted toward what plain retrieval cannot do from one message: updates and
  corrections (a stale value is retrievable and wrong), cumulative mega-span and
  aggregation (more items than one budget of raw messages holds), temporal order,
  abstention, and long-gap retention repeats;
- questions asked late in B and C (730K-1.2M tokens), so the history far exceeds
  any memory budget;
- **ceiling check on the development dry run:** a question type on which full ICE
  scores above 85% is made harder for every system (longer gaps, more required
  items, later cutoffs) before B/C generation. The check uses full ICE's score,
  never a competitor's failure.

**D11b — Token evidence: budget curves.** ICE may help by needing fewer tokens
for the same quality. Full ICE, ICE warm vector and bare vector are each read and
answered at four memory budgets (2k, 4k, 8k tokens and the derived default) on
every Z2 question. Report quality per budget and the smallest budget at which
each system reaches its own default-budget quality. Read passes are local and
identical prompts reuse answers.

**D11c — When a component earns its place (pre-registered, maintainer
2026-10-15).** A component keeps its place only if, in the leave-one-out
ablations, removing it costs answers beyond the noise floor (repeated identical
prompts, measured in Z2 as in tuning) on at least one question type, **or**
keeping it saves ≥ 10% of tokens at equal quality (D11b). Otherwise it is
reported as "no measured benefit", and the default becomes simplification, which
is a maintainer decision. ICE as a whole is judged the same way against bare
vector and warm vector: better answers beyond noise on some type, or equal
answers at ≥ 10% fewer tokens.

**D12 — LongMemEval oracle.** As V3_POST_RESEED_REPAIR §2.6: pinned release, all
500 instances, sessions sorted and checked, isolated per-instance replay; full
ICE, ICE warm vector, bare vector, and the all-evidence oracle control with the
same answerer. Muse grades; results are labelled **within-study, non-official**
(the original GPT-4o judge is not used). Native question types are kept.

**D13 — FINAL handoff (direction, specified at Z3).** FINAL reuses the question
machinery, replay runner, read-pass runner, scoring and background audit without
redesign. FINAL's inputs: the sealed B/C families; the maintainer's other v1
conversations (private, reported separately, never published raw); public and
synthetic data as the reproducible part; the second judge. FINAL's data roster
and public-dataset adapters are decided in Z3, not here.

## 2. Algorithm and data model

**Stages and artifacts** (each stage resumable, each artifact immutable and
hash-bound to its inputs; private content stays under `logs/`):

| Stage | Output | Calls |
|---|---|---|
| S0 speed profile and fixes on a 50-pair slice of B | profile report; fixes as reviewed production changes | local only |
| S1 ledger extraction, both writer families, 25-pair windows with 5-pair overlap | `ledger-raw-{writer}.jsonl` | ~2 × (45 + 13) windows |
| S2 ledger merge, citation check, tie-break, review queue | `ledger.json`, `review-queue.json` | disagreements only |
| S3 question families and cutoffs from the ledger by quota | `families.json` | ~1 per family |
| S4 occurrence ground truth computed from ledger state | `occurrences.json` | none |
| S5 maintainer review (USER-REQUIRED) | `review-decisions.json` | — |
| S6 Z2/FINAL split by family, FINAL part sealed | `split.json` + sealed hash | none |
| S7 replay B and C on the declared clock | checkpoint stores, event log | local |
| S8 background-content audit | `bg-audit.json` | Muse + sample |
| S9 read passes (controls, ablations, decay overlays) | per-condition contexts | local |
| S10 answers | per-condition answers | Luna |
| S11 grading and retrieval diagnostics | grades, semantic credits, report | Muse |
| S12 LME oracle | per-instance contexts, answers, grades | local + Luna + Muse |

**Ledger entry:** `{id, history, kind (fact|preference|decision|plan|advice|procedure|event), subject, value, speaker, asserted|suggested|hypothetical, citations[], valid_from_turn, superseded_by?, writers[], verified}`.

**Question family:** `{id, history, type, ledger_entries[], phrasing_variants[], split}`.

**Occurrence:** `{family, checkpoint_turn, clock_time, question, reference_answer, required_content[], qualifications[], citations[], expected_abstention, repeat_index}`.

**Citation check (mechanical, every entry and occurrence):** the cited span exists
byte-for-byte in the original role text; its turn is at or before the checkpoint;
the source sha256 matches. A failure removes the item; it is never repaired by
hand-editing the model output.

**Abstention ground truth:** an occurrence is a valid negative only if neither
writer's ledger holds a matching entry up to the checkpoint, **and** an embedding
search over all originals up to the checkpoint returns no passage the verifier
judges as answering it.

**Call budget** (per 5-hour limits, maintainer's opencode plan): S1–S3 about
2,000–3,000 writer calls; S8 about 2,000 judge calls; S10/S11 about 10,000 answers
and 10,000 grades across conditions (identical prompts deduplicated); S12 about
2,000 answers and grades. Writers and Muse have five-digit limits; Luna (4,230 per
5 hours) is the pacing limit, so answering runs in resumable batches.

**Compute** (from the 5.5 h, 251-pair development replay, before speed fixes):
replay B + C ≈ 26 h; LME oracle replay ≈ 4 days (≈ 11,000 messages); read passes
≈ 1 day. Speed fixes (S0) are measured before committing to these.

## 3. Files and integration points

New package `experiments/z2/` (research code; production stays in `src/`):

| File | Holds |
|---|---|
| `clock.py` | declared session clock (D3) and its table |
| `ledger.py` | window extraction prompts, merge, citation checks, tie-break |
| `questions.py` | family generation by quota, cutoffs, repeats, phrasing variants |
| `ground_truth.py` | occurrence answers from ledger state; abstention checks |
| `split.py` | Z2/FINAL family split and seal |
| `replay.py` | full-system replay on the declared clock; event log (wraps `chronological_replay.py` and the decay/strengthen writers) |
| `decay_overlay.py` | per-profile decay from the event log, applied to restored copies |
| `bg_audit.py` | artifact inventory, sampling, judging, ledger matching |
| `read_passes.py` | conditions on checkpoint stores (reuses `run_tuning_contexts`/`run_full_arm_coordinates` patterns and `bare_vector.py`) |
| `score.py` | answers/grades via the existing Luna/Muse paths; semantic credit |
| `oracle.py` | LongMemEval oracle adapter (V3_POST_RESEED_REPAIR §2.6) |

Production changes (maintainer-approved 2026-10-14, each gated and reviewed):
read switches for graph reading, turn summaries, conversation fold, procedures,
learned gate, cluster scoping, maintenance outputs and decay neutrality; the
answer output cap (D7); speed fixes from S0.

## 4. Failure modes and resumability

- A writer returns uncited or wrongly cited entries: dropped by the mechanical
  check, counted, never silently fixed.
- Two writers agree on a wrong entry: the S5 review sample and the background
  audit's ledger matching are the catch; report the measured disagreement rate.
- A question leaks future information: the cutoff check refuses it.
- Phrasing variants change the answer: kept, and reported as a style-invariance
  failure (G28), not removed.
- Decay overlay cannot reproduce a cold move: that profile is refused for that
  row, never approximated silently.
- A profile hits its 5-hour limit: the run pauses with saved units; resuming
  the identical command continues.
- Every stage writes durable, content-hashed units; an identical rerun resumes or
  verifies; changed inputs require a new output directory.

## 5. Validation checklist

1. **Machinery dry run on the 251-pair development history first.** Its ledger and
   questions are compared with the 67 reviewed content contracts: how many reviewed
   facts the ledger finds, and how many generated ground truths a reviewer rejects.
   Development data never enters Z2 or FINAL results.
2. Citation checker: planted wrong offset, wrong turn, future turn and paraphrased
   span all refuse.
2a. Prompt qualification (D6a) passed and recorded for every generation, verifier
   and judge prompt, including the ranking and 1-5 rubrics.
3. Ground truth: a planted correction yields different answers before and after it.
4. Split: sealed FINAL families are unreadable by every Z2 stage (test asserts the
   Z2 code paths never open them).
5. Replay: a short B slice reproduces the declared clock, decay passes and event
   log; decay overlay on a restored copy reproduces the live scores exactly for
   the default profile.
6. Read switches: each turns off only its own component (prompt diff on a fixed
   checkpoint), and neutral decay changes only decay-dependent ranking.
7. Background audit: the Muse/DeepSeek agreement rate is reported before any
   quality number is quoted.
8. Full smoke suite (disposable database) before the long runs.

**USER-REQUIRED:** approve this spec; review the S5 queue; approve the clock
table (D3); launch the long replay, answer and grading runs; confirm profile
availability.

## 6. Look-ahead

- FINAL reuses every stage unchanged except data roster, sealed families and the
  second judge (D13). Anything Z2 hard-codes to B/C is a parameter.
- The same machinery can later replay other memory systems through the same
  questions; nothing in the stages assumes ICE internals except S7–S9.
- Public datasets (FINAL) need their own adapters; the ledger format is chosen so
  a synthetic planted-fact ledger can be loaded directly.

## 7. Traps

- **TRAPS 32:** a metric that cannot see a subsystem credits nothing to it.
  Semantic credit (D11) exists for this reason.
- **TRAPS 128:** confirm a switch reaches a decision before measuring it
  (validation 6).
- A source-ID hit is not the required content; verbatim presence is not meaning.
- A model grading its own writing: writers never answer or judge (D6).
- Ground truth written freely by a model is how v2's question set went wrong;
  here it is computed from a cited, two-family ledger.
- More checkpoint observations are not more independent samples: intervals are
  over question families.
