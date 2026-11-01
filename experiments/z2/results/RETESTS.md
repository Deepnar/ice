# Z2 retests — what must run before any fix or finding becomes a number (v3)

Every fix from [FINDINGS.md](FINDINGS.md) that changed behaviour, and every switch built for one,
is **unvalidated until the retests below say otherwise**. Run from the fix-pass worktree
(`cd ~/Programs/ice-worktrees/fix-pass`). **GPU** items wait until Z2's S12 is done; **CLOUD**
items wait for the Go allowance (weekly reset ~Nov 2). Long runs are started by the maintainer.

**⚑ Grading rule (maintainer, 2026-10-29):** Z2 is **not** re-graded. Its grading finishes under
grade-1 on `main` (R6) so B, C and S12 stay complete and comparable; the new judge (grade-2) is
measured on a 250-answer sample graded under both rubrics (R5). Corrected Z2 comparisons use the
per-point rescoring of grade-1's own verdicts — if R5 shows that rescoring agrees with grade-2.
Retests that compare settings (R3, R4, R7) grade both sides with grade-2. FINAL uses grade-2 only.

## Timetable from 2026-10-31 (supersedes the dates below)

**Non-ICE local work starts early (maintainer, 2026-10-31).** The three memory systems on LoCoMo
(Graphiti, Mem0, A-Mem) do not depend on ICE's freeze: their slow part — ingesting LoCoMo's 2,951
pairs through each system's own write path on gemma4:e4b, then searching all 1,540 questions — runs
in the GPU-free windows before FINAL (the cloud week, the briefing week). One thing ties them to
ICE: each system's results are placed in **ICE's recent-turns prompt frame** (ICE's `recent_only`
envelope for the same question), so that only the memory differs — and the freeze may change that
frame. So `run_baseline.py` is split (build item 6): **search now** (stores and hits saved per
question), **wrap later** (minutes, after FINAL's LoCoMo read writes ICE's envelopes). Rules: they
use the GPU, so they pause for R7's reads; the baseline harness commit is recorded when they run and
not changed after; if the briefing's feedback changes how a system is run, that system is re-run.
Started and stopped by hand (Ctrl-C, any time); whatever is unfinished when FINAL's local phase
starts resumes inside it — in LoCoMo's lane, after ICE's own LoCoMo replay and read, beside the
other lane's ICE histories and the cloud stream. Ingest time per conversation is unmeasured; the
meter shows it after the first conversation. **`prep-locomo-controls` first** (revision 5, 2026-11-01):
~10 no-memory controls per conversation, ~200 cloud calls (minutes, ~$0.2) — so it needs the cloud plan
(from Nov 2); every search waits for it, because A-Mem's store lives in memory and a control added
after its search would re-ingest that conversation.

```bash
cd ~/Programs/ice-worktrees/fix-pass
uv run --no-sync python experiments/final/run_final.py run --only prep-locomo-export,prep-locomo-controls,prep-amem-vendor,local-baseline-search-mem0-LOCOMO,local-baseline-search-amem-LOCOMO,local-baseline-search-graphiti-LOCOMO
uv run --no-sync python experiments/final/run_final.py status --watch
```

1. **Now → Oct 31 night (GPU):** x3 reranker-input reads (~3 h; done — neither input joins the
   answers) → the two live trials, `experiments/final/trial.py lme` (~2 h) then `trial.py beam`
   (~40 min, Nov 1 if late). Cloud: nothing (weekly limit until Nov 2).
2. **Nov 1 → ~Nov 8 (GPU free):** the memory systems' ingest + search, all three, all 10 LoCoMo
   conversations. **Beside it, cloud from Nov 2**, one phase after another: Z2's own cloud run (R6) →
   Z3 judging run 1 (judge check) → the maintainer approves → run 2 (answers + grade-2) → cloud prep
   (synthetic conversations, A's and ICE2's ground truth) → the maintainer reviews A's and ICE2's packets.
   R10 (F4) once S12 is graded.
3. **~Nov 8–10:** R7 — the chosen settings together on B and C (GPU reads + cloud answers/grades);
   the memory systems pause for its reads if not done.
4. **~Nov 10:** Z3 freeze — switch defaults, D9 simplification, merge fix-pass into `main`, the freeze commit.
   Stop the memory systems' run first (Ctrl-C) if it is still going; at the merge, move fix-pass's other
   `logs/final-*` folders (questions, ledger, synth) into `main`'s logs as final-run-main and final-locomo were.
5. **~Nov 10–17:** research briefing (GPU free: the memory systems finish here if still running).
6. **~Nov 17:** buy Go Plus; **FINAL local starts** (2 lanes, ~9–11 GPU days → ~Nov 28), **beside it**
   `stream_cloud.py` answering and grading each checkpoint as its reads finish; once ICE's LoCoMo read
   is done, the memory systems' results are wrapped in its frame and answered in the same stream.
7. **~Nov 28 → early December:** the cloud tail (Go Plus weekly cap ~$60; ~$92 plus ICE2's share),
   then the report.

## What happens, in order (written 2026-10-29)

**Why no Z2 re-run is needed.** Every fix and switch so far acts at read time (what reaches the
prompt) or at grading. The expensive part of Z2 — replaying B and C turn by turn with the
background model (~11 h and ~14 h of GPU) — builds the stores, and none of the fixes changes what
is stored. So each retest re-reads the same saved stores for the same questions, answers and
grades only those, and compares settings question by question (paired, so a few points show).
A change to what gets *stored* (e.g. not writing some summaries) would need a new replay; none is
built.

1. **S12 finishes** (~3 h from 2026-10-29 afternoon). Nothing else needs the GPU until then.
2. **GPU day (~10–12 GPU hours):** R1 database tests + the full disposable rails; R8 fix-pass tests
   and the three classifier variants (B7); R2 candidate sweep on B (coverage + seconds per read →
   picks the candidate setting); the reads of R3 and R4 on B and C; R9 live trials.
3. **Cloud from the Go reset (~Nov 2), three phases, one after another, never mixed (maintainer,
   2026-10-29):**
   - **Phase 1 — Z2's own cloud run** (R6): `main`'s Z2 runner as it stopped — grade-1 grading of B's
     remaining phases, C, and S12's answers and grades. Run to the end.
   - **Phase 2 — Z3 judging** (`experiments/z2/z3_judging.py`, its own runner; it waits while any Z2
     run is active): R5 judge panel and judge checks on B and C, the R3/R4 retest answers and their
     grade-2 verdicts, R10's answers and LongMemEval grades (`r10-answers`, `r10-grade`: 308 answers +
     438 grades, ~$1.1; then `r10_report.py --grades` gives correct answers per bar), and anything Z2 or the GPU reads turn up (R7 is added once R2–R4 choose).
   - **Phase 3 — cloud prep** (`experiments/final/cloud_prep.sh`): the synthetic conversations and
     A's private ground truth, then A's review.
   Budget at measured prices: phase 1 ~$11.5, phase 2 ~$9–10 (answers ~$6.3, verdicts ~$1.5,
   judge checks ~$1), phase 3 ~$9 — ~$30, the month's whole remainder; phase 3 may spill onto Go
   Plus, which FINAL's cloud phase needs anyway. Phase 3 last does not delay FINAL: its local
   phase starts after the Z3 freeze.
4. **R7 (~Nov 8–10):** the chosen settings together on B and C — the fix pass's number. Needs a few
   dollars beyond Go's month if the budget is spent; else waits for Go Plus.
5. **Z3 freeze (~Nov 10):** set the switch defaults from R2–R4/R7, D9 simplification, merge fix-pass
   into `main` (only after R6), record the freeze commit.
5½. **Research briefing (~1 week, maintainer 2026-10-29) — gate before FINAL's local run:** a for external review (ICE v3 architecture, Z2→Z3 findings, the FINAL plan; a 4–6 page PDF, appendices if useful; task in the gitignored docs/PUBLISHING.md [PRIVATE:research-briefing]). Made from the frozen configuration and the retest results, sent for critical feedback while the
   design can still change; feedback goes into the pre-registration as a revision. No code, config or runs change.
6. **FINAL local (~Nov 17 → ~Nov 28, after the briefing):** 2 lanes, ~9–11 GPU days.
7. **FINAL cloud (Go Plus, ~$92):** answering can start per history as its local reads finish
   (~Nov 22 onward); at Go Plus's weekly cap (~$60) it takes two weeks or more → ~early December.

**The GPU day is one command** (`experiments/z2/gpu_day.py`, Z2's runner — meter, resume, lock):
gates (DB tests, full rails) → R8 (G60, G75, B7's four trainings; run even if a gate fails) → R2
sweep (7 settings on B, one stage each for timing) → coverage report → R3/R4 reads on B and C →
R9 LoCoMo slice. A failed gate holds every read and the slice; it waits by itself while S12 uses
the GPU. LongMemEval-S full-context and BEAM trials are not in it (short, run by hand after).

```bash
cd ~/Programs/ice-worktrees/fix-pass
uv run --no-sync python experiments/z2/gpu_day.py run
uv run --no-sync python experiments/z2/gpu_day.py status --watch
```

**Decisions wait for C (maintainer, 2026-10-29).** The GPU day measures; it decides nothing. C's
grade-1 results (R6) can arrive during the cloud week and may disagree with B — on the summary
policy, on fold notes, on any part. Every setting is chosen at R7 from B **and** C together (code
checks plus grade-2 answers), after R5; where C disagrees with B, that part's default waits or is
re-read, never set from B alone.

| | What | Needs | Status |
|---|---|---|---|
| R1 | F1 fix — database tests of the source check | GPU/DB | ✅ 93 + full rails |
| R2 | F2 — how many candidates to rank, and keep the rest | GPU | ✅ 256 chosen (default since 2026-10-30) |
| R3 | F10 — conversation order for "which came first" | GPU + CLOUD | ☐ |
| R4 | F6/F13/F14 — turn summaries in reads: off, or a margin of 1–2 | GPU + CLOUD | ☐ |
| R5 | F12 — judge the judge: 250 answers under both rubrics, planted panel, invariance | CLOUD (~$0.6) | ☐ |
| R6 | Z2's grade-1 grading finishes on `main` (B's remaining phases, C, S12) | CLOUD (~$11) | ☐ |
| R7 | All fixes together on B and C, answered and graded | GPU + CLOUD | ☐ |
| R8 | fix-pass ⏳ items (G79, G60, G75, B7 classifier variants) | GPU/DB | ✅ G60 replaced (F20), G75 off, B7 no change |
| R9 | Live trials stopped earlier (LoCoMo slice, LongMemEval-S full context) | GPU | LoCoMo ✅; LME-S ✅ (2026-10-31, `trial.py lme`); BEAM ✅ (2026-11-01, `trial.py beam`) |
| R10 | F4 — B2 "no memory" on a third of LongMemEval | GPU | ✅ read (2026-11-01): reranker rescue reaches the answer on 94/136 vs 17 shipped; bar and ordinary-chat cost open |
| R11 | F18 — clusters dropped from FINAL's parts test (audit A12); decay tested on S4 + S9 | design | ✅ |

**Cloud budget (Go, ~$30 usable Nov 2–27; measured per-call prices):** Z2 grade-1 finishing — B's
remaining claim checks and diagnostics ~$4, C ~$4, S12 answers + grades ~$3.5 — ~$11.5; finish cloud
prep ~$9; R5 ~$0.6; retest answers + grade-2 grades (6 settings × 174) ~$5.7 — **~$27 in all**, ~$3
slack. Order: R6 (Z2's cloud run) → Z3 judging (R5, retest answers and grades) → cloud prep. FINAL's own cloud
(~$92) and any private-question top-up need Go Plus.

## Cloud plan from the Go reset (recosted 2026-10-31 at measured prices; answers included)

Per call: an answer $0.0026 on B (~21k input tokens; ~$0.0045 at 256-candidate prompts, ~36k), $0.0015 on C
($0.0020 at 256); a grading call $0.0010. Phases, never mixed:
1. **Z2's own cloud run** — B's remaining ~4,000 grading calls ($4.0), C's grading ~4,060 ($4.1), S12's 1,982
   answers and ~2,500 grades (~$2.8): **~$10.9**.
2. **Z3 judging, run 1 — the judge check**: planted panel (932 calls) + B/C judge checks (~320): **~$1.2**.
   The maintainer reads the report and blind sheet; `touch logs/z3-judging/judge-approved` to continue.
3. **Z3 judging, run 2 — answers + grade-2**: six 256-candidate settings × (174 B + 75 C) = 1,494 answers
   (~$4.7 B + ~$0.9 C) and 1,494 grades (~$1.5): **~$7.1**.
4. **Cloud prep** (synthetic conversations, A's and ICE2's chains): **~$10**.
Total **~$29** of the ~$30 Go allows until Nov 27; a $5 top-up (Extra Usage) covers any overrun. **Go Plus is
bought only when FINAL starts** — its month then covers FINAL's cloud phase, run beside FINAL's local phase:
each history's answers start as soon as its reads finish (a cloud lane that waits for reads).

## To build before FINAL (maintainer, 2026-10-31) — in this order

1. ✅ Reranker input switches (query context = the classifier's prior-turn prefix; topical instruction) and their
   B/C reads in `gpu_extra.py` (x3-*) — d22ed60. Coverage decides whether they join the Z3-judging answers.
2. ✅ Score-blend simulation (`blend_sim.py`): no blend helps (B level, C worse) — FINDINGS F23; nothing built.
3. ✅ (`experiments/final/stream_cloud.py`) **FINAL's cloud runner waits for reads** — a cloud lane that stays alive, polls, and starts each history's
   answers the moment its reads finish, then its grades the moment its answers finish (per history, step by
   step), so the Go Plus month covers FINAL's local and cloud phases together. Approved design (2026-10-31):
   the local phase stays snapshot-then-read (LSREP's independent reconstruction); the cloud streams by
   **whole checkpoint** (cache sessions kept together), grades partially (never marked finished, no
   report) and finally once inputs are complete. Stops, plan limits and cut lines disrupt nothing: answers
   and judge calls are durable per call; the judge waits out limits instead of counting them as failures;
   a line cut mid-write is skipped and redone. Never run FINAL's own cloud phase while it streams.
4. ✅ **ICE-2 in FINAL** (history `ICE2`: the chat after C in the same project; 331 pairs, ~1.09M tokens —
   C is 1.13M — from the later DeepSeek export through paper 1's own extractor, which rebuilds C's 325 pairs
   exactly): `private_originals.py` writes it, registered in `histories.py`, in FINAL's roster, report suite
   and `private_chain.sh`/`cloud_prep.sh` (its chain runs with cloud prep: ~5M tokens like C's, ~$1), review
   packet as for A/SW/LAP. Pre-registration revision 4 states why and when (after Z2's C results).
5. ✅ Live trials by hand: LongMemEval-S full context, BEAM slice — both `experiments/final/trial.py`
   (`lme`: 2 questions of the pinned slice; `beam`: one 128K conversation's first 30 turns through the
   FINAL runner's own stages — the `--limit 30` run would replay all 20, ~8 h). Ctrl-C stops either
   cleanly; the same command resumes. LongMemEval-S's first attempt found 252 empty sessions in the
   slice that crashed the replay (fixed: left out and counted); its first question then ran clean
   (261 turns, 66 min, all five reads) — and showed F4 on LongMemEval-S (B2 "no memory", so full ICE's
   prompt equalled recent-only's). LongMemEval-S ✅ 2026-10-31: both questions done, no failures; the
   second (247 turns, 72 min) retrieved and carried the answer as a fact and a summary; every control
   (plain search, all evidence, full context) held the evidence turn and recent-only none. One of 46
   source-proof calls exceeded the support model's 512 tokens (WARNING, "unknown") — count it in FINAL. BEAM ✅
   2026-11-01: BEAM-100K-1's 30-turn replay (23 min) and its 10 questions under all four conditions, no
   errors; B2 refused memory on 2 of 10 (F4). Its export step had crashed first on 5 event_ordering
   questions whose source ids group one event's messages in nested lists (fixed: flattened); the trial
   ran on conversation 1, exported whole before the crash, so its result stands.
6. ✅ **Memory systems: search now, wrap later** — `run_baseline.py search|wrap`; runner stages
   `local-baseline-search-<system>-LOCOMO` (no ICE input; last in the local order, in LoCoMo's lane) and
   `local-baseline-<system>-LOCOMO` (the wrap, needs ICE's LoCoMo read). Stop anywhere: a half-ingested
   store is rebuilt from its first session; after ingest, Graphiti's and Mem0's on-disk stores and every
   saved hit are kept (A-Mem's in-memory store is ingested again); each file is written whole or not at
   all. `logs/final-run-main` and `logs/final-locomo` now live in `main`'s logs (fix-pass symlinks), so
   what runs now is where FINAL, run from `main`, looks. Live stop/resume trial (2026-10-31, two LoCoMo
   sessions, three questions per system) passed for all three after fixing three bugs no fake could
   show (TRAPS 154): the runner's stray `--dataset` argument, Mem0 2.2.1's `search(filters=…, top_k=…)`,
   and Graphiti 0.31's never-created Kuzu full-text indexes (now created from its own definitions).
   **Mem0's session date (maintainer, 2026-10-31):** open-source Mem0 cannot be told a session's date
   (`timestamp` is platform-only and raises), so its extraction anchored "yesterday" to the run day — a
   2023 LoCoMo event was stored as "October 30, 2026". Graphiti (reference_time) and A-Mem (note time)
   get each session's date. Mem0 now ingests each session with the clock its modules read set to that
   session's time (`session_clock`; its code and prompts untouched) — as if run live that day; the
   trial then stored "May 7, 2023", LoCoMo's own answer. Disclosed in the pre-registration's §4.
7. ✅ (d017a0f … fb1f701) **The memory decision in FINAL** (pre-registration revision 5, maintainer 2026-11-01) — MDM misses,
   MDF false alarms, NMH no-memory harm, POS position contrast:
   - every read records the gate's breakdown (each term) and the memory read's wall seconds
     (`production_parity` / `seed_v3` stage record — harness, not `src/`);
   - no-memory controls for S1–S8 from `questions.py`'s control machine (sized from measured calls);
   - ~100 no-memory controls for the LoCoMo conversations, through ICE (full, 2K) and the three memory
     systems' search, answered and compared blind with recent-only;
   - the report: MDM by suite and by each benchmark's question types, MDF/NMH, synthetic quiet turns
     from the replay's per-turn gate record, and POS (paper 1's token-crossover table, extended).
8. — **The gate inside running chats (FINDINGS F24): no change** (maintainer, 2026-11-01). B and C are in
   the classifier's training data (in-sample); FINAL's unseen histories measure the gate (MDM/MDF/NMH).

## R1 — F1 fix, database tests (GPU/DB, ~20 min)

```bash
uv run --no-sync python tests/support/disposable_database.py -m pytest tests/smoke/test_source_need_route.py tests/smoke/test_temporal_axis_path.py -q
uv run --no-sync python tests/support/disposable_database.py -m pytest tests/smoke -q
```
Pass: all green; `positive` route test shows memory retrieved with no source-judge call.
Then R7 shows the effect on answers (expected from Z2's `loo_gate`: +47 points on B's 18 skipped).

## R2 — candidates ranked, keep the rest (GPU, reads only, ~1–2 h)

Each read of B's 174 questions under one setting; full ICE as it runs on fix-pass (F1 fixed), so
`retest_cand64` is the new baseline. Time each condition separately (rank time per read is the
cost of a larger pool).

```bash
for c in retest_cand64 retest_cand128 retest_cand256 retest_cand512 retest_cand64_keep retest_cand128_keep retest_cand256_batch16; do
  /usr/bin/env time -f "$c %e s" uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
    --replay logs/z2-run-main/s7-B --asks logs/z2-questions-final/B/z2-asks.json --out logs/z2-retest-B --conditions $c
done
uv run --no-sync python experiments/z2/retest_coverage.py --history B --units logs/z2-retest-B/units \
  --reference logs/z2-run-main/s9-B/units --occurrences logs/z2-questions-final/B/occurrences.json
```
Measures: cited-turn coverage (Z2 reference: full ICE 71.3%, plain search 87.5%), pieces, memory
tokens, seconds per read. **Pick** the smallest setting within 1 point of the best coverage whose
read time is at most ~3 s above `retest_cand64`; batch 16 is kept if it is faster at equal
coverage. Coverage is presence, not quality — the chosen setting goes on to R7 for answers.

**R2 RESULT (GPU day, 2026-10-29; B, 159 questions with cited turns; fix-pass with F1 fixed):**
cited-turn coverage / seconds per read — 64 candidates 81.1% / 7.3 s; 128: 85.5% / 10.7 s; **256: 90.8% /
15.2 s**; 512: 90.8% / 15.2 s (the pool is ~250); 64 + keep the rest 82.2% / 7.4 s (fills the budget,
+1 point); 128 + keep 85.9%; 256 with batch 16 90.8% / 22.1 s (slower, rejected). Reference: plain vector
search 87.5%, Z2's ICE as run 71.3% (the F1 fix accounts for 71→81). 256 is the first setting above plain
search, at +7.9 s per message — over this plan's own +3 s rule, so the pick is the maintainer's (128 is the
middle ground); answers decide at R7.

**R2 follow-up (`experiments/z2/gpu_extra.py`, after the GPU day):** the corrected B7 (moved here from the
GPU day), and at 256+ candidates on B — wider legs (BM25/vector 300, facts 192, rank up to 512) for the
never-found turns, originals first, the reranker keeping the policy's text, wider + originals — plus C at
64 / 256 / wider; each with `funnel.py` (loss stage per cited turn, per-leg shares) and seconds per read.

**Decided 2026-10-30 (maintainer): 256 candidates** (`retrieval_rerank_candidates` default). Keep-the-rest and
wider legs stay off (no net coverage gain). Best combination measured: wider legs + originals first, B 92.9%.
The order and summary answers (R3/R4, Z3 judging) are therefore read at 256: `gpu_extra.py` stages x2-*
(B: order, no summaries; C: order, originals first, policy, wider + originals, no summaries).

## R3 — conversation order (GPU read + CLOUD answers/grade)

```bash
uv run --no-sync python experiments/z2/read_passes.py --history B --database ice_z2_main_b \
  --replay logs/z2-run-main/s7-B --asks logs/z2-questions-final/B/z2-asks.json --out logs/z2-retest-B \
  --conditions retest_cand64,retest_conversation_order
uv run --no-sync python experiments/z2/answers.py --units logs/z2-retest-B --out logs/z2-retest-B-answers --answerer muse12
```
Then grade both settings with fix-pass's `experiments/z2/grade.py` (grade-2) into a new `--out`,
with the same arguments as Z2's B grading (`--units logs/z2-retest-B --answers logs/z2-retest-B-answers`). Pass: temporal questions improve (ICE reversed 7
of 21) and no other type falls by more than its interval. Same for C when C is graded.

## R4 — turn summaries in reads (GPU read + CLOUD)

Z2 already shows (grade-1, all-points): adding summaries −5.8 (±6.0), summaries off in full ICE
+4.6 (±5.5); where ICE was partial and plain search correct, ICE carried a median 59% of the cited
turns' words (96% when both correct). Three settings against `retest_cand64` (preset `retest_summaries`): `retest_no_summaries` (turn
summaries off in reads; stored ones untouched), `retest_original_margin1` and `…margin2` (F14: a
summary replaces the turn's text only when it out-scores it by 1 or 2 logits). Plus F17's two: `retest_policy_text` (the reranker
keeps choose_representation's pick) and `retest_originals_first` (that, with the summary-first
intents off — summaries only as the budget fallback; the allowance B's evidence supports). Answer
and grade as R3; run `retest_coverage.py`, and `summary_allowance.py` on the retest units to see
the policy land. Decide with R7 on B and C.
**A v3 change that removes something v2 shipped is a maintainer decision** ("memory is earned").

**R3/R4 CODE CHECKS (GPU day; B reads done, C running):** conversation order — the same pieces, now
100% in turn order (0% before), coverage unchanged (81.1%). Summary share of single-turn pieces in the
prompt: shipped 50%, margin 1 44%, margin 2 37%, policy text 26%, originals first 10%, summaries off 0%;
coverage 79.5–81.1% throughout (summaries change detail, not which turns). Memory tokens rise as
originals replace summaries (20.7k → 28.6k). Answers and grade-2 decide (Z3 judging).

## R5 — judge the judge (CLOUD, ~$0.6)

Light, but every part runs: the largest single loss Z2 found was the judge (F12).
1. **250 answers graded under grade-2**, already graded under grade-1 — about 60% from the strata
   F12 found mis-graded and 40% controls, spread across ICE and plain-search settings, B and C:
   - issue strata (~150): grade-1 *partial* with every required point covered (60), grade-1
     *incorrect* with every point covered (40), downgrades whose reason cites dates/timestamps (30),
     C answers of the same kinds once C's grade-1 exists (20);
   - controls (~100): grade-1 *correct* (35), *incorrect* with points missing (35), *partial* with
     points missing (30) — grade-2 must not inflate these.
   Reported per stratum: how many verdicts change, and **how often grade-2 agrees with the
   per-point rescoring of grade-1** (the number that decides whether Z2's rescored comparisons
   stand in for a re-grade). The answers that change go into a blind sheet (answer, REQUIRED,
   SOURCES, both verdicts; condition hidden) for the maintainer to read; a change the maintainer
   rejects is a rubric defect, fixed before FINAL grades anything.
2. **Planted panel** — `experiments/z2/qualify.py` under grade-2 beside its grade-1 results: planted
   wrong stays wrong, planted correct stays correct.
3. **Invariance** — 20 answers graded as given and with one true extra detail from the
   conversation appended: the grade must not change.
4. FINAL's cross-family judge sample (pre-registration §4, ~450 answers) stays as planned.

Script to build before the cloud returns: `experiments/z2/judge_check.py` (strata draw from the
saved grade-1 verdicts, grade-2 calls through `grade.judge_caller`, report + blind sheet).

## R6 — Z2's grade-1 grading finishes (CLOUD, ~$11.5)

`main`'s Z2 cloud lane, restarted as it stopped (B's claim checks, ranking, controls, context
precision; C; then S12's answers and grades) — grade-1, unchanged. **It must finish before
fix-pass merges into `main`**, which brings grade-2.

## R7 — everything together (GPU + CLOUD)

Full ICE with F1 fixed + R2's chosen candidate setting + R3/R4 if they passed, read, answered and
graded under grade-2 on B and C, against plain search and long context. This is the number the
fix pass reports; FINAL's design freeze waits for it.

## R8 — fix-pass ⏳ items (GPU/DB)

As listed in the ROADMAP's FIX-PASS checklist: G79 (replay writes batch notes), G60
(`logs/fix-pass/reconcile_eval.py`, then `tests/test_codex_facts.py`), G75
(`logs/fix-pass/claim_check_eval.py`), **B7** classifier variants — shipped, standardize,
layernorm, kaiming (`uv run --no-sync python /home/deepnar/Programs/ice/logs/fix-pass/b7_variants.py`;
keep a variant only if it beats shipped on both F1 and the style flip rate).

## R9 — live trials (GPU)

A LoCoMo replay slice (sessions ending with an empty assistant side — never exercised), one
LongMemEval-S full-context read, short BEAM / LongMemEval-S / baseline trials.

## R10 — F4 (after S12 is graded)

Of LongMemEval's questions where B2 said no memory (97 of 298), check whether the answer was
already in the recent window (oracle histories are short) before calling any of it a miss.
**Done 2026-10-31 (FINDINGS F4):** no window exists in these reads — 130 of 456 got no memory and
plain search held the evidence on every one checked. R10 is now the fix, not the check.

**The fix (2026-11-01, off by default until Z3):** `memory_source_rescue_mode="reranker"` — on a
negative, retrieve and admit when the best reranker score reaches the bar (0 = the reranker's even
odds); no judge or proof call. The re-read (GPU, ~6.5 h on two workers, ~320 s a question; stops
cleanly — the questions in flight are redone; the same command resumes):

```bash
cd ~/Programs/ice-worktrees/fix-pass
uv run --no-sync python experiments/z2/gpu_extra.py run --only r10,r10-report
uv run --no-sync python experiments/z2/gpu_extra.py status --watch        # the meter, from another terminal
```

The report runs by itself at the end (`logs/gpu-extra-main/r10-report.log`).

Reads both rescues in the same run (`full`, `full_rescue_reranker`); the report gives, for the
questions the gate refused, how many reach the answer under each, and at each bar how many are
admitted (abstention questions apart), beside no memory, plain vector search and all answer
sessions, and split by question type. If it recovers most and the bar holds, the setting joins
Z3 judging run 2's answered settings and R7. FINAL's LongMemEval-S report gets the same
question-type split (maintainer, 2026-11-01).

**Result (2026-11-01, v3, LongMemEval oracle, code check on prompts, no answers graded).** 146
refused questions read, 0 failures (136 answerable, 10 abstention). The answer reached the prompt on:
no memory 3 · shipped proof rescue 17 · **reranker rescue at bar 0: 94** (112 admitted) · bar −4: 105
(131 admitted) · bar +3: 77 (86 admitted) · no gate (always fetch) 109 · plain vector search 136 ·
all answer sessions 136. By type at bar 0 (shipped → reranker, of refused): knowledge-update 3 → 22
of 22, single-session-user 5 → 18 of 21, single-session-assistant 5 → 8 of 8, temporal 3 → 8 of 15,
multi-session 1 → 27 of 47, preference 0 → 11 of 23.

What the numbers can and cannot say:
- The shipped-vs-reranker comparison is like for like (same store, same words, same check).
- The 27 always-fetch "misses" are mostly invisible to the check, not retrieval failures: 10 are
  preference questions whose reference is a rubric ("The user would prefer…"), and most of the rest
  are counts, sums or short answers ("two", "$750", "3.83") that ICE carries reworded as facts. Plain
  search stores the original turns, so it matches word for word; ICE vs plain search here is not a
  fair comparison (oracle histories hold only answer sessions, and the check sees words, not meaning).
- Every question in this slice needs memory, so a lower bar always looks better here. The bar's cost,
  memory added to ordinary chat that needed none, is **not measured**.
- "Abstention admitted" is not a cost: those questions are still about the past.

**The other side — memory nobody needed (built 2026-11-01).** 146 of the maintainer's own messages
labeled "needs no memory" (`labeled_prompts.jsonl`, personal, Zero_Shot; those stored word for word in
B left out), each read in a fresh chat on a copy of B's store (`ice_z2_nomem_b`; the original is never
written) with the rescue's bar off. The report gives, at each bar, memory let in on R10's needed-memory
questions beside these, plus the tokens added and seconds per read. ~30 s a message, ~75 min:

```bash
cd ~/Programs/ice-worktrees/fix-pass
uv run --no-sync python experiments/z2/gpu_extra.py run --only nomem,nomem-report
uv run --no-sync python experiments/z2/gpu_extra.py status --watch        # the meter, from another terminal
```

The report: `logs/gpu-extra-main/nomem-report.log`. Trial on 3 messages: the main gate itself fetched
memory on 2 (counted apart — not the double-check's doing); the third was refused and the double-check
let ~1,900 tokens in at best score 2.3.

**Result (2026-11-01, v3, store B, fresh chat, code check, nothing answered).** 146 read, 0 failures, 17
candidates left out (stored word for word). The main gate itself fetched memory on 67 (46%) — not
the double-check's doing; many are follow-ups pointing at something earlier in their original chat, missing in a fresh
chat. The double-check saw the other 79. Memory let in, needed (R10, of 146) vs not needed (of 79):
bar −4 95% / 87% · −2 88% / 66% · −1 84% / 54% · **0 80% / 37%** · **+1 77% / 27%** · +2 71% / 22% ·
+3 62% / 13%. Checked by eye at bar 0: 1 of 29 found its own turn (a two-word message); the other 28
pulled the maintainer's earlier talk on the same theme — related, not needed. ~2,000 tokens added each.

Reading: the reranker scores how related memory is, not whether the message needs it, so it cannot
separate the two cleanly. **Bar 0 chosen (2026-11-01).** Between bars only a few needed-memory questions
move, and most are ones the literal check cannot see: 0 → +1 drops 4 (2 preference, 1 GPA average, 1
commute — only the last counted as "reached") to save 8 of 79 unneeded; 0 → −1 adds 5 (and 1 abstention)
for 14 more unneeded. A miss costs a wrong answer; unneeded related memory costs ~2,000 tokens and an
unmeasured, likely small, harm — and unneeded cases scale with ordinary chat. (A first reading called +1
the knee from the "reached" counts alone — 93 vs 94 — which miss 3 of those 4.) The LongMemEval grading
(always-fetch answers + no-memory answers, every bar replayed offline) confirms or moves it. Not measured:
whether unneeded related memory makes an answer worse (nothing was answered).
