# Handoff — ICE v3, 2026-10-15 (Claude Code; resumed from a Codex session)

**State, not a queue.** ROADMAP owns work; the v2 paper remains frozen.

## Told → did

**Previous handoff (2026-10-08) told:** implement the eleven post-reseed repair
packages, then bounded tuning, freeze, then Z2. Codex then ran 77 commits of repairs
and instrument qualification (10-08 → 10-14), and the maintainer stopped it for
expanding pre-tuning gates instead of tuning.

**This session was told:** stop the loop, finish tuning, act on what tuning exposes,
then close out and push. **Did:**

- **Tuning finished and closed.** Four-arm current-reader baseline (67/67), 30
  sealed + 8 leg-profile full-arm read coordinates, two Luna answer plans (855
  answers), Muse grades. No survivor beat the ±3/67 noise floor; settings frozen
  unchanged ([results](../experiments/v3_repair/results/tuning-freeze-2026-10-15.md)).
- **Exposed and recorded:** base leg weights inert (G29, kept as fallback, TRAPS 128);
  incomplete answers = 1,500-token output reserve eaten by reasoning (P9; Z2 uses
  4,000); the old development store's conversation fold is refused (frozen v1
  proofs; PROVENANCE 10-14); development set near ceiling.
- **Bare-vector control built and tuned** (whole messages): 64/67 reviewed-span
  availability vs full ICE 60-61 at matched budget (PROVENANCE 10-14).
- **Z2 specified and approved:** [Z2_EVALUATION](specs/Z2_EVALUATION.md). FINAL
  roster rewritten in [FINAL_experiments](specs/FINAL_experiments.md) §2.1/§2.7.
- **Tests grouped** quick/research/db (`tests/smoke/conftest.py`); CLAUDE.md test
  commands and work-on-main line updated (maintainer-approved).

## Current measured state (v3, development history, not a score of record)

Full ICE 61/67 correct; ICE vector-only 63; ICE without graph 63; recent-only 41.
Re-asking the identical prompt flips up to 7 answers (net ±3). Memory clearly helps;
on this easy, lookup-heavy, 183K-token set ICE's extra machinery shows no answer
benefit over its own vector leg. That is a warning, not a verdict — Z2 is designed to
settle it.

## Position and next work

**⚑ Do first, before the Z2 replay:** the short list in ROADMAP's
[CURRENT POSITION](ROADMAP.md#-current-position--2026-10-15-v3) block — G27 (wrong
background model in shared mode), G12 (timeouts), G4 + G32 (explicit Ollama control,
inside S0), G68 truncation check on C. That block also sorts every other open fixing
item into after-Z2 / conditional / parked.

**Next: build Z2** in the spec's order (S0 → S12): replay speed profile on a 50-pair
slice of B; question machinery (two-family cited ledger, computed evolving ground
truth, quotas incl. cumulative mega-span) **dry-run on the 251-pair development
history first** and checked against the 67 content contracts; prompt qualification
(D6a) for every generation/verifier/judge prompt; read switches (production code,
maintainer-approved 2026-10-14); declared session clock (USER-REQUIRED review);
replay with live decay and event log; background-content audit; read passes with
ablations, bare vector and budget curves; LME oracle.

Decided, do not re-derive: Z2 ≠ FINAL; Muse judges, Luna answers (4,000 reserve in
Z2), writers DeepSeek V4.1 Flash + MiMo Flash, LongCat tie-break, second judge FINAL
only; ablations are read-time switches on saved checkpoint stores; decay live (option
A) with an overlay computed from logged events; questions retrieve like real use with
exposure logged; difficulty by construction, never by filtering on a competitor's
failures; component keep rule D11c; B/C timestamps are compressed synthetic imports
(declared clock); C is only ICE-1 of the 1,737-turn stitched project chat (FINAL uses
the full chat); FINAL roster tiered (synthetic, private groups, native LME oracle+S,
BEAM; MemoryAgentBench subsets tier 2; LoCoMo and LSREP-adapted public sets dropped).

USER-REQUIRED pending: approve the declared clock table; review the S5 ledger queue;
launch long replay/answer/grading runs; confirm opencode profile availability.

## Validation and propagation

Full disposable smoke suite before push: 1,467 passed (17 min 45 s). Quick group 598 tests
~20 s; research-not-db ~570 ~1 min; db 292 ~17 min. Propagated: ROADMAP (current
phase, Z1/Z2/Z3/G29), PROVENANCE (10-14 baseline/floor, bare vector; 10-15 tuning),
TRAPS 128-129, FEATURE_INVENTORY, MODELS, CLEANUP, README, repair spec, Z2 and FINAL
specs. ICE_Architecture n/a (no production behaviour change). No `src/` change this
session. Alembic head unchanged (`80f1ac593d72`). Private evidence stays under
`logs/`; the private conversation roster is in gitignored `docs/PRIVATE_CONTEXT.md`.
Pushed at the end of this session by maintainer request; the push freeze then
resumes.
