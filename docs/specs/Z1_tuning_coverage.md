# Z1-prep — Staged parameter tuning + whole-system coverage matrix

Assumes decided specs: ALL earlier S1 specs (this pass runs against the system
they describe, after implementation); `FINAL_experiments.md` (the synthetic
ledger auto-scorer is the fast tuning loop — built there, used here FIRST);
`C13_C14_caching.md` (its two early pieces + profile land inside this pass).
Grounded: the knob population enumerated from this cycle's code reads (§2.1)
and the completion-note caveats scattered across the roadmap ("pending live
validation" items).

**Why this exists (user, 2026-07-10):** constants affect behavior everywhere and
exhaustive combination testing is impossible; and the cycle touched nearly every
subsystem — whatever it DIDN'T touch must be explicitly reviewed, not silently
shipped. Z1 = the tuning gate + the coverage gate, strictly before FINAL.

## 1. Decisions

- **D1: greedy coordinate descent by pipeline stage — the explicit answer to
  combinatorial explosion.** Knobs are grouped by pipeline stage; each stage is
  swept one-factor-at-a-time against the fast loop while upstream stages stay
  frozen at their tuned values; downstream stages inherit. One final
  **interaction jitter pass** (20 random ±20% perturbations over the top-5
  load-bearing knobs) checks for gross interaction cliffs. No grid search,
  ever.
- **D2: the fast loop is auto-scored, judge-free.** 150 synthetic-ledger probes
  (exact scoring) + 40 flaw probes (keyword-anchored GT) = ~190 probes/run;
  generation is the only cost (~30 s/probe on the laptop). Sweep budget ceiling:
  **≤4 nights**; shortlist discipline: only load-bearing-*suspect* knobs get
  3–5-value sweeps, plateau-suspects get 2-point checks, cosmetic-suspects get
  frozen unexamined (verdicts still recorded).
- **D3: every knob gets a written verdict** in `docs/tuning_report.md`:
  `load-bearing` (moves mean score >0.1 across its range — tuned + watch-listed),
  `plateau` (±0.05 flat — frozen at center), `cosmetic` (no measurable effect —
  frozen, marked "do not tune"). Keep-rule per change: ≥ +0.05 mean score, or
  Pareto (equal score, fewer tokens).
- **D4: G9 lands as this pass's first commit** — the module-constant sweep:
  every §2.1 knob not already in settings moves there (one commit, behavior
  frozen — defaults = current values), so sweeps are config-driven and the
  packaged app inherits tunability. (T/B2/C16 already put theirs in settings —
  this finishes the job.)
- **D5: background dynamics are solved, not swept.** Decay rates, strengthen
  increments, promotion/archive/cold thresholds, agent caps are *temporal*
  behaviors invisible to a probe loop. They're tuned by **invariant math**:
  write the target half-lives ("a weekly-revisited memory never archives; an
  untouched casual turn reaches cold in ~6–8 weeks; a reinforced codex edge
  promotes after ~3 corroborations"), solve the rates closed-form (the
  exponentials make this algebra), assert them in a pure-logic test
  (`tests/test_dynamics_invariants.py`) that FAILS if anyone retunes a rate
  into violating an invariant. The invariants document IS the tuning.
- **D6: the coverage matrix is hand-verified, script-drafted.**
  `scripts/coverage_matrix.py` drafts `docs/coverage_matrix.md`: every module
  under `src/` × the roadmap items whose completion notes / specs / git log
  touched it. Untouched modules get the **review pass**: read it, G22 import
  smoke, one behavioral probe, one log audit during Z1's live run — with a
  one-line written verdict each (ok / bug filed / dead code → G20 list).
  Expected untouched candidates (verify, don't trust): `drop_zone.py`,
  `ner_utils` beyond A2's fixes, `db.py`, SSE streaming internals (G5 pending),
  `dataset.py`, registry internals pre-B3. *(ner_utils note: if the A2
  over-rejection data collected here shows under-coverage, the decided remedy
  is A9's GLiNER swap on the background tier — pre-flight keeps the micro-NER;
  see the A9 roadmap entry, 2026-07-11.)*
- **D7: the "pending live validation" ledger is Z1's entry checklist.** Collect
  every completion-note caveat into `docs/coverage_matrix.md` §pending (A2
  over-rejection rate, A6 real-LLM reconciliation, A7.3 enrichment backlog
  (~1,150 entities), C1 summary quality, C5 naming quality, B4 promotion run,
  G21's is_user_active in real use, plus each S1 spec's own deferral rules).
  Z1 is done only when each has a measured verdict or an explicitly-accepted
  risk note. **This is the user's "test every single thing" made enumerable.**
- **D8: stage order** (candidate → fusion → bonuses → budget → decision):
  1. candidate generation: EPISODIC_RECENCY_BOOST/TAU, leg LIMITs, wide-net
     fraction/floor, chunk CHUNK_TOKENS/OVERLAP (2-point only — re-chunk cost);
  2. fusion: RRF k (60), base leg weights, PROFILES rows (incl. B1's two new
     intents, D9 there);
  3. bonuses: BONUS_KEYWORD/LENGTH family, PENALTY_SHORT, META_DOWNWEIGHT,
     MAX_TOTAL_BONUS_MULTIPLIER, A4 RELATION_OVERLAP_BOOST;
  4. budget/representation: C16 fractions + brackets, C1 coverage threshold
     (0.7) + entropy gate (0.35), degrade behavior spot-checks, T timeline caps;
  5. decision thresholds: B2 τ/bias/bumps (incl. ltm_bump_timescope), B1 tag
     threshold (**resolved by B1 2026-07-27 — fitted to 0.65 and stamped inside the
     checkpoint, so Z1 re-sweeps it by re-running `sweep_threshold.py`, not by editing
     `settings.classifier_threshold`; B2's `ltm_*` weights were also swept and left
     unchanged — see `tune_b2.py` and keep its recall-first objective**), C15
     confidence_fallback_threshold, T2 joint-gate
     strictness, codex trust floors (0.5/1.0);
  6. dynamics: D5's invariants (no sweeps).
- **USER-REQUIRED (rule 11):** machine on for the sweep nights (≤4, resumable —
  the runner is FINAL's `run_condition.py` with `--knob k=v` overrides); ~30 min
  reviewing the tuning report's load-bearing table (the user should know which
  five numbers matter).
- **Empirical deferral:** none — this spec IS the empirical protocol.

## 2. Files & integration points

`docs/tuning_report.md` + `docs/coverage_matrix.md` (deliverables) ·
`scripts/coverage_matrix.py` · `tests/test_dynamics_invariants.py` · the G9
settings commit · FINAL's runner gains `--knob` overrides (one kwarg → settings
override dict) · C13's embedding LRU + the Z1 profile (its §D2 rules execute
here).

## 3. Edge cases & traps

A knob whose sweep flips a validation suite red → the suite wins; the knob's
range is constrained, not the test deleted. Score ties → prefer the current
default (stability bias). Synthetic-only wins that flaw disagrees with →
verdict `dataset-sensitive`, keep default, note for FINAL. **Traps:** don't
sweep two knobs at once "to save nights" (attribution dies — the whole design);
don't tune on the judge (auto-scorer only — judges drift, ledgers don't); don't
let tuning re-open settled design decisions (a bad number is retuned, a bad
mechanism goes back to its spec via README rule 12); don't skip the jitter pass
(greedy descent's known blind spot, 20 runs is cheap insurance).
