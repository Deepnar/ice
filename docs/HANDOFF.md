# Handoff — 2026-08-10, 23:5x IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Everything else here is a
pointer — if a doc already records it, it does not get repeated. Overwritten
every session, committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

---

## TOLD → DID

**Told:** `NEXT: G30, then G28.`

**Did:** G30's ground check, then **the user redirected the session**. G30 is not
finished and was not the shape the entry describes by the end. What happened:

1. G30's ground check found the entry **wrong about its own subject in five
   places** (recorded in G30's entry) — the eighth consecutive entry to be so.
2. Found and fixed an unrankable all-zero leg-weight state (`de64ba4`).
3. Found the `Null_Noise` row is **G15's frozen decision**, stopped, and wrote
   the measurements into G15 instead of acting (`f2d82ee`).
4. **The user declared Z1 STARTED and the push freeze ON** (`71ccb9e`).
5. Rebuilt the approach to tuning from the user's own argument (below), then
   built the first instrument: the retrieval ground-truth derivation
   (`a229504`).

## ⚑ THE PUSH FREEZE IS ON

**Five commits are local and unpushed** (`de64ba4` → `a229504`). Do not push
until the user lifts it. This is Z1's experiment-phase rule from CLAUDE.md.

## WHERE THINGS STAND

→ [ROADMAP.md](ROADMAP.md)'s **CURRENT POSITION** block, rewritten this session.

Facts it does not carry:

- **Alembic head `d5c81a37e9b2`, unchanged** — no migration this cycle.
- **Store: 1 row** — the deterministic `ice://mcp-notes` shell. Not residue.
- **Background model is `gemma4:26b-a4b-it-q4_K_M`**, ~2.3 s per constrained
  call warm (6 s cold). 91 probes derive in ~3 minutes.

## ⚠ THE NEXT ACTION NEEDS THE USER, AND NOTHING SHOULD BE BUILT ON TOP FIRST

`experiments/curation_files/VERIFY_ME.txt` (gitignored, 391 lines, 25 samples)
asks one question per sample: **does the gold turn actually contain what the
question asks for?** Regenerate any time with
`uv run python scripts/z1/derive_retrieval_gt.py --verify-only --sample 25`.

**The key is NOT trusted until this passes.** If it is wrong, every tuning
number built on it is wrong *invisibly* — which is the single failure the whole
instrument exists to prevent. Do not build the scorer on an unverified key.

Also pending: **10 of 91 probes derived no gold turn.** Triage which are
genuinely unanswerable from the loaded history versus a derivation miss.

## THE DECISIONS THIS SESSION MADE — the tuning argument, which changed the plan

The user's objection, and it is quantitatively correct: **tuning delicate
parameters against ~190 probes is not trustworthy.** With ~190 probes the
standard error on a mean score is ≈0.036, so the Z1 spec's `+0.05` keep-rule is
~1.4 SE — inside noise. Greedy coordinate descent then compounds it, because
taking the max over several noisy options systematically selects upward noise.

What was decided against, and why: **doing it "all in one big experiment" makes
this worse, not better** — an end-to-end run adds answer variance on top of
retrieval variance, costs more so affords fewer configurations, and cannot
attribute a movement to a knob.

**The resolution: split by whether an LLM is in the loop.**
- Retrieval / scoping / fusion / budget have **no LLM** — deterministic, exact
  scoring (did the gold turn rank top-k), milliseconds per probe. The
  190-probe ceiling is self-imposed here; this half can run thousands.
- Extraction / summarisation / answering need a judge — **choose**, don't
  finely tune, and that is where the integrated run belongs.

**And the answer to "how do we KNOW the tuning works": a held-out split, opened
once**, plus a **measured noise floor** (same config, resampled probe sets, ~50
runs) taken *before* any tuning, so "better" has a threshold that was not
chosen after seeing results.

**The user's own simplification, accepted:** Z1's live run and Z2's mini-exp are
**one harness**, built once. They differ only in what is done with the output —
Z1 scores it, Z2 reads it.

**Superseded:** the earlier worry that Z1 must not tune on the 259 curated
probes because FINAL claims them. **The user will regenerate probes for FINAL**,
which frees the existing set for tuning. The roadmap's CURRENT POSITION block
still states the sealed-probe rule and **should be updated to record this**.

## WHAT WOULD OTHERWISE BE LOST

| What | Where |
|---|---|
| G30's five wrong claims; the all-zero guard's reasoning | [ROADMAP.md](ROADMAP.md) G30, and `de64ba4` |
| The three measured `Null_Noise` re-homings + the 0.0-≠-off finding | [G15](ROADMAP.md#g15) |
| Z1 started, push freeze, probe budget (681 raw → **259 unique**, 19 convs) | [ROADMAP.md](ROADMAP.md) CURRENT POSITION |
| Corpus choice + the variety sweep behind it | `a229504`, and the script's docstring |

**Three things found in passing that have no home yet:**

- **`PyTorchClassifier()` bare loads the v1 checkpoint** (`classifier.py:25`
  defaults to `ice_classifier.pt`) while `settings.classifier_model_path` is
  `ice_classifier_v4_schema2.pt`. Production passes the path explicitly
  (`core.py:39`) so it is safe; **scratch scripts and new tests are not**, and
  it silently produced a wrong variety ranking in this session before it was
  caught. Worth an entry.
- **`probe_type` is the literal placeholder `ENTER_TYPE` on 660 of 681 probes**,
  and on 21 the *question text* was pasted into that field.
- **`orchestrator.py:566-571`** carries a stale comment block ending
  `# (delete the old leg-diversity block entirely)` describing code that is gone.

## NEXT

1. **User verifies the key** (`VERIFY_ME.txt`), and the agreement rate is
   recorded. Nothing downstream is valid before this.
2. Triage the 10 gold-less probes.
3. Build the scorer: seed the three conversations' turns as episodic rows with
   real embeddings, run the real orchestrator per probe, report recall@k and
   MRR against the key. Deterministic seeding, **not** the LLM ingestion path —
   an instrument for tuning must not carry generation variance.
4. **Measure the noise floor before tuning anything.**
5. Only then sweep — screening pass first (which knobs move the score at all),
   held-out conversation opened once at the end.

G30's other three gaps (`main.py` request path, the `RUN_LLM_TESTS=1` lane, the
`test_retrieval.py` rename) are **untouched** and still owed; the user asked for
all four, and the session went deep on the tuning half by their redirection.

## WHEN DONE

Propagate on completion per the roadmap's rules, update the docs the change
invalidates in the **same** session, then rewrite this file — carrying the NEXT
above into TOLD → DID — and commit it last. **Do not push while the freeze
holds.**
