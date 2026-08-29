# LongMemEval on ICE — the external anchor

**Status: harness under construction. No score has been produced. Do not quote a
number from this directory until this line says otherwise.**

## What this is, and which ICE it measures

⚑ **This measures ICE v2 — the frozen system at git tag `v2-paper-eval` (commit
`0521df9`, 2026-07-02), which is the system the paper reports.** It does *not*
measure `main`. Artifacts live under `runs/v2-paper-eval/` and are named after
the git tag on purpose: `v2-paper-eval` is a unique string in this repo and
cannot be misread the way a bare "v2" can (see [CLEANUP.md](../../docs/CLEANUP.md),
2026-08-29 naming rule).

**Why it exists.** ACM TIST rejected the paper on 2026-08-28, editorially and
without review. One of the AE's accurate criticisms was that the evaluation used
no *"commonly used evaluation metrics in the literature."* LongMemEval is the
answer to that: it is public, MIT-licensed, third-party, and the systems ICE
positions against (mem0 and others) publish numbers on it.

**What it is NOT.** It does not replace LSREP and is not intended to. LongMemEval
supplies a fixed history up front and asks its questions at the end, so nothing in
it ages, decays, or is revised *between* questions — the processes LSREP exists to
observe. The two measure different things; the paper says so in Related Work §2.5.

## ⚑ The Z1 rule, and why this run does not violate it

[ROADMAP.md](../../docs/ROADMAP.md) **Z1** carries a standing decision (2026-08-04):

> LongMemEval-S IS A Z1 FIXTURE, NOT A SCORE. Do NOT score it pre-Z1 — an untuned
> system yields a number about tuning, not design.

**That rule governs v3**, the in-flux system on `main`. This run scores **v2**,
which is frozen and is the system whose numbers are already published. A number on
v2 is a number about the paper's system.

⚠ **There will therefore be two LongMemEval numbers**: this one on v2, and later
FINAL's on v3 ([specs/FINAL_experiments.md](../../docs/specs/FINAL_experiments.md)
§2.8). **They are not comparable.** Every artifact here carries the tag in its path
and in its manifest so the two can never be conflated.

## The corpus

Fetched by `fetch_dataset.py` at a **pinned revision** — `xiaowu0162/longmemeval`
@ `2ec2a557f339b6c0369619b1ed5793734cc87533`, MIT. `data/` is gitignored (293 MB);
`data/MANIFEST.json` records the revision and a sha256 per file, and
`--verify-only` re-checks them.

| file | size | role |
|---|---|---|
| `longmemeval_s` | 278 MB | the run corpus |
| `longmemeval_oracle` | 15 MB | **control** — evidence sessions only |
| `longmemeval_m` | 2.7 GB | **not fetched.** ~500 sessions/instance is far beyond this hardware. |

### Real shape, measured — not assumed

500 instances. Per instance: `haystack_sessions` (list of sessions, each a list of
`{role, content}` turns), `haystack_dates` (one timestamp per session),
`haystack_session_ids`, `answer_session_ids` (which sessions hold the evidence),
and `question_date`.

| question type | n |
|---|---|
| multi-session | 133 |
| temporal-reasoning | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |

**Abstention** is not a `question_type`; it is a flag — 30 instances whose
`question_id` ends in `_abs`, spread across four of the types. The correct answer
is that the corpus does not contain one.

## ⚑ THE COST FINDING — read this before starting any run

Measured on the real corpus, not estimated:

| | |
|---|---|
| sessions per instance | min 39, **median 50**, max 66 |
| turns per instance | min 396, **median 492**, max 616 |
| history per instance | **~123k tokens** (median 490k chars) |
| **total turns, all 500** | **246,930** |
| **total turns, spec's 150-instance subset** | **~74,000** |

Each instance needs its **own** memory state, so histories cannot be shared — the
haystack is replayed from empty, per question.

**74,000 turns through ICE's full post-flight path** (codex extraction, procedural
extraction, clustering, decay) on a thermally-bound 24 GB laptop GPU is **days of
continuous compute**, not an overnight job. FINAL_experiments.md §2.8's
"~150 instances" was written before anyone measured the per-instance turn count.

### The phased plan this implies

CLAUDE.md: *"Size the run to the question and compute what it can resolve before
starting"* and *"a long run started on an unverified change does not fail; it
returns a number, and the number is wrong."*

- **Phase 0 — oracle control (cheap, hours).** Replay `longmemeval_oracle`, whose
  instances carry only the 1–5 evidence sessions. Ingestion is a rounding error.
  This proves the adapter, the ingestion path and the judge end-to-end **before**
  any expensive run. ⚑ If ICE cannot answer from the evidence sessions alone, the
  defect is in this harness, not in ICE's memory — and every later number would
  have been noise.
- **Phase 1 — the abstention subset in full (30 instances, ~15k turns).** A
  complete, non-arbitrary unit, and the one where a curation-first system should
  most visibly differ: does it decline, or confabulate from retrieved noise?
- **Phase 2 — stratified widening**, only if 0 and 1 are sane, toward the spec's
  150. Reported explicitly as a subset with its stratification stated.

**Accepted going in: ICE may simply lose to published numbers.** That is a
publishable result and consistent with this paper's character — it already reports
Experiment 1, where the system was *worse* than its baseline. The decision to
publish either way was made before the run, not after.

## Layout

```
experiments/lme/
  fetch_dataset.py     pinned fetch + sha256 manifest    [done]
  data/                gitignored corpus                 [fetched]
  runs/v2-paper-eval/  artifacts for THIS run            [empty]
```

The harness lives on `main` and talks to the system under test over HTTP
(`/v1/chat/completions`), so it never has to live inside the frozen tree. v2 runs
in a worktree at `/home/deepnar/Programs/ice-worktrees/v2-paper-eval` on branch
`lme/v2-paper-eval`.

## ⚠ Standing-up v2 — known costs, not yet paid

- **503 commits** between the tag and `main`.
- **Schema drift: 12 migrations at the tag vs 36 on `main`.** The run needs its
  **own database at the v2 schema**; it cannot share the current one.
- **`background_model_mode` defaults to `"dedicated"` at the tag** — it wants a
  separate vLLM on :8002, which CLAUDE.md calls a manual power-user path.
- The classifier checkpoint it pins, `models/classifier/ice_classifier_v3_qwen_ft3.pt`,
  **is still present**.
