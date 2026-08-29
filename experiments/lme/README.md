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

## ⚑ Why LSREP is not reused, and where the production path still is

LSREP replays **one** conversation chronologically into **one** deployment,
deliberately *keeping* memory across checkpoints and scoring against an evolving
ground truth. LongMemEval is 500 **independent** instances, each needing a
**fresh** store, each one question asked once against a fixed answer. Bending
LSREP to that shape would mean disabling its checkpoint loop, its evolving-GT
pipeline and its state retention — everything except the HTTP call — and a
harness rebuilt that far measures itself ([TRAPS #32](../../docs/TRAPS.md)).

So the **control flow is written fresh**. What is *not* reinvented is the path a
turn actually takes, per CLAUDE.md's first correctness question — *did the harness
call what the real path calls?* A harness writing rows into Postgres directly
would measure storage, not ICE.

**⚑ The split, and why it is where it is.** LongMemEval's haystack carries
**fixed assistant replies**, and the evidence often lives in them —
`single-session-assistant` is 56 instances on its own. Regenerating those replies
with ICE would destroy the haystack. So:

| stage | path | why |
|---|---|---|
| haystack turns | v2's **storage/post-flight** path, assistant text supplied verbatim | this is what "a turn happened and was stored" *is* in production; nothing is being answered yet |
| the question | full **`POST /v1/chat/completions`** | classify → retrieve → fuse → budget → assemble → route, exactly as production serves it |

`post_flight.evaluate_turn(self, batch_id, prompt, response, conversation_id,
model_used)` takes prompt **and** response explicitly, so verbatim injection is
supported by the real function rather than by a harness shortcut.

**Consequence for layout:** ingestion must run **in-process inside the v2
environment** (it imports `src.*`), so the harness cannot be a pure HTTP client.
It still lives here on `main` as the single copy, and is *executed* from the
worktree so `uv run` picks up v2's lockfile and `src.*` resolves:

```bash
cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval
uv run python /home/deepnar/Programs/ice/experiments/lme/lme_run.py --phase oracle
```

## Reproducibility — checked, and it holds

The question was whether a run today reproduces the July system, given that
embeddings silently define every stored vector.

| what | value | how pinned |
|---|---|---|
| embedder | `Qwen/Qwen3-Embedding-0.6B`, **`truncate_dim=384`**, `device="cpu"` | identical at all four v2 call sites |
| HF snapshot | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | present in the local HF cache |
| `sentence-transformers` | **5.5.1** | `uv.lock` **at the tag** |
| `torch` | **2.11.0** | `uv.lock` at the tag |
| `transformers` | **5.9.0** | `uv.lock` at the tag |
| `pgvector` | **0.4.2** | `uv.lock` at the tag |
| schema | 12 migrations | `alembic/versions/` at the tag |

**⚑ The trap this closes:** v2 stores **`Vector(384)`**; `main` is 1024-dim. Both
use the string `Qwen/Qwen3-Embedding-0.6B` — *same model name, different vectors*,
because v2 matryoshka-truncates. A run that picked up `main`'s embedder would
produce vectors that are wrong in a way nothing would flag. Always `uv sync`
**inside the worktree**; never share `main`'s venv or `main`'s database.

Residual risks, recorded rather than solved: the HF repo could be re-fetched at a
different revision if the cache is cleared (pin `revision=` if that ever happens),
and GPU/driver differences are outside our control.

## ⚠ Standing-up v2 — known costs, not yet paid

- **503 commits** between the tag and `main`.
- **Schema drift: 12 migrations at the tag vs 36 on `main`.** The run needs its
  **own database at the v2 schema** — plan is a separate `ice_lme_v2` database via
  `DATABASE_URL`, so the live `ice_db` is never touched.
- **`background_model_mode` defaults to `"dedicated"` at the tag** — it wants a
  separate vLLM on :8002, which CLAUDE.md calls a manual power-user path.
- **⚑ v2 STILL HAS CELERY.** `evaluate_turn` is a bound Celery task (`self.retry`),
  so post-flight at the tag needs **Redis + a Celery worker**. CLAUDE.md's "there is
  no Celery, no Redis and no separate worker process" describes `main` after C7 —
  it is **false for the tag**, and this is the single biggest surprise in standing
  v2 up. Anyone reading CLAUDE.md while working in the worktree will get this wrong.
- Post-flight carries an `is_gpu_busy()` gate that reschedules with a 15s backoff,
  and an `is_user_active()` gate in `shared` mode. Both matter for an unattended
  overnight run: the harness must tolerate ingestion that legitimately stalls.
- The classifier checkpoint it pins, `models/classifier/ice_classifier_v3_qwen_ft3.pt`,
  **is still present**.

## Running it yourself

```bash
./experiments/lme/run_lme.sh oracle          # the control — do this first
./experiments/lme/run_lme.sh abstention      # all 30 _abs instances
./experiments/lme/run_lme.sh stratified 60   # 60, spread across question types
./experiments/lme/run_lme.sh oracle --plan   # show the plan, touch nothing
```

**Safe to kill at any point** — Ctrl-C, lid close, power cut. Re-run the same
command to continue. Progress is one answer file per instance under
`runs/v2-paper-eval/<phase>/answers/`, written atomically (tmp + `os.replace`), and
**the presence of that file *is* the state** — there is no progress file to fall out
of sync. An instance interrupted mid-flight simply has no answer file, so it is
redone from a full store wipe next time, which is what correctness requires anyway.
Ctrl-C stops *after* the current instance rather than mid-write. Everything is
tee'd to `runs/v2-paper-eval/<phase>/run.log`.

`--plan` needs no database, no GPU and no worktree — the heavy imports are deferred
past it, so it is always safe to run just to see what a phase would cost.

### Measured phase costs

| phase | instances | turns to ingest | notes |
|---|---|---|---|
| `oracle` | **500** (all) | **~10,960** | ~22 turns each — evidence sessions only |
| `abstention` | 30 | ~15,144 | the full `_abs` set, ~505 turns each |
| `stratified 60` | 60 | ~29,782 | seeded, deterministic, even across the six types |

⚑ **The oracle phase is cheap enough to run in full**, and it is not merely a smoke
test. Scoring all 500 with only the evidence sessions loaded gives an upper bound —
it separates *can ICE find and use the evidence* from *can ICE survive the
haystack*. If the two numbers are far apart, the gap is distractor robustness; if
oracle itself is low, the problem is retrieval or the adapter, and no haystack run
would have told you which.

### Guardrail against the embedding trap

The runner **asserts the embedding dimension at startup** and refuses to run if it
is not 384:

```
⛔ embedder returned 1024 dims, expected 384.
   v2 stores Vector(384); `main` embeds at 1024 with the SAME model name.
```

This is the one failure that would otherwise be silent — no crash, no warning, just
wrong geometry in every stored vector.

## What the runner deliberately does not do

- **No decay simulation.** LSREP ages memory between checkpoints because it is
  measuring accumulation. LongMemEval asks once, at the end, so there is nothing to
  age *between*; simulating decay here would model something the corpus does not
  contain. Clustering runs once after ingestion so cluster-scoped retrieval has
  something to scope to.
- **No Celery, no Redis, no vLLM.** Ingestion calls the post-flight functions
  in-process, mirroring `experiments/mature/run_mature_experiment.py` — the harness
  that produced every number in the paper. It needs Postgres and the background
  model, nothing else. *(An earlier note in this file claimed Redis and a Celery
  worker were required, on the strength of `evaluate_turn` being a bound task. The
  mature harness bypasses that task entirely and calls `is_lossless`,
  `generate_summary`, `extract_triplets` and `handle_triplet` directly. Corrected.)*
- **The answer pass is not written yet.** Instances currently stop at `"status":
  "ingested"` with an empty `answers` block. The answer path will mirror the mature
  harness's in-process probe — `classify` → `find_best_model` →
  `HybridRetrievalOrchestrator.retrieve` → `set_budget_from_turn_count` →
  `assemble_prompt` — which is what produced the paper's numbers, and which calls
  the budget setter that CLAUDE.md warns a scorer must not skip. The `vector_rag`
  baseline arm comes nearly free from the same loop.
