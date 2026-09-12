# LongMemEval on ICE — the external anchor

**Status: frozen ICE v2 matched oracle and full-S evaluation complete.**
ICE/vector correctness is 50.8/72.8% in the oracle and 43.0/69.5% in full-S.
These are matched within-study scores, not official-judge leaderboard scores.
See [the result report](results/matched_cloud_v2.md),
[aggregate paired/cost analysis](results/matched_cloud_analysis.json), and
[paper artifacts](../paper/ARTIFACTS.md). Raw `runs/` and downloaded `data/` stay
local and gitignored; only code and aggregate reports are released. The old
local oracle is historical, and flattened-adapter results are invalid.

## What this is, and which ICE it measures

⚑ **This measures ICE v2 — the frozen system at git tag `v2-paper-eval` (commit
`00d3d35eee99843fd12790d2bf704c177f3097d1`), which is the system the paper reports.** It does *not*
measure `main`. Matched artifacts live under `runs/v2-paper-eval-cloud-v1/`; historical records under `runs/v2-paper-eval/` and are named after
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
  lme_run.py           adapter-v2 resumable runner
  cloud_provider.py    pinned Chat-Completions/Responses profiles
  calibrate_cloud_models.py  bounded public-oracle model selection
  run_lme_cloud.sh     matched Luna-answer/Ollama-background generation
  score_lme_cloud.sh   separate Muse judging
  trace_retrieval.py   read-only per-leg/stage trace
  repair_invalid_adapter.py  reversible v1 artifact repair
  runs/v2-paper-eval/  active + archived run evidence
```

The harness lives on `main` but executes from the v2 worktree at
`/home/deepnar/Programs/ice-worktrees/v2-paper-eval`, branch
`lme/v2-paper-eval`. Ingestion calls the frozen storage/post-flight functions
in-process; answer generation uses an explicit pinned provider profile. The
historical oracle used local Ollama Gemma; the matched oracle/full-S rerun uses
GPT-5.6 Luna through OpenCode's Responses API.

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

### ⚑ Adapter v2: what a LongMemEval session means to ICE

LongMemEval provides many timestamped **history sessions** and asks the question
after all of them. They are not turns in one giant chat. For each independent
question, adapter `ice-v2-lme-sessions-v2` therefore:

1. truncates every mapped v2 table — no question can leak into the next;
2. creates one deterministic `memory_scope_type="auto"` ICE conversation per
   history session and ingests that session's turns into it;
3. creates a separate empty auto-scoped conversation for the question;
4. passes the query conversation as a **string** and `scope={}`, matching v2
   production, so retrieval searches globally across the history conversations;
5. derives v2's budget from the new query chat (zero turns), not from the sum of
   all historical sessions.

Adapter v1 flattened all sessions into one conversation and passed a UUID object
where retrieved fragments carry strings. `_session_diversify` misclassified every
current fragment as external and collapsed 111 RRF candidates to three. Its 500
oracle and first 20 stratified ICE answers are invalid. Originals are preserved
under each phase's `invalidated_adapter_v1/`; valid direct-SQL vector answers stay
active so adapter v2 reruns only ICE. **Never score or cite adapter-v1 output.**

**⚑ The split, and why it is where it is.** LongMemEval's haystack carries
**fixed assistant replies**, and the evidence often lives in them —
`single-session-assistant` is 56 instances on its own. Regenerating those replies
with ICE would destroy the haystack. So:

| stage | path | why |
|---|---|---|
| haystack turns | v2's **storage/post-flight** path, assistant text supplied verbatim | this is what "a turn happened and was stored" *is* in production; nothing is being answered yet |
| the question | v2's in-process classify → budget → retrieve → assemble components, then one pinned answerer profile | preserves the frozen algorithm while allowing paired ICE/vector conditions against one isolated store |

The mature paper harness established the in-process post-flight seam used here:
`is_lossless`, `generate_summary`, `extract_triplets`, and `handle_triplet` receive
the corpus's fixed prompt/response rather than regenerating benchmark evidence.

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

## Standing-up v2 — resolved run environment

- v2 has its own `ice_lme_v2` database at the tag's 12-migration schema. The live
  v3 `ice_db` is never touched.
- The branch restores shared-mode background routing to reachable Ollama and
  records the resolved model in every manifest.
- No Celery or Redis process is required: the mature paper harness established
  the in-process path used here (`is_lossless`, `generate_summary`,
  `extract_triplets`, `handle_triplet`).
- The v2 classifier checkpoint and 384-dimensional embedder snapshot are present.

## Running it yourself

**Setup is already done on this machine** — worktree, venv, database, corpus. It is
recorded here so it can be rebuilt, not because it needs repeating.

<details>
<summary>One-time setup (already complete)</summary>

```bash
# 1. worktree at the tag, on its own branch
git worktree add -b lme/v2-paper-eval /home/deepnar/Programs/ice-worktrees/v2-paper-eval v2-paper-eval

# 2. v2's exact environment from its own lockfile — NOT main's
cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval && uv sync

# 3. link the gitignored artifacts (models/ and data/ are not in git)
ln -sfn /home/deepnar/Programs/ice/models models
mkdir -p data && ln -sfn /home/deepnar/Programs/ice/data/labeled data/labeled
ln -sfn /home/deepnar/Programs/ice/data/ner data/ner

# 4. isolated database + schema
docker exec ice_postgres psql -U ice -d postgres -c "CREATE DATABASE ice_lme_v2 OWNER ice;"
uv run python /home/deepnar/Programs/ice/experiments/lme/setup_v2_db.py

# 5. corpus, pinned
uv run python /home/deepnar/Programs/ice/experiments/lme/fetch_dataset.py
```

The worktree's `.env` points at `ice_lme_v2`, and its `alembic.ini` is repointed
there too — `alembic/env.py` reads the URL only from that file, so left alone every
alembic command in the worktree targets the **live** `ice_db`.
</details>

### Current matched oracle + full-S run

Selected after the bounded calibration in
`results/cloud_stack_calibration.md`:

- answerer: `gpt-5.6-luna` through `/responses`;
- judge: `muse-spark-1.3-contributor` through `/responses`;
- background: exact `qwen3:4b-instruct-bg` through Ollama, kept resident;
- output root: `runs/v2-paper-eval-cloud-v1`, separate from the historical
  local-oracle artifacts.

Luna tied the best answerer point score at 13/14, was fastest among the tied
models, and passed an 88K-input needle check. Omen Alpha also scored 13/14 and
passed 88K input, but was about four times more verbose. Muse is the judge
because it has the strongest prior human-labelled judge calibration and passes
the LongMemEval discrimination self-test. The official benchmark uses a
different judge, so these remain controlled within-study numbers rather than
leaderboard-comparable scores.

The vLLM Qwen3-4B-AWQ substitution was tested and rejected: it reversed the
fixed extraction control or leaked reasoning depending on template. Cloud
answering already eliminates local large-model swaps, so the exact evaluated
Ollama Qwen can stay resident. The strengthened preflight requires both expected
directed facts, a non-thinking summary, and a real long-turn endurance pass.

From any directory, run in this order:

```bash
/home/deepnar/Programs/ice/experiments/lme/run_lme_cloud.sh oracle
/home/deepnar/Programs/ice/experiments/lme/score_lme_cloud.sh oracle
/home/deepnar/Programs/ice/experiments/lme/run_lme_cloud.sh full
/home/deepnar/Programs/ice/experiments/lme/score_lme_cloud.sh full
```

Generation and judging are deliberately separate. Wait for generation to print
`0 instance(s) still outstanding` before starting its scorer. The first
`Ctrl-C` requests a graceful stop after the active instance; a second `Ctrl-C`
immediately abandons that instance. Rerun the identical command to resume from
its last atomic condition answer or complete ingestion checkpoint. Before each
answer request the runner prints the active condition, model, timeout, and retry
count. Cloud SDK retries are disabled: a transport failure becomes a visible,
resumable instance failure instead of silently extending one request.
The generation wrapper starts the existing Docker Postgres container if needed,
checks Ollama, evicts other resident Ollama models, pins
`qwen3:4b-instruct-bg` with infinite keep-alive, and never requires an activated
venv. It does not start or use vLLM. Every OpenCode answer and judgement sends
the required `x-opencode-session` header; the value is a deterministic UUID for
that phase/question/condition, so a retry is stable while independent benchmark
conversations never share an id.

If OpenCode returns HTTP 401/403/402/429 or a quota, usage-limit, billing, or
credit-limit message, generation/scoring stops immediately instead of failing
every remaining item. Completed atomic files stay valid. Replace only
`PROBE_API_KEY` in `/home/deepnar/Programs/ice/.env` (currently line 55), then
rerun the identical command. If the terminal had exported an older value, run
`unset PROBE_API_KEY` before restarting so the edited `.env` is loaded. Key
rotation helps only when the replacement key/account has available quota.

Plan-only, with no DB/model/provider work and no output-directory creation:

```bash
/home/deepnar/Programs/ice/experiments/lme/run_lme_cloud.sh oracle --plan
/home/deepnar/Programs/ice/experiments/lme/run_lme_cloud.sh full --plan
```

### Historical local-oracle reproduction

The commands below reproduce the already reported local Gemma answer/judge run;
they are not the commands for the new matched study.

#### Before you start a run

1. **Docker postgres up** — `docker ps | grep ice_postgres`
2. **Ollama up** with `gemma4:26b-a4b-it-q4_K_M` (answerer), `qwen3:4b-instruct-bg`
   (background) and `gemma4:12b` (judge) pulled. All three are already present.
3. That is all. **No Redis, no Celery, no vLLM, no SGLang** — ingestion calls the
   post-flight functions in-process.

#### The commands

```bash
./experiments/lme/run_lme.sh oracle --plan   # costs nothing, needs nothing
./experiments/lme/run_lme.sh oracle          # the control — run this first
```

Then score it:

```bash
cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval
uv run python /home/deepnar/Programs/ice/experiments/lme/score.py --phase oracle
```

Later phases, once oracle looks sane:

```bash
./experiments/lme/run_lme.sh abstention      # all 30 _abs instances
./experiments/lme/run_lme.sh stratified 60   # 60, spread across question types
```

#### Measured throughput, and what a run actually costs

With the recorded CUDA-embedder deviation, full-haystack ingestion measured
roughly **1.8–2.0 s per user/assistant pair** on this laptop, dominated by Qwen
background extraction/summarisation rather than embeddings. Model swaps and
answer generation add phase-level overhead.

| phase | pairs | rough wall-clock |
|---|---|---|
| `oracle` | ~5,480 | hours; rerun preserves 500 valid vector answers |
| `abstention` | ~7,570 | several hours |
| `stratified 60` | ~14,900 | most of a day |

These are days, not evenings — which is why resumability is the design constraint
rather than a nicety. Start it, close the laptop, run the same command again.

#### The historical local judge

Pinned to **`gemma4:12b`**, matching the paper, which judged every Exp-1/2/3 probe
with a Gemma-4 12B. ⚠ Recorded deviations: this is the **Ollama GGUF** build rather
than the paper's `mattbucci/gemma-4-12B-AWQ` on SGLang, and the judge token cap is
raised from LongMemEval's `max_tokens=10` to 256 because this build reasons before
answering and returns an empty string at 10. The **decision rule is untouched**, and
reasoning never reaches it — the API returns reasoning separately from
`message.content`. For exact parity, serve the AWQ build on SGLang and pass
`--judge`.

**Safe to kill at any point** — re-run the same command to continue. Each answer
condition is persisted atomically (`tmp` + `os.replace`), so a valid vector answer
can survive while ICE is rerun. An interrupted ingestion writes no answer and is
redone from a full store wipe; a complete matching store can be reused after an
answer-side interruption. First `Ctrl-C` is graceful; second `Ctrl-C` exits the
active instance immediately. Interactive stdout is tee'd to the phase log. Under a
supervisor set `LME_DIRECT_JOURNAL=1`, which executes Python directly; stopping a
supervised `python | tee` pipeline can close the logger before Python handles its
signal.

`--plan` needs no database, no GPU and no worktree — the heavy imports are deferred
past it, so it is always safe to run just to see what a phase would cost.

### Measured phase costs

| phase | instances | turns to ingest | notes |
|---|---|---|---|
| `oracle` | **500** (all) | **~10,960** | ~22 turns each — evidence sessions only |
| `abstention` | 30 | ~15,144 | the full `_abs` set, ~505 turns each |
| `stratified 60` | 60 | ~29,782 | seeded, deterministic, even across the six types |
| `full` | **500** (all) | **246,930** | complete LongMemEval-S; matched cloud wrapper only |

⚑ **The oracle phase is cheap enough to run in full**, and it is not merely a
smoke test. Scoring all 500 with only the evidence sessions loaded tests whether
ICE can find and use supplied evidence before distractors are introduced. It is
not a mathematical upper bound: retrieval can change non-monotonically when
candidates are added. The matched oracle/full-S gap is still the cleanest
available diagnostic of distractor sensitivity under one fixed stack.

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
- **No parallel Ollama requests.** Ingestion is temporally ordered because graph
  updates can supersede earlier facts, and one laptop GPU was already 95–96%
  utilised. Parallel calls would add memory/thermal pressure without adding a
  second compute lane.
