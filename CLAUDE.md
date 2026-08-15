# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

**This file is loaded into every session, so it holds only what is (a) true of
the system as built and (b) not written anywhere a session reads naturally.**
Everything else is a pointer. When a rule's evidence or worked example lives in
another doc, that doc owns it and this file keeps the one-line imperative.
⚠ It rots the most expensively of any doc here — it was once found describing
DI3, Celery, Redis, a 384-dim encoder and a 25-logit head, all long deleted.
It is item (5) on the deletion-sweep checklist in [docs/CLEANUP.md](docs/CLEANUP.md).

**⚑ Keeping it true is a standing job, and changing it is USER-GATED.** When a
session touches a subsystem this file describes, **verify the claim here against
the code** — same sweep as any other doc. But **ask before editing CLAUDE.md**,
every time, even for a one-word correction: it is loaded into every session, so
an edit changes how all future sessions behave, and a rule quietly dropped here
is gone everywhere. *(That is not hypothetical: the `942d5a9` shrink compared a
rule against its ROADMAP twin **by heading rather than by text** and deleted a
clause the twin never had. Report what you found and what you propose; wait.)*

## What ICE is

The **Infinite Context Engine (ICE)** is local-first AI memory middleware. It
runs as a **FastAPI proxy** (`src/api/main.py`) between any OpenAI-compatible
chat frontend and a local inference backend (**Ollama**, port 11434). Every
`POST /v1/chat/completions` passes through ICE, which classifies the prompt,
retrieves relevant memory, assembles a context-enriched prompt, routes to a
model, streams the response, and asynchronously processes the finished turn
into a structured memory store.

Guiding principle in the code: **"memory is earned"** — a turn is stored
losslessly only if dense enough (`lossless_flag` / `inject_raw`, set by the
post-flight evaluator); otherwise it is summarised by the background model.

*(Open WebUI was the interim frontend and is **no longer the intended path** —
user decision 2026-08-03; Track F's packaged app is the destination. Existing
design-rationale citations of Open WebUI's architecture stay: citing someone's
design is not depending on their product.)*

ICE is also a **research project**. v2 is finished, the experiments are done, a
paper is written for later arXiv posting, and the work now is the post-paper
cycle — the experiments exposed gaps and closing them is the job. Experiment
results live in `experiments/*/results*/` as `.md` summaries.

## Start here — read these, in this order

| Read | What it is |
|---|---|
| [docs/HANDOFF.md](docs/HANDOFF.md) | **Where the last session left off, and what it was told to do.** Rewritten every session. Read it first; it is state, never a queue. |
| [docs/TRAPS.md](docs/TRAPS.md) | Mistakes this project has actually made. Read once at session start. Add to it the same session something bites. |
| [docs/ROADMAP.md](docs/ROADMAP.md) | **The queue, and the only queue** — 56 open items plus a one-line stub per finished one. Its "How to use this file" block is the rules for working it and its "HOW TO EXECUTE THIS ROADMAP" section carries the current position; neither is repeated here. Check items off there as they complete. |
| [docs/ROADMAP_DONE.md](docs/ROADMAP_DONE.md) | The finished items in full — what shipped, what each entry got wrong about its own subject, look-ahead, propagation, validation. Every stub in the roadmap links to its record here. |

Then, as the work requires:

| Doc | Owns |
|---|---|
| [docs/ICE_Architecture.md](docs/ICE_Architecture.md) | How each subsystem actually works. The authoritative design reference. **Where any doc conflicts with code, code wins.** |
| [docs/specs/](docs/specs/) | Decision-complete specs. Since S1 every design-heavy roadmap item has one — read the item's spec **and its `Assumes decided specs:` chain** before coding, and obey [specs/README.md](docs/specs/README.md) rules 11–12 (USER-REQUIRED steps; the divergence protocol: code↔spec mismatch ⇒ stop, re-ground, fix the spec first, never improvise past it). |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | What was *done* — model revisions, corpora, checkpoints, run parameters. It states its own standing rule; follow it when a run produces an artifact. |
| [docs/CLEANUP.md](docs/CLEANUP.md) | Cleanup rules (incl. the deletion sweep) + the move/rename ledger. |
| [docs/VISION.md](docs/VISION.md) | Intent — memory for human–AI thinking sessions. A separate Coding Mode is planned post-paper. |
| [docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) | **Every feature the system actually has** — ~433 of them, each with a verified `file:line`, its controlling setting, that setting's **default**, and whether it is **ON by default**. Plus 40 items that are implemented and **cannot currently fire**. |

**⚑ CHECK [FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) BEFORE CLAIMING THE
SYSTEM DOES SOMETHING — AND BEFORE BUILDING SOMETHING.** It prevents the two
failures that have each already happened here: a capability rebuilt because
nobody knew it existed (the coding core, forgotten within 45 days), and a
capability described as active in a write-up while its setting defaults to off.
Its `On by default?` and `DEAD OR INERT` columns are the load-bearing parts — an
inert path attributed a measurement is how several wrong conclusions were drawn
on 2026-08-15. **Update it in the same session as anything that adds, removes or
re-gates a feature**; like every doc here it rots, and the `file:line` is there
so re-deriving an entry costs seconds. Code wins over it, always.

`docs/ICE_Architecture[real_v2].md` is the **frozen** technical report for the
system as evaluated in the paper (git tag `v2-paper-eval`) — never update it to
match current code; its known inaccuracies are part of the historical record.
Superseded docs live in `docs/outdated/` and are never edited.

Code comments across `src/` often explain *why* something is the way it is.
Neither docs nor comments are guaranteed current — verify against the code.

> **Maintainer note:** some working files at the repo root and under `docs/` are
> **gitignored on purpose** (personal planning, publishing strategy). Read them
> for context if present; never commit them, never quote them into tracked files.

## Standing rules

### RESEARCH FIRST — the measurement is the deliverable (2026-08-12)

**A product is wanted; research is what is being done.** Those are not in
tension, they are in *order*: nothing gets productised until it is known to
work, and "known" here means measured, not believed. So the unit of progress in
this repo is **a measurement someone can trust**, not a feature.

**⚑ Be certain the measurement is CORRECT before believing what it says.** Not
"be thorough" — *be correct*. Before reporting a number, ask:

- **Is everything there?** Did the harness populate every part the real path
  populates, and call what the real path calls? A store missing three of eight
  legs, or a scorer skipping the budget setter `main.py` calls, measures
  something — just not the system.
- **Would this number look the same if the thing under test were broken?** If
  yes, it is not a measurement. A presence-based score (coverage, in-vocabulary
  rate, hit count) answers *"is the expected thing in there"* and never *"is
  what is there any good"*.
- **Does the instrument reproduce production, or my idea of it?** The instrument
  becomes part of what it measures the moment it diverges. **So change it one
  part at a time, and prove each part TWICE before it feeds a run** — once in
  isolation (does the piece do what it claims?), once through the path
  production actually takes (does the real caller still agree?). The second
  catches what the first cannot: a fixture's sitting layout passed its own check
  and still aged the whole corpus by fifteen days, visible only once the
  assertion asked what production would see. **A long run started on an
  unverified change does not fail — it returns a number, and the number is
  wrong.**
- **⚑ Did I QUERY the running system, or reason about the code?** Reading source
  and reasoning forward produces conclusions that feel verified and are not —
  **four wrong ones in a single session, 2026-08-12**, each disproved by one
  query taking seconds. The comments describe intent; the database describes
  what happened. **Any claim about behaviour gets a query before it gets
  stated**, and a claim that cannot be cheaply checked says so in its own words
  rather than in a caveat afterwards. TRAPS #23.

**Nothing may be skipped, and no result is exempt.** A step left out is a
variable left uncontrolled, and it will be discovered later as a wrong
conclusion rather than a missing step. Where something genuinely cannot be
measured yet, **say so in the number's own words** — "off-production",
"unconfirmed", "inert on this corpus" — rather than letting it read as clean.

**But this is not an argument for long experiments.** A four-night sweep on an
unverified instrument wastes four nights; a ten-minute check that would have
invalidated it is worth more than all of them. Nor is it an argument for
short ones: a probe set too small to resolve the effect you are looking for
produces a confident number about noise. **Size the run to the question**, and
compute what the run can actually resolve *before* starting it.

**There is no hurry. There is also no room for waste.** Being unhurried is what
makes it affordable to re-check; it is not permission to re-derive what is
already written down, to re-run what a cheaper measurement settles, or to
explore adjacent questions nobody asked.

**Worked evidence, and it is the reason this rule exists:** on 2026-08-12 five
conclusions were drawn and corrected within one session — four of them from
reasoning about code instead of querying the running system. The instruments
built that day carried nine defects of their own. **Every finding that survived
scrutiny came from reading actual output**, and the two that mattered most (a
ground-truth key that looked fine, an eight-model ranking that was backwards)
were caught by a human and an agent *reading*, against metrics that were green.
See TRAPS #20, #21 and roadmap [G46](docs/ROADMAP.md#g46).

### Name the item, then SAY WHAT IT IS — every time (2026-08-13)

**A bare roadmap id is not a reference, it is a lookup the user has to perform.**
There are 56 open items; nobody holds them in their head, and the person reading
your message has been away from the queue longer than you have.

So every mention of a roadmap item, spec, TRAPS entry or setting carries **its
number AND a one-line plain explanation with a concrete example** — every time
it appears, including the second and third time in the same message. Repetition
here is not clutter; it is the only thing that makes the id mean anything.

> ❌ "G44 is next, then G45."
> ✅ "**G44** — codex subjects are not real entities: the store holds nodes
>    literally named `8` and `3`, typed as *person*, which nothing can ever look
>    up. Then **G45** — the relation vocabulary is a closed 197-word list, so a
>    true fact like `i --didnt_get--> csi` is thrown away for using a word that
>    is not on it."

The example is the load-bearing part. "G44 (entity quality)" is still a lookup;
`nodes named 8, typed as person` is recognisable a week later.

### Ask before changing production code (2026-08-03)

The user's instruction, after stopping a session mid-flight: *"if doing anything
to the production code ask always."* Investigation, measurement, scratch scripts
and reading are free — **changing `src/` is not.** Before touching production
code: say what you intend to change, why, and what it affects, **in plain
language they can act on** — mechanism in short sentences, what it is NOT, then
the decision. Then wait. Docs, tests and scratch work do not need the gate.

This is not ceremony: the alternative failure mode is real — an evaluation turns
into a half-migration, and a benchmark session ships code nobody agreed to.

Same rule from the other side: **analysis is not a decision.** The user has said
*"I understood what you said, but not the decision from it."* End with "do X",
not "X is worth considering".

### Check where a signal LANDS, not just that it exists (2026-07-27)

ICE computes a lot of signals. A signal can be trained, accurate, stored on
every result — and still change nothing, because it is wired to a decision that
was already made. Three were found in that state on one day.

**When a new classifier label or score is added, trace it to the decision it
changes and measure the delta with it on vs off.** "It's wired up" is not the
test; "turning it off changes N decisions" is. Beware especially that **a signal
which is a subset of another signal cannot improve that signal's own decision**.

Worked examples and the measurements behind them: `ICE_Architecture.md` §2.2 and
§2.4. Corollary on deletions: when a measurement says a designed hook or seam is
not paying off, **that is evidence, not permission — ask the user first.**

### No decision may depend on HOW a thing is written (2026-07-28)

A rule keyed on punctuation, word order, or a fixed vocabulary is a **bet on
writing convention**, and it is the measured reason Codex underperformed. ICE
was validated on corpora that all share one convention ("people typing at a
chatbot"), so the bets never looked broken.

**The goal is INVARIANCE, not personalization.** We have no idea how any given
user writes and no corpus can tell us. The requirement is that the same intent,
written any way, yields the same decision.

**The test that works** holds meaning fixed and varies only form: write one
prompt several ways (± question mark, "ok so"/"like" prefixes, interrogative
buried, lowercase, typos, terse vs rambling) and measure the **decision-flip
rate**. A rule that flips is measuring typography. **Apply it to the replacement
too** — the trained head is not automatically the fix. Parsers that RESOLVE a
value (a date → a datetime) may stay; lexicons that INFER INTENT must pass
invariance.

The measurements, and why firing-rate-by-source is a smell detector rather than
an acceptance test: `ICE_Architecture.md` §2.6. Roadmap **G28** is the systematic
sweep and owns the style-variant probe set; **D8** is the worked deletion protocol.

### A silent fallback hides an outage (2026-08-03)

When a component substitutes a default for a real answer, **it emits at WARNING
with the reason — every time, not on first occurrence.** A fallback that fires
on 100% of calls is not resilience, it is an outage wearing resilience as a
costume. Check the *rate* before shipping any new warning, and when a subsystem
produces plausible-but-thin output, verify the model was actually called before
tuning anything about it. Worked example and corollaries: **TRAPS #11**.

### Boy-scout cleanup (2026-07-10)

Every implementation session leaves the files it touches cleaner than it found
them. Rules and the ledger: [docs/CLEANUP.md](docs/CLEANUP.md) — the short form
is touched-files only (never a repo-wide reformat), imports sorted and unused
dropped, dead code and lying comments fixed in place, one-off scripts *moved*
(never deleted) to `scripts/oneoff/`, and every move logged.

### Git, commits, and the public repo

**⚑ THE REPO IS PUBLIC** (`origin` → `github.com/Deepnar/ice`), and has been
for a while — TRAPS #12's exposure incident, closed 2026-08-04, was already a
public clone. It is not a private backup: **every push is a publication.**
Pushing during normal development stays pre-authorized — push at natural points
without asking — but write every commit, message and file knowing it is read as
soon as it lands, by strangers, with no window to take it back.

- **⚑ SMALL COMMITS *IN THIS FORMAT*. Granularity and format are ONE rule.**
  Many small commits split by *concern* — never one end-of-session commit; if a
  message needs bullets for unrelated changes it should have been several
  commits. **And every one of them, however small, takes the full shape below.**

  ```
  area: what changed (ITEM)

  ITEM — what it does now:
  - concrete change: function/file/migration names, measured numbers
  - concrete change

  SECOND-ITEM — ...

  Also fixes: the thing found in passing.

  Validated N/N (tests/test_x.py: what was checked); regressions green:
  a 13/13, b 31/31. Architecture §6.10 updated; roadmap C6 checked.
  ```

  **Subject:** lowercase area prefix (`retrieval:`, `codex:`, `memory:`,
  `workers:`, `classifier:`, `clustering:`, `budget:`, `roadmap:`, `specs:`,
  `docs:`, `tooling:`) · what changed · the roadmap item id in parens · ≤ ~70
  chars. **Body:** organised **by item**, dense and concrete, closing with a
  **`Validated`** line and a **docs/roadmap** line. **The log is the reference —
  read it before writing one:** `git log 6bac35d 1a30484 051cea9 5124efb`.
  - ⚠ **It is a record, not an explanation.** No paragraph essays, no rhetorical
    beats ("That is not an edge case."), no argument structure. Impersonal, in
    the repository's voice; **never narrate the session** ("the user asked…",
    "as requested…", "we decided…"). **No AI attribution / Co-Authored-By.**
  - *(This drifted badly on 2026-08-09/10 — a dozen commits with no area prefix,
    no item id, no validation line, and prose bodies. The rule was already here;
    the log was not read. Read the log.)*
- **Freeze at the experiment phase:** once **SEMIFINAL (Z1)** or **FINAL**
  begins, **stop pushing** until the user says otherwise — a half-run
  experiment published mid-flight is a result nobody chose to publish.

**⚑ IT IS PUBLIC SINCE A GOOD WHILE NOW — the tree, the docs and the commit log.** This used to
read "going public" long after it already had; corrected 2026-08-10.

- **Never commit personal content** — private planning, career notes,
  conversation corpora, third-party emails, credentials. **Git history is
  forever, and on a public remote it is forever the moment you push**: a file
  committed once and gitignored later is still there, and even a rewritten
  history keeps serving the old objects by SHA until the host garbage-collects
  them (TRAPS #12 — that took a support ticket last time). **Check before the
  commit, not after the push.**
- **Verified clean 2026-08-10:** `check_history_clean.sh --clone`
  green against the live remote, no personal or planning file tracked, no
  credential-shaped string in the tree, README links the canonical
  venue-agnostic paper, and `v2-paper-eval` points at `0521df9` (post-rewrite).
  Re-run the `--clone` check after anything that touches history.
- **A history rewrite must rewrite TAGS, and `git log` on `main` cannot verify
  it.** Verification is `scripts/git/check_history_clean.sh`, wired to a
  `pre-push` hook by `scripts/git/install_hooks.sh` (**run once per clone**);
  `--clone` checks what the public actually receives. Never resolve a tag
  mismatch by forcing one side to win without first asking which side predates
  the rewrite. Full story: **TRAPS #12**.
- **README.md is a first-class deliverable** — the first and often only thing a
  visitor reads. Update it in the same session as anything that changes what the
  project *is* or how it is run.
- **⚑ README links the VENUE-AGNOSTIC paper, never a venue submission.**
  `experiments/paper/` holds one canonical paper (`ICE_paper_v2.tex`, generic
  `article` class) plus venue twins (`_tmlr`, `_tist`, …). Only the canonical one
  may be linked or offered publicly — a twin carries that venue's branding, line
  numbers, placeholder volume numbers and a dummy DOI, which on a public repo
  reads as *"published there"*, a false claim about work merely submitted. This
  has happened once and was reverted. When a twin's content is better,
  **back-port it into the canonical file**; never repoint the link.
- **README and ICE_Architecture follow the same contract.** Both describe `main`
  as it is now, both updated in the same session as the change that invalidates
  them. README is the outside view (what ICE is, what it does, headline numbers,
  how to run it) and stays short; ICE_Architecture is the inside view and carries
  the detail. A number appearing in both must change in both or neither. Where
  either quotes evaluation results, say which snapshot (the paper's numbers are
  the `v2-paper-eval` tag, not `main`).

### End every session by rewriting docs/HANDOFF.md

One file, overwritten in place, committed as the session's last commit. It
carries the datetime, what the *previous* session was told to do next (so the
next session can compare intent against the git log), current position, the
decisions made, and anything that would otherwise be lost. **It is state, never
a queue** — the roadmap is the queue, and a handoff that starts listing work
becomes a second source of truth that drifts. Format and rules are in the file.

**⚑ It is written LAST, and there are exactly two things that make it time.**
Either **the work the session was given is DONE**, or **the context is genuinely
running out** — and in that second case it is not a shortcut: write the handoff
*and* do the full end-of-session job with it, propagation included (the docs the
change invalidates, the roadmap boxes, PROVENANCE, TRAPS, the cleanup ledger).

**A convenient stopping point is NOT one of the two.** Writing it early turns an
unfinished session into a finished-looking one: the next session reads the
handoff, trusts the position it states, and the outstanding items silently
become nobody's. **If work is outstanding and context is not the reason, say so
in chat and let the user decide** — never narrate a stop into HANDOFF.md as
though it were an ending. Stating what is unfinished, plainly and specifically,
is worth more to the next session than a tidy summary of what is not.

## Commands

Package/deps are managed with **uv** (Python 3.11.9, pinned in `.python-version`).
Always run project code through `uv run`, **from the repo root** (model paths and
`.env` resolve relative to it — see TRAPS #10).

```bash
./ice          # start everything: docker postgres, uvicorn proxy; tails logs
./stop_ice     # stop all services
./setup.sh     # first-time install (Arch/CachyOS): pacman deps, pyenv, uv sync, docker up, alembic upgrade, model pull

# Individual services (what ./ice runs under the hood)
docker compose -f docker/docker-compose.yml up -d
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Database migrations (SQLAlchemy sync + Alembic; postgres+psycopg on localhost:5432/ice_db)
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "message"   # ⚠ read every generated migration — TRAPS #14
```

There is **no Celery, no Redis and no separate worker process**: C7 replaced
them with an in-process maintenance runtime that the proxy owns. Background
model mode is `shared` by default (reuses the main Ollama model); `dedicated`
starts a separate vLLM on :8002 and is a manual, power-user path.

**⚠ Assume `./ice`, `./stop_ice` and `setup.sh` DO NOT WORK.** They are dev
scaffolding with a decided fate (Track F end-state: one packaged app, plus the
separate headless boot path for ICE-as-MCP that shipped with E7, `ice-mcp`), so
they are unmaintained and are not kept in step with the code. **Never debug
them, never build on them, and never conclude anything from one of them
failing** — bring the stack up with the individual commands above instead. Fix
one only if the user asks. Logs go to `logs/{proxy,vllm_bg}.log`.

### Tests

Tests in `tests/` are **standalone scripts, not a pytest suite** — they
`sys.path.insert` the repo root and run directly against a live Postgres. The
exception is `tests/smoke/` and a few `test_*_freeze`/`invariants` files, which
are pytest.

```bash
uv run python tests/test_retrieval.py     # a behavioural suite
uv run pytest tests/smoke -q              # the rails — run before every commit
```

They need docker up, and for pipeline tests Ollama running. No lint config;
match existing style. **Suites leak rows and the residue fails a *different*
suite later — TRAPS #6, #15, #16.** Check the store before debugging a strange
scoping or retrieval failure.

## Code map

Where things live. (How they *work* is `ICE_Architecture.md`'s job — kept as
pointers here so this section cannot rot into a description of a deleted system,
which is exactly what happened to its predecessor.)

| Path | Holds |
|---|---|
| `src/api/main.py` | The request lifecycle: classify → decide → retrieve → assemble → route → stream → post-flight. |
| `src/api/config.py` | **Pydantic Settings**, loaded from `.env` — every tunable value (G9 consolidated ~160 of them here). The DB URL is duplicated in `alembic.ini`. |
| `src/api/prompt_assembler.py` | Builds the system prompt from retrieved fragments, slots, bookmarks, summaries. |
| `src/api/memory_decision.py` | B2 — the one place the retrieve/don't-retrieve decision is made. |
| `src/api/routers/` | `memory_slots.py`, `user_control.py`, `adapter.py` — thin adapters over `src/services/`. |
| `src/classifier/` | The MLP head over a frozen `Qwen/Qwen3-Embedding-0.6B` encoder at native 1024 dim. Schema v2: **27 logits — 11 topic + 12 intent (multi-label sigmoid) + 4 independent context sigmoids**. Live checkpoint in `settings.classifier_model_path`; schema in `data/labeled/label_schema.json`. |
| `src/retrieval/orchestrator.py` | The hybrid retrieval legs, RRF fusion, scoping, budgeting. Leg weights live in `src/retrieval/leg_weights.py` + settings. |
| `src/memory/models.py` | Every ORM model, one file. `EpisodicMemory` (+pgvector), `Codex*` (the knowledge graph), `ProceduralMemory`, `ContextCluster`, `MemorySlot`, documents, plus operational tables. DB is **pgvector** (`pgvector/pgvector:pg16`). |
| `src/workers/runtime.py` | The in-process maintenance runtime. **`JOBS` is the source of truth for what runs and how often** — register a new background job there. |
| `src/workers/` | The jobs themselves: `post_flight` → `codex_extractor` / `procedural_extractor`; periodic `clustering`, `decay`, `reflection`, `batch_summarizer`, `maintenance_agent`, `fine_tune`. |
| `src/services/` | HTTP-free service layer shared by the REST routers and the MCP server. |
| `src/mcp/server.py` | ICE-as-MCP (`ice-mcp`), with its own headless boot path. |
| `src/paths.py` | Anchors every model/data path (G31); `ICE_HOME` overrides. |

Logging is **structlog** (`ice.*` loggers), consistent with the no-silent-failure
principle: surface what the system decided, don't swallow it. Trained artifacts
(`models/`), datasets (`data/`) and `scripts/` are versioned; classifier
checkpoints are suffixed by version/fine-tune generation, and `config.py` is
updated when one is promoted.
