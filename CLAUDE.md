# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

**Loaded into every session — so it costs context on every prompt, and holds
only what is (a) true of the system as built and (b) not written anywhere a
session reads naturally.** Everything else is a pointer: where a rule's evidence
lives in another doc, that doc owns it and this file keeps the imperative.

⚠ **It rots the most expensively of anything here**, because a stale line is
believed by every future session. It was once found describing Celery, Redis, a
384-dim encoder and a 25-logit head — all long deleted. Item (5) on the
deletion-sweep checklist in [docs/CLEANUP.md](docs/CLEANUP.md).

**⚑ Keeping it true is a standing job, and changing it is USER-GATED.** When a
session touches a subsystem this file describes, **verify the claim here against
the code** — same sweep as any other doc. But **ask before editing CLAUDE.md**,
every time, even for a one-word correction: it is loaded into every session, so
an edit changes how all future sessions behave, and a rule quietly dropped here
is gone everywhere. *(That is not hypothetical: the `942d5a9` shrink compared a
rule against its ROADMAP twin **by heading rather than by text** and deleted a
clause the twin never had. Report what you found and what you propose; wait.)*

## What ICE is

Local-first AI memory middleware, and a research project. A **FastAPI proxy**
(`src/api/main.py`) sits between an OpenAI-compatible frontend and **Ollama**;
every `POST /v1/chat/completions` is classified, given retrieved memory, routed,
streamed, then processed into the memory store asynchronously. The same store is
reachable through **ICE-as-MCP** (`src/mcp/server.py`).

Guiding principle in the code: **"memory is earned"** — a turn is stored
losslessly only if dense enough (`lossless_flag` / `inject_raw`, set by the
post-flight evaluator); otherwise the background model summarises it.

Why it exists, what it is *not*, and the design principles: [docs/VISION.md](docs/VISION.md).
Experiment results live in `experiments/*/results*/` as `.md` summaries.

### ⚑ WHICH VERSION YOU ARE WORKING ON — say it, every time

| | what it is |
|---|---|
| **v1** | the paper's Exp 0 and Exp 1 — the immature system. Historical only. |
| **v2** | the paper's mature run and the ablations. Frozen at tag `v2-paper-eval`; `docs/ICE_Architecture[real_v2].md` describes it and is **never updated**. Every number in the paper is v2. |
| **v3** | **what `main` is now, and what you are working on.** |

**Name the version whenever a number or a behaviour is discussed** — v2 and v3
numbers are not comparable and nothing in the output says which you hold. A v3
change that removes something v2 shipped is a decision, not a regression, and
needs the same gate as any other production change.

## Start here

**⚑ READ IN FULL — both are short and both are about *this* session:**

| Read | What it is |
|---|---|
| [docs/HANDOFF.md](docs/HANDOFF.md) | **Where the last session left off, and what it was told to do.** Rewritten every session. It is state, never a queue. |
| [docs/TRAPS.md](docs/TRAPS.md) | Mistakes this project has actually made — failure *shapes*, so you recognise one from inside it. Add to it the same session something bites. |

**⚑ CONSULT, NEVER READ WHOLE.** These are reference works, not reading. Opening
one end-to-end burns the context the actual work needs:

| Doc | Read only… |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | **the item you are working**, plus its `Assumes` chain. It is the queue and the only queue; its "How to use this file" and "HOW TO EXECUTE" blocks carry the rules and current position. Check items off there. |
| [docs/ROADMAP_DONE.md](docs/ROADMAP_DONE.md) | the record for one finished item — what shipped, and **what that entry got wrong about its own subject**. Every stub links here. |
| [docs/ICE_Architecture.md](docs/ICE_Architecture.md) | the section for the subsystem you are touching. Authoritative design reference. **Where any doc conflicts with code, code wins.** |
| [docs/FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) | the row for one feature — its `file:line`, controlling setting, that setting's **default**, and whether it is **ON by default**. |
| [docs/specs/](docs/specs/) | the spec for your item **and its `Assumes decided specs:` chain**, before coding. Obey [specs/README.md](docs/specs/README.md) rules 11–12 — USER-REQUIRED steps, and the divergence protocol: code↔spec mismatch ⇒ stop, re-ground, fix the spec first, never improvise past it. |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | the entry for the run you are questioning. It owns what was *done* — models, corpora, run parameters — and states its own standing rule: follow it when a run produces an artifact. |
| [docs/CLEANUP.md](docs/CLEANUP.md) | the deletion-sweep checklist, or the move/rename ledger. |
| [docs/VISION.md](docs/VISION.md) | why ICE exists, what it deliberately is **not**, and the design principles. Stable — check it before assuming a capability is in scope. |

**⚑ CHECK [FEATURE_INVENTORY.md](docs/FEATURE_INVENTORY.md) BEFORE CLAIMING THE
SYSTEM DOES SOMETHING — AND BEFORE BUILDING IT.** Its `On by default?` and
`DEAD OR INERT` columns are the load-bearing parts: a capability has been
rebuilt because nobody knew it existed, and inert paths have been credited with
measurements. **Update it in the same session as anything that adds, removes or
re-gates a feature.** Code wins over it, always.

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

**A product is wanted; research is what is being done** — in that *order*:
nothing gets productised until it is measured, not believed. The unit of
progress here is **a measurement someone can trust**, not a feature.

**⚑ Be certain the measurement is CORRECT before believing it.** Four questions,
each of which has caught a wrong number in this repo:

- **Is everything there?** Did the harness call what the real path calls? A
  scorer skipping the budget setter measures something — just not ICE.
- **Would this number look the same if the thing under test were broken?** A
  presence-based score answers *"is the expected thing in there"*, never *"is
  what is there any good"*.
- **Can the metric even SEE the subsystem?** Recall credited only episodic
  fragments for months while four other legs returned thousands
  ([TRAPS #32](docs/TRAPS.md)).
- **⚑ Did I QUERY the running system, or reason about the code?** Reasoning
  forward produces conclusions that feel verified and are not — four wrong ones
  in one session, each disproved by a query taking seconds
  ([TRAPS #23](docs/TRAPS.md)).

**Change the instrument one part at a time, and prove each part TWICE** — once
in isolation, once through the path production takes. **A long run started on an
unverified change does not fail; it returns a number, and the number is wrong.**

**Nothing may be skipped and no result is exempt.** Where something cannot be
measured yet, say so **in the number's own words** — "off-production",
"unconfirmed", "inert on this corpus" — rather than letting it read as clean.

**Neither long nor short experiments are virtuous.** Size the run to the
question and compute what it can resolve *before* starting. **There is no hurry
and no room for waste**: being unhurried is what makes re-checking affordable,
not permission to re-derive what is already written down.

Worked evidence: [PROVENANCE.md](docs/PROVENANCE.md) 2026-08-12 onward, and
TRAPS #20–23, #32–35. Every finding that survived scrutiny came from reading
actual output.

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

**⚑ THE REPO IS PUBLIC** (`origin` → `github.com/Deepnar/ice`) and has been for a
while. It is not a private backup: **every push is a publication.** Pushing
during normal development is pre-authorized — push at natural points without
asking — but write every commit, message and file knowing it is read as soon as
it lands, with no window to take it back.

- **⚑ SMALL COMMITS *IN THIS FORMAT*. Granularity and format are ONE rule.**
  Many small commits split by *concern* — never one end-of-session commit; if a
  message needs bullets for unrelated changes it should have been several.

  ```
  area: what changed (ITEM)

  ITEM — what it does now:
  - concrete change: function/file/migration names, measured numbers

  Validated N/N (tests/test_x.py: what was checked); regressions green.
  Architecture §6.10 updated; roadmap C6 checked.
  ```

  **Subject:** lowercase area prefix (`retrieval:`, `codex:`, `memory:`,
  `workers:`, `classifier:`, `budget:`, `roadmap:`, `docs:`, `tooling:`) · what
  changed · roadmap id in parens · ≤ ~70 chars. **Body:** organised by item,
  dense and concrete, closing with **`Validated`** and a **docs/roadmap** line.
  **The log is the reference — read it before writing one:**
  `git log 6bac35d 1a30484 051cea9 5124efb`.
  - ⚠ **A record, not an explanation.** No essays, no rhetorical beats, no
    argument structure. Impersonal, in the repository's voice; **never narrate
    the session** ("the user asked…", "we decided…"). **No AI attribution.**
- **Freeze at the experiment phase:** once **SEMIFINAL (Z1)** or **FINAL**
  begins, **stop pushing** until the user says otherwise.
- **Never commit personal content** — planning, career notes, conversation
  corpora, third-party email, credentials. **Git history is forever the moment
  you push**, and a rewritten history still serves old objects by SHA until the
  host garbage-collects (TRAPS #12 — a support ticket last time). **Check before
  the commit, not after the push.** ⚠ A cleanup commit *message* that describes
  what was removed is itself a signpost; rewrite messages too, not just diffs.
- **⚑ PRIVATE DETAIL IN A TRACKED DOC — USER-GATED, every time.** The tracked doc
  states the technical claim and stands alone; the specifics go in
  `docs/PRIVATE_CONTEXT.md` (gitignored — verify with `git check-ignore` **before**
  writing it) under a marker like `[PRIVATE:g43-example]`; the tracked doc points
  at the marker. **Then ask the user to eyeball the exact lines before
  committing** — not a summary, the lines. **What belongs where is the user's
  call, not a judgement to make for them.**
- **A history rewrite must rewrite TAGS, and `git log` on `main` cannot verify
  it.** Verification is `scripts/git/check_history_clean.sh` (`--clone` checks
  what the public receives), wired to `pre-push` by `install_hooks.sh` — run once
  per clone. Full story: **TRAPS #12**.
- **README.md is a first-class deliverable** — the first thing a visitor reads.
  Update it in the same session as anything that changes what the project *is*
  or how it is run.
- **⚑ README links the VENUE-AGNOSTIC paper, never a venue submission.**
  `experiments/paper/` holds one canonical paper (`ICE_paper_v2.tex`) plus venue
  twins. A twin carries that venue's branding and a dummy DOI, which on a public
  repo reads as *"published there"* — a false claim about work merely submitted.
  When a twin's content is better, **back-port it**; never repoint the link.
- **README and ICE_Architecture follow the same contract** — both describe `main`
  as it is now, both updated in the same session as the change that invalidates
  them. README is the outside view and stays short. A number in both must change
  in both or neither, and must say which snapshot it is (the paper's numbers are
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
