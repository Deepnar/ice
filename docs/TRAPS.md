# Traps — mistakes this project has actually made, and how each was caught

**Why this file exists.** These were carried in the session handoff note for
weeks, growing one entry at a time. That made them fragile: a handoff that got
trimmed, or a session that ended badly, would lose them. They are hard-won and
several have re-occurred, so they belong in the tree.

**How to use it:** read it once at the start of a session. Each entry is a
*failure mode*, not a rule — the point is to recognise the shape when you are
inside it.

**How to add:** when something bites, add it here in the same session, with the
concrete case. An entry with no worked example is a platitude.

---

### 1. Two eval scripts cannot see a change to the pre-classifier path
`score_hard_probes.py` and `eval_probes.py` both call `load_checkpoint` and
never run `classify()`. A change upstream of the head is invisible to both, so
a green run means nothing about it.

### 2. A spec can contradict itself, or go stale
`docs/specs/README.md` rule 12: stop, verify against current code, record the
divergence in the entry, fix the spec first, then code. Never improvise past a
mismatch — that is what the specs exist to prevent.

### 3. Measuring is not permission
A designed seam that measures useless is **evidence**, not authorisation to
delete it. Ask the user — and first explain what the thing was *meant* to do.
Applied repeatedly: `growth_cap` was kept despite coverage superseding it,
because deleting it before coverage is measured to bind makes ICE *more*
expensive.

### 4. Test the direction you didn't think of
The C12a suite passed `runtime=None` at every call site, so 48 green checks
never touched the path where a runtime *is* passed — which was the only path
any live surface uses, and it was completely broken.

### 5. A two-sided assertion, or it proves nothing — and check the negative side isn't vacuous
A tight-budget check once passed on three-word fixtures that fit *any* budget.
A knee assertion asserted the wrong thing (`-1` means "no cut", which trivially
satisfies "never cuts below min_keep").
**Re-earned 2026-08-03:** the first negation assertions in
`test_codex_write_path.py` were wrong about the design — negation does not just
delete, it expires the positive edge **and** writes an active `negated=True`
edge, so "we decided against X" stays retrievable. An absence-only check would
have passed if negation had written nothing at all. Also: assert on **sections**,
not bare substrings — `"Negations: NOT uses → x"` legitimately contains
`"uses → x"`.

### 6. A crashed or leaky test leaves rows behind, and they fail a *different* test later
`test_documents`' cleanup once selected its rows from a hardcoded filename
allow-list, so every new fixture leaked.
**Re-earned 2026-08-03, and this time it broke an unrelated suite:** two orphan
conversations (one `kind='document'`) left by a run killed mid-flight made
`test_session_scoping` fail 39/40 — `_apply_document_visibility` adds every
non-enabled document conversation to `exclude_conversation_ids`, and the
assertion compares that list exactly. `test_documents` does **not** leak on a
clean run; the residue came from the killed one.
**⇒ If a scoping or retrieval test fails oddly, check the store for orphan rows
BEFORE debugging the code.**
**Re-earned a THIRD time, 2026-08-08 — and this time the residue was partly
self-inflicted.** Two live turns driven through the proxy to validate G5, plus
an orphan `kind='document'` conversation, took `test_session_scoping` to 39/40
on exactly the check named above. The rule held: the store was inspected first
(all rows were empty shells — 0 turns, 0 `documents`), they were deleted, and
40/40 returned with no code touched.
**⚠ But the CAUSE was then misdiagnosed, which is the more useful half.** The
orphan was a *document* conversation, so `test_documents` was blamed — with no
evidence beyond the matching vocabulary. Bisecting suite-by-suite against a
cleaned store showed `test_documents` leaks **nothing** (53/53, 0 rows, and
`test_session_scoping` passes right after it); the leaker is
**`tests/test_longevity.py`**, whose `doc_conv` fixture has no cleanup.
Three additions to the habit: **(a)** end-to-end validation against the live
stack is itself a source of residue — clean up after it in the same session;
**(b)** **bisect before naming a suite** — one `for` loop over the suites with
a row count after each is cheaper than the wrong docs it saves you writing;
**(c)** a cleanup that verifies "0 rows remaining" scoped to its *own* recorded
ids cannot catch a row it created but never recorded.

### 7. A dead test can be promoted to load-bearing, and an import check will never notice
Re-check retired tests by **running** them. Corollary: do not write a test that
pins ground a scheduled item is about to change — flag it instead.
(`test_codex_2_0.py` was deliberately left unreplaced for exactly this reason,
and was rebuilt as `test_codex_write_path.py` once A9b/A12 had settled.)

### 8. Inserting a parameter mid-signature breaks positional callers silently
`_cold_lookup` started receiving a scope dict where an embedding was expected.
Related and reassuring: G23's fail-loud guard refused an unregistered vector
column and `test_longevity` caught a missing `store_meta` stamp — both were the
system telling the truth. **Do not route around a guard that fires.**

### 9. A benchmark harness that forgets `model_override` benchmarks the default model
Identical medians across "two different models" is what gave it away.
**Always print the resolved model name.** Two companions from the same session:
`pgrep -f <pattern>` matches the **waiter shells** whose own command line
contains the pattern (four wait-loops reported RUNNING for a process that had
already died — use `[a]9b_...`); and a harness that writes results only at the
end loses everything when a later arm crashes, so write after each arm.
**The `pkill` half re-earned 2026-08-08, and it is worse than `pgrep`:**
`pkill -f 'uvicorn src.api.main:app'` inside a shell command whose own text
contains that string **kills the shell running it** (exit 144), taking the rest
of the compound command with it. The bracket trick is not optional here —
`pgrep -f '[u]vicorn ...' | xargs -r kill`.

### 10. Running from the wrong directory silently changes what the system IS
Model paths in `src/` are CWD-relative (roadmap **G31**). Outside the repo root
the micro-NER falls back to a capitalized-word regex and `load_registry()`
returns `{}`, so `get_fallback_model()` drops to `default_fallback_model` —
which is a model measured to produce word salad. Both fired accidentally on
2026-08-03 and **both produced results that looked like findings** until the
working directory was checked. **Run scripts from the repo root.**

### 11. A silent fallback hides an outage
**The standing rule, and its worked example** (was a CLAUDE.md section until
2026-08-09; the rule stays there in one line, the evidence lives here).

The measured case: **every background LLM call in ICE was returning nothing**,
and the system looked fine. Reasoning models spend the whole `max_tokens`
budget inside a hidden thinking block, so Ollama returns `content=""` — and
every caller had a default. `clustering._generate_cluster_name` returned
`"Unnamed Cluster"`. `detect_blob_kind` returned `document` via its
`blob_kind_unparsed` branch. `post_flight` turned the empty summary into `None`
and let raw text win. Each of those defaults is individually *correct*
engineering. Together they made a dead subsystem indistinguishable from a
working one, for an unknown length of time.

⇒ **A fallback must be observable.** When a component substitutes a default for
a real answer, it emits at WARNING with the reason — **every time, not on the
first occurrence**. A fallback that fires on 100% of calls is not resilience,
it is an outage wearing resilience as a costume.

Two corollaries, both earned the same day. **(1)** Check the *rate*, not the
existence: `ner_utils` has a regex fallback whose own docstring says it is
"log-worthy if this fires in normal operation" — and there was no log line, so
nobody could have known it fires whenever the process starts outside the repo
root. **(2)** When a subsystem produces plausible-but-thin output, **verify the
model was actually called before tuning anything about it** — the first
explanation is usually "it never ran".

**Swept through retrieval by G36 (2026-08-09):** 22 silent `except` handlers in
`orchestrator.py` now report through one `retrieval_leg_failed` event. The
measured firing rate before shipping was zero across 476 checks, which is what
makes the line meaningful — see #13b for why that check is not optional.

### 12. `git log` on `main` cannot verify a history rewrite
A rewrite must rewrite **tags**. `refs/tags/v2-paper-eval` pointed into
pre-filter history and a plain `git clone` served ~5,700 personal prompts from a
public repo for two days. Verification is
`scripts/git/check_history_clean.sh --clone`, which checks what the remote
actually hands out. Also: **never resolve a tag mismatch by forcing one side to
win without first asking which side predates the rewrite** — doing exactly that
is what overwrote the good local tag with the bad remote one.
**Closed 2026-08-04:** re-pointing the tag fixed reachability, but GitHub keeps
unreachable objects fetchable by direct SHA until it garbage collects, so a
Support ticket was needed for the server-side sweep. Confirmed done — the
pre-filter commit and tag object now return 422 while the good tag target still
returns 200 (the control that stops the check passing vacuously).
⇒ **Re-pointing a ref is not deletion.** After any history rewrite on a hosted
remote, the objects survive until the host GCs them, and only the host can do
that.

### 13. A hand-authored probe set grades the author's imagination, not the system
The relation-matching ladder scored **24/24, zero errors** on 31 probes written
by the same session that designed it. On 1,152 real out-of-vocabulary relations
harvested from 300 turns, the identical cascade scored **5.7%**. The probes
tested the failures the author *expected* — space-vs-underscore, typos, helper
verbs — and the model's actual failure is **inventing new concepts**
(`is_exam_of`, `exists_in`, `has_task`), which no string method can map. The
top three real misses were `is`, `has`, `includes`: ordinary English the
197-word vocabulary simply lacks.
⇒ **Before a hand-built probe set is allowed to decide anything, harvest the
real distribution and check the probes look like it.** A probe set authored
alongside the solution is a mirror, and it will agree with you.

### 13b. …and the same trap fires on the fixtures you write for a bugfix
Same shape as #13, one week later, caught only because a live run happened.
G5's new SSE parser was validated by 8 hand-written fixtures, all green. The
first real turn through the proxy then logged `dropped=1, lines=14` on a
**perfectly healthy stream**: the terminal usage chunk carries
`"choices": []`, which raises `IndexError`, and the parser counted that as
damage. So the brand-new "this fallback is now observable" warning would have
fired on **100% of turns** — noise that trains everyone to ignore it, which is
the exact failure the logging was added to prevent.
The fixtures missed it because their author wrote what a *damaged* stream
looks like and never pasted in what a *healthy* one actually contains.
⇒ **A bugfix's own test fixtures are hand-authored probes too.** Before
trusting them, capture one real sample of the thing being parsed and put it in
the suite verbatim. And for any new warning, ask what fraction of NORMAL
traffic trips it — a fallback that always fires is not observability.

### 13c. A same-second, same-size edit is silently ignored (stale `.pyc`)
Python invalidates cached bytecode by comparing the source's **mtime at
one-second granularity and its size** against the header in
`__pycache__/*.pyc`. Change `1.2` to `1.3` — identical byte count — within the
same second as a previous run, and the interpreter serves the **old** module.
No error, no warning.
It bit twice inside one command during G9, in both directions: a negative
control that nudged a leg weight **passed when it should have failed** (the
edit never loaded), and the freeze suite then **failed after the file was
restored** (the pre-restore value was still cached). Both results were
believed for several minutes, and one of them was about to be written up as a
real finding.
The tell is the pair `stat -c '%y %s' <file>` against `ls -la
__pycache__/*.pyc` showing the same second. The fix is
`find src tests -name __pycache__ -type d -exec rm -rf {} +` between edit and
run.
⇒ **This is aimed squarely at Z1.** Its sweep is a loop of "edit one constant,
re-run, record the score" — same file, same size, fast iterations. A sweep that
does not clear bytecode between arms will silently record **the previous arm's
number** for the current arm's label, which looks exactly like a knob measuring
`plateau`. Clear the cache per arm, or mutate `settings` in-process rather than
editing files.

### 14. `alembic --autogenerate` proposes deleting every index the models don't declare
Adding one table generated a migration that also emitted **20 `drop_index`
calls and five `alter_column`s loosening NOT NULL** — including all seven
`idx_*_embedding` HNSW indexes G23 created. None of it was drift: those indexes
were made by earlier migrations and raw SQL rather than declared on the ORM
models, so autogenerate cannot see them and reads their presence as something
to remove. Applying it unread would have dropped every vector index in the
database, and retrieval would have silently gone from indexed to sequential
scan — slower, but still *correct*, so no test would fail.
⇒ **Read every autogenerated migration and delete everything that is not the
change you asked for.** Verify after applying:
`SELECT count(*) FROM pg_indexes WHERE indexname LIKE 'idx_%_embedding'` (8).

### 15. A deterministic id is not evidence of a test fixture
Cleaning residue on 2026-08-09, an empty `conversations` row with a **UUID
version 5** id was classified as test leakage — the reasoning being that only
`test_ingestion` computes uuid5s — and deleted. It was
`services/bookmarks.py::NOTES_CONVERSATION_ID`, the fixed conversation
`ice_remember` writes MCP notes into, created by **production** code. Cost was
nil (the row was an empty shell, the id is derived from a constant string, and
the service does get-or-create), and it was restored with its original
`created_at`. But the check that would have prevented it costs one command and
was run only **after** the delete: `grep -rn uuid5 src/`, which prints the
generator of every deterministic id in the system.
⇒ **Before deleting a row you did not create, grep `src/` for whatever produced
its id.** "Which suite leaked this?" quietly assumes a suite did. Three genuine
orphans in the same sweep — a `memory_slots` row, four `review_queue` rows, one
`procedural_memory` row — made the fourth look like more of the same, which is
exactly how the assumption survives.

### 16. Residue is not always inert, and "is_active" is where to look
The same sweep found a `memory_slots` row left by `test_services` on
2026-07-28: `slot_name='session_patterns'`, content `"svce0e62d24 content v1"`,
`is_active=True`, `scope_tier='global'`. `main.py` fetches **every**
`is_active=True` slot with no scope filter and the assembler renders every
global one, so that test marker had been injected into the system prompt of
**every turn for twelve days**. Any Z1 or Z2 measurement taken in that window
would have carried it.
The `procedural_memory` orphan in the same sweep is the subtler shape: it is
`is_active=False`, therefore invisible to both retrieval paths (`_procedural_lookup`
and `retrieval_svc` both filter `is_active=True`) — but `is_active=False` is
the **normal birth state** for a new pattern, and `procedural_extractor.py:86`
and `reflection.py:244` flip it to True on the next matching observation. It
was a dormant seed that a future turn could promote into a live memory whose
evidence turn had already been deleted. It was also never pruned because
C10's cascade only prunes patterns whose batches belong to a *deleted
conversation*, and this turn was removed by an ad-hoc cleanup instead.
⇒ **Rate the residue by what reads it, not by how much of it there is.** One
row can be in every prompt. And deleting turns outside `delete_conversation`
leaves procedural patterns orphaned — use the cascade.

**The leaker was found by bisect the same day, and its defect is #6c's exact
shape.** `test_mcp_server` snapshot-and-restores the `session_patterns` slot —
but the restore was guarded by `if slot_snapshot:`, and on a clean store there
is no slot to snapshot. The suite then *creates* one through `ice_slots set`
and the cleanup does nothing, so **every run left a global slot behind**. A
snapshot-and-restore cannot clean a row that did not exist to be snapshotted;
it needs a `created` flag and a delete branch, which `test_c10_c11` already had
for `pending_items` and was copied from. Fixed 2026-08-09.
⇒ **Any "snapshot the live row, restore it after" fixture needs the
did-not-exist branch.** Verify by clearing the table, running the suite alone,
and counting — which is also the bisect that names the leaker.

### 17. A config value can be present, correct, frozen — and inert
Two instances on 2026-08-11, from one afternoon, and they are the same mistake
in two costumes: **the value was verified, the effect was not.**

*Setting a knob to its off-value is not turning the feature off.* The plan for
Z1's tuning runs was `codex_relation_overlap_boost = 0.0`, to stop G34's
fires-on-every-prompt relation detector from distorting the tuned weights. That
setting is read at **one** site (a score bump). A detected relation has three
other effects, all of which survive 0.0: its fact lines are appended to the
fragment text and still consume budget; its fact edges reach
`_reinforce_codex_edges`, which writes edge strength, promotes `pending` →
`active`, and **commits**; and it still gates `_codex_enumeration`. So the
"neutralised" run would have kept mutating the store it was measuring. This is
the **second** 0.0-≠-off finding in four days — [G15](ROADMAP.md#g15) was the first.

*A literal that equals its setting is a dead knob with no symptom.*
`_session_diversify`, `_relevant_cluster_ids` and `_apply_rrf` each read their
setting only when the caller omits the argument — and every call site passed a
literal. `retrieval_max_per_conversation`, `retrieval_cluster_top_k` and
`retrieval_rrf_k` were unreachable. Nothing looked wrong because in every case
the literal *equalled* the setting (3=3, 10=10, 60=60), so the system produced
the right number by coincidence. **`test_settings_freeze.py` had a row for all
three and passed throughout**, because the declarations never moved — a suite
vouching for knobs it cannot reach.
⇒ **Never accept a config value on inspection. Change it and observe a
decision change** — which is CLAUDE.md's check-where-a-signal-LANDS rule applied
to settings rather than to classifier labels. Two consequences worth stating
separately: an off-switch must be checked at the **source** of the thing it
disables, not at the last site that consumes it; and a freeze/declaration test
proves a value was not edited, **never** that anything reads it.

### 20. The instrument you build to measure a system becomes part of what you measure
2026-08-12: three of the day's headline numbers were partly about the scorer
rather than about ICE. The scorer never called `set_budget_from_turn_count`, so
it ran at a default budget production does not use; the snapshot restored the
database but not the probe→row map, so a stale file produced "every retrieval
leg returns 0" and nearly shipped as *"retrieval is completely broken"*; and the
probe generator's ambiguity guard scored **question + answer** words when
retrieval only ever sees the **question**, letting 74% of ambiguous probes
through the check meant to stop them.

**The rule: before believing a measurement, check that the instrument reproduces
the production path.** Cheap tests that would have caught all three — does the
harness call what `main.py` calls; do the IDs in the map exist in the store; does
the guard test the same information the system has.

### 21. A metric can be green on output that does not exist
`summary_coverage` measures whether must-preserve TERMS appear. A model that
emits `Key terms: a, b, c` and one generic sentence scores **0.9375–1.0** while
producing no summary at all — measured on `granite4:tiny-h`, 12 of 12 items,
2026-08-12. The metric was working exactly as designed; "the terms are present"
and "a summary exists" are different claims and only one was being tested.

**Generalises past summaries:** any presence-based score (coverage, in-vocabulary
rate, hit counts) answers *"is the expected thing in there"* and never *"is what
is there any good"*. Pair every one with either a structural check (is this prose
or a word list?) or a human reading a random sample. On the same day the eyeball
pass overturned the metric ranking of eight models and found fabricated facts
(`kael --role--> fire mage`) that every number had scored as healthy.

### 22. `pgrep -f "<pattern>"` matches the shell that is running it
Three background waiters in one session hung forever because
`until ! pgrep -f "seed_store.py"` matched **its own command line**. The process
being waited on had exited; the waiter kept finding itself.

Use the bracket form — `pgrep -f "[s]eed_store.py"` — which matches the target
but not the literal pattern text in the watcher's own argv. Same trap applies to
`ps | grep`.

### 23. Reasoning about the code instead of querying the running system
**The most productive failure mode in this repo, measured 2026-08-12: FOUR wrong
conclusions in a single session, every one from reading source and reasoning
forward instead of asking the live system.** Each was stated confidently, each
was wrong, and each took minutes to disprove once actually queried.

| claimed, from reading code | what one query showed |
|---|---|
| "the pgvector query has no index, so it full-scans" | `idx_codex_entities_embedding` exists **and** is migration-covered. Read the diff, never opened the database |
| "`retrieval_max_per_conversation=3` is capping recall" | `_session_diversify` passes **90 of 90**. The token budget is the cap. Read the setting, never traced the pipeline |
| "the vector leg is dead — every hit is bm25" | vector returns **87–89 candidates alone**. Read the leg-stamping code, never ran the leg |
| "procedural extraction produced 0 for every arm" | 25–59 per arm. Read my own script's output under the **wrong dict key**, never queried the table |

**Why it is so seductive here:** the source is well-commented, the comments
explain *why*, and a chain of correct-looking reasoning over correct-looking
code produces a conclusion that feels verified. It is not. The comments describe
intent; the database describes what happened.

**The rule: any claim about behaviour gets a query before it gets stated.** Not
"read the function" — run it, print the count, select from the table, trace the
stages. If a claim cannot be cheaply checked against the running system, say it
is unverified *in the claim itself* rather than in a caveat afterwards.

**The corollary that makes this affordable:** these checks are seconds. Every one
of the four above was disproved by a single query. The cost of the habit is
nothing; the cost of skipping it was four wrong conclusions, one of which nearly
shipped as "retrieval is completely broken".

### 24. A guard that only fires in the harness makes production look broken

**2026-08-13, and it moved a headline number by 2×.** Fifteen probes returned
zero fragments and the recorded hypothesis was that the token budget stopped
them. Wrapping the live `_enforce_token_budget` showed **it was never called at
all**: retrieval exits ~120 lines earlier at `retrieve()`'s
`context_reliance == "Zero_Shot"` guard — a *defensive* guard whose own comment
says the decision is made upstream, and which `main.py` makes unreachable by
setting `context_reliance = "Long_Term_Memory"` before it calls.

The scorer called `retrieve()` directly, so it hit a branch production cannot
reach. Correcting that, plus calling the budget setter `main.py` calls, moved
recall@10 from **0.250 to 0.508** on the same store and the same probes. **A
legacy-mode control reproduced 0.250 exactly** — `recall@1` identical to 17
decimal places — which is what makes the delta attributable to the harness
rather than to luck.

**The rule: when a component has a "this should never happen here" guard, the
harness must reproduce whatever makes it unreachable.** Otherwise the harness
measures the guard, and the guard is not the system.

⚠ **Neither endpoint is a retrieval result (added 2026-08-22).** The 2× swing is
sound evidence about *instruments* — the control reproducing 0.250 to 17 decimal
places is what makes it attributable — and it is no evidence at all about ICE.
Both 0.250 and 0.508 credit only episodic fragments ([#32](#32-a-metric-can-be-structurally-blind-to-most-of-the-system)),
both were taken on a probe set later measured **64% contaminated**, and both came
from a scorer that classifies without the conversation, moving the fusion weights
on 65% of probes ([#44](#44-the-harness-classified-without-the-conversation-and-it-moved-65-of-the-fusion-weights)).
**Cite this entry for the lesson, never for the number.**

### 25. `.env` keys that Settings does not declare take the whole application down

**2026-08-13.** Three `PROBE_*` lines were added to `.env` for an experiment
script that reads `.env` directly. `Settings` is configured `extra="forbid"`, so
**every import of `src.api.config` raised `ValidationError`** — the proxy, the
workers, and all 347 tests at once. The experiment script kept working
perfectly, because it never imported Settings.

That asymmetry is the trap: the new thing was fine and everything else was dead.
It surfaced only when an unrelated command touched the production import path.

**The rule: any new `.env` key gets a declared field in `Settings` in the same
edit.** And: after touching `.env`, run something that imports the app.

### 26. Embeddings cannot tell a converse from a synonym

**2026-08-13, measured on the live encoder.** Opening the relation vocabulary
(G45) meant canonicalising near-duplicate relations by cosine. Calibration
against 30 hand-built pairs found the classes **overlap**, and the worst
offenders were exactly the pairs that must never merge:

| pair | cosine | |
|---|---|---|
| `before` / `after` | **0.8791** | above the 0.86 merge threshold in force at the time |
| `parent_of` / `child_of` | 0.8569 | |
| `teaches` / `learns_from` | 0.8159 | |

Converses share every context word, so a similarity measure places them on top
of each other. Merging `before` into `after` reverses every temporal fact in the
graph, silently.

**The rule: direction and polarity are settled deterministically, before
similarity is consulted — never by a threshold.** A curated converse map plus a
passive-marker check do that job; the threshold only ever decides between pairs
that are already known to be safe. With the guards in place the highest
unsafe pair fell to 0.787 and the threshold could be *lowered* to 0.82, catching
more true merges with fewer wrong ones.

### 27. Things that LOOK broken and are not — check here before debugging

**Every line below cost real time on 2026-08-13 and was settled by a
measurement.** They are collected in one place because each presents as an
obvious bug, and the obvious fix is wrong in every case.

**"The procedural leg returns nothing."** It is not retrieval. `_procedural_lookup`
requires `is_active = true`; activation needs `reinforcement_count >= 3`;
reinforcement needs a cross-session match above `procedural_similarity_threshold`
(0.85) — and two extractions of the SAME habit **measure 0.708**. Patterns are
born at 1 and never activate. Extraction can be working perfectly while the leg
returns nothing forever. See [G49](ROADMAP.md#g49).

**"The probe API is down / the key is wrong" (HTTP 403, body `error code: 1010`).**
Cloudflare fronts `opencode.ai` and rejects anything that looks like a scripted
client. **Send a browser `User-Agent`** and the same key, endpoint and model work
immediately. Nothing is wrong with the credentials.

**"Retrieval is non-deterministic."** Two separate causes, and only the first is
a defect. (1) `access_count` was written on every retrieval and read by nothing —
now gated by `retrieval_strengthen_writes`; that took identical runs from 26/40
to 40/40. (2) What remains is **process warm-up**: pass 1 differs from passes 2
and 3 on ~2/60 fragment counts with **no rank changes**, and passes 2 and 3 agree
exactly. `_relation_gloss_cache` is a lazily-built module global and the prime
suspect (unverified). **Issue one throwaway retrieval before scoring** rather
than hunting it.

**"`== None` in the ORM filters is a bug ruff caught."** It is **required**
SQLAlchemy: `is None` does not generate SQL. E711/E712 in a `.filter()` are false
positives, and "fixing" them silently breaks the query. Same for the late imports
(E402) in `codex_extractor.py` — they look deliberate.

**"The model returned nothing, the call failed."** An empty reply is a
**legitimate answer** when the prompt says "return an empty list if there is
nothing here". `json.loads("")` raises at char 0, which reads as a failure;
retrying it three times with backoff wastes ~12s per honest non-answer and makes
a correct model look broken. Related: a **truncated** reply still contains
complete objects before the cut — salvage them instead of discarding the
generation. Discarding cost 87% of one probe class.

**"`retrieval_max_per_conversation` is capping recall."** It is not, for
within-conversation probes: `_session_diversify` exempts the CURRENT
conversation entirely, and every generated probe asks about its own. This was
claimed, struck, and is struck again.

**"`.env` says the background model is `gemma4:26b-a4b-it-q4_K_M`, so that is the
model."** That is a stale pin, not a decision. The A12 read ranked it **third**
and named **`qwen3:4b-instruct`** the practical pick — tied on summary quality at
**2.5 GB against 9.6 GB**. Check [PROVENANCE.md](PROVENANCE.md) before trusting
the pin.

**"`data/` is untracked, so commit it."** `data/` is **871 MB** across 102 files
and `data/labeled/` is gitignored for good reason (conversation-derived training
data, on a **public** repo). Only `data/relation_seed.json` is meant to be
tracked. **Never `git add data/` without `git add -n` first.**

**"The coverage metric says 1.000, retrieval is perfect."** Coverage-based scores
(`codex_multihop`, `summary_synthesis`) are **trivially satisfied on a small
store** — if the store holds only the gold turns, everything comes back.
Measured 1.000 on a 2-turn fragment where nothing useful was happening.
`score_typed.py` now refuses to let that read as a result.

### 28. A roadmap entry is not a reliable description of its own subject

**Nine consecutive entries were found wrong about themselves** — the count is not
rhetorical, it was tracked across sessions in 2026-08 (`G30`'s ground check found
five errors in one entry; `G29` made it nine, and was the first where the entry
*overstated* remaining work rather than understating it).

The failure is structural, not careless. An entry is written when a problem is
noticed, and then the code moves: the bug gets fixed by a neighbouring change,
the setting is renamed, half the work ships inside another item, or the entry's
premise was a guess nobody re-checked. Nothing forces the text to follow.

Both directions cost, differently:

* **Understating** — the entry says "add X"; X already exists, and a session
  builds a second one.
* **Overstating** — the entry lists work that is already done, and a session
  spends itself confirming that. **This is the one that wastes a whole session**
  rather than breaking a system, and it is the harder one to notice, because
  finding nothing to do feels like being wrong rather than being finished.

⇒ **GROUND-CHECK THE ENTRY BEFORE WORKING IT.** Read the item, then read the code
it names, then query the running system — *before* planning against its
description. Record what the entry got wrong in the entry itself; that is what
`ROADMAP_DONE.md`'s "what it got wrong about its own subject" field is for.
Budget for this: it is routinely the first hour of an item, and it is not
overhead — it is the item.

### 29. Derive ground truth in the EASY direction

Z1's first retrieval key was built **backwards**: take a curated answer, work
back to the turns that support it. Two rounds of fixes later it still
**mis-grounded ~20% of claims**, and hand-verification by the user **failed it
twice**. Worse, the failures were invisible to every automated check — the key
looked fine.

Two distinct problems, and only one was fixable:

1. **The direction.** Reversed — *pick a real turn first, then write a question
   whose answer is in it* — the gold turn is correct **by construction**. No
   shortlist, no confirmation step, nothing to be wrong.
2. **Some probes cannot be scored at all, and no care in the derivation fixes
   it.** Questions asking for *synthesis* ("go through my entire story") have no
   single answer-bearing turn: **38 of 91 spanned ≥6 gold turns, up to 17**.
   That is a property of the TASK, not of the derivation (user, 2026-08-11).
   Scoring them with recall@k measures nothing — they need their own metric, or
   they must be excluded. This is the same root as [G48](ROADMAP.md#g48).

⇒ When building an evaluation key, ask which direction makes the answer true by
construction, and take it. Then ask which probes the metric can *physically*
score, and type the rest rather than averaging them in.

### 30. A test that seeds stub embeddings cannot test retrieval

Codex suites seeded entities with `[0.05]*1024` — a constant vector. Every
cosine comparison against it returns the same value, so vector ranking is
uniform and the retrieval path under test is not exercised at all: the suite
passes whether ranking works or not. Named as a blind spot in
[G30](ROADMAP.md#g30) and fixed in `tests/test_retrieval_failopen.py`, the first
suite to seed **real** embeddings.

⇒ A fixture value chosen for convenience becomes part of what the test measures.
If the thing under test consumes an embedding, the fixture must contain a real
one — otherwise the green is about the stub.

### 31. "Zero edges" is not "safe to delete" — the in-flight endpoint

G44's node promotion folds a zero-edge stub into a more specific name: `plan`
becomes `the big plan`. Its safety argument was written down and is sound as far
as it goes — *only stubs are eligible, and a stub has no facts to re-attribute,
so the merge cannot destroy anything.*

It is still wrong, because **the degree count sees committed edges and the
caller is holding an uncommitted one.** `handle_triplet` resolves the subject,
then the object, then writes the edge between them. Resolving the object can
promote the SUBJECT away — at that instant the edge that would have given it a
degree does not exist. The caller then writes `source_id` pointing at a deleted
row, Postgres rejects it, and `evaluate_turn` rolls back and **re-raises**, so
the turn loses its codex *and* procedural extraction. The seeder counts the
failure and continues, so a 293-turn run completes and looks finished.

Measured 2026-08-15: 1 turn in 6 on a smoke seed with `CODEX_NODE_PROMOTION=true`;
0 in 6 with it off, same turns, store cleaned between. Reproduced deterministically
at the function level — resolve `zzrepro`, then `the big zzrepro`, and the first
entity's row is gone. Fixed with a `protect_ids` exemption for the endpoint the
caller is still holding.

⚠ **Two things this cost, and both are general.** The first repro used a
two-word generic (`zzrepro plan`) and did not reproduce — promotion only matches
an entity whose WHOLE name is one token of the incoming name, so the fixture was
never eligible and the clean result was about the fixture (TRAPS #13b again).
The second is the handler underneath: promotion's `except` called a bare
`db.rollback()` inside the caller's open transaction, which would discard the
whole batch's uncommitted work — now a SAVEPOINT, so a failed promotion undoes
only itself.

⇒ Before deleting a row on the grounds that nothing references it, ask what the
**current call stack** is about to reference. A "read the database" check cannot
see the caller's intent, and an operation that repairs the store must not run in
the middle of a write to it.

### 31b. Unusual content is indistinguishable from hallucination — unless you look

**Three times in one session (2026-08-15) something judged real corpus content
to be a model error and suppressed it.** Each time the suppression was written
by someone reasoning about what a normal conversation looks like, and each time
one SQL query would have settled it.

* `november --eye_color--> golden black` was cited across three sessions as an
  invented value, and [G43](ROADMAP.md#g43) was partly justified by it. The
  corpus owner has synesthesia; months carry colours. **The value was present in
  the source turn.** The real defect was the closed property vocabulary
  underneath.
* [G44](ROADMAP.md#g44)'s name rule refused `8`, `19`, `2023`, `9.65`, `3/80` —
  **94 numeric refusals of 2,273** — on the reasoning that a node named `8`
  cannot be looked up. Versions, years, prices, sizes and scores are numeric for
  every user; it was deleting the subject of `3.80 --score--> maths`.
* A subagent sent to read output declared a set of triplets fabricated and made
  that the **"dealbreaker"** deciding its model ranking — a verdict one query
  reversed, taking the agent's overall model preference with it. The subject
  matter is sensitive and stays out of this file: **`[PRIVATE:traps31b-third-case]`**
  in `docs/PRIVATE_CONTEXT.md` (gitignored) has the case, and names a real
  design gap it exposed — ICE stores an assistant's *hypothesis about the user*
  as a *fact about the user*, at full confidence, with nothing marking which is
  which.

The failure is not carelessness — it is that the check *feels* unnecessary. An
implausible-looking fact reads as obviously wrong, and "obviously" is doing all
the work.

⇒ **Any claim of the form "the model made this up" gets a corpus query before it
is stated, and the query goes in the write-up.** This binds subagents too: brief
them to verify against the source, because a confident wrong verdict from a
reader is more expensive than a wrong metric — it arrives with reasoning
attached. Corollary for design: a suppression rule written from an outside view
of "normal" content will be wrong for the users who most need memory.

### 32. A metric can be structurally blind to most of the system

Recall@k credited a fragment only when its `source_batch_id` matched the gold
turn — and **only episodic fragments ever set that field**. Measured on one
harvest: 4,124 episodic fragments earned 249 gold credits while **474 codex, 400
procedural and 207 timeline fragments earned zero**. Not because they were
wrong. Because nothing could attribute them to a turn.

So every recall number this project has produced — including the 0.508 that a
whole cycle celebrated, and the 0.664 called "the only solid one" — is an
**episodic-leg score reported under the whole system's name**. Fixing the
attribution alone moved 25 probes from 60% to 84% with ICE untouched.

The tell was available and ignored for months: a per-leg breakdown showing
hundreds of fragments from a leg that never once appears in the numerator.

⇒ **For any metric, ask which components can physically appear in the
numerator.** If a subsystem contributes output that the metric has no way to
count, the metric is not measuring the system — it is measuring the one part
that happens to be countable, and it will read as evidence that the other parts
do not work.

### 33. Two id spaces with near-identical names produce a confident false finding

`ContextFragment.source_batch_id` holds the episodic **row id**.
`codex_edges.source_batch` and `procedural_memory.source_batch_ids` hold that
turn's **batch id**. Verified: 9,662 edges join on `batch_id`, **0** on row id.

Comparing them can never match. When the leg-attribution fix above was first
written it compared the two spaces directly, and the result — codex fragments
scoring zero gold credits — read exactly like a substantive finding: *"the graph
does not cover the gold turns."* That sentence was one step from being written
into the roadmap.

What separated bug from finding was asking the cheap structural question — *do
these two columns even live in the same space?* — rather than interpreting the
zero.

⇒ **Before believing a zero, check that the two things being compared could ever
have been equal.** A join count settles it in seconds, and near-identical field
names are the warning sign, not a reassurance.

### 34. Fixing one failure mode can cause a worse one in the same call

The cloud judge returned empty content on long inputs while a toy prompt worked
perfectly — a reasoning model spending its whole `max_tokens` inside the hidden
block (TRAPS #11). Setting `reasoning_effort="none"` fixed it completely.

It also made the judge stop deliberating, and it **over-called `both_failed` on
30 of 73 verdicts**, several at 100% content overlap between the answer and the
expected answer. The failure mode moved from *visibly broken* (empty strings) to
*confidently wrong* (a clean-looking verdict table), which is strictly worse:
the first stops you, the second gets published. Restoring reasoning and removing
the ceiling took `both_failed` from **49% to 31%**.

Cost of the detour: a full 150-probe run on a paid API, spent on verdicts that
had to be discarded.

⇒ **A fix that silences a symptom on a judgement task deserves the same
verification as the feature it fixed.** Reach for budget before capability: pay
for the thinking rather than switching it off. And when a fix works instantly on
a small case and the real case is expensive, test on the real case anyway.

### 35. A generator wrote an empty result over an irreplaceable input

`generate_typed_probes.py --types temporal --limit 6` built 0 prompts (the limit
excluded everything) and **wrote its output file anyway**, overwriting 420 typed
probes with an empty set. `experiments/curation_files/` is gitignored, so there
was no history to recover from. 372 were rebuilt by merging an older 328-probe
file with records recovered out of the answer-run JSONs; **48 codex probes were
lost permanently**, and those had cost a salvage fix to obtain (12 → 89).

Two mistakes stacked: a **writing** command used as a smoke test, and no copy
taken of the artifact every measurement depends on.

⚠ **THE SALVAGE LOST A FIELD, AND IT WENT UNNOTICED FOR A DAY (found 2026-08-17).**
The rebuild is worse than "89 → 41": records recovered from the answer-run JSONs
kept `question` and `gold_turns` but **not `anchor_entity`**, which only the
generator ever wrote. So **29 of the 41 surviving codex probes have no anchor**,
`score_typed.py` reads `(p.get("anchor_entity") or "")`, and those probes score
the anchor half False *by construction* — ceiling 0.5 instead of 1.0. The mean
then read as the graph regressing 0.538 → 0.317 when it had improved.
⇒ **A partial recovery is not a recovery until you diff the KEY SET, not the
record count.** `sorted({k for p in probes for k in p})` against the previous
generation would have caught it immediately. Restore `anchor_entity` when the
lost codex probes are regenerated.

⇒ **Treat a generated corpus as an ASSET, not an output.** Guards now in that
script and worth copying to any generator: refuse to write zero over a populated
file, merge partial runs instead of overwriting, and keep the previous
generation beside the new one. And run smoke tests with `--limit` against a
**temp output path**, never the live one.

### 36. Killing the client does not stop the GPU, and a long run that only writes at the end writes nothing

Two failures that compound, both hit on 2026-08-16.

**The work was lost.** `generate_typed_probes.py` collected every model reply in
memory and wrote the output file only after the last one returned. A ~2-hour
generation was killed near the end and produced **exactly what a run killed
after one second produces: nothing.** The judge had per-probe checkpointing by
then and survived its own kill with 65 usable verdicts; the generator did not,
and lost everything.

**The GPU did not stop.** The Python parent was gone from `htop`/`nvtop` and
**Ollama kept generating** — queued requests outlive the client that queued them,
and a `ThreadPoolExecutor` with N workers means N of them are already in flight.
The machine ran at **105 °C** until a full restart. Killing the process you can
see is not the same as stopping the work you started.

⇒ **Three habits for any run measured in hours:**

* **Checkpoint every unit, not every run.** Append raw results to a sidecar as
  they land. A killed run should be resumable or salvageable, never repeated.
  Cost is one `open(..., "a")` per call; the alternative is hours.
* **Stop the server, not just the script.** `pkill` on the Python parent leaves
  Ollama working. Unload the model explicitly:

      pgrep -f '[g]enerate_typed' | xargs -r kill
      ollama stop <model>            # the part people forget
      ollama ps                      # verify nothing is resident

* **Keep concurrency low on a local GPU.** `--workers 3` against a 27B model is
  three simultaneous generations on one card. Thermals, not throughput, are the
  binding constraint, and there is no backpressure telling you so.

⚠ **Related, and it bit twice the same day:** `/tmp` is cleared aggressively on
this machine. Three run logs vanished mid-session, including the one holding a
two-hour generation's progress. **Write run output under
`experiments/curation_files/`, never `/tmp`.**

---

### 37. A presence metric turns one row into 0.000 or 1.000, and reports both as a result

`score_typed.py` scores the `procedural` class as *"did any procedural fragment
come back at all"*. On the two seeded arms, that metric produced:

| arm | procedural rows | `is_active` | fragments returned | **reported score** |
|---|---|---|---|---|
| 2 — `fixed-gemma4-e4b` | 58 | **0** | 0 | **0.000** |
| 1 — `fixed-qwen3-4b-instruct` | 61 | **1** | 40 of 40 probes | **1.000** |

**Same defect, opposite extremes, one row apart.** Activation requires
`reinforcement_count >= 3`; on arm 1 exactly one pattern reached 4 and every
other sits at 1. That single row is the leg's whole candidate pool, so it is
returned to every query — verified against the leg's own SQL, the nonsense
string `zzzq wumpus glorbnax fleeb` retrieves it at cosine 0.4147, identical to
what a real question gets.

**The zero was predicted; the 1.000 was not, and it is the dangerous half.**
[PROVENANCE](PROVENANCE.md) 2026-08-16 recorded *"the procedural leg will score
zero and it is NOT retrieval"* — correct, on arm 2, and it reads as a flag to
investigate. The identical defect on arm 1 reads as a **finished subsystem**. A
zero gets chased. A 1.000 gets celebrated and then built on.

⇒ **Three habits:**

* **A presence test is not a quality test.** *"Is the expected kind of thing in
  there"* cannot answer *"is what is there any good"* — CLAUDE.md's second
  question, and this is the worked case. Score what the fragment says, not that
  it exists.
* **Before believing a perfect score, ask what the candidate pool is.** A metric
  over a pool of one is a constant. `select count(*) … where is_active` is the
  whole check and it takes seconds.
* **⚑ Feed the subsystem something it must fail.** Nonsense input is the cheapest
  instrument in this repo: one query against `zzzq wumpus glorbnax fleeb`
  distinguished "retrieval is working" from "the pool has one row" instantly,
  after a 372-probe run could not. **Do this before quoting any score at or near
  a ceiling.**

⚠ **Corollary — the same run's `codex_multihop` failed the mirror-image way.**
29 of 41 probes had lost the `anchor_entity` field the metric reads, so the
anchor half scored False *by construction* and capped those probes at 0.5. The
mean (0.317) read as the knowledge graph regressing from 0.538; on the 12 probes
that could actually be scored the graph had **improved**, 68.5% → 75%. **A field
missing from the probe file is indistinguishable from the system failing, unless
you split the population by whether the field is there.**

---

### 38. An undeclared `.env` key takes the whole application down

Adding `COE_API_BASE_URL` / `COE_API_KEY` / `COE_MODEL` to `.env` — and nothing
else — broke **the proxy, the workers and every test at once**:

```
pydantic_core.ValidationError: 3 validation errors for Settings
coe_api_base_url
  Extra inputs are not permitted [type=extra_forbidden]
```

`Settings` is a pydantic-settings model with extras **forbidden**, so it reads
every key in `.env` and refuses any it does not declare. There is no partial
failure: `settings = Settings()` runs at import, so the first thing that touches
config dies, and so does everything downstream of it.

⚠ **This is the second time.** `config.py:549` already carried the note — *"took
the application down — the proxy, the workers and every test at once. Found
2026-08-13, seconds after they were added."* It was found in seconds that time
because a test ran immediately; it was found the same way here, by a test that
had nothing to do with config.

⇒ **An env var is not optional configuration — declare it in `Settings` or do
not put it in `.env`.** The two steps are one change, never two. And the reason
it stays cheap is that *something ran straight afterwards*: the failure is
instant and total, so any test at all catches it. Add the var, then run
anything.

⚠ Note the shape it hides behind: the traceback names **pydantic**, not your
edit, and points at `config.py` rather than at `.env`. If config suddenly fails
to validate and you did not touch `config.py`, look at `.env` first.

---

### 39. A subsystem can produce output that never reaches the prompt

Two independent instances the same day, and they look identical from the score:

| subsystem | produces? | retrievable? | reaches the prompt? | reported as |
|---|---|---|---|---|
| procedural | 1 active pattern of 61 | yes, on every query | yes — but it is the *same* one always | **1.000**, a tautology |
| batch summaries | 3, covering 89 turns | **yes** — `_batch_summary_lookup` returns 2 fragments at cosine 0.52 / 0.48 | **no** | **0.322**, unchanged after fixing the producer |

The batch-summary case is the sharper one. `batch_summaries` sat at **0**, so the
obvious diagnosis was "the summariser is broken". It was not: `seed_store` calls
it and the call had failed **once, transiently**, inside a `try/except` that
prints a warning nobody read. Running it produced summaries immediately.

**And the score did not move.** The producer was fixed, the leg demonstrably
works when called directly, and `summary_synthesis` stayed at 0.322 — because
`leg_budget_share` gives **episodic 95–99%** of every request and the summary
fragments are trimmed before assembly.

⇒ **"The subsystem produces output" and "the output reaches the prompt" are
different claims.** Three questions, in order, before crediting or blaming any
leg:

1. **Does it produce?** `select count(*)` on its table.
2. **Does its leg return it?** Call the leg directly with a real embedding — not
   through `retrieve()`, which fuses and trims.
3. **Does it survive the budget?** Read `leg_budget_share` on a real request.

A failure at (3) looks exactly like a failure at (1) from the score, and fixing
(1) changes nothing. ⚠ The corollary bites the other way too: a subsystem can be
fully *broken* at (1) and still score **perfectly**, if the metric only asks
whether *something* came back — see [#37](#37-a-presence-metric-turns-one-row-into-0000-or-1000-and-reports-both-as-a-result).

⚠ **And a `try/except` around a periodic job converts a transient failure into a
permanent one.** The seeder logs and continues, which is right; what was missing
is that nothing ever re-ran it or checked the count afterwards. A worker whose
output is a *precondition for a measurement* needs its row count asserted, not
its exception caught.

---

### 40. A result written outside the tracked tree is a result you will lose

**What happened (2026-08-20).** A 293-turn seed's log was written to
`/tmp/ice_ner_arm_seed.log`. The machine rebooted between sessions and `/tmp`
was cleared, so that arm's per-chunk grounding lines — the only record of how
often extraction grounded nothing — are gone permanently. The comparison they
existed for cannot be run. Nothing errored; the file simply was not there the
next day.

**It was recoverable only by luck.** The same outcome happens to be persisted in
the database as `codex_edges.extraction_confidence`, so the question could be
re-asked a different way. That is not a plan.

**The trap is wider than `/tmp`, and this is the part that surprises people.**
In this repo the *durable* set is much smaller than it looks:

| location | status |
|---|---|
| `experiments/curation_files/**` — score runs, judgements, snapshots, probe sets | **gitignored** (it carries the corpus) |
| `logs/` | **gitignored** |
| `docs/SESSION.md` | **gitignored, and emptied at session end** |
| `/tmp` | cleared on reboot |
| `docs/PROVENANCE.md`, `ROADMAP.md`, `TRAPS.md`, `FEATURE_INVENTORY.md` | **tracked — the only durable home** |

So "I wrote it down" is ambiguous, and three of the five places a number
naturally lands are erased on a different schedule. A judged run whose verdicts
sit in `experiments/curation_files/judgements/` is one `git clean` from gone,
and the session file that summarised it is emptied on purpose.

**The rule.** *When a run produces a number anyone might cite later, copy the
number into a TRACKED doc in the same session — normally
[PROVENANCE.md](PROVENANCE.md), which already owns "what was done".* The JSON
artifact stays where it is and is still worth keeping; it is the **evidence**.
The tracked doc carries the **claim**, so the claim survives losing the evidence.
Write experiment logs under `logs/` or `experiments/` rather than `/tmp` — they
are still gitignored, but they survive a reboot, which `/tmp` does not.

**The tell.** You are about to say "it's in the session file" or "it's in the
judgements folder" about a number that will be quoted next week. Both are local
and both are swept.

---

### 41. Six instruments in one day ran clean and returned the wrong number

**2026-08-20, during the A9b two-arm comparison.** Every defect below was in the
MEASUREMENT, not the system. None raised an exception. Each returned a plausible
number that would have been reported as a finding.

| # | instrument | what it actually measured | how it was caught |
|---|---|---|---|
| 1 | arm-B config mirror | full TURN vs production's CHUNK — 24 spans against 13 | its own equality assert |
| 2 | stratified probe sampler | target recomputed inside the loop against DRAINING pools, so a 40-probe type stopped at 20 | output count looked short |
| 3 | codex quality sampler | `left join conversations c on true` — a CROSS JOIN for a column never printed; every triplet rendered 3× | sample looked like duplicate extraction |
| 4 | answer judge | source double-truncated (3 turns × 1,200 chars, then 4,000) — judge saw a **median 12.7%** of the gold | **the user asked whether the GT was too small** |
| 5 | gold consistency check | `evidence` numbering is type-dependent — absolute for `codex_multihop`, RELATIVE to a 14-turn window for `procedural`; read as absolute it declared **34 of 40 valid probes broken** | the failure pattern was too uniform to be real |
| 6 | ablation patch | filtered a `(fragments, classification)` TUPLE as if it were a list | read the patched line back |

**⚑ 2026-08-26 — DEFECT 4 RECURRED IN A NEW SCRIPT, AND THE SAME PERSON CAUGHT
IT THE SAME WAY.** `judge_summaries.py` was written with `source[:4000]` and
`summary[:2000]`. The sampled turns run to **22,694 chars** (median 3,572), and
**3 of the 6 turns in the first calibration were over the cut** — two at 9,292
and 10,655, so the judge read ~38-43% of the source while grading a summary of
all of it. It returned **13 of 18 clean summaries flagged as `fabricated`**, a
result that read as a dramatic finding about local models inventing, and was an
artifact of not showing the judge the evidence.

Worse: the fix already existed *in this repo*, three lines of comment in
`judge_answers.py:154` beginning **"⚑ NO CAP. This was `source[:4000]`"** — the
identical constant, in the identical role, with the reasoning written out. The
maintainer's words on catching it: *"are we giving the judge the truncated
source truth to compare as we have gone thru this bs before."*

⇒ **A cap on evidence is not a performance detail, it is a silent change to what
the judge is allowed to know.** Grep any new judge for `[:` before trusting one
number it produces, and treat a fix recorded in one file as a rule for all of
them — a lesson written in a comment does not travel to the next script by
itself.

**What they have in common.** Not one crashed. Each produced output of the right
SHAPE — a percentage, a count, a table — and the shape is what gets believed.
Defects 1, 3 and 5 were caught only because a human or an assert found the
*pattern* implausible, never because anything failed.

**The cost when it is not caught.** Defect 4 alone produced three successive
readings of the same run that each reversed: "arm A wins decisively" → "arm B
wins the graph type" → "dead heat". Two full interpretations were written and
withdrawn. Defect 5 would have deleted 34 valid probes and declared the ground
truth broken.

**The rules that actually work, in order of what they saved:**

1. **Write the assert that would FAIL.** Defect 1 was caught by a mirror-equality
   check written specifically because a mirror is an adjacent system until
   proven otherwise. That is the only defect caught by design rather than luck.
2. **Prove the instrument DISCRIMINATES before spending the run.** Before the
   3-hour ablation, all three conditions were run on 2 probes to confirm they
   produce different fragment counts. An ablation whose conditions are
   accidentally identical returns a clean, meaningless null.
3. **Do not interpret a comparison until the metric has been checked against its
   own ground truth.** Both withdrawn interpretations came from reading a
   comparison before checking what the judge could see.
4. **A number whose failure pattern is too uniform is an instrument bug.** 34 of
   40 probes broken *in exactly the same way* is not data about the world.

**The tell.** You are about to report a number from an instrument that has never
once returned an error. Ask what it would look like if the instrument were wrong,
and check that specific thing.

---

### 42. A new session proposes work the project already did, or work on things it no longer has

**Two instances in one afternoon, both caught by the user, neither by me.**

1. **Proposed re-running a settled experiment.** After measuring the codex graph
   at 20% triplet correctness, the obvious hypothesis was "the extractor is a 4B
   model — try a bigger one." That experiment is **A12**, run 2026-08-12: eight
   models, 4B through 26B, same 60 turns. **The 26B ranked THIRD**, and A12's
   conclusion is explicit — the defects were *"present in ALL SEVEN arms.
   Universal ⇒ prompt/design, not model capability."* The answer was on disk,
   in PROVENANCE, under a heading naming the item.
2. **Nearly ablated a deleted subsystem.** Queued `mera` as an ablation leg.
   MERA was deleted by A4; the flag survives only because it was **re-homed**
   onto the graph-enumeration path. Right answer by luck — the leg does measure
   something — but the reasoning was "the flag exists so the thing exists."

**Why this shape recurs.** A session starts with no memory of what was tried.
Everything looks unexplored. The unexplored thing is *interesting*, so it gets
proposed with confidence — and a confident proposal to re-run settled work is
expensive twice over: it burns the hours, and it implies to the user that the
earlier result did not exist or is not trusted.

**The rule: CHECK THE BLAST RADIUS BEFORE PROPOSING OR CHANGING ANYTHING.**
Both directions, and both are cheap:

- **Backwards — has this been done?** Before proposing an experiment, grep
  PROVENANCE and ROADMAP_DONE for the *subject*, not the item id you have in
  mind. `grep -n "model comparison\|eight-arm" docs/PROVENANCE.md` would have
  cost seconds and saved an hour.
- **Forwards — what does this touch?** Before changing or removing anything,
  find every consumer. A flag existing does not mean its subsystem exists
  (`mera`), and a subsystem existing does not mean anything reads it — the
  codex leg feeds BM25 query expansion and the timeline leg, neither of which
  is obvious from `_codex_graph`'s signature. Disabling codex silently disabled
  timeline; nothing recorded that until an ablation exposed it.

**The tell.** You are about to say "we should test whether X" or "we can just
remove Y" in a session that started less than an hour ago, about a subsystem you
have only read today. Both sentences are a cue to grep first.

---
### 43. Two harnesses disagreed with production about one argument — and the diagnosis of WHY was wrong for a day
<!-- Rewritten 2026-08-22. The original entry's rule was right and its mechanism
     was wrong; both halves are kept, because the correction is the lesson. -->

**2026-08-21, corrected 2026-08-22.** A full day of retrieval measurement — a
two-arm comparison, a three-condition codex ablation, a five-leg ablation, ~2,000
scored requests — was declared invalid because the harnesses call
`orchestrator.retrieve()` with `scope=None` while *"production passes a populated
scope"*.

`retrieve()` does take both a `conversation_id` parameter and a `scope` dict, and
it does derive its conversation filter from **the scope only**
(`orchestrator.py:515`):

```python
conv_id = None
if scope and "conversation_id" in scope:
    conv_id = scope["conversation_id"]
```

**⚑ But the "production" column of the table that proved it was built BY HAND.**
It compared `scope=None` against a hand-written `scope={"conversation_id": …}`,
and cited `services/retrieval_svc.py:152` as the authority — which is the **MCP
`ice_context` pull**, where the scope is *supplied by the caller*. The evaluation
harnesses model the **chat proxy**, whose scope comes from
`resolve_retrieval_scope(db, conv_row)` at `main.py:421`.

Run against the three conversations actually under measurement:

```
f8192c3f  scope_type='auto'  project=None  ->  scope = {}   conv_id = None
33681aa4  scope_type='auto'  project=None  ->  scope = {}   conv_id = None
def5743a  scope_type='auto'  project=None  ->  scope = {}   conv_id = None
```

`services/scoping.py:114+` sets `conversation_id` for `none` and `project` modes
and `conversation_ids` for `manual`/project-attached — and for **`auto`, the
default mode, it sets neither.** The only scope mutation between `main.py:421`
and `:606` is `scope["timescope"]`. So `conv_id` is `None` in chat production
too, and the claims the retraction withdrew were **true of ICE as it runs**:
`_batch_summary_lookup`'s own-conversation half really never fires, and
`batch_summary` appears in **zero** of the 15 recorded `legs_seen` dicts.

**What the defect actually is.** `scope=None` and `scope={}` differ at exactly
**one** reachable site — every other scope read is `if scope`-guarded and treats
them identically:

```python
# orchestrator.py:534-536
cluster_ids = self._relevant_cluster_ids(...)
if cluster_ids and scope is not None:      # <- None skips it; {} applies it
    scope["cluster_ids"] = cluster_ids
```

Measured on 30 stratified probes, both ways: production sets cluster ids on
**90%** of probes and the returned fragment set differs on **17%**. Real, and
two orders of magnitude smaller than the retraction claimed — bounded here only
because `_cluster_filter`'s `OR NOT EXISTS` arm keeps unclustered turns and 268
of 293 turns are unclustered.

**⇒ Two rules, and the second is the one that was missed twice.**

1. **Find the production call site and diff its arguments** — the original rule,
   and it stands.
2. **⚑ Diffing a call site's SIGNATURE is not diffing its ARGUMENT'S VALUE.**
   `retrieval_svc.py` passes `scope=scope`, which reads as "production passes a
   scope" and is useless; the question is what `scope` *contains* on the path you
   are modelling. Evaluate it — `resolve_retrieval_scope(db, conv_row)` is one
   line in a REPL — and do it for the conversations actually under test, because
   the answer depends on their `memory_scope_type`.
3. **And when a system has more than one production path, name which one the
   experiment models.** ICE has two callers of `retrieve()` and they disagree
   with each other. "Production" was ambiguous, and the ambiguity is what let a
   hand-built column pass for a measurement.

**The tell, restated.** The original entry ended: *"a zero that clean is nearly
always a switch, not a design."* Half right. The zero **was** a switch — the
switch is just also thrown in production, which makes it a design defect in ICE
rather than a defect in the instrument. Before attributing a perfect zero to the
harness, check whether production produces the same zero.

---

### 44. The harness classified without the conversation, and it moved 65% of the fusion weights

**2026-08-22, and it is the defect #43 was standing in front of.** Every Z1
harness calls:

```python
c = clf.classify(question[:2000])            # score_typed.py:202, and three siblings
```

Production calls (`main.py:448-451`):

```python
result = classifier.classify(user_message, conversation_id=str(conversation_id))
```

That argument is not decoration. It makes the classifier build a **CL7 context
prefix** from the conversation's recent turns and embed *that*, so the head sees
a different vector. Measured on 100 stratified probes against the live store:

| | differs |
|---|---|
| `topic_tags` | 69 / 100 |
| `intent_tags` | 66 / 100 |
| **RRF leg-blend weights** | **65 / 100** |
| routed model (⇒ context window ⇒ token budget) | 42 / 100 |
| B2 retrieve/don't-retrieve decision | 0 / 100 |

The blend weights **are** the fusion ranking (`orchestrator.py:588-596`). So 65%
of every scored probe fused with weights production would never use, and 42%
derived their memory budget from the wrong model's context window.

**Three things make this worse than #43, which got all the attention:**

* It is **~4× the perturbation** of the scope defect (65% vs 17%), and it was
  invisible while #43 was being written and retracted.
* The **older paper-era runners get it right** —
  `experiments/mature/run_mature_experiment.py:547` and both flaw-ablation
  runners pass `conversation_id=cid`. This is a **regression introduced with the
  Z1 harnesses**, not an inherited habit.
* B2 flipped on **0 of 100**, so the one signal anybody would have spot-checked
  looked perfectly stable while everything downstream of it moved.

⇒ **When a production call passes an argument the harness omits, ask what that
argument CHANGES, not whether the call succeeds without it.** An optional
parameter that alters an embedding is not optional. And check the *older* code
before assuming a divergence is inherent: a harness that used to be right and
is now wrong is a regression with a commit behind it.

---

### 45. A metric measured whether the model obeyed a formatting instruction

**2026-08-22, on the write path — the first time it had ever been examined.**

`summary_coverage` (`turn_density.py:172-180`) is the **trust gate**: its own
code comment calls it *"precisely the trust gate that decides whether a summary
may be injected at all"*, and `post_flight.py:228` uses it to decide whether the
stored summary **replaces the raw turn** in the prompt:

```python
inject_raw = not (summary and coverage >= settings.turn_summary_coverage_threshold)
```

It scores the fraction of must-preserve terms — named entities, figures,
identifiers — appearing anywhere in the summary.

**And the summariser's own prompt orders the model to append them**
(`post_flight.py:75-78`):

> *"End with two final lines formatted exactly as: `Key terms: <comma-separated
> list of the named entities, figures, and identifiers that appear in the …>`"*

Same three categories. So the metric largely measures compliance with a
formatting instruction, and a model that writes a poor summary followed by an
obedient term list scores the same as one that writes a good summary.

Measured on the live store: **193 of 212 summaries (91%) carry the block**,
coverage averages **0.971 with 79% at exactly 1.000**, and every stored summary
ends in a comma-separated list rather than a sentence. At that distribution the
0.7 threshold effectively never binds — **99 turns have their raw text replaced
in the prompt** by a summary nothing verified, and 2 of those are bare
comma-separated lists with no prose at all.

**This is the MECHANISM behind [#21](#21-a-metric-can-be-green-on-output-that-does-not-exist),
which blamed a weak model.** #21 measured `granite4:tiny-h` emitting
`Key terms: a, b, c` plus one generic sentence and scoring 0.9375–1.0, and read
it as that model failing. Every model does it, because the prompt asks for it.
The scoring bug and the prompt were written to serve each other and nobody read
them side by side.

⇒ **Read the metric and the prompt that produces its input in the same sitting.**
A presence metric over terms the generator was told to list is not a measurement,
it is a receipt. And when a metric saturates — 79% at the ceiling — that is the
symptom, not a sign of health: ask what it would take to score *low*, and if the
answer is "disobey the format", the metric is measuring the format.

---

### 46. A floor measured for one source, applied to a comparison it does not cover

**2026-08-22/23, four times in one session, by the session fixing measurement.**

Each time the shape was identical: measure a variance, establish a floor, then
apply that floor to a comparison the measurement never touched.

| the floor measured | what it actually covered | what it was applied to |
|---|---|---|
| judge self-consistency **±2.5 pts** — same store, same seed, same 200 triplets | the judge alone | a comparison of two DIFFERENT stores, where sampling and extraction also vary |
| **1.1%** volume spread between two arms | one pair, one configuration | called a "noise floor" for all runs |
| model deterministic — **12 warm calls identical** | within ONE process | concluded the model was deterministic, full stop. Across processes it is not |
| **grounded 2.5% correct** | a **40-triplet** subgroup | quoted as a settled finding in three tracked docs |

The last one is the sharpest. At n=40 the 95% interval on a rate near 15% is
about **±11 points**, which spans every number in that table — and
`extraction_confidence is INVERTED` was written into PROVENANCE, HANDOFF and
the Z1 index on that basis. Re-measured over 124 turns on two seeds, the runs
**disagree on the direction**. Withdrawn.

**Why it keeps happening.** A floor feels like a property of the system once
you have one number for it. It is not — it is a property of *one comparison
under one set of held-constant things*. Change what varies and the floor no
longer applies.

⇒ **Before using an interval, say out loud what varied when you measured it and
what varies in the comparison you are about to make.** If the second list is
longer, the interval is too small. And **never quote a subgroup rate without
its n** — a percentage over 40 rows is not a finding, it is a hint.

**⚑ FIFTH INSTANCE, 2026-08-23 — and this one had already been written down as
a finding.** "The direction rule is worth **+9.4 points**" compared a control
judged with the **flat sampler** (11.0%, 191 triplets from ~76 turns) against
treatments judged **`--per-turn 3`** (17.1% / 14.4%, 368 from 124). Two
samplers, one subtraction. Re-judging the *same control store* — same judge,
same seed, nothing changed but which turns were sampled — gives **16.7%**. The
entire claimed effect was the 5.7-point gap between the two samplers, and the
real effect is **+0.28 pts, z=0.09**.

What makes this instance worth its own paragraph: the previous four were caught
by the session that made them. This one **survived a session, got written into
FEATURE_INVENTORY, ROADMAP and the Z1 index as ⚠ UNRESOLVED, and shaped the
queue** — the small-model sweep was ordered behind it on the theory that fixing
the prompt was the live lever. It was flagged as unresolved, which is what saved
it; had it been flagged as *settled* it would have been load-bearing.

⇒ **A number produced by two different instruments is not a difference, it is a
difference between instruments.** Before subtracting two rates, check they were
sampled the same way — not just judged by the same model with the same seed.
Rule 4 in the Z1 README said *"same judge, same seed, or the comparison is not a
comparison."* That was necessary and **not sufficient**; it now says sampler too.

---

### 47. Temperature 0 is deterministic WITHIN a process, not across them

**2026-08-23.** Twelve alternating warm calls in one process returned
byte-identical text, so extraction was declared deterministic. Two seeds in
separate processes then shared only **6 of 9** raw responses, diverging on
call 1 with the model already resident:

```
X: [{"object":"detroit","relation":"manufactured_by","subject":"ford"}]
Y: [{"object":"detroit","relation":"lives_in","subject":"ford"}, …]
```

Temperature 0 makes *sampling* deterministic — it does not make the *logits*
bit-identical. Batch composition, kernel selection and memory layout differ
between processes, and two near-tied tokens then resolve differently.
`manufactured_by` against `lives_in` is exactly that kind of tie.

⚠ **It matters more here than the token difference suggests**, because
canonicalisation feeds accepted relations back into the vocabulary and every
minted entity changes what later turns resolve against. One flipped token on
turn 1 changed **31 of 48 turns**.

⇒ **Verify determinism ACROSS PROCESSES, never within one.** And when a system
turns out to be irreducibly variable, stop trying to remove the variance and
start reporting it: repeated runs, stated intervals, and no claim that a
difference is real unless it exceeds the observed spread. ⚠ Check *which*
quantity is unstable before despairing — here **which triplets exist** varies
substantially while the **aggregate correctness rate** moves only 2.7 points
across two seeds, and only the second one most claims depend on.

---

### 48. An unordered SQL read is a coin flip that decides your data

**2026-08-23, three instances in one file.**

`SELECT DISTINCT relation FROM codex_edges` had no `ORDER BY`. Postgres returns
rows in whatever order the plan produces, and that shifts as the table grows
and is vacuumed — so the relation vocabulary arrived **in a different order on
every run**. `canonical_relation` takes an argmax over that list, and real
candidates sit thousandths apart (`has`/`have` **0.9469**, `fails`/`failed`
**0.9431**), so list order decides near-ties. Fixing it made turns 1–19
reproduce exactly where previously turn 1 diverged.

Three more in `get_or_create_entity`, all `.first()` over sets that can hold
several rows: the alias lookup (`aliases` is an array and nothing makes it
unique across entities), the merge_key tier (**many-to-one by design** — that
is its whole purpose), and the promotion scan (which iterates a `set()` *and
mutates the store as it goes*).

⚠ **Picking the ordering key is its own trap.** The first attempt used
`created_at`, which does not exist on `CodexEntity` — caught only by a test.
The obvious substitute, `last_updated`, is **worse than useless**: it MUTATES
on every touch, so it cannot express "which came first" and would have looked
stable while silently reordering. `id` is a random uuid that differs per run.
The right key was `canonical_name` — UNIQUE, so a total order, and derived from
content rather than from history.

⇒ **`.first()` or an iterated `.all()` without `ORDER BY` is a defect wherever
more than one row can match**, and doubly so when the loop writes. Order by
something CONTENT-DERIVED and immutable; a timestamp that mutates and a random
id are both traps that look like fixes.

---

### 49. The metric a change was aimed at moved, and the change made things worse

**2026-08-24/25, twice in one day, on the same investigation.**

Two guards were added to the extractor and each **hit its own target metric**:

| guard | its target | result |
|---|---|---|
| relation vocabulary in the prompt | in-vocabulary relations | **50.9% → 75.2%** ✅ |
| canonicalisation rule | capitalised subjects | **54% → 30%** ✅ |

Both were called wins. Then a blind round measured **truth**: the vocabulary
scored **22% correct against the bare prompt's 60%** — a 38-point drop that
clears zero. It had hit its target *by forcing facts onto dictionary words that
did not fit*: `MIT/Stanford/CMU --works_at--> Google India`,
`authors --cites--> authors`, `AI/ML paper --writes--> AI/ML paper`.

⇒ **Hitting the metric you aimed at is not the same as improving the thing.**
A guard that optimises a proxy will optimise the proxy.

**And the same day, the mirror of it.** The shape table said
`ice_baseline` was best on nearly every column — 1.0% capitalised subjects,
74% in-vocabulary, zero parse failures, most facts per turn. It measured **15%
correct with 30% reversed**. **Structure is not truth, and a system can top
every structural column while being mostly wrong.**

⇒ Before believing a table, ask which column would still look good **if the
thing under test were broken**. Every column in that one would.

**The bad diagnosis this also produced, worth keeping:** the jamming hypothesis
was checked with a *relation-concentration* test — do a few words dominate? They
did not (19.5% vs 18.2%), and that was read as exonerating. It was not: the
jamming spread across **many** dictionary words, which concentration cannot
detect. **A null from a test that could not have detected the effect is not
evidence of absence.**

---

### 50. A guard added for one model silently broke the next one

**2026-08-25, wiring G63/P1.**

Three defences, each added for a real measured failure, each firing correctly —
and together they made a better model look broken:

* **`reasoning_effort: "none"`**, added 2026-08-03 because a reasoning model
  spent its whole budget thinking and returned empty content. NuExtract3 *is* a
  reasoning model and needs that budget.
* **`max_tokens = 1200`**, itself a raise from 500 for exactly this defect —
  still one size too small. Template mode lost **30 of 60 turns** at 1200.
* **`chunk_tokens = 550`**, from an era of small context windows, splitting a
  1,178-token turn into three so entities introduced in one chunk were described
  in another.

⇒ **A guard encodes an assumption about the thing it guards.** Swap that thing
and the guard is a bet on a model that is no longer there. **When replacing a
component, audit what was added to compensate for the old one** — those are the
first things to break, and they break quietly, looking like the new component's
fault.

**The corollary that cost the most time here:** the first three hours of
debugging blamed NuExtract3. The bug was ours — a parse filter testing
`all(k in item ...)`, key PRESENT rather than value a STRING, so a `null`
admitted a triplet that killed a `.strip()` two hundred lines later and lost the
**whole turn's** extraction. Invisible for months because the JSON schema
guaranteed strings on the only path anyone used.

### 51. A threshold the tester invented turned a working instrument into a false negative

**2026-08-26, choosing whether 174 probes could be salvaged.**

`gt_feasibility.py` asked whether the anchorless probes have enough lexical
signal to shortlist a candidate turn. It cut at a top-candidate score of **1.0**
— a number chosen by the person writing the script, checked against nothing —
and reported:

> **171 of 174 probes have no lexical signal.**

That would have condemned 171 recoverable probes as needing regeneration. The
calibration takes one extra query: run the same cut against the probes whose
gold turns are **already known to be correct**.

| population | median top-candidate score |
|---|---|
| anchored (gold known good) | **0.78** |
| anchorless (under test) | 0.64 |

The threshold rejected most of the *known-good* probes too. And the stage it was
judging already **recovers the true gold turn in the top 6 for 343 of 368 =
93%** of the cases where the answer can be checked.

⇒ **A cut-off invented by whoever is running the test is not a measurement.**
Every threshold needs a population where the right answer is known, run through
the same cut, before it is allowed to classify anything. Ask *"what would a
KNOWN-GOOD case score here?"* — and if that is unknown, the threshold is not
ready to be used.

**Why this shape is expensive:** it fails toward "the data is bad", which feels
like diligence and quietly discards work. The opposite error announces itself;
this one is congratulated.

### 52. `summary_coverage` rewards a model for NOT summarising

**2026-08-26, the background-model bake-off. A SECOND hole, after
[#45](#45-a-metric-measured-whether-the-model-obeyed-a-formatting-instruction).**

#45 caught the metric scoring compliance with a formatting instruction; that was
fixed by `strip_generated_index`, which scores the prose and ignores the model's
own `Key terms:` index. The metric is no longer circular. It is still a
**presence** metric, and presence scales with length:

| model | coverage | prose chars | % of source turn |
|---|---|---|---|
| `lfm2.5:8b` | **0.973 — best of 11** | 3,372 | **95%** |
| `mistral-nemo` | 0.896 | 425 | 12% |
| `ministral-3:8b` | 0.859 | 866 | 24% |
| `qwen3:4b-instruct` | 0.814 | 917 | 26% |

Median source turn: 3,553 chars. **The model that summarised least scored
best**, because a near-verbatim copy contains every must-term by construction.

⚠ **This is the trust gate.** `post_flight` uses coverage to decide whether the
summary **replaces the raw turn** in the assembled prompt. So the gate cannot
distinguish a good summary from the absence of summarisation — and the same
model was independently the worst in the field on the reconciler (0/9), which
is what prompted the check.

⇒ **A presence metric needs a compression term, or a length bound, or it is
partly measuring output length.** More generally: when a metric's best scorer is
also the worst performer elsewhere, look at the metric before crowning it.

**⚑ AND IT IS WORSE THAN BLIND — MEASURED THE SAME DAY.** Once the faithfulness
judge passed its own controls (planted fabrications caught 15/16, verbatim
copies called faithful 16/16), 42 real summaries from three models were graded:

| verdict | n | mean coverage | clears the 0.7 trust gate |
|---|---|---|---|
| faithful | 29 | 0.803 | 23/29 = 79% |
| **fabricated** | 13 | **0.914** | **13/13 = 100%** |

Fabricated summaries score **higher** coverage than faithful ones inside every
model tested, and the fabrication rate tracks coverage across models
(0.859 → 43%, 0.814 → 29%, 0.768 → 21%). **Every invented summary cleared the
gate; a fifth of the honest ones did not.**

**The mechanism is the prompt feeding the metric, again.** `_summary_llm_call`
orders *"every one of these MUST appear verbatim"*. A model that cannot ground a
must-term invents context to carry it, and coverage then rewards exactly that.
So the instruction manufactures the defect the metric scores well.

⇒ **A gate can be worse than absent.** An absent gate admits everything; this one
admits invention preferentially, which is a filter running backwards. Whenever a
generator is instructed to include terms that a metric then counts, assume the
pair is coupled and measure the coupling before trusting either.

### 53. A resident local model can be reachable, deterministic, and still be in a bad generation state

**2026-08-30, resuming the v2 LongMemEval-S run.** The same
`qwen3:4b-instruct-bg` blob and template that had produced valid triplet objects
ten minutes earlier began returning this on the extractor's fixed capability
probe:

`["user","lives_in","berlin","user","works_at","vertex labs"]`

The response was non-empty, valid JSON, semantically recognisable, and identical
across five `temperature=0.0` calls — but it was a flat list, so the v2 parser
correctly produced zero triplets. Reachability, a one-word `ok` probe, full GPU
residency, and deterministic output all looked healthy. A clean
`ollama stop qwen3:4b-instruct-bg` plus reload restored the expected object array
immediately, without changing the model definition or code. The root cause of
the resident bad state remains unknown.

⇒ **Preflight the output contract, not merely model reachability or non-empty
text, on every resumed run.** When a previously passing local model starts
failing that fixed contract repeatedly, preserve the raw response, cleanly
reload that one model, and re-run the same probe before changing prompts,
parsers, or evaluation code. A coherent response is not proof of a healthy
structured-output path.

### 54. `nohup` inside an agent shell is not proof that a process survived the shell

**2026-08-31, restarting the v2 LongMemEval-S run under Codex.** A launch using
`nohup run_lme.sh ... &` returned success, but the process was gone immediately
and its redirected file was empty. The execution sandbox reaped the background
process when its shell ended. The previous attached PTY launch had likewise
disappeared when the assistant turn ended, halfway through an instance.

No completed answers were lost because the runner writes answers atomically and
rejects partial stores, but hundreds of pair-level model calls were repeated.
A transient user systemd service survived the launching tool call and exposed a
stable unit state, PID, journal, and stop operation.

⇒ **Verify liveness after the launching shell has exited.** For a long local
experiment started by an agent, a returned PID or zero exit status is not the
test. Query the process from a fresh command and require new durable progress.
Use an external supervisor (here, `systemd-run --user`) when the run must outlive
the agent turn; keep the experiment's own resumability as the second line of
defence.

### 55. Equal identifiers with different Python types turned the current conversation into an external one

**2026-08-31, v2 LongMemEval adapter.** The adapter created its conversation id
with `uuid.uuid5()` and passed the `UUID` object into `retrieve()`. Retrieved
fragments carry `conversation_id=str(row.conversation_id)`. The values print
identically, database filters work, and every scoping check looked healthy — but
`_session_diversify` used a direct Python comparison.

It therefore saw every current-conversation fragment as external and applied
the three-per-conversation cap. The live trace was unambiguous: BM25 45 + vector
100 → RRF 111 unique → diversify **3** → budget 3. Changing only the current id
to `str` gave diversify 111 → budget 31. This invalidated 500 oracle ICE answers
and 20 stratified ICE answers while leaving the direct-SQL vector arm untouched.

⇒ **Identifier equality needs a type-invariance control at every in-memory
boundary.** A UUID that works in SQL is not proof it works in Python. For any
scope, owner, tenant, conversation, or project id, run the same fixture once as
the ORM/native type and once as the API/string type and require identical
decisions — especially before interpreting a suspiciously exact cap as a system
property.

### 56. Stopping a supervised shell pipeline can kill the logger before the worker handles its signal

**2026-08-31, thermal safety stop.** The systemd unit ran
`uv run python ... | tee run.log`. `systemctl --user stop` signalled the entire
control group. `tee` exited while Python's SIGTERM handler was finishing the
current pair; the next structlog write hit `BrokenPipeError`, and even the
runner's failure message hit the same closed pipe.

Atomic answer files still protected completed work, but the operation described
as "graceful stop" was not graceful and its own diagnostics became unavailable.

⇒ **A graceful worker signal and a control-group stop are not equivalent when a
pipeline is involved.** Either supervise the worker directly and let the journal
own logging, or signal the worker first, wait for its clean exit, then stop the
wrapper. Verify the runner's normal stop footer; process absence alone proves
only that it stopped.
