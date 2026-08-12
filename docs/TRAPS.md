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
