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
Promoted to a CLAUDE.md standing rule on 2026-08-03 after every background LLM
call in ICE was found returning nothing while the system looked healthy. Kept
here too because it is the failure *shape*, not just a policy: when a component
substitutes a default for a real answer, that substitution must be observable.
⇒ **When a subsystem produces plausible-but-thin output, verify the model was
actually called before tuning anything about it.**

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
