# ICE v3 campaign fault handling and speed audit

Updated 2026-10-06. This audit concerns current v3; paper v2 remains frozen.
No complete development campaign, live cloud call or new answer-quality score
was produced by these repairs.

## R4 temporal-control pause

R4 reached 1434 durable originals (1119+251+64) and 213 checkpointed probes,
then its third-history turn65 probe batch hit the recent-only integrity guard.
The temporal empty-window helper queries nearest source eras directly and
adds a fragment after fusion. Disabling seven retrieval legs leaves that helper
and the separate wide-net SQL reachable. This was comparison contamination,
not an unavailable model; removing the assertion would hide the defect.

Recent-only now keeps shared recent budgeting and assembly while skipping the
entire search. Vector-only suppresses the temporal note and follows the normal
warm-vector leg under low confidence. Full/no-Codex retain their production
helper behavior. All 213 committed controls were inspected: no recent-only
fragments/ranks/notes, no off-vector selected legs and no Codex/timeline evidence
in no-Codex. The screenshot's 228 includes 15 unfinished provisional probes,
which normal recovery archives/rebuilds; it does not discard 213 saved contexts.

The second exact registered instrument transition changes only the seed control
wrapper and checkpoint receipt handling. Snapshot Git provenance reproduces the
previous code hash; each transition has a distinct receipt/archive, and completed
trace validation requires an ordered registered chain with unchanged non-code
identity. The original 1120-turn receipt remains intact. Normal resume retains 1434
turns and starts third-history turn65. No memory writer, model, configuration,
label, DB schema, inference parallelism or source history changes.

## R4 summary pause and progress-preserving correction

R4 reached 1120 durable originals (1119 in its first history, one in its second)
and 82 frozen prompts. Five summary jobs hit completion `length`; the last three
were at first-history turns 1070/1095 and second-history turn 1. The outage guard
incorrectly counted responsive truncation as provider unavailability. Correct
only that instrument classification: retain all degraded receipts and never
accept truncated summaries. All memory writers and output budgets stay frozen.
The 29 other degraded post-flight calls (mostly extraction length, one missing
quote) remain quality evidence, not silently repaired or discarded.

A closed exact-code registry authorizes only the two recovery-tool changes.
Preparation archives the old checkpoint files and independent committed trace
prefix without modifying the store, original trace, pointer or manifest. Normal
resume still binds identical non-code identity, checks every SQL row and records
the instrument transition. Unreviewed code changes, settings/models/labels or
writer changes still refuse. The current r4 archive/receipt is prepared; resume
the existing r4 command below rather than starting again. The next source is
second-history turn 2. Final controls: 578 disposable smoke checks in 37.66 s,
36 focused controls and actual four-turn local transport-outage/resume passed.
No full campaign or cloud call was launched by the repair.

Concurrency/residency is unchanged. Models can share residency if memory fits;
parallel contexts increase memory and a busy single GPU can merely contend.
[Ollama's concurrency contract](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests)
is consistent with bounded within-turn overlap as a future measured option, not
parallel source histories or a reason to change this frozen campaign.

## Actual failure and shared repair

The maintainer's r2 seed failed on turn 54 after 53 processed/51 durable turns:
a copied source quote collapsed whitespace. R3 later failed twice on turn 80,
restoring/replaying from its 70-turn checkpoint each time. The third automatic
attempt was stopped gracefully after its campaign-only process group was
verified; its interruption receipt and all old checkpoints remain preserved.

Actual local background-NER/NuExtract role-unit capture reproduced the turn 80
case in 29.467 s: one user row and 32 assistant rows. Of the assistant quotes, 31
were literal; the final 212-character quote capitalized its first letter while
the remaining 211 characters matched the original clause. This was a copying
error, not hallucinated content or a model outage.

The shared production source selector now compares unique case/whitespace copies
and returns original bytes, including capitalization and spacing. Ambiguous
copies and changed words/numbers/punctuation/polarity are never accepted quotes.
Complete typed unsupported proposals are withheld from graph writes; their
original chunk remains searchable evidence alongside valid rows. An ambiguous
chunk mapping preserves the complete original role unit. Missing/malformed or
truncated execution output still fails atomically. No partial triple creates an
edge; source matching is not semantic entailment or corroboration.

An actual saved-output writer/reader control stored 33 exact original claims,
wrote 20 source-supported relations and withheld two unknown relations in 20.458 s.
The repaired sentence retained original case and was retrieved. These counts
are one regression control, not independently graded graph truth or answer gain.

## Current recovery boundaries

| Boundary | Production v3 / new manual bundle | What remains visible |
|---|---|---|
| Copied/unsupported Codex quote | Shared original-byte alignment and proposal isolation; valid rows continue, original evidence retained. | WARNING counts/reasons; no unsupported graph assertion. |
| Failed post-flight model call | Production runtime retries background jobs without undoing an already sent answer. Manual `continue` mode has two idempotent local attempts, then processes the next original. | Degraded source/worker/attempt receipt, truthful source counts; no invented completion key. |
| Failed periodic model call | Live scheduler survives job errors. Manual job gets one attempt per due cadence; committed changes are inspected, failed-attempt time is separate from last success. | Before/after semantic rows and degraded status; no immediate replay of partial commits. |
| Repeated local outage | Three successive transport failures of one worker checkpoint the current turn, then pause; responsive output errors remain degraded and reset this streak. | Fault streaks and cadence survive resume; original turns/probes remain durable. |
| Cloud answer failure | Two attempts per question; isolated exhausted transport/server/empty/incomplete response is an ungraded terminal error and later questions continue. | Planned/success/error counts; exact inputs; no prefix becomes a successful answer. |
| Cloud judge failure | Two attempts per display order; preserve successful first orders. Exhausted format/completion/temporary transport errors remain ERROR, never ties or incorrect grades. | Both-order receipts, unresolved pairs retained in denominators. Failed-answer pairs consume no judge call. |
| Interrupted cloud request | Reserve the attempt before sending; an unknown response outcome remains unconfirmed and consumes its attempt allowance. | Durable call ledger; resume cannot silently reset the two-call cap. |
| Cloud outage/access failure | Three successive exhausted questions pause; quota/authentication/unknown errors pause immediately. | Persisted operator-required reason, earlier answers/orders reusable. |
| Integrity, preflight/probe, schema, disk or unknown failure | Controlled pause; no continuing with untrustworthy state. Parent uses at most two total eligible child attempts, with one 2-second backoff. | Snapshot restore before seed process retry; fresh structured failure required; fsynced attempt/pause receipts when storage is writable. |

New manual configs select worker/cloud `continue`; standalone commands default
to `strict`. Resume binds policy/code/settings/models/corpus/labels. Large final
LME-oracle/semi-LSREP runners still need their own frozen contracts; this repair
supplies tested reusable primitives, not an already-built final campaign.

Original-turn coverage and memory-processing health are separate. Trace
validation binds degraded receipts to historical writes and exact fault counts.
Probes/answers record prior failures and affected gold turns, without putting
fault metadata into answering/judging prompts. Reports keep clean,
degraded-gold and other degraded groups. `processing_complete` means every
planned cloud question was attempted; `complete` requires all successful.
A degraded replay cannot be reported as clean memory processing.

## Speed and checkpoint cost

Earlier actual native frame calls with unloading versus warm residency took
3.54/6.77/6.41/3.21 s; the second unloading/resident calls spent 3.52/0.31 s loading.
Native proof/need now retain owned models between calls; existing idle release
and serial-seed exit cleanup release only ICE-owned models. Reranker/NLI weights
return to CPU between calls. No all-model GPU residency guarantee or additional
model was introduced. Cloud answering/judging adds no local answer model.
Provider load/prompt/evaluation/total durations and wall times remain recorded.

The screenshot's 28:03/51 represented about 33 s per processed pair, including
local preflight, extraction/summaries/NLI, maintenance and checkpoint probes,
not merely database insertion. Earlier maintenance 148.03 s / observer 0.81 s
through turn 53 isolate those costs. Laptop contention and maximum VRAM remain
unmeasured.

New manual bundles snapshot every completed turn; standalone default remains 10.
A copied actual 70-turn store (35 tables, 17.45 MB) took 4.214/4.125 s per checkpoint.
After a synthetic 71st completed turn and unfinished later mutation, restore
matched every table/hash, kept turn 71, retained the unfinished tail and exactly
two rolling generations. This is not final-store throughput: snapshot cost
grows with store size. Three fixture setup attempts were not credited (ORM
column order versus frozen migrated DDL, lost vector extension after resetting
the fixture schema, and a missing nonnull fixture field). Corrected fixture
used the frozen DDL and complete synthetic fields; no old manifest was weakened.

## Validation and operator restart

Final disposable smoke: 564 passed in 38.06 s, five existing warnings. Focused
controls 70/47/77/95 overlap the final suite and are not independent samples.
Actual four-turn synthetic fault control exercised real preflight, writing,
post-flight, source/probe capture and checkpoint restore: three injected Codex
failures exhausted six attempts, paused after the third durable turn, and
resumed only turn 4. Gold-source degradation reflected only its own prefix.
Actual SQL control captured a committed partial periodic rewrite and retained
failed cadence without declaring success or repeating inside its interval.
Cloud controls used controlled transports/outputs, not live API requests.
Earlier actual route/cancellation and 49 runtime checks remain recorded in
[PROVENANCE](../PROVENANCE.md); they were not rerun as independent samples here.
These controls do not prove exhaustive robustness or supported answers/token.

At preparation on 2026-10-04, fresh r4 reused byte-identical reviewed packets, 117 admissions,
1471 recorded pairs and 259 frozen prompts. That repair launched no experiment
or cloud stage; the maintainer subsequently ran r4. R3's 70-turn recovery remains unchanged; changed writer identity
requires a fresh start, never editing its old manifest.

From the repository root, without activating a venv:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

Fresh r4 started from turn 1; the current guard-repair resume retains 1120 turns.
Later Ctrl+C and rerun of the same r4 command resumes
with unchanged code/settings/models/inputs. The [manual guide](2026-10-01-v3-manual-campaign.md)
owns scope, counts and limitations. Tuning, output-specific truth review and
independent judge qualification remain research gates; final runs are LME oracle
and semi-LSREP only, not a full LME-S campaign.
