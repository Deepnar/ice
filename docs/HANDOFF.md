# Handoff — ICE v3, 2026-10-08 15:00 IST

**State, not a queue.** ROADMAP owns work; the v2 paper remains frozen.

## Told → did

**Told:** close out the first complete v3 reseed analysis, propagate the findings,
merge the JMLR submission branch into main, commit and push once, and specify
the evidence-led repair/tuning/final-experiment phase. Retain SESSION until an
explicit instruction to clear it.

**Did:** completed aggregate evidence and documentation propagation, merged
`work/jmlr-v2-submission`, and wrote the
[post-reseed repair, tuning and Z2 contract](specs/V3_POST_RESEED_REPAIR.md).
No runtime repair, model call or new campaign was performed in this closeout.

- `d0b8565`: JMLR branch merge. All eight paper files match the branch and the
  pre-merge local files; the archive's PDFs match the merged PDF bytes. The
  backup stash and separate paper worktree remain. This is manuscript packaging,
  not a new v2 or v3 result.
- `207c7a7`: completed-r4 evidence, interpretations and limits propagated to
  PROVENANCE, architecture, feature inventory, models, README and failure traps.
- `cd25bb1`: eleven repair packages, all 29 dependent roadmap owners, bounded
  tuning and the final experiment contract. Older execution docs now point to it.

## Current measured state

R4 completed all five stages **diagnostically**, not as a score of record:
1,471 original turns, 259 question occurrences and 39 saved as-of states.
All 255 base source reviews have dispositions: 113 valid, 102 invalid and
40 uncertain. Four repeated occurrences yield 117 admitted occurrences per
answer arm. Across four arms, 456 answers returned and 12 were incomplete.
The 115 older-question panel supports a large advantage over recent-only
context; it does not establish a reliable win over warm vector or fewer tokens.
Same-full-answer grades differ across comparator contrasts on 36/111 fully
graded cases. The current judge still needs independent qualification.

Confirmed v3 compression failure: all 986 nonempty turn summaries received
unknown support receipts because the whole-source NLI input exceeded its
contract. This is not evidence that those summaries were contradicted. The
report's zero-unknown summary counter is also wrong. A retrieved gold source
ID can reach the prompt while its required answer content is absent. These
measurement and long-source mechanisms lead the next repair pass.

Graph precision, procedural usefulness and semantic maintenance quality remain
unqualified. Synthetic-date eligibility blocked the observed reconciliation
model calls; the run did not test their accuracy. No cold-memory transitions
were observed. Full results, denominators and limitations live in
[the completed campaign provenance](PROVENANCE.md#2026-10-07--v3-completed-r4-development-campaign).

The immutable baseline remains `logs/z1-v3-manual-2026-10-04-r4`, including its
39 checkpoints and original answer/judge artifacts. All campaign-report hashes
were rechecked; the independent read-only aggregate recount matches. New writer
behavior must use new versioned development state, never overwrite this bundle
or bypass its resume identities. Read-only repairs may reuse its checkpoints.

## Position and next work

**Closeout/planning is complete; the new repairs are not implemented.** The
[roadmap's current phase](ROADMAP.md) points to the
[eleven-package execution contract](specs/V3_POST_RESEED_REPAIR.md).
Start with trustworthy nested-status/semantic evidence measurements, then
source-linked bounded verification that permits useful long-source compression.
The spec owns the remaining repair order and acceptance checks; this handoff
is not a second queue. Focused experiments are allowed, broad repeated model
searches and product edge-case expansion are not the current objective.

Formal roadmap count: 160 anchored entries, 93 checked and 67 open. The open
partition remains 29 dependency-sensitive entries, 35 later items and three
experiment gates. Completed collection/review/cloud-processing substeps are
checked; no whole quality item was closed merely because r4 finished.

After repairs: bounded hyperparameter tuning → frozen configuration → Z2,
meaning one long-conversation semi-LSREP with full lifecycle/decay, repeated
cutoff-reviewed probes, additive and independent leave-one-out ablations,
plus LongMemEval oracle. Both ICE warm-vector and independent bare-vector
controls are required. There is no full LME-S haystack run in this scope.
Recorded original replies drive replay; only evaluation probes generate new
cloud answers. The future runner, review packet and final fixture manifest
still need implementation. Do not present the r4 command as their launcher.
The maintainer will manually launch the long experiments after readiness.

Model roles are unchanged: local E4B general background and NuExtract3
extraction, GPT-6-Luna answers, Muse Spark 1.3 Contributor Responses judge
subject to qualification. Laya-like models are a bounded source-backed decision
candidate, not an adopted resident model or default NLI/classifier replacement.

## Validation and propagation

Isolated smoke suite: 618 passed with five existing warnings, after the paper
merge (38.37s) and again for the documentation pass (45.47s). These are repeated
runs of the same suite, not 1,236 independent behaviors or memory-quality tests.
All 59 introduced relative links resolve; the spec has seven required sections,
11 packages and 29 unique owner mappings. Roadmap counts match. Whitespace and
history-sensitive-path checks pass. No src/test changes in this closeout, no
working database restore and no cloud calls. Private evidence stays ignored;
tracked results contain aggregate findings, not conversation text.

Propagation complete: ROADMAP, PROVENANCE, TRAPS, FEATURE_INVENTORY,
ICE_Architecture, MODELS, README, affected specs and execution guides.
ROADMAP_DONE needs no entry because no whole parent item closed. CLEANUP has
no new move/rename to record. AGENTS is unchanged. SESSION retains its earlier
checkpoints and explicitly says **do not clear until requested**, overriding the
usual session reset for this continuing chat.

Work remains on main. This closeout includes one explicitly authorized normal
push to origin/main; verify the remote tip after publishing. The experiment
push freeze applies again afterward. HANDOFF is this closeout's final commit.
