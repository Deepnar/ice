# Handoff — ICE v3, 2026-10-07 10:49 IST

**State, not a queue.** ROADMAP owns work; paper v2 remains frozen.

## Told → did

**Told:** inspect the r4 crash during judging; fix it so the maintainer can
resume without restarting the long seed/answer work.

**Did:** actual judge log refused changed answer-file hashes, not a Muse outage.
Successful cloud retries retained temporary UnconfirmedCloudCall metadata.
Completed answer resume falsely added degraded_final to successes, rewrote
files and doubled some progress error counts. Removing ONLY those false fields
reproduced BOTH original judge input SHA256 values exactly. Applied a locked,
archived recovery; no answer/source/real fault, judge identity or grade changed.
The seven returned judge orders and eighth unconfirmed reservation remain.

Completed answer reruns now validate frozen receipts without writing bytes;
new successful responses clear temporary fault metadata. Promotion requires a
real error. Progress counts actual failed rows. A closed exact runner-hash
transition admits completed old outputs read-only, with remaining identities
intact; it does not authorize old partial files or any cloud call. New outputs
pin compatibility helper/registry bytes. No production/seed/judge code,
settings, model role or global .env changed. Implementation commit84baffd.

## Current measured state

R4 has1471 durable originals,259 checkpoint contexts and four117-record answer
arms. Returned answers/errors: full113/4, no-Codex114/3, vector-only113/4,
recent-only116/1. Twelve real terminal cloud faults among468 planned records
remain ungraded; these are availability counts, not factual accuracy.
Partial no-Codex judge: three complete pairs/seven returned display orders;
the second order of the next pair has one unconfirmed reserved attempt.

Full/no-Codex restored input hashes9039f97f/33925244 match the ORIGINAL saved
judge pins. Original judge state remains fa36594c unchanged. Private archive
`answer-input-recovery-7kqv_t9p` retains all pre-repair inputs/judge bytes and
receipt. Seed code digest still0870efc; its two earlier registered instrument
boundaries, archives and as-of snapshots remain intact.

Actual all-four-arm resume traversed real trace/label/frozen-input checks,
kept byte/mtime identity and never constructed a model caller. A private judge
copy traversed actual paired-input/pending-order validation up to a network
trap, retaining all seven returned orders. Original judge state never changed.
No API calls, campaign launch or experiment database restore occurred.

Final618 disposable smoke checks passed in34.70s with five existing warnings.
Sixteen new controls cover successful retry → interrupted judge → upstream
resume → missing-order continuation, true terminal errors, registered read-only
old artifacts, changed identities/text, active operators and partially restored
inputs. Earlier89 focused/617 pre-final smoke passes overlap. The final suite
includes the later campaign-lock/archive-parent-fsync controls.

Answers remain GPT-6-Luna; judge remains Muse Spark1.3 Contributor Responses
(`opencode-muse13`). DeepSeek-v4.1 Flash stays an explicit alternative, never
an automatic switch. Independent current-v3 judge qualification and semantic
memory-quality review remain open; score_of_record=false.

## Position and next operator action

This resume repair is complete. The maintainer can rerun from the repo root:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

Use the SAME directory. Seed/snapshot/answer stages verify saved work; judging
continues with the missing order. A registered-old-runner read-only warning is
expected. In-flight cloud calls can incur cost without a saved response and
still consume their reserved attempt. Do not change frozen code/models/labels,
reset pins, delete outputs, or launch the full campaign on the agent's behalf.

Combined Z1/Z2 remains the current development-quality/tuning phase, not a
passed gate. Preserve the research focus: evaluate actual memory output and
answers before targeted repairs/tuning; final work remains LME oracle and
semi-LSREP, not full LME-S. Queue and acceptance rules live in ROADMAP and
RESEED_PLAN. This instrument repair closes no memory-quality roadmap item.

## Propagation / publication

RESEED_PLAN, manual guide, architecture, inventory, ROADMAP execution note,
PROVENANCE and TRAPS #86 updated. Models/README/AGENTS unchanged; no files moved.
Session findings archived privately, then SESSION reset. Work is on main;
push freeze remains active, NO PUSH. Next public publication requires explicit
one-time authorization. HANDOFF is the final commit of this repair session.
