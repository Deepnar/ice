# Handoff — ICE v3, 2026-10-06 21:26 IST

**State, not a queue.** ROADMAP owns work; paper v2 remains frozen.

## Told → did

**Told:** before resuming r4, explain why the judge was DeepSeek rather than
Muse, which had been preferred; if DeepSeek is used, prefer v4.1 Flash.

**Did:** actual v3 paired judge retained global PROBE_MODEL=deepseek-v4-flash,
while frozen-v2 LongMemEval had selected Muse1.3 through Responses. Prior small
Muse-family human calibration led73% versus gateway DeepSeek v4 Flash60%; that
older task/model evidence does not qualify the current v3 rubric. The v3
DeepSeek pilot's three all-correct human pairs likewise do not prove superiority.
Current provider catalogGET200 and official OpenCode Go docs list Muse1.3 and
DeepSeek-v4.1 Flash. Decision: Muse1.3 Responses for new v3 bundles/current r4;
explicit v4.1 Flash alternative, never automatic fallback.

The paired judge now reuses the existing TextGenerator Responses/Chat adapters,
with ICE user-agent, stable session, completion status/usage and no hidden SDK
retries. Complete-source rubric and both display orders unchanged. Model/profile/
endpoint/decoding/transport identity bind judge and calibration resume; changing
these refuses old outputs. Raw verdicts retain response IDs/status/usage.
Parent status/report expose selected judge. Caller attempt/error/access/outage
policies remain bounded and unchanged.

## Actual r4 selection and retained progress

`logs/z1-v3-manual-2026-10-04-r4` now has explicit
`judge_profile=opencode-muse13`; its previous private campaign configuration is
archived. No earlier judge output existed. Only judge children receive
ICE_JUDGE_PROFILE. Global .env/PROBE_MODEL, every production Settings field,
local writers, GPT-6-Luna answering and source labels remain frozen.

**1434 originals/213 checkpointed probe contexts retained** (1119+251+64).
The prior control-integrity repair still has its verified continuation/archive:
recent-only skips all search helpers, vector-only excludes temporal notes and
wide-net bypass, full/no-Codex behavior unchanged. Both earlier instrument
boundaries remain retained. Normal resume begins third-history turn65;37 turns
remain. Its15 unfinished provisional probes are rebuilt after tail archival.
No original checkpoint/manifest/pointer/trace/store was edited.

Actual seed resume readiness was rechecked after the judge change: seed code
0870efc unchanged, all8 input digests/259 planned prompts/two installed local
writer manifests and full settings identity match the verified continuation.
No actual experiment DB restore or full campaign launched; no running parent
or seeder was found. Do not edit pinned seed code while the operator runs it.

From the repo root, without activating a venv, use the same existing bundle:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-04-r4 --run
```

One command continues seed → snapshot → answers → judge → report. Stage2
verifies/freezes all35 tables and complete1471-turn/259-prompt replay. Stage3
uses GPT-6-Luna for117 admitted probe instances across four arms, up to468
successful answers before retries. Stage4 uses **Muse1.3 Responses**, with up
to702 judge-order requests if all answers succeed. Stage5 combines source/rank,
answer-quality, paired effects, token cost, retention and degradation receipts.
Ctrl+C then the same command resumes with unchanged identities.

## Validation and scope limits

Two actual cloud requests on one synthetic public port-choice pair returned
correct/incorrect with A preferred, then incorrect/correct with B preferred
when swapped. This tests current endpoint/schema/basic discrimination through
the shared v3 judge request, not independent population accuracy. No personal
corpus was sent by this check. V4.1 has catalog/mock transport evidence only,
not live judgment or comparative accuracy. Independent current v3 judge
qualification remains pending; score_of_record=false.

602 disposable smoke passes42.36s, five existing warnings;89 focused overlap.
Final10 profile controls pass after import cleanup. Tests cover exact model/
endpoint, full-source marker, stable session/UA, incomplete completion refusal,
500 one-call behavior, quota pause, model/profile identity and judge-child-only
routing. SDK retries0; caller reservations remain durable. All test databases
removed. Prior592 control/restore checks and saved213-control contamination
audit remain documented in PROVENANCE; no new memory-quality rate is claimed.

Quality progression remains completed development evidence → output review/
targeted repair/tuning → frozen **LME oracle and semi-LSREP only**. No full LME-S.
Summary/extraction truncation and earlier degraded workers remain explicit
quality evidence, not repaired outputs. Existing final runner/vector-baseline
and broader judge/graph/summary qualification contracts remain in ROADMAP.

## Git and propagation

Main-local `ab8e233` provider/phase routing, `be0fd66` model roles/audit docs;
no push under the publication freeze. Spec, MODELS, architecture, inventory,
README/manual guide, provenance, roadmap note and TRAPS58 endpoint/role
recurrence reconciled. No production source, AGENTS, schema, global .env or
local model assignment changed; CLEANUP n/a (no moves). Requested judge
correction/readiness complete. HANDOFF committed last; private SESSION archived
and reset after propagation.
