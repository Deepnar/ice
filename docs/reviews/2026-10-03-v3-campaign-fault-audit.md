# ICE v3 campaign fault handling and speed audit

The r2 manual seed exited on turn 54 with an unsupported-source-quote error.
It had 53 fully processed turns, 51 durable turns and 5 saved checkpoint probes.
The agent did not stop it. An actual NuExtract/background-NER reproduction
returned three nullable source-only rows; one copied quote differed only by
whitespace. The corrected writer resolves that copy to original bytes and
retains three attributed claims with zero inferred graph edges. Changed words
and values remain failures.

## Recovery boundaries

| Boundary | Live v3 / manual behavior | Evidence and limit |
|---|---|---|
| Invalid/truncated Codex output | Strict atomic graph/key contract; live runtime retries. Coordinator retries fresh local output failures at most twice, restoring its snapshot first. | Actual quote control, parser and DB negative controls; not exhaustive model-output qualification. |
| Independent post-flight stages | Codex failure permits procedural progress, then propagates for retry. Stage completion keys remain separate. | Real post-flight DB control with failed then healthy extraction. |
| Retrieval / source proof | Existing observable leg fallback and unknown-proof behavior preserve scope and prior recall; no invented proof. | Current source/transport smoke controls; some per-turn model calls may still be expensive. |
| Periodic maintenance | Live runtime catches job errors and stays alive; manual retry restores the entire checkpoint before any committed partial job can run again. Unknown failures pause, never silently skip a job. | 49 runtime checks plus recovery controls; serial scheduling remains different from asynchronous production. |
| Stream cancellation / upstream failure | Initial status cancellation no longer leaves generation active; actual generation uses a matching finally. Unsuccessful replies remain unstored. | Actual route failed before repair; 6 route/transport checks passed afterward, including both cancellation points. |
| Cloud answers | Save each successful call; fresh connection/timeout/server failures get two coordinator retries. | Profiles use zero SDK retries; quota/authentication and incomplete completion pause instead of becoming successful answers. No live cloud requests made here. |
| Paired judges | Preserve a successful first order; retry only its missing failed order for eligible temporary failures. | Existing order-resume and coordinator controls. Quota/authentication/invalid verdicts pause; no grades are fabricated. |
| Snapshot / reporting / startup | Unknown, schema, identity, resource and filesystem problems give a clean operator error/pause; artifacts and stage state are retained when writable. | Failure controls cover every parent stage and corrupt configuration. Disk failure can prevent new receipts; prior checkpoints remain the recovery basis. |

Each parent attempt is fsynced to `campaign-attempts.jsonl`. Retry uses 2/8-second
backoff and at most three child attempts. `campaign-pause.json` describes an
unresolved failure; no later stage starts afterward. Only fresh persisted failure
metadata can authorize retry, so an old extraction failure cannot mask a new
identity refusal. Ctrl+C retains recovery and returns 130. This is bounded
recovery, not immunity to all possible errors. Missing memory work never counts
as a complete experiment.

## Speed finding

Native source calls already registered their model with ICE's guarded release
manager but requested immediate unload. Four identical actual frame requests
under two residency policies took 3.54/6.77/6.41/3.21 seconds. The second
unloading call versus the second resident call reported 3.52 versus 0.31 seconds
of model loading. The first retained call still had to load the model. This
isolates roughly 3 seconds of avoidable reload cost in this control, not a
whole-run speedup or a clean controlled comparison of competing laptop use.

Native calls now retain their owned model between requests and leave release
to ICE's existing idle manager. Serial seed exit releases its owned models
because it has no idle scheduler. Reranker and NLI weights stay cached on CPU
between calls; the shared encoder and lazy background NER keep their existing
policies. This does not force all six substantial components onto the GPU.
Cloud answering/judging adds no local model residency. Local calls log wall time;
native proof/need calls also log provider load, prompt, evaluation and total durations. The
[Ollama chat API](https://docs.ollama.com/api/chat) defines those durations in
nanoseconds; the [FAQ](https://docs.ollama.com/faq) documents keep-alive control.

The user's 28:03/51 screenshot represented about 33 seconds per fully processed
pair, including local preflight, summaries/extraction/NLI, maintenance and
checkpoint probes. Maintenance contributed 148.03 seconds through turn 53;
semantic observation overhead was 0.81 seconds. This was progressing before
the source-quote failure, not merely inserting rows or generating foreground
cloud answers. Ordinary laptop use can affect resource contention, but this
audit did not measure that contribution or certify maximum simultaneous VRAM.

## Validation completed

The final disposable smoke suite passed 546 checks, with five existing
SQLAlchemy deprecation warnings. Twelve extraction/claim database checks and
49 live-runtime checks also passed. Focused coordinator controls ran real
Python child failures, resume and three-attempt exhaustion; cloud boundaries
used controlled responses. An actual local-model source-unit replay of the
failing turn stored and retrieved all three original claims, with zero graph
edges. These checks overlap and should not be added together. They do not
establish whole-history answer quality, maximum VRAM or full-run fault coverage.

## Manual restart

Preserve r2's files, store and 51-turn checkpoint unchanged. Production and
recovery changes alter pinned code, so the corrected writer needs a fresh
campaign. r3 is initialized only, with byte-identical reviewed source/repeat
packets: 1471 recorded pairs, 259 prompts, 117 reviewed admissions. No seed, cloud
answer or judge run has been launched by this audit.

From the repository root, without activating a venv:

```bash
uv run python scripts/z1/run_v3_campaign.py --run-dir logs/z1-v3-manual-2026-10-03-r3 --run
```

This restart begins at turn 1. Later Ctrl+C interruptions use that same r3
command while code/settings/models/inputs remain unchanged. The corrected
quote must not be mixed into an old checkpoint by editing its identity. The
[manual guide](2026-10-01-v3-manual-campaign.md) retains the scope, ground-truth
and measurement limits. Tuning, independent judge qualification and final LME
oracle/semi-LSREP remain research gates, not outcomes of this crash repair.
