# ICE v3 summary field and optional-retry audit —2026-10-11

This diagnoses the completed251-turn development baseline and bounded local
checks. It does not generate cloud answers, inspect validation or measure a
population precision rate. Complete original source text remains private.

## Internal support is not faithful value assignment

The read-only inventory finds38 summaries and41 abstracts with79 current
internal supported receipts and236 assertions. Agent inspection of six complete
original/output pairs finds a ranking assertion that omits whether a value
belongs to the overall or subject column. The correct numbers can occur in both
source and claim while their intended fields become ambiguous. The table survives.

That particular old summary has coverage0.68, below the0.7 read prerequisite.
Its internally current support receipt is not proof of use in a final prompt.
The diagnostic is a summary qualification weakness, not evidence that this
statement caused a wrong answer. Original-source availability, required content
and final answer support must remain separate measurements.

Existing E4B author-checker diagnosis uses its unchanged deployed native helper,
model, limits and strict JSON parser. The original assistant-only input withholds
the ambiguous assertion; seven unambiguous field/value controls classify correctly.
With both original messages under attributed JSON or raw role headers, the same
checker accepts the ambiguous assertion while accepting the three healthy controls.
The second form does not fix the miss. Context and batching also differ; these
checks cannot isolate a serialization cause. The graph's separately qualified
typed subject/relation/object gate is not being replaced or invalidated by this
argument-free summary pilot.

**Decision: do not add this checker as an extra summary pass.** A positive small
pilot cannot establish a useful guard through the actual evidence form. No new
model, resident weights or verification call was adopted. Native E4B pins and
all complete requests, responses, source/code hashes and decisions are private:
`logs/v3-summary-qualifiers-2026-10-11-r1/`. The first8-case check makes four
native calls37.18s; each four-case compiled form makes three calls32.51s/31.10s.
These overlap sources and are not independent accuracy observations.

## Bounded wording decision

[FActScore](https://aclanthology.org/2023.emnlp-main.741/) and
[FENICE](https://aclanthology.org/2024.findings-acl.841/) motivate checking atomic
facts rather than scoring a mixed statement as one unit. They do not certify
ICE's NLI, this prompt, source completeness or answer quality.

One wording candidate requests independently qualified facts and explicit
value/field/unit/time bindings. It restores explicit ranking columns and
coverage0.68→0.76 on the retained example, but lengthens a healthy clarification
and loses a previously supported long-input abstract at the unchanged900-token
ceiling. One shorter revision names the column but reduces ranking coverage
to0.56 and again loses the long-input result. Both use the real post-flight
generation/parser/NLI/chooser path, unchanged windows/caps/model and no store
writes. They do not demonstrate better answers per token.

**Decision: retain the existing production wording.** Do not tune a trust floor,
install an unqualified critic or start another prompt/model search. Explicit
field/attribution/completeness qualification remains open. Preserve all native
packets and healthy failures under `logs/v3-summary-bindings-2026-10-11-r1/`,
`r2/` and`r3/`. The first diagnostic shadows its SDK callable after the first
source: three later attempts fail locally before provider calls. That attempt
and corrected remaining-only run are retained; these are instrument failures,
not production model rates. The final revision makes seven live calls59.18s.

## A complete first result is lost when an optional retry truncates

The final revision's longest source returns a complete first provider reply
at692 output tokens. Its optional coverage retry reaches length900. The typed
IncompleteCompletion escapes the shared retry helper; generate_summary catches
it and discards the complete first candidate as well. This is a direct loss
mechanism, distinct from the failed wording improvement.

The approved one-file repair catches only the existing typed incomplete-output
exception at that optional retry boundary, warns every time with finish/usage/
traceback, and returns the exact first candidate and coverage. Attached citations
and abstract remain available for their existing independent source/NLI checks.
It never parses a partial retry or grants source support, and it adds no call,
model, cap, wording or trust change. First-call failure and unrelated errors keep
their existing behavior. Current v3 conversation notes no longer use this
coverage helper; this change affects the turn-summary caller.

Candidate source/patch and focused controls:
`logs/v3-summary-retry-2026-10-11-r3/`. The maintainer explicitly approved the exact final patch on2026-10-11; it is
now applied unchanged to src/workers/turn_density.py. The first private candidate omitted the newly required
logger binding; four focused failures exposed it before any production edit.
Its retained native healthy pass does not test that failed branch. Corrected
candidate qualification and recorded native failure replay are separate receipts.
No historical proof, producer trace, checkpoint or original table is overwritten.

The final candidate uses traceback.format_exception without frame locals;
exc_info through the rich console would expose those locals. Forty-seven focused
checks pass13.92s. Four recorded native response panels run through actual parser,
NLI, chooser and100-token budget. Two healthy original-wording paths retain exact
text/coverage/citations/budget selections. The observed incomplete retry retains
its first result, but that result still fails NLI capacity; neither representation
is admitted. A separate controlled coverage1.0 trigger plus native length-response
injection restores an independently supported47-token abstract, where baseline
admits nothing. That trigger is off-default and the second native packet comes
from another request. It proves a useful guarded recovery path, not a live gain
rate or improved answers. Three healthy live original-wording replies31.98s and
all35 original table fingerprints remain bound; playback adds no provider calls.
The first playback's incorrect expectation that the default first abstract would
pass NLI is retained and corrected. Actual-source focused47 checks pass14.13s; permanent failure/reader controls25
pass13.23s. The exact fifth reader transition exports a separate profile, stable
on identical reuse, while ordinary and earlier-profile commands refuse. Planning
validates67 pending questions with no model/DB call. Full actual-source
smoke/dynamics1153 pass541.23s/346 existing warnings; disposable store removed.
Separate applied receipt: logs/v3-summary-retry-applied-2026-10-11-r1. All35
baseline tables and original trace remain unchanged; no corpus or answer-quality
promotion follows.

No whole roadmap owner closes. **G57 — useful representations**, for example
a compact note preserving the requested value and its field, and **G75 —
compression trust**, for example keeping a supported independently usable
abstract after a failed improvement, still require broad qualification. **G29 —
shared consumer behavior** owns the retry-helper preservation mechanism; neither
reachability nor a focused recovery test establishes answer benefit.
