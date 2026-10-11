# ICE v3 procedural development audit —2026-10-11

This reads the completed251-turn/67-probe development baseline. It does not
rerun its writer, produce answers, alter a setting or inspect validation.
Private packet, original source bodies, internal proofs, rendered selections
and all-table preservation receipts:
`logs/v3-procedural-review-2026-10-11-r1/`.

| Observation | Actual denominator and limit |
|---|---|
| Stored patterns |35;12 active at confidence0.8,23 pending at0.3. |
| Current internal source proofs |35/35 resolve against unchanged originals and the exact deployed proof policy. This is currency, not independent semantic accuracy. |
| Supported source observations |271; all typed observed_request, none explicit_instruction or self_report. Repeated references are not unique turns. |
| Source sessions |One per pattern; no cross-session reinforcement is established by this store. |
| Saved merge comparisons |160;157 unknown,3 different,0 equivalent. Unknown is not proof that all pairs should merge. |
| Full-arm ranked procedural material |51 fragments on51/67 occurrences,5394 tokens. |
| Full-arm final procedural material |37 fragments on37/67 occurrences,3899 tokens. |
| Task labels |35/67 occurrences include the overlapping procedural label. A task label does not identify which store should answer it. |

The agent inspected all35 complete descriptions and115 distinct complete cited
user originals. This is a diagnostic source review, not blinded human labels,
per-citation precision or an answer-quality result. The descriptions are broad
observations about requests; they contain no explicit instruction proof in this
baseline. Historical advice questions can require the assistant's original
instructions, which a source-linked behavioral description does not reproduce.
Crediting its origin-turn IDs cannot establish that required advice survived.

Eight current final descriptions match the37 historical selected texts. Those
matches are descriptive grouping only: the old trace has no stable procedural
artifact ID. Do not report eight uniquely used historical database rows or
infer lifetime usage from current state. Rendered versions and source overlap
remain separate from stable artifact/version identities.

Three current patterns cite identical source sets while using different
descriptions. Some other retained comparisons have high cosine but unresolved
bidirectional entailment. Shared sources and similarity nominate inspection;
they do not authorize merging different scopes, conditions or behaviors. This
audit does not lower the NLI floor, replace it with cosine or add another model.

[LangMem's primary concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
separate stored facts/experiences from procedural instructions that change agent
behavior. That distinction motivates a qualification panel; it does not prove
ICE should adopt a prompt optimizer or convert an observed request into a rule.

**Decision:** retain the current source/activation/equivalence guards. The
original zero-selection diagnosis no longer describes this repaired baseline.
Current material is reachable; utility and instruction specificity are still
unqualified. Continue matched-budget reader/content qualification and final
component comparisons rather than forcing activation or declaring the leg useless.

Before activation or merge tuning, distinguish three expectations: a recalled
user behavior, an explicit scoped instruction, and historical assistant advice.
Require the actual requested content in final context and supported answers;
measure complete source preservation, false global instructions, lost scope,
irrelevant selected tokens and quiet/revisited retention separately. Use original
episodic/claim evidence for historical advice where the procedure lacks it.
The full event parent must capture stable procedural IDs/versions and eligible
inventory; these historical text matches cannot supply that missing telemetry.

No complete roadmap owner closes. **G49 — procedural contribution**, for example
a relevant workflow reaching an answer rather than merely sharing a source ID,
still needs useful-answer/context evidence. **G55 — truthful evidence scoring**
must keep content support separate from source-turn hits. **G70 — write/read/use**
still needs stable artifact identities and final answer use.
