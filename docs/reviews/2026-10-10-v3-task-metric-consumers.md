# ICE v3 — reviewed task and metric consumers, 2026-10-10

**G40 — task labels**, for example a progression question scored as one-turn
lookup, owns this prerequisite audit. Generated catalog origin/target types and
reviewed semantic intent are different namespaces. Neither a graph-targeted
probe nor a source-ID hit certifies graph benefit or the answer's correctness.

## Actual consumers and disposition

| Consumer | What it reads/does | Current decision |
|---|---|---|
| `unify_probes.normalise` / `seed_v3` catalog generation | Normalizes `ENTER_TYPE`; creates origin/target-bearing `probe_type` values, including native/generated types. | Preserve catalog provenance. Do not infer semantic intent from these strings or a `target_leg`. |
| Legacy `score_retrieval` | Source-ID rank over its legacy probe list; no reviewed semantic-type admission. | Historical/diagnostic only; not the repaired tuning instrument. |
| Legacy `score_typed` | Uses generated typed probes; single-source rank versus turn-set/procedure/temporal coverage. Source-span coverage is not semantic support; temporal score also assumes the question requests the older value. | Preserve historical method/artifacts. Do not reuse mixed means or turn-ID coverage as current quality or final tuning acceptance. |
| `answer_as_of` / `label_review` | Checks reviewed through-cutoff source labels, knowledge scope and semantic task vocabulary; retains catalog type separately. | Existing admission remains. Preserve valid, invalid and uncertain dispositions; no guessed replacement type. |
| `report_v3_replay` | Raw source funnel grouped by catalog type across the259-probe replay. | Diagnostic provenance coverage, distinct from the117 admitted answer occurrences and their reviewed intent. |
| Historical `judge_answers.reviewed_outcome_strata` | Overlapping semantic task/scope outcomes; separates two before controls, but old absolute grades depend on the comparator. | Preserve old results; rubric remains unqualified. No automatic promotion from old grades. |
| New `score_frozen_answers` | Semantic labels already entered blind judge input, but collection/report dropped them and repeat phase. | **Repaired:** retain scope/tasks/phase/source metric contract, refuse mismatched arm metadata before calls, report own-arm absolute grades and explicit age/task/error denominators. Independent rubric qualification still required. |
| New `prepare_inputs.development_keys` | Previously stored valid labels but had no executable metric-admission contract. | **Repaired:** attach scorer-only contract per key and counts per tuning/validation partition; bind the helper in immutable preparation identity. |

## The implemented contract

`scripts/z1/metric_contract.py` uses the existing nine reviewed task labels.
Single-source rank tuning needs **lookup-only**, one source turn, reviewed
older-history answerability and private/assistant-history scope. Synthesis,
multi-hop, temporal/update, procedures, negative/abstention, public/mixed
questions and recent controls remain outside that rank cohort. Their source
coverage and qualified semantic answers are separate measurements. Missing
source identity or age proof is explicit, not an ordinary miss or a guessed hit.

Answer reports retain correct/partial/incorrect/uncertain counts and
pending/ungraded/unavailable statuses. Before controls are separate; missing
age review is a separate stratum. Task groups overlap, scope groups are
exclusive, and missing family IDs do not imply independent questions. The
contract is scorer-only: expected answers, source IDs and reviewed types never
guide production ranking. Grade-cache completion/attempt rules are unchanged.

## Real-input audit and qualification limits

The offline development plan retains73 reviewed occurrences:67 tuning and6
validation. Only **5 tuning and2 validation** qualify for precise single-source
rank diagnostics. This is7 occurrences, not7 certified independent families or
a sufficiently powered claim of optimized retrieval. Other tasks remain in the
plan for separately qualified answer/support outcomes; vague synthesis coverage
cannot select rank settings. No new labels are guessed or old keys deleted.

Actual read-only independent planning over all four r4 arms has454 unique
absolute tasks, all pending;456 successful answer occurrences share two tasks.
Each arm retains117 records =115 older-source occurrences +2 before controls,
with0 unknown-age records and9 rank-eligible tasks. Unavailable answers remain
in the record denominator. No judge request/cache write is made by planning.

61 focused controls pass, including the actual offline preparation and actual
Responses SDK→controlled provider→cache→report/resume paths. These overlap the
full smoke suite and do not establish judge semantic accuracy. Current final
B+C references remain792 unqualified legacy occurrences, pending source/cutoff/
correction/task review. The oracle's official labels do not qualify its adapter.
The whole G40 parent remains open for final taxonomy/source qualification.

Private audit/plan receipts are under
`logs/v3-taxonomy-consumer-audit-2026-10-10/`; the revised immutable preparation
is `logs/v3-offline-input-plan-2026-10-10-r4`, also binding declared local models.
Earlier preparations and r4 remain
unchanged; code/helper drift requires a new bundle, never an ignore-hash resume.
