# ICE v3 named policy-consumer audit —2026-10-10

This reconciles the twelve historical clusters in G29 — duplicated policies,
for example a tunable limit hidden behind a literal argument. It is a consumer
audit, not a memory-quality result or closure of its whole original scope.

| Historical cluster | Current core path and disposition |
|---|---|
| Token accounting | Prompt assembly, retrieval and chat import `src.memory.tokens`. The deleted second budget loop remains absent. Required total-input/cap checks remain part of tuning. |
| Privacy/scope | Core legs use the existing conversation, source and exclusion predicates. The later vector-anchor, notes and archived-source repairs have their own opposing SQL controls; this audit does not re-prove all product readers. |
| Cluster filter | `_cluster_filter` has seven consumers, including claims, source notes, cold and wide retrieval. Inclusion and exclusion answer different questions and remain separate. |
| Read representation | `src.memory.representation.choose_representation` owns verified selection. Retrieval, classifier prior context, recent assembly, explicit pull serialization and **bookmarks** call it. Chat calls `bookmarked_turn_texts`; the old two bookmark bypasses are not current. Explicit memory slots are a distinct user-authored store. |
| Model JSON | Generic object/array salvage lives in `llm_json`; maintenance/reflection call it. Native typed extraction and cited-note contracts validate different schemas and must not be collapsed into generic salvage. Loud parse failure is distinct from a valid empty decision. |
| Stopwords/referential cues | **Still distinct and unresolved for broad invariance.** `_extract_prompt_keywords` has a fixed common-word set; B2's referential cues answer a different question. Merging sets would silently change meaning. Existing source-proof/style qualification and subsequent gate controls own behavioral limits; no claim that this audit removed lexical intent assumptions. |
| Second budget loop | Historical `0.9 * 4096 / 1.33` executable loop is gone. Final assembly/budget controls remain required; an old comment is not an active formula. |
| Job keys | Post-flight, procedural and Codex use `job_key`. The feature inventory's claim that Codex still hashes by hand was stale. Completion-negative receipts are separate from supporting occurrences. |
| Recent-window curve | B2, retrieval allocation and the conversation-fold creation gate share `recent_window`. Retrieval's density/label modifier remains explicit. Opposing fractions move actual callers. |
| Coverage retry | Post-flight uses `retry_on_coverage_miss`. Conversation folds now use cited assertions, complete-source manifests and separate completion/support outcomes; treating that schema as the old duplicate retry would undo later repairs. |
| Manual decision insertion | `reconcile_and_insert` remains the shared writer; product controls are later scope, not a new memory-repair detour. |
| Literal-shadowed settings | Existing diversification/cluster/ablation checks retain the shared setting/default seams. Fusion diversity was separately repaired and demonstrates opposing main/wide selection deltas. Static reads still cannot certify every one of the173 tuning candidates. |

Current focused verification: **60 passing controls**,14.79s, two existing
warnings, using a disposable database removed afterward. Suites cover settings
reachability, recent-window callers, cluster filter, verified representations,
coverage retry, namespaced keys, fusion-diversity callers and ablation seams.
Some are structural or controlled-provider tests; this is not60 actual model
decisions, a full privacy audit or a semantic/tuning score. No model/API calls,
working-store change or original r4 rewrite occurred.

Decision: do not rebuild the already shared bookmark/job/cluster mechanisms.
Proceed to fresh versioned development state and meaningful content controls.
Keep G29 open for live tuning-consumer effects and its wider original scope;
keep keyword/style limits explicit instead of merging unrelated policies.
