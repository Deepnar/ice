# ICE_Architecture §4 — Codex knowledge graph as of v2/v3 (superseded 2026-10-16)

Moved verbatim from `docs/ICE_Architecture.md` when the conversation codex was rewritten as
verified facts ([CODEX_V4](../specs/CODEX_V4.md)). Superseded docs are never edited.

## **4. Codex Knowledge Graph (v2)**

ICE v3 Codex retains graph navigation beside attributed source sentences. Relation categories remain extraction/type hints; no category establishes cardinality or authorizes expiry. Source-backed reconciliation decides whether assertions coexist or replace one another.

### **4.1 Entity and edge tables**

codex_entities (PK id — a deterministic UUIDv5 in CODEX_NAMESPACE for new entities) holds canonical_name (unique), aliases ARRAY(Text), tags ARRAY(Text), **entity_type Text (A7 — structural node type; ICE is general-purpose, so the vocabulary spans all domains: universal person/place/organization/event/concept/object, coding/research software/function/class/file/module/dataset, academic/business document/product, creative character/location/item/creature/faction; inferred from an entity's relations via _infer_entity_type, or set deterministically by the code graph E1b)**, **description Text (A7 — the enriched "note body", written by the reflection enrichment worker)**, properties JSONB, an auto-regenerated context_payload Text, embedding Vector(1024), and last_updated. **A7 rich notes:** _regenerate_context_payload assembles context_payload as a bidirectional Obsidian-style "note" — the description (note body), then Properties, then **Links** (outgoing edges, strongest-first) *and* **Backlinks** (incoming edges — what points *to* this entity, previously invisible) — so an entity's full connection picture reaches retrieval in both directions. codex_edges holds id, source_id/target_id FKs to entities, relation Text, strength Float (default 1.0 for all new extracted edges), source_batch UUID (the batch that asserted the edge), confidence Text ∈ \{pending, active\} (ordinary edge promotion requires a distinct source batch; reads never promote), **extraction_confidence Float ∈ [0,1] (A3)** — how much the extraction itself is trusted: seeded by NER grounding at write time (0.9 grounded, 0.7 no-NER chunk, 0.35 grounding-rejected; legacy edges 1.0), raised to the max seen on corroborating re-extraction — valid_from, and valid_until — **`NULL` means not retired, not verified true**; non-NULL means historically superseded — plus **learned_at / unlearned_at (bi-temporal, migration `505f12031434`, 2026-08-13)**. The two pairs answer different questions and conflating them is what made a wrong memory unauditable: `valid_*` is when the FACT was true in the world, `learned_*` is when ICE came to believe it. They diverge constantly — a fact true since 2019 learned yesterday, or a fact believed for months that was never true. Only the second pair can answer *"when did we start believing this?"*, which is the question asked after a fabrication is found. An invariant in models.py mirrors `valid_until` into `unlearned_at` on every retraction path (seven call sites across two modules), so a new retraction site cannot forget it. An edge's **effective trust** used by retrieval is extraction quality with bounded retention/recency ranking (strength carries usage dynamics; extraction_confidence carries extraction trust). Two auxiliary tables complete the graph: codex_events is an append-only event log (event_type ∈ \{edge_added, edge_expired, edge_strengthened, property_updated, context_appended\}, payload JSONB, batch_source, compacted Boolean), and codex_snapshots holds point-in-time full_state JSONB produced by the compaction worker.

### **4.2 Controlled relation vocabulary**

The vocabulary is defined in workers/codex_extractor.py as three disjoint dicts keyed by category, with an import-time assertion guarding against accidental overlap. Four design rules (quoted verbatim from the source comment) govern it: (i) one canonical relation per meaning (near-synonyms collapsed, look-alikes kept distinct); (ii) generic dumping-ground relations (is, has, applies_to, connects_to) removed entirely so the extractor is forced to be specific; (iii) relations grouped into labeled categories so the extraction prompt renders them grouped, not as a flat 150-item list; (iv) single-vs-multi-valued cardinality reassigned by real-world cardinality (e.g. works_on, founded_by, created moved to multi-valued).

- **`PROPERTY_RELATIONS`** (11 categories, written to the properties JSONB): identity (name, alias, nickname, full_name, title, description), demographics (age, gender, species, nationality, religion, birthday, blood_type), appearance, professional (role, occupation, profession, affiliation, status), contact_location, metadata_generic (type, genre, format, license, version, language), task_management, attribution (author), technical_specs, narrative_metadata, abstract_metadata. A new property observation expires the prior active edge and overwrites the JSONB key.

  **A property VALUE is also a node, deliberately (G33, 2026-08-08).** `handle_triplet` creates the object entity before it checks whether the relation is a property, so `kael --role→ fire mage` writes the property, an edge, **and** a `fire mage` entity. That node is kept rather than suppressed, because the edge behind it is what lets T4 build a timeline when the property *changes* — remove the node and the FK, the `edge_expired` event, and the property's history go with it. What was wrong was that its `context_payload` stayed empty forever: this branch was the only one of `handle_triplet`'s three that refreshed the subject alone (the negation and non-property branches always refreshed both ends), so the value node was a matchable pre-flight anchor that §6 then skipped for being empty — an anchor slot spent on nothing. Both ends are now refreshed and the value renders as `Backlinks: kael --role→`. ⚠ Regenerating exposed a second defect, also fixed: `_infer_entity_type` maps a relation to the type of the entity that has it **outgoing** (`role` ⇒ person, because a thing that *has* a role is a person), but voted over a flat list of both directions — so read from the far end it typed `fire mage` as a *person*. Incoming relations now vote only when the relation is **symmetric** (`married_to`, `colleague`, …); asymmetric incoming edges yield `entity`, which is unknown rather than wrong.

- **`MULTI_VALUED_RELATIONS`** (16 categories): technical_dependency (uses, imports, depends_on, supports, integrates_with, calls, returns, references, cites, extends, implements), technical_distribution, structural_containment, social_relationship (friend, ally, enemy, colleague, knows), social_action, organisational_collab, support_endorsement, activity_participation, categorisation, works_on_projects, founding_creation_multi, data_lineage (ICE-specific: derived_from, trained_on, configured_with, evaluated_on, benchmarks_against), code_structure, research_relations, narrative_structure, conceptual. Multiple active edges coexist; new edges never expire prior ones.

- **`SINGLE_VALUED_RELATIONS`** (11 categories): organisational_position (part_of, works_at, reports_to, managed_by, …), executive_role, succession, education, production_singular, personal_relationship (married_to, parent_of, child_of, sibling_of, …), biography_location (lives_in, born_in, died_in, …), deployment_ownership, logical_requirement, narrative_singular, conceptual_singular (instance_of, subtype_of). These are historical vocabulary hints; v3 no longer expires another value from this category alone.

### **4.3 Temporal versioning and automatic property updates**

handle_triplet is the heart of the write path. **G44 (2026-08-13): it first refuses triplets whose subject or object cannot name a node at all.** The measured store held entities literally named `8`, `3`, `6`, `2`, `d` — **typed `person`** — plus `i`, `the` and `~`; 155 of 3,671 entities were ≤3 characters. A node called `i` collects every speaker who ever said "I" and can never be looked up, so the edge hanging off it is not a weaker fact but an unreachable one. ⚑ The test is **functional, not a length rule**: `ai`, `ml`, `q4`, `eq` are two characters and perfectly good nodes, while pronouns, articles, bare numerals and punctuation refer to nothing whatever the writing style — which is what keeps this clear of the style-invariance rule G28 exists to enforce (a length rule is exactly the bet on convention that rule forbids). Rejection is logged at WARNING every time: the rate is the finding, and a silent skip would read as a clean graph.

Entity resolution (get_or_create_entity) is **three-stage** (G50, 2026-08-17): exact canonical_name match, then alias match, then a **normalisation-key match** — `properties.merge_key` (casefold · split digit/letter runs · collapse separator punctuation · **token order preserved**), matched against live entities only, with the arriving surface form recorded as an alias so the exact tiers catch it next time. Otherwise create with a deterministic UUIDv5 and a fresh embedding, stamping `merge_key` at birth. The third tier exists because the first two are exact-string and `canonical_name` is UNIQUE, so `gemma-4-e4b q4` and `gemma-4-e4b-q4` each minted their own row; relations have had `canonical_relation`'s three-tier ladder since G45 and entities had none, which was never a decision. It is an equality test, not a judgement: it never deletes a row, is not gated behind `codex_node_promotion`, and honours `protect_ids` exactly as promotion does. Migration `a1c4e7b90d22` backfilled 8,280 rows and added a partial index; 51 groups on the live store collapse (`20gb`/`20 gb`, `ai ml`/`ai/ml`). ⚠ Token order is preserved deliberately — sorting finds more groups and two of them are converses (`shiva without brahma`/`brahma without shiva`, live at cosine 0.9639). **Node promotion (G44's second half, `codex_node_promotion`, default OFF):** when a strictly more specific name arrives for a generic node already stored — `master plan` against a bare `plan` — the specific node becomes canonical and the generic name resolves to it thereafter, so later mentions land on the node worth traversing instead of sprouting a second one. ⚠ **Only zero-edge stubs are eligible.** Promotion merges two identities, and `plan` is not always `master plan`; doing it to a node that already carries facts would silently re-attribute them, whereas a stub has none to re-attribute. It is off by default for that reason — the aggressive form (promoting nodes with edges, migrating those edges) changes stored history and is a separate decision. All extracted assertion types now share one writer (v3, 2026-09-19):

- Exact directed relation and polarity observations reinforce once per source
  batch; repeated reads or the same batch never promote confidence.
- A new edge starts pending at strength1. Property and negative assertions have
  no privileged expiry or confidence path.
- Every same-relation target/polarity alternative and known opposition is a
  candidate comparison. Complete current linked source units must share a
  conversation and authoritative role, with original source timestamps strictly
  ordered, before automatic reconciliation. Unknown authority retains both.
- Reconciliation compares complete passages and speaker/time metadata. A rejected
  replacement rolls back prior expiries from its comparison set. Property JSON
  projects all live positive values, not last-write-wins state.
- Maintenance shares the evidence boundary and prompt; application rechecks it.
  Manual reconciliation requires an explicit keep-edge subset. Expiry journals
  the decision and refreshes both endpoint notes and property projections.

Current `valid_*` expiry remains a recorded system decision, not an inferred
real-world event date. Older imports cannot overwrite newer source assertions;
source times remain separately labeled. Autonomous event-time interpretation,
cross-conversation author identity, semantically different open-vocabulary
relation candidates and long-source verification remain unqualified.

### **4.4 Extraction pipeline**

> **⚑ EXTRACTION HAS ITS OWN MODEL SETTING (G63, 2026-08-27).** `codex_extraction_model` resolves as `model_override or settings.codex_extraction_model or get_bg_model_name()`, defaulting to `hf.co/numind/NuExtract3-GGUF:Q8_0`. It exists because **one background pin served ten jobs** — turn summary, batch summaries, conversation summary, cluster naming, reflection, the maintenance agent, procedural extraction, document-kind, the raw slicer, and extraction — and only the last wants a specialist. Pointing the shared setting at NuExtract3 would hand a JSON-template filler to the summariser and the cluster namer, which fails *silently*: the summaries keep arriving, just wrong.
> ⚠ **NOT the conflict reconciler.** `make_llm_reconciler` lives in the same file and is a one-word **reasoning** call (`expire_old` / `keep_both` / `reject_new`); it deliberately follows the general background model. It carries a docstring saying so, because it is the natural thing to sweep along when "the extractor" is repointed.
> ⚠ **Moves with `codex_extraction_mode`.** Each model is usable only in its own prompt shape: NuExtract3 on the instruct prompt emits garbage; qwen on the template emits unparseable clause-triplets.


Bookmark dispatch retains the `priority` argument for compatibility. In v3, the runtime owns scheduling and the extractor cooperatively yields at chunk boundaries; the argument does not bypass user-activity gating.

The Codex Extractor (workers/codex_extractor.py::extract_codex, plain callable since C7 — a direct call inside the post-flight job for every non-private turn; user_control's bookmark endpoint enqueues it standalone as the "codex_extract" runtime job with priority=True) runs the following pipeline:

**v3 completion repair (2026-09-12):** extraction accepts complete arrays or
explicit facts/triplets envelopes with valid string fields and boolean polarity.
Empty JSON is a successful no-fact result; empty text, malformed output or a
provider truncation raises to the runtime. No partial graph or completion key
commits until every chunk succeeds. Cooperative yields propagate. The old
emotion-object blacklist and equality-of-endpoints filter are removed: “feels
happy” and reflexive relations can be supported facts. Parser and live writer
checks cover these mechanics; this is not a new semantic-quality measurement.

> ⚑ **§4 STEPS 4–6 CHANGED ON 2026-08-25 (G63/G68) — read this before the numbered steps below, which describe the pre-change path.**
>
> **The background extractor is now `NuExtract3-Q8_0`, not `qwen3:4b-instruct`.** Measured over 60 random turns with the maintainer's own blind labels: **63% of stored triplets true vs 15%, and 0 reversals in 20 judged facts against 6.** That is the only intervention in this project's history to move graph correctness. See [PROVENANCE](PROVENANCE.md) 2026-08-24 and [ROADMAP G63](ROADMAP.md#g63).
>
> Four things in §4 are now different:
>
> * **Two prompt SHAPES exist** (`codex_extraction_mode`). `instruct` is the nine-rule prompt in step 5 below — built for a generalist that does not know the task. `template` sends a JSON schema to FILL and nothing else, strips a `</think>` reasoning block, and accepts a `{"facts": […]}` envelope. **Each model is usable only in its own shape**: NuExtract3 on the instruct prompt produces garbage, and qwen on the bare template produces clause-triplets it cannot parse 30% of the time. ⚠ **The v3 default is `template`** (since August 27); historical instruct runs require their original configuration.
> * ⛔ **The relation vocabulary must NEVER be put in the prompt.** Isolated in a blind round: vocabulary-in-prompt scores **22% correct against the bare template's 60%**. Given a list, the model reaches for a listed word when none fits — `MIT/Stanford/CMU --works_at--> Google India`, `authors --cites--> authors`. It is applied **after** extraction by `canonical_relation` instead, at no prompt cost. See [G69](ROADMAP.md#g69).
> * **Chunking is no longer a fixed 550.** It is sized from `serving_window()` minus prompt and output budget, clamped by `codex_extraction_chunk_max` (4096), falling back to 550 **with a warning** when the probe fails. Measured over 71,256 turns: at 550, **only 23% of corpus tokens fit in one chunk** — 91% of *turns* did, but the median turn is 24 tokens. ⚠ The ceiling is unresolved; nothing above ~1,500 tokens has been measured ([G68](ROADMAP.md#g68)).
> * **`codex_extraction_max_tokens` is 3000, not 1200.** At 1200 template mode lost **30 of 60 turns**, and truncation was logged on qwen too. Couples to the timeout: 72s → 180s.
>
> **Step 4b's tagger changed too:** `codex_extraction_ner_tier` now defaults to **`background` (NuNER Zero)**, not the micro-NER — junk entity names **8.7% vs 19.5%** over 60 turns. ⚠ Decided on shape; never judged for truth.
>
> ⚠ **Everything downstream of step 6 is UNCHANGED and still earns its place** — grounding, confidence tiers, canonicalisation, the entity-name gate, merge keys, supersession, the context payload. Verified end-to-end on 2026-08-25: 94 edges written, zero expiries, canonicalisation and merge-key both fired. **The model swap replaced the *shaping* machinery, not the *checking* machinery.**
>
> ⚠ **Every graph number in this document predates that swap** and is being re-measured — [G71](ROADMAP.md#g71) carries the register, [G72](ROADMAP.md#g72) the reseed that supplies the new baseline.

4. **Chunking (roadmap A1).** The turn is split into **sentence/code-line-aware chunks of ≈`settings.chunk_tokens` = 550 tokens** with `settings.chunk_overlap_words` = 50 words carried into each subsequent chunk. `_chunk_text` isolates fenced code blocks from prose (`_split_segments`), breaks each into atomic units that are never split across a chunk boundary — sentences for prose, non-blank lines for code (`_atomic_units`) — and greedy-packs those units up to the token budget; a single unit larger than the budget is hard word-split as a last resort. **C16 (2026-07-29): the packer now counts REAL tokens** (`memory/tokens.py`). The old `words × 1.33` estimate measured 0.53× on fenced Python, so a chunk the packer believed was 550 tokens was really ~1,000 — `CHUNK_TOKENS` meant nothing for exactly the content where chunk size matters most. A no-whitespace blob (minified JS, base64) also never split at all, because the last-resort splitter was word-based; it now falls through to characters. Short turns come back as a single chunk. Each chunk is extracted independently and triplets are concatenated then deduplicated. *(Rationale: a 3–4B extractor's attention dilutes past ~1k tokens; the previous ≈4,511-word windows caused mid-passage entity dropping and self-referential hallucinations like `fastapi uses fastapi`. The 550-token size is shared with the NER-grounding step so both run off one chunking pass. `MAX_EXTRACTION_TOKENS` = 6000 is retained only for import compatibility.)*

4b. **NER grounding — entity confirmation (roadmap A2).** Before the LLM call, each chunk is run through the shared CPU micro-NER (`extract_entities`, §4.6) to produce a confirmed entity list. When non-empty, that list is injected into the extraction prompt as a **CONFIRMED ENTITIES** block instructing the model to use only those as subjects/objects and introduce no new named entities. This splits the cognitive load — the LLM reasons about relations, not entity discovery.

5. **Background-model extraction.** The extractor calls the background model via `get_bg_client()` / `get_bg_model_name()` (mode-selected: dedicated vLLM :8002 Qwen2.5-3B, or shared Ollama; see §10.5). The prompt renders the relation vocabulary as grouped \# category: … headers, imposes **nine** strict rules — ⚠ **corrected 2026-08-23, this said six** — (individual relation words only; canonicalise subjects/objects; property facts use the property relation; the relation must make logical sense; never output a category header as a relation; NEGATION uses the positive word plus `"negated": true`; an optional noun-phrase shape rule; **rule 8 DIRECTION**, added by G59 — which argument goes in `subject`, the `_by` mirror case, and a read-back check; and output only a JSON array). Before G59 the prompt had **two rules numbered 7** whenever the shape rule was enabled, and **no direction instruction at all** while all three of its examples were active SVO, appends a code-specific sub-prompt when Software_&_Tech is in the topic tags, and appends the CONFIRMED ENTITIES block from 4b. Decoding: temperature=0.0, **max_tokens=`codex_extraction_max_tokens` (1,200)**, timeout scaled by G12. **The output SHAPE is constrained at the decoder** (G32/a1): `response_format` carries a JSON schema pinning the triplet array and its required keys, which removes malformed-by-confusion output entirely. `relation` is deliberately left **free text** — see §4.4b. The budget rose from 500 because `finish_reason == "length"` was truncating whole turns into unparseable JSON; a schema guarantees a valid *prefix*, not a complete document, so the constraint does not cover truncation and `bg_client_factory` now warns on it.

6. **Triplet validation.** (i) Markdown-fence strip; (ii) json.JSONDecoder().raw_decode with a regex fallback r'\\\{\\s*"subject"\\s*:\\s*"([^"]+)"\\s*,\\s*"relation"\\s*:\\s*"([^"]+)"\\s*,\\s*"object"\\s*:\\s*"([^"]+)"\\s*\\\}'; (iii) shape filter (all three keys present); (iv) **vocabulary mapping + gap ledger (G32/a1, replaces a silent filter)** — see §4.4b; (v) verb-phrase filter dropping triplets whose object is in \{"blush","laugh","cry","smile","angry","sad","happy","mad"\}; (vi) self-reference filter dropping triplets whose normalised subject equals its object (kills `fastapi uses fastapi`); (vii) **NER grounding → extraction confidence** (`_ground_triplets`, A2+A3) — each triplet's subject (and, for non-property relations, object) is checked against the confirmed entities by normalised-exact or token-subset comparison, and the result sets its confidence: CONF_GROUNDED = 0.9 on pass, CONF_REJECTED = 0.35 on fail (**stored anyway** — retrieval's trust gates keep it out of context until corroborated, and nonuse never declares it false), CONF_UNGROUNDED = 0.7 when NER returned no entities for the chunk (nothing to ground against). Cross-chunk dedup keeps the highest confidence seen per (subject, relation, object). `handle_triplet` stores the value on new edges as extraction_confidence and raises it to the max seen on corroborating re-extraction.

7. **Deduplication.** Keyed on the case-normalised (subject, relation, object) triple, so multi-chunk repeats across overlap windows collapse.

8. **Write + reconcile (A6, v3 repair 2026-09-13).** Canonicalization's relation-separation map prevents inverse/polarity merges; it does not authorize expiry. The narrower opposition candidate map and legacy supersession cues nominate pairs for a source-aware reconciler. Every candidate uses that reconciler or review; none expires deterministically from an opposition label. The model consumes full supplied source within `codex_reconcile_input_tokens` (default8192); over-bound input, missing source, incomplete response or anything other than an exact decision retains both claims for review. The ordinary writer reinforces only the same direction, relation and polarity; sharing endpoints with a different relation does not authorize expiry. As of 2026-09-19, property/single-valued/negative assertions use the shared two-source boundary described in §4.3. Background polarity/opposition pairs become deduplicated `codex_contradiction` proposals, never automatic processing-time expiry. No claim of complete conflict semantics from these controls.


9. **Idempotency.** idempotency_key = sha256("codex:" + batch_id); on a hit the callable returns immediately. The key insert shares the worker's transaction, so a crash before commit leaves no key (retry re-runs) and a crash after commit leaves the key (retry is a no-op) — turning the runtime's at-least-once retries into effectively-once side effects.

### **4.4b Relation vocabulary — mapping, and the gap ledger (G32/a1, 2026-08-04)**

> **⚑ SUPERSEDED 2026-08-13 — THE VOCABULARY IS NOW OPEN (G45). Read this box
> before the three subsections below, which describe the closed-list era.**
>
> The closed list was the defect, not the filter around it. Re-measured on the
> 293-turn store: **1,259 relations destroyed in a single seed**, including true
> facts — `i --didnt_get--> csi`, and `my father --t_get--> first` where an
> apostrophe split `didn't get` into `t get`. Neither branch of the old choice
> was acceptable: dropping kept 32% of what the model produced, forcing the enum
> kept 100% and got ~78% of them wrong.
>
> **`canonical_relation` replaces the gate**, in three tiers: an exact
> vocabulary hit (unchanged, keeps existing relations stable); else
> canonicalisation against relations **the graph already holds** — embed,
> compare, reuse above `codex_relation_canonical_threshold`; else **accept it as
> a new relation**. `ALLOWED_RELATIONS` survives as tier 1 and as a kill-switch
> (`codex_relation_open_vocabulary=False`). `_relation_gloss_cache` now reads
> the open vocabulary, so G34's relation fit does not go inert on relations
> coined after it was written. 111 multi-arm candidates ship as
> `data/relation_seed.json` so a fresh store canonicalises from turn one.
>
> **⚑ DIRECTION IS SETTLED DETERMINISTICALLY, BEFORE SIMILARITY IS CONSULTED — and as of 2026-08-23 that is finally true of POLARITY and ARGUMENT ROLE too. The guard previously checked only a curated antonym map and "exactly one side is passive", which compares True against True for any `is_*`/`is_*` pair and so could never fire on `can`/`cannot` (0.9134), `is_used_by`/`is_used_for` (0.9327) or `results_from`/`results_in` (0.9266). Replaying one seed's 4,012 real merges, the added checks block 1,611 — led by `is` → `in` ×942 — while `has` → `have` ×958 and `is` → `are` ×715 still merge. ⚠ Blocking them did NOT reduce the reversal rate, which is what showed reversals are generated rather than merged in.**
> Embeddings cannot separate a converse from a synonym — they share every
> context word. Measured on the live encoder: `before`/`after` **0.8791**,
> `parent_of`/`child_of` 0.8569, `teaches`/`learns_from` 0.8159, all above the
> 0.86 threshold then in force. Merging `before` into `after` reverses every
> temporal fact in the graph, silently. `_ANTONYM_PAIRS` (converses) and a
> passive-marker check run first; the threshold only chooses between pairs
> already known safe. ⚠ **The live threshold is `0.90`, not the 0.82 this line claimed** (`config.py:521`, corrected 2026-08-23) — and the guard it describes was structurally unable to fire on `is_*`/`is_*` pairs until G45's polarity and argument-role checks were added. Negations stay
> out — A8's `negated` flag owns those, precisely so the vocabulary does not
> double. See TRAPS #26 and [specs/G45_open_relation_vocabulary.md](specs/G45_open_relation_vocabulary.md).

Step (iv) of the pipeline used to be one line — `[t for t in triplets if t["relation"] in
ALLOWED_RELATIONS]` — with no log. Measured over 300 real turns (PROVENANCE, 2026-08-04)
**it was discarding 67.8% of everything the model produced**, and the three most frequent
casualties were `is` (92×), `has` (68×) and `includes` (29×): ordinary English the 197-word
vocabulary does not contain, not near-misses of it. Extraction spent the GPU, looked
healthy, and mostly deleted its own output.

Three things replace it.

**1. Normalisation (`normalize_relation`).** Case, separators and leading helper verbs only —
`Works On` / `works-on` / `works on` → `works_on`, `is located in` → `located_in`. This is
safe because **no two vocabulary entries collide once separators are stripped** (verified
across all 197, and pinned by a test). It is deliberately narrow: the fuller ladder
(lemmatisation, fuzzy matching, token containment, embeddings) was built and measured at
**5.7%** recovery while introducing *direction inversions* — `is_used → uses`,
`is created by → created` — which write a fact backwards. Those tiers are **premature, not
useless**: once the vocabulary is repaired the residual failures genuinely will be
near-misses, and they become worth adding.

**2. The gap ledger (`codex_relation_gaps`).** What still does not map is written as a row
carrying the model's own wording *and* the subject/object, rather than dropped. Two
consequences: the table ranks what the vocabulary is missing — the input the Z2 vocabulary
experiment needs and previously could not obtain, because the evidence was destroyed at the
moment of creation — and a relation added later can be replayed straight into the graph with
**no new LLM call**, since the fact was already extracted. Rows are `pending` until promoted
or dismissed.

**3. A loud drop.** `codex_relation_out_of_vocabulary` is emitted at WARNING on every chunk
that loses a triplet, with the count and a sample. The rate *is* the finding.

**Why the relation is not enum-constrained**, when the shape is: the vocabulary has no way
to say "none of these". Pinning `relation` to the 197 values keeps 100% of triplets, but on
the same 300 turns **~78% of the ones it rescued were wrong** — a few relations act as
attractors for anything inexpressible (`is → is_employed_by`, `is_family_member_of →
is_dating`, `is_sibling_of → is_separated_from`). A dropped fact costs recall; a wrong fact
is retrieved and served as truth. The capability is wired and switchable
(`codex_constrain_relation_enum`, default off) because **the decision belongs to Z2**, with
the ledger's data in hand. Contrast the enums that *are* used — `DOCUMENT|TRANSCRIPT`, the
four reconciliation verdicts, `user|assistant` — all genuinely closed sets where declining is
either impossible or itself a legal answer.

### **4.5 Relation-aware retrieval and enumeration (A4 — replaces MERA)**

*MERA (retrieval/mera.py) was removed in the post-paper A4 rework: it scored −0.21 in the buildup ablation (loose triggers, an LLM tag/relation mapping that often missed the DB's actual tags, and a flat 15-entity payload dump). Its capability — answering category/enumeration and relation-shaped queries from the graph — was re-homed into the primary Codex leg as follows.*

**Relation detection** (`_detect_relations`) maps the prompt onto the controlled relation vocabulary through two LLM-free channels: (1) a **lexical channel** — a relation's own content words appearing in the prompt after crude two-sided stemming ("who inspired X" → inspired_by, "who is X married to" → married_to); multi-word relations require all content words, single-word relations one; and (2) an **embedding channel** — true-cosine similarity between the prompt embedding and once-per-process cached gloss embeddings (both natively unit-norm since C17; A4's re-normalisation workaround is deleted) of all ~200 relations, top-k=5 above a 0.45 floor, for paraphrases ("who is X's wife"). Empirically the joint set is recall-oriented and *deliberately not trusted alone*: neutral prompts also score ~0.69 absolute against random glosses, so detected relations only ever act **jointly** — paired with matched entities (fact surfacing, §4.8) or with explicit enumeration cues (below). Precision comes from the join, never the detector.

**Enumeration** (`_codex_enumeration`) fires only for entity-less prompts when an explicit enumeration cue is present (list, all, who are, what are, every, each, which, name the, tell me about, enumerate) **and** a grounded signal exists: (a) a prompt token (singularised) matching an actual entity tag ("list all the characters" → tag character) — surfacing up to 8 tagged entities' payloads; and/or (b) a detected relation — surfacing up to 15 trust-gated edges as explicit `[Fact: a --rel--> b]` lines. Both channels honor project scope (§4.8) and the A3 trust floor. No LLM call, no separate module — score 1.0, no reinforcement (enumeration edges are not query anchors).

### **4.5b Cluster naming and description — ⛔ UNMEASURED, and one attempt to measure it was invalid**

`_create_cluster_from_members` generates BOTH a `name` and a `description` from the same members, each grounded with `_recurring_member_entities(turns)`. **Neither has a qualified real-cluster aptness rate.**

**V3 source/label repair, 2026-10-09:** generated entity names previously
supplied assignment/merge overlap. An actual SQL opposing control shows a
planted cast causes an otherwise ineligible merge. Cast now comes from a
deterministic cohort of at most ten distinct actual warm/cold originals, using
existing NuNER plus literal mention grounding. Generated labels have no merge
authority. Complete attributed member input includes short/chapter/late text;
unknown roles remain unknown. Existing E4B capacity/finish guards and typed JSON
use160/500-token name/description reserves; incomplete/invalid outputs preserve
previous labels. Accepted labels are topical metadata, not NLI-supported facts.
Worker/replay receipts distinguish admission attempts and failures from model
calls or semantic accuracy. A real two-member NuNER/E4B/SQL path produced two
complete labels; this is path validation, not corpus naming/coherence quality.

⛔ **A 2026-08-27 attempt produced a 62-76% "wrong" rate and is RETRACTED**: the harness fed the namer *five arbitrary consecutive turns* with `recurring_entities=None`, so it asked for the shared subject of turns never established to share one, without the grounding production always supplies. It measured the harness. ⇒ **G74**, and the open question underneath it is whether the defect is the naming prompt or the CLUSTERING — if members are weakly grouped, there may be no right name available. Judge cluster coherence first, then naming conditional on it.

### **4.6 Entity extraction — two models, split by path (A9b, 2026-08-03)**

`retrieval/ner_utils.py::extract_entities(text, embedder, max_chars=None, tier="preflight"|"background")` serves two latency classes. **Pre-flight (`tier="preflight"`, the default) keeps the micro-NER** on the synchronous user-prompt path. **Background (`tier="background"`) uses NuNER Zero** (`settings.background_ner_model`, default `numind/NuNER_Zero`) for post-flight key-term extraction, clustering and, by the current `codex_extraction_ner_tier` default, Codex grounding. The Codex tier is independently configurable; earlier micro-vs-NuNER grounding counts were judged on their own whitelists and do not measure graph truth. NuNER is loaded lazily on its configured device; the runtime idle drain calls inference-guarded `release_background_ner()`. Complete-stack peak and reload latency remain unqualified. It uses ICE's entity labels, excluding `concept` and `object` by default but restoring them for the permissive Codex grounding list. Adjacent word spans are merged before entity cleanup.

The background caller keeps its configured 250-whitespace-word coarse pieces, then windows each piece using the loaded model's **own** word splitter and `config.max_len`, with overlap. This matters because punctuation/code can make 250 whitespace words exceed NuNER's 384 model-word limit; the earlier path silently truncated the tail inside the model. Missing model-token bounds cause a warned micro-NER fallback. This preserves input coverage, not proof that the added entities improve Codex facts or answers. The micro-NER's poor short-prompt recall remains a separate pre-flight activation problem; see [PROVENANCE.md](PROVENANCE.md).

The micro-NER model (classifier/ner_model.py) is a BIO tagger — a 3-layer MLP 384 → 128 → 64 → 3 with two ReLU + Dropout(0.2) blocks — that classifies per-token embeddings (not token IDs) from the same shared embedder — since C17 the slice384 MRL prefix of its native output (bit-identical to the old truncate_dim input) — into B-ENT, I-ENT, O. Each token string is embedded **in isolation** (context-free): the model learns which tokens tend to be entities, but cannot use surrounding words to disambiguate — a known limitation (good recall, noisy boundaries; roadmap A9 tracks a context-aware rework). Operating on embeddings lets it generalise across the embedding space (e.g. recognise a misspelling close to a known entity) at the cost of requiring the embedder at inference. A shared singleton in retrieval/ner_utils.py::extract_entities consolidates both the orchestrator's live-prompt NER and the clustering worker's full-chapter NER onto one loaded model; if the .pt file is missing it falls back — **loudly, at WARNING with the working directory** (the path is CWD-relative, roadmap G31) — to a regex \\b[A-Z][a-zA-Z]\{2,\}\\b minus a stoplist (The, This, User, Assistant, Chapter, …). **Output post-processing (roadmap A2 NER cleanup):** decoded entity spans are snapped to whole-word boundaries (`_snap_to_words`, fixes subword truncation like `Pyd`→`Pydantic`), trimmed of leading/trailing function words (`_clean_entity` / `_EDGE_TRIM`, e.g. `on Pydantic`→`Pydantic`), filtered of pronoun/boilerplate junk (`_NER_STOP`), and deduplicated. Empirically ~95% entity recall on a mixed prose/technical probe set; residual verb-led bleed (`uses PostgreSQL`) and descriptor false-positives (`fire mage`) are absorbed by A2's token-subset grounding and await the context-aware rework. The training pipeline (scripts/ner/*) is offline-only: extract_turns combines simulation + labelled + synthetic prompts; label_entities calls mattbucci/gemma-4-12B-AWQ at SGLang port 8003 to extract verbatim entity strings; generate_bio aligns entities to token offsets (longest-first, no-overwrite) producing BIO labels; train_ner trains with Adam(lr=1e-3), BATCH_SIZE=16, EPOCHS=10, CrossEntropyLoss with inverse-frequency class weights clamped at 100.0, early stopping PATIENCE=3.

### **4.7 Vector fuzzy matching for entity resolution**

The Codex retrieval leg resolves prompt-extracted entities to stored CodexEntity rows in three stages (A4): (1) _match_entities_by_similarity(threshold=0.85) — embeds the prompt entity strings and asks Postgres for the nearest entity above 0.85; the source/project visibility predicate is applied before ranking and LIMIT, so hidden project vectors cannot become anchors or crowd out visible ones (G29, v3 2026-09-23). Each prompt entity matches at most one stored entity and vice versa (seen_ids), with an inline exact canonical_name / aliases.any(norm) fallback per string. (2) The base-class _match_entities_exact (exact/alias only, no vectors) — used when use_fuzzy_match=False (the ablation fuzzy_match flag). (3) **Payload descriptor fallback** _match_entities_by_payload — when name-based matching finds nothing, content words (≥4 chars, minus a small stoplist) from the NER strings are ILIKE-searched inside context_payload, ranked by hit count, top 2 accepted: "main fortress" resolves to The Obsidian Citadel because its payload mentions "fortress". This closes part of the semantic-vs-lexical gap without a schema change (payload *embeddings* remain future work).

### **4.8 Retrieval and event-sourced compaction**

The v3 Codex retrieval leg (`_codex_graph`) extracts prompt entities, matches visible graph nodes, and emits a separate fragment for each matched anchor. It traverses time-valid edges in both directions with direct/deep trust floors and bounded fan-out; negated relations render as facts but are not navigated. The anchor's own relations are ranked for query fit, and an entity-less enumeration path can surface tagged entities or relation facts. Grounded names and aliases can expand the episodic full-text query. A closed conversation/project scope resolves eligible warm and cold source batches and graph endpoints; traversal and fact surfacing require an eligible live source, and only that source's current exact quote and timestamp are rendered and credited. Under `auto`, the graph searches all visible non-private sources except excluded conversations. Retrieval does not strengthen or promote graph claims: separately counted context exposure affects retention ranking, while independent source observations determine corroboration. Legacy edges without a linked current claim remain explicitly labeled unverified. The bounded graph and source checks have implementation controls; the resulting answer quality and per-token benefit are not yet measured on a representative v3 conversation.

The codex_events append log is checkpointed by
`workers/compaction.py::compact_entities` at the existing 100-uncompacted-event
threshold and 24 h runtime cadence. **V3 repair, 2026-10-09:** the old parser
rebuilt only the newest batch from an empty set, omitted actual ID-only
expirations and lost polarity. Version2 JSON now snapshots the existing current
entity/edge projection in a repeatable-read transaction. It records all outgoing
edge identities, polarity, source/trust/temporal metadata, unexpired edge IDs and
entity note/properties/aliases; these are stored graph state, not new proofs.
Only observed pending events are marked compacted atomically with snapshots.
All events, graph rows and old snapshots remain. The journal is preserved;
legacy snapshots are not retrospectively certified. No production retrieval
consumer reads these checkpoints, so this repair provides no measured answer
gain or historical field-level replay. The timeline still reads retained events.



**v3 graph source scope (G76, 2026-09-24).** A conversational edge's first
`source_batch` is its write origin, not its only evidence. Scoped graph reads
admit a current edge when its primary or an independently observed batch is
allowed, including warm and cold originals. The reader chooses a non-private,
non-excluded source within the requested time, prefers a current linked exact
claim, renders that source's quote and recorded time, and credits the same batch
to the fragment. An edge with no available source does not render. Cached entity
notes remain labeled unverified and are suppressed when private edge sources
exist. This protects source visibility and attribution; it does not establish
the graph's answer quality or re-verify old edges.

