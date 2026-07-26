# B1 — Classifier retrain: multi-label context head, coding + temporal labels, real methodology

Assumes decided specs: `T_temporal.md` (Temporal_Recall label joins this retrain;
the T2 detector remains the window-resolver), `E_coding_core.md` (coding intent
labels land here, D11), `C7_scheduling.md` (fine-tune/retrain runs are consent-
gated jobs), `FINAL_experiments.md` (two-labeler cloud lane + human-audit pattern
reused for labeling; schema version recorded in run manifests),
`G23_C17_data_longevity.md` if already written — else the C17 half of the roadmap
(train at 1024-dim). Grounded at commit `0dc5d89`: `model.py`
(Linear(384→128)→ReLU→Dropout(0.3)→Linear(128→25)), `classifier.py` (the two
inference prompt templates — with/without context prefix — and the 11/11/3
slicing at lines ~146–152), `schemas.py` (B2 scalar seam already in place),
`data/labeled/` (~25.3k labeled prompts, 708 probes, failure sets),
`scripts/training/{build_training_data,train_classifier,fine_tune,test_classifier}.py`.

**Timing (unchanged from the roadmap):** this runs once F7/F11 give Real_Time
consumers and E2/T2 need their labels — ONE bundled retrain. The spec makes it
mechanical whenever that moment comes.

> **[rev 2026-07-25] — RE-GROUNDING (session paused mid-B1; recorded BEFORE coding per
> README rule 12).** The spec above was grounded at `0dc5d89` (S1, 2026-07-10) and
> predates C7, Track T, E0/E7, D1/D2, E-core, C4/C9, C10/C11, G23/C17 and F10/F14.
> Eight corrections, all verified in code on 2026-07-21/25:
>
> 1. **D4.1's "real usage from `episodic_memory`" is WRONG for today's world.** The
>    store was EMPTIED at the C17 1024 cutover and F10 imported nothing for real.
>    **B1 trains from FILES ON DISK, not the DB** — which is cleaner anyway (no
>    import, no store pollution, no decay interaction). Corpus sources are (i)
>    `data/labeled/labeled_prompts.jsonl`, (ii) fresh online pulls, (iii) the
>    `data/simulation/` exports parsed by F10's adapters, (iv) synth. If the user
>    ever does a real F10 import, episodic becomes an OPTIONAL extra source, not a
>    dependency.
> 2. **The 25k corpus, measured (not guessed):** 25,354 rows = lmsys 5,255 +
>    wildchat 5,257 + sharegpt 5,159 (≈15.7k online breadth) + **personal 7,711** +
>    synthetic 1,972. v1 context-reliance: Zero_Shot 18,844 / Long_Term_Memory 5,891 /
>    Real_Time_Search 619. So B1 does NOT start from a data desert; "the user's chats"
>    in the roadmap **means this corpus**, not an unbuilt personal-history set.
> 3. **The real gaps are six specific label slices, not "not enough data":**
>    `Needs_Memory` inherits 5,891 (healthy) · `Needs_Live_Info` inherits 619 (thin) ·
>    `Temporal_Recall`, `High_Complexity`, `Codebase_Query`, `Code_Change` have **zero**
>    existing positives. Diversity is the ONLINE layer's job (re-pull more, weighted to
>    thin topics) — **not** the personal exports'. The exports' job is (a) real
>    multi-turn CONTEXT rows (the 25k is mostly standalone; D3 wants ≥40% context-
>    prefixed + ≥1k hard-negative pairs), (b) the coding/temporal labels, (c) personal
>    calibration. Narrow-technical personal data is therefore FINE.
> 4. **`Temporal_Recall` gets free weak supervision:** run T2's deterministic detector
>    (`src/retrieval/timescope.py`, shipped) over the whole corpus — every hit is a
>    positive. D7 still holds: labels gate, only the detector sets windows.
> 5. **`High_Complexity` is the cut candidate.** Zero positives, no real-data
>    definition, and its only consumer (the F11/B3 cloud toggle) does not exist yet.
>    Seed it synthetically, but honour §4's rule: <150 real positives ⇒ drop the label
>    rather than train a coin-flip head. Decide explicitly at build time.
> 6. **B1 IS 100% LOCAL — zero cloud calls (user, 2026-07-25, hardened).** D4.2's "two
>    independent LLM labelers" = **two DIFFERENT LOCAL model families** (independence
>    comes from distinct architectures, not from being cloud-hosted). The **third-model
>    tiebreak is ALSO local** (a third distinct family), and rare-label synth is local.
>    Cloud is reserved *entirely* for FINAL ("the final shot") — see FINAL rev
>    2026-07-25. Rationale: the whole ₹5,000 cap should back the paper's judged runs,
>    and labeling volume (25k+ rows × 2 labelers) is exactly the workload that must not
>    be metered. Human review remains the final arbiter (D-USER b/c).
> 6b. **Labeling runs on SGLang (or vLLM), NOT Ollama (user, 2026-07-25).** Ollama is
>    unusably slow for bulk labeling. **SGLang is the recommendation** for this exact
>    job, on two grounds verified 2026-07-25: (i) **RadixAttention prefix caching** —
>    the labeler's system prompt is ~400 lines (the traps/signals rubric) and is
>    IDENTICAL across all 25k rows, i.e. a maximally prefix-heavy workload (reported up
>    to ~6.4× on such workloads; on all-unique prompts the edge over vLLM is minimal —
>    ours is the opposite case); (ii) **first-class constrained decoding** (JSON-schema
>    guided via a compressed FSM, ~3× faster than standard guided decoding), which
>    replaces `instructor`'s retry-on-invalid-JSON loop with schema-guaranteed output.
>    vLLM is an acceptable fallback (the v1 script already used it — precedent:
>    `legacy/promt_labeling/VLLM_label_dataset.py` served `Qwen/Qwen2.5-7B-Instruct-AWQ`
>    at `localhost:8001/v1`, an OpenAI-compatible base_url, so the client code barely
>    changes). **Serve AWQ/GPTQ HF weights, not Ollama GGUF.**
>    **Model picks for 24 GB (grounded 2026-07-25; re-verify availability at build
>    time):** labeler A = **Qwen3.6-27B** (strongest single 24 GB default, ~16 GB at
>    4-bit, best instruction-following); labeler B = a **different family** —
>    Gemma-4-26B-A4B (the registry's long-context champion) — never two Qwen variants,
>    or "agreement" is measuring one model against itself; tiebreak C = a third family
>    (Mistral/DeepSeek-distill class). If throughput dominates, **Qwen3.6-35B-A3B (MoE,
>    ~3B active, ~20 GB at 4-bit)** decodes far faster than a dense 27B — a legitimate
>    swap for labeler A *if* a quality spot-check on ~200 rows holds up. Run labelers
>    **sequentially** (24 GB holds one at a time), each as a full pass over the corpus.
> 7. **Tooling moved (2026-07-21 cleanup, commit `b0d3e5f`).** §2's paths are stale:
>    `scripts/training/*` and `scripts/classifier/promt_*` now live under
>    **`scripts/classifier/legacy/`** (frozen, provenance) and the v2 pipeline is built
>    in **`scripts/classifier/pipeline/`** as named stages —
>    `extract → stitch_icedev → synth → label → build → train → evaluate → promote`.
>    See `scripts/classifier/README.md` for the old→new map. The strong v1 assets to
>    REUSE (not rewrite from scratch): `legacy/promt_labeling/VLLM_label_dataset.py`
>    (source-aware thresholds, 6 immunity traps, signals A–F, reasoning-first) → the
>    base for `label.py`; `legacy/promt_labeling/synthetic_data.py` → the base for
>    `synth.py`; `compare_labeling.py`'s diff logic → the agreement step. The 3 online
>    extractors get rewritten for LARGER, topic-weighted pulls.
> 8. **`stitch_icedev.py` is a new stage (user-requested):** the ice-dev conversation
>    spans 6–7 separate DeepSeek chats; stitch them into ONE chronological mega-
>    conversation. It is a **shared asset with FINAL** (a real long-project memory
>    test), so build it properly once.
>
> Order of work: **schema v2 → the model → the pipeline scripts.** Unchanged by this
> rev: D1's label set, D2's architecture, D3's template discipline, D5's non-regression
> gate, D6's derivation layer, D7, D8's DI3 deletion rule, D9, and every trap in §7.

> **[rev 2026-07-25b] — DIVERGENCES FOUND WHILE BUILDING (recorded per README rule 12).**
> Seven deviations from the text above, each with what forced it. None change a
> decision; they change how a decision is realised.
>
> 1. **Serving backend: SGLang → vLLM.** Rev 6b picked SGLang on the merits (RadixAttention
>    over an identical ~4k rubric; FSM-constrained JSON). The pinned `sglang 0.3.6.post2`
>    cannot import against this environment's Triton (`ImportError: cannot import name
>    'default_cache_dir' from 'triton.runtime.cache'`), and unpinning it would drag the
>    repo's torch stack — a far larger intervention than the spec's own sanctioned fallback.
>    **vLLM 0.22.0 with `--enable-prefix-caching`** runs and keeps the property that
>    mattered (shared-prefix reuse). `serving.py` keeps `--backend sglang` wired; it becomes
>    correct again the day SGLang is upgraded. *The reasoning in 6b was right; the
>    environment just isn't ready for it.*
> 2. **Labeler A's model must be re-picked.** `mattbucci/Qwen3.6-27B-AWQ` (the on-disk
>    requant) fails to load: `ValueError: The input size is not aligned with the quantized
>    weight shape` on `visual.blocks.0.mlp.*` — its **vision tower** is misquantized. This is
>    a property of that repo, not of Qwen3.6-27B. Also: never pass `--quantization`
>    explicitly; community requants disagree about their own format (one ships AWQ, another
>    compressed-tensors) and each config.json already declares it.
> 3. **Templates are VERSIONED, not merely "lifted verbatim".** `templates.py` holds the v1
>    strings frozen AND a v2 pair naming the real v2 categories, and the checkpoint records
>    its `template_version`. Reason: D5's gate must render the OLD model's input the way that
>    model actually saw it, or the comparison is rigged against the baseline. Verbatim
>    extraction alone couldn't express that.
> 4. **The D6 derivation lives in `schema.py`, not on the classifier class.** It is pure label
>    logic; keeping it as a method forced tests to fake a loaded checkpoint (the existing
>    suite called `PyTorchClassifier._finalize_confidence` unbound with `self=None`, which
>    the schema-driven version broke). `finalize_context_scalars(result, schema)` is now
>    testable without torch state, and covers both generations.
> 5. **F10's `parse_jsonl` learned the `{prompt, response, timestamp, conversation_id}` pair
>    shape.** Three of the user's own exports (`claude.jsonl`, `gpt.jsonl`,
>    `simulation_full.jsonl` — ~5k real multi-turn rows, the scarce context layer) use it and
>    were silently yielding zero conversations. Extending the shared adapter beats a private
>    parser in `extract.py`: a real F10 import would hit the same wall. `normalize_file` stays
>    fail-loud on malformed JSON; `extract.py` salvages line-by-line for corpus building only.
> 6. **Hard-negative pairs (D4.4) need no second labeling pass.** Construction: take a row the
>    labelers saw WITH context and judged NOT to need memory, whose text is referentially
>    ambiguous alone; strip the context and the referent is definitionally gone, so the twin
>    needs memory. The flip is *entailed*, not guessed. Referential detection reuses
>    `memory_decision.REFERENTIAL_WORDS` rather than a second private list.
> 7. **Synthetic rows carry no labels.** v1 stamped the intended label onto each generated
>    row; that is self-certification. `synth.py` records `meta.target_label` as provenance and
>    the row goes through the same two-labeler pass as everything else — which matters most
>    for exactly the labels synth exists to seed, since they are the ones near the §4 drop
>    floor. Also: `curated_labels` gained `schema_version` + `corrected_context_labels`
>    (migration `c4d7e91a2b58`) so §6's F9 look-ahead is satisfied now rather than later.

> **[rev 2026-07-25c] — MEASURED RESULTS FROM THE LABELING RUN (these change D4, not just its
> implementation).** Everything below is data, not estimate.
>
> 1. **Labelers, final: A = `Qwen/Qwen3-14B-AWQ` · B = `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
>    · C (tiebreak) = `openai/gpt-oss-20b`.** Three distinct lineages (Qwen / Gemma / OpenAI),
>    all vetted on real rows. Speed decided A: 1.77 vs 1.20 rows/s, ~6.2 h vs ~8.7 h for a full
>    pass, with equivalent agreement — so gpt-oss takes the tiebreak slot, which is a fraction of
>    the corpus. **Two candidate requants are BROKEN and marked in `serving.PROFILES`:**
>    Qwen3.6-27B-AWQ (misquantized vision tower, won't load) and Mistral-Small-3.2-24B-awq-sym
>    (loads, serves, and emits token soup behind schema-valid JSON — the reason
>    `label.is_degenerate` and its abort gate exist).
> 2. **Agreement, measured over two independent labeler pairs (n=184 / n=142):** the three
>    memory signals agree **90.8–98.6%** (`Needs_Memory` 90.8/94.4, `Temporal_Recall` 96.7/98.6,
>    `Needs_Live_Info` 97.3/95.1). **The methodology works.** Topic agrees 78–80%, intent 64–66%
>    — both fuzzy multi-label heads, as expected.
> 3. **D4.2's "disagreement → tiebreak" was costed wrong by two orders of magnitude.** Requiring
>    all three heads to align simultaneously compounds topic/intent noise onto a context head
>    that was never the problem: only **36–38%** of rows settle, so ~24k rows reach the tiebreak,
>    not the "~200–400" the spec assumed. That is a COMPUTE question, not a quality one — a local
>    third pass is ~4 h of GPU. What must stay small is the HUMAN queue, i.e. rows where the
>    tiebreak *also* fails.
> 4. **`High_Complexity` is resolved separately and never blocks settlement** (`SOFT_CTX_LABELS`
>    in `label.py`). It is the weakest-agreeing context label (83–84%) and alone caused ~16% of
>    context disagreements. **Decided on its error asymmetry (user, 2026-07-25), which depends
>    on which of three deployments the user is in — recorded here because it was nowhere:**
>    - **local-only** — routing has a single class; the signal changes nothing.
>    - **cloud-only** — every turn goes to cloud; the signal changes nothing.
>    - **mixed local + cloud** — the signal decides, and a **false positive spends the user's
>      real credits** on a prompt that never needed the strong model. A false negative costs
>      answer quality on one turn.
>    Money is the harder error to undo, so the label resolves conservatively: majority vote when
>    a third labeler exists, and a 1-1 split resolves to ABSENT. This is also the right shape for
>    a threshold consumer — B3 reads `p_complex >= 0.6` and Z1-prep owns that number, so a
>    precise head can be made liberal by lowering the threshold, while a noisy head cannot be
>    made precise by raising it.
> 5. **Prompt cap:** the labeler sees at most 6,000 chars of a prompt. 116 rows overran an 8k
>    context window on the first pass (rubric ~3.2k tokens + context + prompt); code/CJK text
>    tokenizes near 3 chars/token, well under the 4 English suggests.
> 6. **Synth moved AFTER merge+tiebreak.** A gap measured from one pass is provisional in a known
>    direction — agreement keeps the intersection, so a single pass always overstates. Measured
>    from Gemma alone, only `Codebase_Query` was short (221 vs the 300 floor); every other label
>    cleared it, **including `High_Complexity` (9,141)**, so §4's "drop the label" rule does not
>    fire and the cut candidate survives.

> **[rev 2026-07-25d] — "DON'T USE THE SYSTEM TO VALIDATE THE SYSTEM" (user principle, and it
> adds a gate this spec was missing).** The user's objection, recorded because it corrects a
> habit visible throughout this spec: pointing at B4's fine-tune and F9's thumbs as the path
> that fixes label mistakes later. **The shipped model must be good on its own**, and the
> feedback machinery is not the mechanism for getting it there. Verified 2026-07-25 — that
> machinery is triply inert today: `settings.auto_finetune` defaults **False**;
> `MIN_ROWS_TO_PROMOTE = 20`; and `CuratedLabel` has exactly ONE writer (the manual
> tag-override endpoint in `services/scoping.py`) because F9's thumbs UI does not exist.
>
> Three reasons it is the wrong mechanism *for this component specifically*, beyond being unbuilt:
>
> 1. **The held-out split shares the labelers' bias.** train/val/test all come from the same two
>    labelers under the same rubric, and those labelers agree 90–98% on the context signals —
>    i.e. they share a lot, including their mistakes. A held-out split encodes the same
>    misconception, so §5's per-label F1 table can look excellent while the head is confidently
>    wrong exactly where both labelers were. **No quantity of held-out data from the same
>    process can detect this.** Only independently-authored probes can.
> 2. **This classifier's failures are invisible to the user.** A wrong derived-Zero_Shot means
>    retrieval silently does not run and the answer is quietly worse; nothing signals it. A
>    thumbs-down loop therefore under-samples precisely the errors that matter most. Feedback is
>    a fine quality signal for an *answer*; it cannot be the primary one for a *silent gate*.
> 3. **Corrections are drawn only from what the system surfaced**, so regions where it is
>    confidently wrong never enter the curated set. Bootstrapping converges toward the blind
>    spot, not away from it.
>
> **Consequence — a SECOND promotion gate, beside D5's.** A hand-authored adversarial probe set,
> independent of the labelers, stratified over the *valid* intent × context cells (~40–60 cells,
> 5–10 probes each), deliberately targeting: context/no-context twins, memory+live combinations,
> temporal reference without a parseable date, the Codebase_Query vs Code_Change boundary, and
> **Zero_Shot controls that must NOT trigger retrieval**. Strictly EVAL — never trained on (house
> rule). D5 asks "is it worse than before?"; this asks "is it actually right?" — different
> questions, and a model must pass both. The 708 v1 probes in `data/labeled/probes_*.jsonl` are
> reusable as TEXT only; their labels come from the v1 process and carry its bias.
>
> Authorship note: probes drafted by the assistant are independent of Gemma/Qwen (the
> contamination being guarded against) but not of "written by an LLM" — so the intended shape is
> assistant-drafted for matrix coverage, then adversarially edited by the user, who knows their
> own phrasing habits. Weight/threshold sensitivity sweeps (pos-weights, the 0.3 tag threshold)
> belong with Z1-prep's tuning stage.
>
> **TWO PILES OF SYNTHETIC DATA — keep them separate (user, 2026-07-25).** `synth.py` currently
> treats every generated row the same way: strip the intended label, send it through the
> two-labeler pass. That is right for one pile and WRONG for the other.
>
> * **Pile A — bulk filler.** Hundreds of rows a local model generates to pad a thin label.
>   Generation drift is real (ask for `Codebase_Query`, receive a `Code_Change`), so the label
>   must be earned through the normal labeling path. This is what `synth.py` does today; keep it.
> * **Pile B — hand-authored.** A small set written deliberately, one at a time, by the user or
>   the assistant, *for a specific label combination*. **The label ships WITH the prompt and is
>   never sent to the labelers.** Running Pile B through Qwen/Gemma lets two local models
>   overrule a human-authored ground truth, which makes the data worse, not better. Pile B feeds
>   the adversarial gate above and any manual fine-tune. **Pile B does not exist yet** — that is
>   a missing component, not a disagreement. If a label needs more rows than are worth
>   hand-writing, the assistant contributes a labelled batch alongside Pile A's generated batch,
>   and the two are merged only at the end.
>
> **EXISTING HAND-WRITTEN ASSET: `experiments/curation_files/` (user, do not ask them to
> rewrite these).** 58 JSON files = **19 conversations × 3 checkpoints**, named
> `EC-<conv-hash>-TURN<n>.json`; the same hash means the same conversation and `<n>` is the split
> turn, so probes repeat across a conversation's three checkpoints. Raw total 649 probes.
> ⚠ **The user's stated rule — "keep only the largest-suffix file per prefix" — loses data:**
> it yields 220 unique probes and DROPS 32, because the largest checkpoint does not always carry
> the most probes (`EC-cca73c87` has 27 probes at TURN29 but only 5 at TURN61). **Correct rule:
> deduplicate by probe TEXT across each conversation's files → 252 unique probes.** Note these
> were written to test whether the SYSTEM remembered (`user_injected_prompt` +
> `expected_answer`), not whether the classifier labelled correctly — so they are reusable as
> prompt text and as a rich source of genuine `Needs_Memory` positives, with classifier labels
> to be authored by hand (Pile B).

## 1. Decisions

- **D1: label schema v2 — heads become schema-driven, no magic numbers anywhere.**
  `label_schema.json` gains `schema_version: 2` and explicit head definitions:
  - `topic_labels`: unchanged (11).
  - `intent_labels`: +2 coding intents → 13: **`Codebase_Query`** ("where is X /
    how does Y work in this repo" — navigation/comprehension) and
    **`Code_Change`** ("implement/fix/refactor this"). `Troubleshooting` remains
    the error-diagnosis intent (coding or not). No other duplicates of existing
    generic intents — coding *mode* is scope-driven (E2); these labels only steer
    retrieval weights and routing.
  - `context_reliance` → **4 independent sigmoids** (softmax-3 dies):
    **`Needs_Memory`** (was Long_Term_Memory), **`Temporal_Recall`** (memory
    query with a time dimension — as-of/range/evolution flavored),
    **`Needs_Live_Info`** (was Real_Time_Search — now orthogonal, the item's
    original point), **`High_Complexity`** (prompt benefits from the strongest
    available model — this is the "cloud toggle" signal: F11/B3 consume it when
    the user enables cloud, biggest-local otherwise). `Zero_Shot` stops being a
    label — it is the derived state "all reliance sigmoids low".
  Every consumer reads head widths/offsets from the schema (the audit list in §3
  kills each hardcoded 11/22/25).
- **D2: architecture — shared trunk, three heads, 1024-dim input.**
  `Linear(1024→512) → GELU → Dropout(0.2) → Linear(512→256) → GELU →
  Dropout(0.2)` trunk; heads `Linear(256→11)`, `Linear(256→13)`,
  `Linear(256→4)`; all-sigmoid outputs, per-head BCE with per-label pos-weights.
  ~700k params — still minutes to train on the laptop. Input = the full 1024
  Qwen3 embedding (C17's un-truncation; if C17 hasn't landed yet, the trainer
  encodes at 1024 anyway — classifier input never depended on the DB's stored
  vector width).
- **D3: the train/inference mismatch is fixed by construction.** Every training
  row is rendered through the SAME two templates `classifier.py` uses at
  inference (verbatim strings, extracted into a shared
  `src/classifier/templates.py` so they can never drift again): rows sourced
  from real conversations get their actual prior-3-turn context via the
  `_get_context_turns` logic run offline; standalone rows use the no-context
  template. Target mix: **≥40% context-prefixed rows.**
- **D4: labeling methodology (the "genuinely better" requirement, settled):**
  1. **Corpus:** existing 25k labeled rows (text reused, labels NOT — see trap 1)
     + real usage from `episodic_memory` (user prompts + their live context) +
     the probe sets + FINAL's synthetic transcripts once generated.
  2. **Two independent LLM labelers** (two different Ollama-cloud models, FINAL's
     paced lane), each labeling schema-v2 from scratch, blind to each other.
     Agreement → keep. Disagreement → third model tiebreak. Still split → human
     review queue.
  3. **Per-label floors:** ≥300 positives per intent/ctx label. Rare labels get
     targeted sourcing, not fabrication from thin air: `Temporal_Recall`
     positives = T2's deterministic detector run over the real corpus (weak
     supervision) + templated paraphrases of real hits; coding intents = ICE's
     own dev conversations + labeled probe rewrites.
  4. **Context-dependence hard negatives (the payoff pairs):** the same prompt
     text labeled once WITH its context prefix and once WITHOUT, where reliance
     genuinely differs ("so which should I choose then?" is Needs_Memory
     standalone, Zero_Shot-ish when the context already contains the options).
     ≥1k such pairs — this is what makes context-aware classification real.
  5. **Quality gate:** 5% stratified human audit; any label slice with <90%
     user-agreement gets its slice relabeled with a corrected rubric prompt.
- **D5: evaluation + promotion.** 15% stratified held-out; report per-head
  macro-F1 and per-label F1. Promotion reuses B4's machinery (backup + atomic
  replace of `settings.classifier_model_path`) with an added **non-regression
  gate**: on the shared old labels, new-model F1 ≥ old-model F1 on the same
  held-out rows (old model evaluated by mapping its 3-way ctx to the derived
  3-way — §2). If it loses, do NOT promote; diagnose data first (the roadmap's
  "we cannot know whether the retrain lands better" made operational).
- **D6: backward compatibility is a derivation layer, not a parallel path.**
  `ClassificationResult.context_reliance` (string) becomes derived:
  `Long_Term_Memory` if `p_mem` is the max of {p_mem, p_live, 1−max(p_mem,p_live)}
  … concretely: `zero = 1 − max(p_mem, p_live)`; argmax over
  {Zero_Shot: zero, Long_Term_Memory: p_mem, Real_Time_Search: p_live}.
  `p_ltm = p_mem` (B2 reads it unchanged — its designed-for seam),
  `ctx_confidence` = top1−top2 of that derived 3-way. New fields:
  `p_temporal`, `p_complex`. `raw_probs` becomes 28 wide, schema-ordered.
- **D7: T2 merge rule:** `memory_decision` treats `p_temporal ≥ 0.6` as
  equivalent evidence to a detector-fired TimeScope for the retrieval *bump*
  (OR, never AND) — but **only the deterministic detector ever sets a window**;
  the label gates/boosts, it cannot invent dates.
- **D8: DI3's sentence is executed here (roadmap's firmed default).** After
  promotion, run the model vs DI3 on held-out slices matching DI3's fast-path
  categories (noise / code-detect / sentiment / meta). Per slice: model ties or
  wins → that DI3 path is deleted. Expected end-state: `di3.py` deleted, keeping
  only a trivial inline length/noise guard if (and only if) the noise slice is
  the lone DI3 win. `reference_signal` survives as `memory_decision`'s
  REFERENTIAL_WORDS bump only (DI3's anaphora rule dies with it).
- **D9: new intent labels get retrieval-weight rows** in the orchestrator
  PROFILES: `Codebase_Query → {codex 1.3, bm25 0.9, vector 0.8, procedural 0.6}`,
  `Code_Change → {procedural 1.2, codex 1.0, vector 0.6, bm25 0.6}` (starting
  values; Z1-prep's tuning pass owns their final numbers).
- **USER-REQUIRED (rule 11):**
  (a) one-time consent that local chat history is used as local training data
  (yes/no; nothing leaves the machine except two-labeler cloud calls on prompt
  text — flag rows from `none`-scoped conversations are **excluded
  automatically**, no consent can override that);
  (b) review the labeler-disagreement queue (~200–400 rows, ~1–2 h, via the
  existing labeling CLI in `scripts/classifier/promt_labeling` or F9's UI when it
  exists; done = queue empty);
  (c) the 5% audit pass (~30–60 min);
  (d) run the retrain + approve promotion (one command; minutes).
- **Empirical deferral (rule 2b):** trunk width (512) and the 0.3 sigmoid
  tag-threshold — sweep both in Z1-prep's decision-threshold stage against the
  held-out set (3 values each); everything else about training is
  standard-practice mechanical (early stop on held-out loss, AdamW, lr 1e-3).

## 2. Data & pipeline shape

```
scripts/training/build_training_data.py   (rewritten)
  sources → render templates (shared templates.py) → two-labeler pass →
  agreement/tiebreak/queue → floors + pairs synthesis → train/val/test split
  → data/labeled/v2/{train,val,test}.jsonl  (+ provenance per row)
scripts/training/train_classifier.py      (heads/losses/schema-driven widths)
scripts/training/fine_tune.py             (28-wide via schema; B4 promotion + D5 gate)
src/classifier/templates.py               (the two verbatim prompt templates)
```
Row format: `{text, context_text|null, topic[], intent[], ctx{mem,temporal,live,
complex}, source, labeler_agreement}`.

## 3. Files & integration points (the ripple audit — every magic number)

`model.py` (D2 arch) · `classifier.py` (schema-driven slicing; D6 derivations;
templates import; DI3 gate per D8) · `schemas.py` (`p_temporal`, `p_complex`;
comment updates) · `label_schema.json` (v2) · `_head_confidences`
(orchestrator.py:66 — reads `probs[:11]`/`probs[11:22]`: switch to schema
offsets) · `memory_decision.py` (D7 merge; p_ltm unchanged) · orchestrator
PROFILES (D9) · DI3 modules (D8 deletions post-eval) · `fine_tune.py` /
`build_training_data.py` / `train_classifier.py` · FINAL manifest records
`schema_version`. Grep-gate at the end: `grep -rn "\[:11\]\|11:22\|22:\]\|== 25\|
(25)" src/ scripts/` → only schema-driven reads remain.

## 4. Edge cases & failure modes

Old checkpoints unloadable by the new class → `ICEClassifier.load` dispatches on
a `schema_version` saved inside the checkpoint dict (v1 checkpoints keep loading
for the D5 comparison, via the legacy head shape). DI3 fast-path rows with
all-zero raw_probs → convention updated to 28 zeros; `_head_confidences` fallback
unchanged. Incognito rows excluded from the corpus at source (D-USER a). A label
with irreducibly few positives after floors (<150) → drop the label from v2
rather than train a coin-flip head (decision rule; candidates: none expected,
but High_Complexity is the riskiest — its fallback definition: prompts >150
words with multi-step asks, reasoning verbs, or cross-domain span). Cloud tier
down mid-labeling → resumable per-row JSONL (FINAL's D2 pattern).

## 5. Validation checklist

1) Build pipeline dry-run on 200 rows: template-rendered, two-labeler agreement
recorded, pairs present; 2) training runs to early-stop; held-out per-label F1
table produced; 3) D5 non-regression gate computes both models on identical rows;
4) derived `context_reliance`/`p_ltm`/`ctx_confidence` populate; B2's 23/23
`test_memory_decision.py` still green (it consumes scalars); 5) C15 wide-net
trigger works with 28-wide probs (test with a peaked topic + fuzzy intent);
6) T2 label-merge: a `p_temporal=0.8`, detector-silent prompt gets the retrieval
bump but NO window; 7) DI3 slice eval table produced; deletions applied per D8
with the noise guard verified; 8) promotion round-trip: backup written, path
swapped, proxy reload serves the new head (live smoke); 9) grep-gate from §3
clean.

## 6. Look-ahead constraints

B3 consumes `p_complex` + per-head confidences for routing — keep them on
`ClassificationResult`. B5 (ensemble) stays gated on this retrain's results. F9's
feedback UI writes `CuratedLabel` rows in schema v2 (add `schema_version` to that
table's payload now). Z1-prep tunes the 0.3 threshold + PROFILES numbers. FINAL's
gating-failure metric reads the derived 3-way (unchanged interface).

## 7. Traps

- **Don't map old ctx labels into v2** — the single-label 3-way IS the defect;
  relabel from scratch (text reuse yes, label reuse no).
- **Don't let the labelers see each other** (or the old labels) — agreement is
  the quality signal; contamination fakes it.
- **Don't train on un-templated bare prompts** — the mismatch this item exists to
  fix; the templates module is load-bearing, not cosmetic.
- **Don't promote a model that loses the shared-subset gate** "because the new
  labels matter more" — B2/retrieval run on the shared labels today; regressions
  there are user-visible immediately.
- **Don't delete DI3 before the slice eval** — order is promote → measure →
  delete; and don't keep it "just in case" after it loses (two classifiers
  disagreeing is worse than one).
- **Don't let Temporal_Recall set time windows** — labels gate, detectors parse
  (D7); a sigmoid inventing "two years ago" is a hallucinated filter.
