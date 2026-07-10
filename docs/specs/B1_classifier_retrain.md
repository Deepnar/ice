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
