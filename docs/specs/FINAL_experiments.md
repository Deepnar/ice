# FINAL — Full experiment redo (evaluation redesign)

Assumes decided specs: `T_temporal.md` (temporal probes exercise T-track modes; the
`timescope` ablation flag exists). Grounded in: the FINAL/H sections of ROADMAP.md,
`experiments/mature/results/*paper_summary_full.md` (both variants),
`experiments/unmature/results_phase2/paper_summary.md`,
`experiments/flaw_ablation/buildup/paper_summary.md`, at commit `443533a`.

**Execution order:** FINAL runs LAST (after Z1) — this spec is written now so the
design thinking is pre-paid. Nothing here is implemented before Z1 passes.

**What the redo must fix (the four criticisms + two audit additions), stated as
design requirements:**

| # | criticism / gap | design answer (section) |
|---|---|---|
| 1 | Straw-man baseline (no frontier comparison) | `cloud_longctx` (frontier model on the $20/month tier — the exact product the criticism names) + `full_ice_cloud` (the same model WITH ICE) + LongMemEval external anchor (§2.3, §2.8, §2.9) |
| 2 | Terrible ROI (4.26 vs 4.25 excl. ice_dev) | budget-parity conditions + per-dataset CIs — measure the post-rework system honestly; if the delta is still ≈0, the paper says so (§2.6) |
| 3 | Misleading token-efficiency claim (22,411 vs 21,025) | identical context budget for ICE and baseline by construction; efficiency reported as quality-at-equal-budget (§2.6) |
| 4 | Half-baked ablation (1 dataset, weaker judge) | subtraction ablation on ≥2 datasets with the SAME judge as the main runs (§2.7) |
| 5 | Time axis never measured (Track T) | temporal probe class on ≥2 datasets; auto-generated with perfect GT on synthetic (§2.4) |
| 6 | GT errors + untrusted 70% hallucination metric | ledger-based GT by construction (synthetic), citation-verified GT (personal sets), negative probes replace the hallucination % (§2.5) |

> **[rev 2026-07-25] — CLOUD PROVIDER SWAP + PER-CONDITION RUN LAYOUT (user decision;
> supersedes the Ollama-Cloud clauses in D6 and in the 2026-07-10 line below).**
> Recorded BEFORE implementation per README rule 12. Six changes:
>
> 1. **Provider = OpenRouter (paid credits), not Ollama Cloud.** Reasons (user): any
>    model reachable through one API; a **hard credit cap** (~₹5,000 ≈ $60 total)
>    instead of a recurring subscription; no local VRAM contention; and **prompt
>    caching** with OpenRouter *sticky routing* keeps caches warm across calls —
>    decisive for the judge (one rubric × ~4,500 calls) and `cloud_longctx` (the same
>    long history re-sent per probe). D6's "one Ollama Cloud paid month ≈ $20 flat" is
>    **void**; the cost model is metered-per-token now (see 3).
> 2. **Ollama Cloud free tier survives for ONE-OFF calls only** — B1's disagreement
>    tiebreak, synth spot-fills, ad-hoc checks. NOT for FINAL runs: limits are
>    unpublished and revisable, session quota resets ~5 h, weekly caps, **1 concurrent
>    model**, metering is by **GPU-time not tokens**, and a shifting free tier cannot
>    give the pinned, manifest-recorded model a paper requires (reproducibility).
> 3. **Cost reality (grounded 2026-07-25): the budget is NOT the constraint.**
>    DeepSeek-V4-Flash on OpenRouter ≈ **$0.09/M in, $0.18/M out, cache hits $0.0028/M
>    (~98% off)**. Estimated FINAL total ≈ **$5–10** (judge <$1 with the rubric cached;
>    `cloud_longctx` ~$2–5; `full_ice_cloud` ~$1; GT-gen ~$0.5) — ~6× headroom inside
>    the ₹5,000 cap. Consequence: **do not reflexively pick the cheapest judge.** Run
>    D8's calibration gate first; if the cheap judge clears κ, keep it; if not,
>    escalating costs ~$10, not the budget. (Gemini-3.6-Flash at $1.50/$7.50 per M ≈
>    $27 for judging alone — affordable but unjustified unless calibration demands it.)
> 3c. **PIN REVISIONS, NOT NAMES, IN EVERY MANIFEST (2026-07-26).** "exact version strings" in
>    (4) below means the **revision SHA**, for local weights as much as cloud models. A community
>    quantization can be re-uploaded, silently revised, or deleted, so `cyankiwi/gemma-4-26B-A4B-
>    it-AWQ-4bit` is not a reproducible reference — that repo *at revision `0ef577a5…`* is. HF
>    already stores it as the `snapshots/<sha>` directory name, so it is free to capture. The
>    manifest records: repo + revision, quantization method, serving engine + version, and the
>    sampling parameters. Same for the embedder, whose identity silently defines every stored
>    vector. Running ledger with the values captured so far: **[docs/PROVENANCE.md](../PROVENANCE.md)**.
>    Note this does NOT touch scored results — FINAL's judge and GT generation are cloud
>    full-precision — but it does cover the local answerer conditions, where the rule is to serve
>    the SAME model across conditions so quantization stays a constant in the ICE-vs-baseline
>    delta rather than a confound.
> 4. **Judge/longctx model roster stays deferred to run time** (empirical-deferral (a)
>    holds, new provider): pick the strongest instruct model and the strongest
>    long-context model available on OpenRouter *then*; record both + exact version
>    strings in the run manifest.
> 5. **Per-condition run layout (user): `experiments/final/<dataset>/<condition>/`,
>    local and cloud lanes running in PARALLEL.** Exp-2 ran all conditions in one pass;
>    separating them means (a) the GPU never idles while the cloud judges — different
>    resources, (b) one condition's calls run contiguously, which is exactly what
>    maximizes prompt-cache hits + sticky routing, (c) a single condition can be re-run
>    or re-judged without disturbing the rest, (d) each condition carries its own
>    manifest. §2.9's runner architecture and §2.3's "own artifact dir" already lean
>    this way — this makes it the mandated shape, plus a **shared cloud pacer /
>    budget-guard** across lanes (one hourly call budget + a hard credit ceiling).
> 6. **Config seam:** §3's provider-agnostic knobs (`cloud_base_url`,
>    `cloud_api_key_env`, `cloud_hourly_call_budget`, `allow_personal_cloud`) are
>    already sufficient — OpenRouter is an OpenAI-compatible base_url + key. Add
>    `cloud_model_judge`, `cloud_model_longctx`, `cloud_credit_ceiling_usd`.
>    **`allow_personal_cloud` (default False) still gates every personal-memory byte** —
>    the provider swap changes nothing about privacy.
>
> 7. **CLOUD IS EXCLUSIVE TO FINAL (user, hardened 2026-07-25).** Every other phase —
>    B1's labeling AND its tiebreak, NER, synth generation — is **100% local** (see B1
>    rev 2026-07-25). FINAL is "the final shot": the entire ₹5,000 cap backs the judged
>    runs and the frontier baseline, nothing else.
> 8. **DUAL JUDGE — every probe is judged TWICE by two DIFFERENT model families
>    (user, 2026-07-25).** Single-judge scoring carries two documented biases this
>    kills: (i) **self-preference** — a judge systematically favours text from its own
>    family, which is fatal here because `cloud_longctx` and `full_ice_cloud` are
>    *answered by* a cloud model; (ii) idiosyncratic rubric drift. Rules:
>    - **The judge must never share a family with the answerer.** If DeepSeek-V4-Flash
>      answers `cloud_longctx`/`full_ice_cloud`, DeepSeek must NOT be a judge of those
>      conditions. Record the answerer↔judge family matrix in the manifest and assert
>      disjointness at run start (fail loud, don't warn).
>    - **Two judges of different families score every probe** (e.g. a Gemini-class and
>      a Qwen/Llama-class model on OpenRouter — exact roster deferred to run time per
>      empirical-deferral (a)).
>    - **Inter-judge agreement (κ) becomes a REPORTED statistic**, alongside D8's
>      existing anchors (auto-score agreement, κ vs the human-audited set, the 5% audit).
>      D8's calibration gate now applies **per judge**: a judge failing κ is replaced,
>      not averaged in.
>    - **Score of record:** where the two judges agree, that score stands; where they
>      disagree, the auto-score decides if one exists (synth fact/negative probes),
>      else the disagreement is surfaced for the human audit sample and reported as a
>      disagreement rate. **Never silently average two judges into a mean** — that
>      hides exactly the disagreement the second judge was added to expose.
>    - Cost impact is negligible (judging ≈ 2 × <$1 with the rubric cached); this is
>      the single cheapest credibility upgrade available and it directly answers
>      criticism #4 ("weaker judge").
>    - **Same anti-bias logic applies to any other generative role** where one model
>      could grade or generate for itself — notably **evolving-GT generation must not
>      use a judge family** (GT-generator, judge A, judge B = three distinct families
>      where reachable).
>
> Unchanged by this rev: which work is local vs cloud (the roadmap's AI-usage map),
> D8's anchors themselves, budget parity, the synthetic/LME datasets.

> **[rev 2026-07-25b] — EXPERIMENT-DESIGN NOTES CARRIED OVER FROM THE B1 SESSION
> (user, recorded for when FINAL's turn comes; nothing here is built yet).** These came
> out of reviewing why the previous experiments under-measured, plus a costing argument
> about who generates what. Four items:
>
> 1. **Probe diversity has a definition now, and the old probes failed it.** The user's
>    critique of the previous runs: *every probe was retrieval-shaped* ("what did I say
>    about X"), and the four seed conversations weren't diverse either. Two consequences.
>    (a) **Gating failure was unmeasurable.** If every probe needs memory, retrieving is
>    always right, so "retrieved when it shouldn't have" cannot be scored — and that is
>    precisely what B2's posterior and B1's 4-signal head changed. FINAL needs
>    **Zero_Shot control probes that must NOT trigger retrieval**, or the headline
>    improvement is invisible. (b) **Diversity becomes a coverage matrix, not a vibe:**
>    schema v2 gives 13 intents × 4 context signals; enumerate the *valid* cells
>    (Casual_Banter × Needs_Memory is real, Null_Noise × High_Complexity is not), quota
>    each, and audit coverage. This is the same artifact Z1-prep's coverage-matrix stage
>    wants, so build it once. Probe families beyond retrieval: **synthesis** (facts from
>    two distant turns), **temporal** (as-of / evolution — now a first-class label),
>    **absence** ("did I ever mention X" where X never occurred — catches false-positive
>    retrieval), **preference application** (procedural memory), **contradiction/update**
>    (what is current after a revision), and the **Zero_Shot controls**. The seed
>    conversations must span the intent space too, not just the topic space.
>
> 2. **Synthetic conversations: script the ground truth FIRST, then generate the
>    dialogue from it.** The user's objection to two-model roleplay is correct — "the ai
>    in that, it itself might not remember correctly", so a transcript generated freely
>    and *then* read for ground truth inherits every drift the generator introduced. The
>    fix inverts the order: (i) author a structured **fact ledger** — entities,
>    decisions, preferences, each with a timestamp *and a revision history* ("chose
>    Postgres in March → switched to SQLite in May because X"); (ii) generate each turn
>    feeding the generator only the facts legal at that point in the timeline; (iii) GT
>    is then **known by construction**, never recovered by re-reading. **Evolving GT
>    falls out for free** — the revision was scripted, so the correct answer at any point
>    is a lookup, not an inference. Two-model roleplay still helps naturalness (one
>    plays human, one the assistant) but neither model is the source of truth, which
>    also kills the failure mode where the "human" invents a fact the ledger never had.
>
> 2b. **THE USER-ROLE MODEL GETS A DIFFERENT PERSONA PER CONVERSATION (user, 2026-07-26).**
>    In the two-model generation loop, the model playing the *human* must be given a fresh
>    persona each time — age, occupation, temperament, writing habits (an older man who types
>    in full sentences and over-explains; a hurried student who sends three fragments in a row;
>    someone who never capitalises and abbreviates everything). **Reason, in the user's words:
>    it relieves the straw-man problem of single-authored conversations.** A generation loop
>    without this produces N conversations in ONE voice, so the evaluation only ever measures
>    the system against a single writing style, and any result generalises no further than that
>    style. That is the same criticism levelled at the straw-man baseline (§ criticism 1), just
>    pointed at the dataset instead of the comparison.
>    Practical requirements: persona is sampled per conversation and RECORDED in the manifest
>    (so a per-persona score breakdown is possible — if ICE only works for one voice, that must
>    be visible, not averaged away); personas vary **typing behaviour**, not just biography,
>    since the classifier and retrieval see surface form; the persona is given ONLY to the
>    user-role model, never to the assistant-role model or the judge, or the assistant is
>    effectively told the answer. The same reasoning applies to the hand-authored probe sets —
>    prompts written entirely in one register test one register.
> 3. **Who generates what — decided on the user's LIMITS argument, not on quality.**
>    Claude subagents (spawnable on the user's plan, isolated cold context, background,
>    model selectable haiku/sonnet) are genuinely usable for *small, judgment-heavy*
>    work. But: **plan usage is a shared resource with the user's own working sessions.**
>    Their words: if GT or judging eats the daily/weekly limit, "then i cant really work
>    with you right", and hitting the cap mid-run loses the rest of the week for both the
>    experiment *and* development. That is a scheduling constraint no quality argument
>    overrides. Therefore: **probe generation → subagents are plausible** (low volume,
>    high judgment, restartable, no deadline coupling). **GT / evolving GT → OpenRouter
>    credits**, because the 4-conversation evolving/corrected-GT script plus its JSON
>    output is high-volume model usage and must not compete with the user's ability to
>    work. **Judging → OpenRouter credits**, and the user notes the judging script has
>    historically finished in **under ~10 h running LOCALLY**, so on cloud it is neither
>    slow nor expensive — it is a smaller job than its prominence suggests. This
>    *narrows* rev 2026-07-25's "cloud is exclusive to FINAL": the exclusivity holds, and
>    within FINAL the split above is now explicit.
>
> 3b. **`experiments/curation_files/` — hand-written probes, reusable, do NOT ask the user to
>    rewrite them.** 58 files = 19 conversations x 3 checkpoints (`EC-<hash>-TURN<n>.json`; same
>    hash = same conversation, `<n>` = split turn), 649 probes raw. Probes repeat across a
>    conversation's checkpoints, so dedupe **by probe TEXT** -> **252 unique**. Do not dedupe by
>    "keep the largest-suffix file", which drops 32 unique probes (the biggest checkpoint is not
>    always the one with the most probes: EC-cca73c87 has 27 at TURN29 and 5 at TURN61). Each
>    probe carries `user_injected_prompt` + `expected_answer` against a `historical_context_block`
>    — i.e. they already have the shape FINAL's evolving-GT design wants, authored by hand.
> 4. **Shared asset already built (B1, 2026-07-25):** `scripts/classifier/pipeline/
>    stitch_icedev.py` produces `data/labeled/v2/icedev_stitched_dialogue.jsonl` — the
>    six ICE-N DeepSeek chats merged into ONE chronological conversation (3,473 turns,
>    2026-06-04 → 07-03), written in the JSONL dialogue shape `formats.parse_jsonl`
>    already accepts, so FINAL can feed it straight through the F10 import engine. It is
>    a **real** long-project memory test — months of continuous work with genuine
>    callbacks across chat boundaries — and needs no synthetic generation at all.

User decisions (2026-07-10; **cloud-provider + judge-tier clauses SUPERSEDED by the rev
above**): cloud via **Ollama Cloud free tier** (rate-limited, fine —
the runner is resumable by design); **synthetic publishable dataset required** (private
sets can't be replicated by reviewers); **LongMemEval yes** ("would love that", spec
provides the mechanical how); **compute = one laptop, intermittent, ~a week of nights
total, not always on** → per-condition modular runs, probe-level resume everywhere;
**judge = biggest model reachable via Ollama cloud tier** (+ optional Claude lane if
the user gets access — seam only, no dependency); **better ground truth is mandatory**.

---

## 1. Decisions

- **D1: One dataset, one memory build, one DB snapshot — conditions differ only at
  query time.** Per dataset checkpoint, LSREP replay builds the mature DB ONCE; a
  `pg_dump` snapshot is taken; every condition restores the same snapshot before its
  run. Kills replay re-runs (the exp2 week-eater), guarantees identical memory state
  across conditions, and makes any condition re-runnable in isolation (the user's
  "separate scripts per condition" requirement — which is hereby a hard rule: **no
  mega-script**; comparisons are assembled afterward from per-condition artifacts).
- **D2: Probe-level resume is universal.** Every stage (generation, answering,
  judging) appends to a JSONL keyed by `(dataset, condition, probe_id, stage)` and
  skips already-done keys on restart. A laptop that sleeps mid-run loses nothing.
- **D3: Budget parity.** `vector_rag` and `full_ice` receive the SAME total context
  budget per probe (the C16-derived budget for the routed model). The baseline fills
  it with top-k chunks; ICE fills it its way. Efficiency claims become "quality at
  equal budget" + "tokens actually used ≤ budget" — criticism 3 cannot recur.
- **D4: Datasets are scored separately; no pooled headline.** Pooling hid the
  ice_dev distortion (4.27 pooled vs 4.26/4.25 fair). Per-dataset tables with
  bootstrap CIs; the paper's summary table is per-dataset rows, not one number.
  ice_dev is retired to a stress-test appendix (the baseline-collapse story), never
  pooled again.
- **D5: The synthetic set is the headline reproducibility artifact.** Planted-fact
  generation (§2.2) with a machine-readable fact ledger = ground truth by
  construction (zero GT labeling errors — the user's #1 judge complaint), auto-
  generated probes including every temporal class, and the whole thing (generator +
  seed + transcripts + ledger + probes) ships in the repo. Private sets stay as
  ecological-validity evidence; the synthetic set is what reviewers replicate.
- **D6 (FINAL revision 2026-07-10 — user cost-comfort decides): the cloud lane
  is one Ollama Cloud PAID MONTH (~$20 flat), bought when FINAL starts; metered
  APIs are an optional appendix, default off.** Reasoning: for a student, flat
  and predictable beats metered foreign-currency billing — and the design keeps
  the cheaper lane rigorous anyway: the judge is **calibration-gated** (D8:
  κ vs the human-audited set + agreement vs the ledger auto-scorer — judge
  trust is *measured*, never assumed), and the frontier baseline stays credible
  because the paid tier serves frontier-class open models (DeepSeek-V3.1-class,
  100B+). Rhetorically this is exactly right: criticism #1 literally says
  "just pay $20/month for an off-the-shelf product" — that product is now the
  baseline. Scattered earlier cloud needs (B1's two-labeler pass, synth
  generation if not adjacent to the FINAL month) ride the FREE tier with
  resumable pacing; if that proves too slow, a second paid month is a bounded
  +$20 (worst case $40 total, ever). **The Anthropic/OpenAI adapter seam stays
  built and OFF** — if ever funded (~$5–10 suffices), it runs a ~100-probe
  prestige appendix (Claude judge + Claude longctx) with prompt caching
  required; skippable forever without touching any conclusion. Pro/Claude Code
  remains the user's interactive coding lane only — never the harness's, so
  nothing clashes. Personal datasets still require the explicit per-run
  `--allow-personal-cloud` flag (default off) regardless of provider.
- **USER-REQUIRED (rule 11):** (a) subscribe one Ollama Cloud paid month when
  FINAL begins + set a cancel reminder (effectively one-time ~$20); (b)
  laptop-on nights (~a week, fully resumable); (c) the `--allow-personal-cloud`
  decision per run; (d) OPTIONAL and freely skippable: a small Anthropic API
  key (~$5–10) for the prestige appendix (**a Pro subscription is not API
  access** — it stays your coding lane).
- **D7: MoE runs only if it earns its slot.** Pilot gate: 50 stratified probes,
  specialist pool vs generalist; run the full `*_moe` conditions only if the pilot
  shows ≥ +0.15 mean score. Otherwise the redo drops MoE and the paper reports the
  pilot honestly. (Exp1 −0.04, Exp2 −0.02: burning nights on a known-neutral
  condition is the old mistake. B3/F15 precede FINAL in the roadmap; the pool this
  gates on is whatever they produced.)
- **D8: Judge validity is itself measured.** Three anchors: (a) the synthetic set's
  ledger allows EXACT auto-scoring for fact probes — the judge is scored against it
  (agreement = judge calibration, reported); (b) the existing 1,211 human-audited
  Exp2 probes re-judge under the new judge → κ vs human; (c) 5% of new-run probes
  get a human spot-check queue. Gate: κ ≥ 0.6 and synthetic agreement ≥ 0.85, else
  escalate judge model and re-run judging only (cheap — D2 keys by stage).
- **D9: Hallucination% is replaced.** Two measurable substitutes: **negative-probe
  pass rate** (questions about facts that provably never existed — ledger-backed on
  synthetic, authored on personal sets; a pass = the system says it doesn't know)
  and **unsupported-claim rate on the audited 5%** (human-verified). The old global
  judge-guessed hallucination number is never reported again.
- **D10: G19 gate before any run.** `configurable_orchestrator.py` is audited
  against the post-rework orchestrator (or its flags folded into the parent,
  settings-driven — preferred). An ablation harness that silently diverged from the
  real system makes every ablation a lie; this is a hard prerequisite, sequenced
  with Z1.
- **D11: Temporal scoring uses the existing "Temporal Score Quality" hook** but
  becomes a first-class probe class with its own table (as_of / range / evolution /
  which-is-current), not a post-hoc judge dimension.
- **Empirical deferrals (rule 2b):** (a) exact Ollama-cloud model roster at run time
  — pick the largest instruct model available on the tier for judge, and the largest
  long-context model for `cloud_longctx`; record both in the run manifest. (b) MoE
  pool composition — comes from B3/F15; the gate (D7) is the decision rule. (c) If
  LongMemEval's official judge prompts require an OpenAI model unavailable to us,
  the decision rule is: reuse their prompts verbatim on the Ollama-cloud judge and
  report ours as "protocol-adapted" (never silently substitute).

---

## 2. Algorithm & design

### 2.1 Datasets

| id | source | role | cloud-allowed |
|---|---|---|---|
| `flaw` | existing (1,119 turns, creative) | temporal-rich personal set (saga evolution = real evolution probes) | no (default) |
| `masters`, `shinchan` | existing personal | ecological validity | no (default) |
| `synth` | **new generator** (§2.2) | headline reproducible set; perfect GT | yes |
| `lme` | LongMemEval-S subset (§2.8) | external anchor vs published systems | yes |
| `ice_dev` | existing | appendix stress test only | no |

### 2.2 Synthetic dataset generator (`experiments/final/synth/`)

The reproducibility artifact. Design:

1. **Fact ledger first, text second.** A seeded generator produces a persona and a
   ledger of facts with lifecycles: `{fact_id, theme, value_v1, introduced_turn,
   revisions: [(turn, new_value, kind: revised|reversed|abandoned)], never_facts:
   [...decoys never stated...]}`. Themes span the label schema (project/technical,
   preferences, relationships, creative work-in-progress, admin/life) so MoE routing
   and intent gating are actually exercised. Targets: **≥1,200 turns across 8
   conversations, 4–6 simulated months, 60–80 tracked facts, ≥35% revised at least
   once, ≥15 never-facts.** Timestamps are scheduled (sessions on a calendar) so
   T-track sees real temporal structure.
2. **Turn generation with fact injection.** Per scheduled turn, the generator LLM
   (bg model or cloud tier) writes a natural user+assistant exchange that MUST
   express the scheduled fact operations. **Verification pass:** each scheduled fact
   value must appear (string/paraphrase check: value string or its NER-normalized
   form present) in the generated turn; on miss → regenerate (≤3 tries, then
   simplify the sentence template deterministically). The ledger is therefore true
   by construction.
3. **Auto-probes from the ledger** (no human labeling, no GT errors):
   current-value (`what is F now?` → last value), **as_of** (`what was F in
   <month>?` → value valid then), **evolution** (`how did F change?` → ordered value
   chain), **which-is-current** (present old value, ask if still true → must flag
   supersession), **negative** (ask about a never-fact → must say unknown),
   enumeration (all facts in a theme), cross-conversation (facts whose intro and
   revision live in different conversations — H1 finally measured). Probe phrasing:
   template + LLM paraphrase pass (paraphrases keep probe_id → GT mapping).
4. Artifacts shipped: `persona_seed.json`, `ledger.json`, `transcripts/*.jsonl`,
   `probes.jsonl`, and the generator itself. A reviewer reruns everything from the
   repo.

### 2.3 Conditions (each = one `run_condition.py` invocation, own artifact dir)

| condition | what | datasets |
|---|---|---|
| `control` | sliding window only, no memory | synth, flaw |
| `vector_rag` | tuned single-leg vector baseline at ICE's budget (D3) | all |
| `full_ice` | post-rework ICE via public `retrieve()` | all |
| `full_ice_moe` | + registry routing | D7 pilot gate first |
| `cloud_longctx` | the cloud lane's frontier model, full-history stuffing (as much as fits its window, newest-first) — the "just pay $20/month" baseline | synth, lme (+personal only with `--allow-personal-cloud`) |
| `full_ice_cloud` | **ICE's retrieval/assembly (identical to `full_ice`) with the SAME cloud model answering** — the memory-system head-to-head at equal answerer strength: ICE's ~20k curated context vs the stuffed window, same brain (new condition, user 2026-07-10; also pre-validates F11's cloud-models-in-ICE path) | synth, lme |

Answer model for local conditions: the same generalist for all (whatever Z1
promoted; record in manifest). MoE uses the registry. One answer per probe
(temperature 0.2, seeded) — the old runs' repetition-cleaning scripts existed
because of duplicate answering; don't recreate that problem.

### 2.4 Probe taxonomy & quotas (per dataset, authored where not auto)

factual-current 30% · temporal (as_of/range/evolution/which-is-current, evenly) 25% ·
anaphoric/ambiguous phrasing (H4 — human-authored, incl. the messy real style) 15% ·
cross-conversation (H1) 10% · enumeration/aggregation 10% · negative probes 10%.
Personal sets: existing probes are reused where they fit the taxonomy; temporal and
negative probes are newly authored (~40/dataset — flaw's saga evolution provides
ready evolution material). `lme` keeps its own question set/types unchanged.

### 2.5 Ground truth & judging

- **GT:** synth = ledger (exact). Personal sets = regenerated GT with **citation
  requirement**: every GT claim carries source turn ids; an automated pass verifies
  the cited turns contain the claim's key strings (NER-normalized); failures go to a
  fix queue before any judging. (This is the "SOO many errors" fix — GT that can't
  cite its source doesn't enter the benchmark.)
- **Judge (final, per D6):** primary = the **strongest instruct model on the
  Ollama paid month that passes D8's calibration gates** (try largest first —
  DeepSeek-V3.1-class), temperature 0, fixed rubric prompt (1–5 +
  per-dimension flags: correctness-vs-GT, era-correctness for temporal probes,
  unsupported-claims list). The 5% human-spot-check stratum is the standing
  second opinion; the optional Claude appendix (D6) adds a third if ever
  funded. Fact/negative probes on synth are ALSO auto-scored from the ledger
  (exact/normalized match) — the judge's agreement with auto-scores is the
  calibration statistic (D8), gating whichever judge runs: **a judge that
  fails the gates gets swapped for a bigger one, so the cheap lane can never
  silently cost rigor.**
- **Score of record:** auto-score where it exists (synth fact/negative), judge
  elsewhere; human spot-check 5% stratified (existing `manual_evaluate.py` flow).

### 2.6 Metrics & statistics

Per (dataset, condition): mean score, 95% bootstrap CI (10k resamples), paired
win/loss/tie vs `vector_rag` (same probe), tokens-used distribution (honest full-
prompt counting both sides — exp1's corrected protocol, kept), SPF, negative-probe
pass rate, temporal-class table, judge-agreement stats. Significance: paired
bootstrap on score deltas; report the CI, not stars. **No pooled cross-dataset
headline (D4).** Longitudinal curves (score vs checkpoint) kept from exp2.

### 2.7 Ablation (subtraction, redesigned)

From `full_ice`, subtract one at a time: `rrf`, `chunk_retrieval (C2/C3)`,
`cluster_restrict (C5)`, `codex (A-track)`, `recency (C8)`, `timescope (T)`,
`dynamic_budget (C15/C16)`, `keyword_boost`. Datasets: **synth (full probe set,
auto-GT) + flaw (40-probe subset)** — two natures, same judge as the main runs
(criticism 4 dead). Runner = same `run_condition.py` with `--ablate <flag>` (the
post-G19 flag mechanism). Budget: 8 flags × ~490 probes ≈ 2–3 nights.

### 2.8 LongMemEval adapter (`experiments/final/lme/`) — the mechanical "how"

1. Fetch the LongMemEval dataset (HF hub; pin the revision in the manifest). Use the
   **S** variant, stratified subset of ~150 instances covering all question types
   (incl. its temporal-reasoning and knowledge-update types — they map 1:1 onto our
   temporal classes).
2. Per instance: create a fresh conversation set, **replay its haystack sessions
   through the LSREP/F10 ingestion path with the dataset's session timestamps** (ICE
   lives through them: post-flight, codex, clustering — same as any import), then ask
   the question through the normal `/v1/chat/completions` path.
3. Score with the benchmark's official protocol (its judge prompts, adapted per
   D-deferral c); report per-question-type accuracy next to the published numbers of
   long-context/Mem0/Zep-class systems from the LongMemEval results table (cite; we
   compare against *published* numbers, we don't rerun competitors).
4. `vector_rag` and `cloud_longctx` run the same instances (baseline + frontier
   anchor on neutral, public data — criticism 1's clean kill).

### 2.9 Runner architecture (`experiments/final/`)

```
run_replay.py     --dataset X [--checkpoint N]      # build memory; pg_dump snapshot per checkpoint
run_condition.py  --dataset X --condition C [--ablate F] [--resume]   # restore snapshot, answer probes → answers.jsonl
run_judge.py      --dataset X --condition C [--resume]                # judge/auto-score → scores.jsonl
run_metrics.py    --dataset X                                          # tables + CIs → metrics.json
run_report.py                                                          # cross-dataset paper summary .md (per-dataset rows)
```
Every script: idempotent resume (D2), a `manifest.json` (git commit, model ids,
settings hash, dataset hash, timestamps), structured progress logging. Cloud client:
shared adapter with pacing/backoff (D6). Nothing imports experiment code into `src/`
or vice versa beyond the public API + configurable orchestrator.

### 2.10 Sizing (fits "a week of nights, laptop, intermittent")

Main runs: ~1,450 probes (synth 400, flaw 250, masters 200, shinchan 200, lme 150,
control extras 250) × ~4 local condition-passes ≈ 5,000 local generations ≈ 40–55
GPU-hours → 5–6 nights. Ablation ≈ +2–3 nights. Judging rides the cloud lane
(paced, resumable, zero GPU). **Cost (plan of record): one Ollama Cloud paid month ≈ $20 flat** — covers
judging (~4,500 calls), `cloud_longctx`, AND `full_ice_cloud` (~550 ICE-context
calls); worst case +$20 if B1's labeling can't ride the free tier. Optional
Claude prestige appendix: ~$5–10 of API credit for a ~100-probe subset. (The
metered-API estimate — $30–60 — stays recorded for comparison; the flat lane
won on predictability, and the calibration gates keep it equally rigorous.) Replay happens once per dataset (snapshots reused).
If the week overruns: the priority drop order is fixed — ablation flaw-subset first,
then `control` on flaw, then lme to 100 instances. Never drop: synth main runs,
budget parity, temporal probes, judge calibration.

---

## 3. Files & integration points

All new code under `experiments/final/` (+ `experiments/final/synth/`, `lme/`).
Reuse, don't rewrite: LSREP replay machinery (unmature/mature runners),
`manual_evaluate.py` flow, `generate_paper_summary.py` table style. `src/` changes:
**none** beyond what earlier roadmap items already landed — the experiment consumes
the public API and the (post-G19) configurable orchestrator. Settings additions:
`cloud_base_url`, `cloud_api_key_env`, `cloud_hourly_call_budget`,
`allow_personal_cloud` (default False).

## 4. Edge cases & failure modes

- **Laptop sleep / power loss mid-stage:** JSONL append + skip-done keys; a killed
  run resumes with zero loss. pg_restore is idempotent per snapshot.
- **Cloud 429 / tier exhaustion:** backoff + hourly pacer; judge lane can trail the
  answer lane by days without blocking anything (stages are decoupled).
- **Judge calibration gate fails (κ < 0.6):** escalate judge model, re-run
  `run_judge.py` only; answers are never regenerated for a judge problem.
- **Synthetic generation drift** (fact not expressed): verification-regenerate loop,
  deterministic template fallback at 3 misses — the ledger is never silently wrong.
- **LongMemEval instance too big for replay time:** per-instance turn cap with the
  instance id logged as truncated; excluded from headline lme numbers if truncated.
- **MoE pilot inconclusive (|Δ| < 0.15):** MoE dropped, pilot reported (D7) — the
  neutral result is a finding, not a failure.
- **Snapshot/restore version skew:** manifests carry the git commit + alembic head;
  `run_condition.py` refuses to run against a snapshot from a different alembic head.
- **Probe leakage:** probe generation reads transcripts only (never retrieval
  internals); paraphrase pass keeps ids; synth probes derive from the ledger, not
  from generated text.

## 5. Validation checklist (of the experiment machinery itself, before real runs)

Dry-run gate on a 10-turn mini-dataset committed as a fixture:
1. replay → snapshot → restore round-trip leaves identical row counts.
2. `run_condition.py --resume` after a mid-run kill produces no duplicate/missing
   probe ids.
3. Budget parity: baseline and ICE prompt token counts within ±2% of the same budget.
4. Honest token counting: recomputed from the actual assembled messages, both
   conditions, matches the logged numbers.
5. Ledger auto-scorer: hand-checked on 20 synth probes (incl. as_of and negative).
6. Judge rubric returns parseable JSON on 20 probes; calibration stats compute.
7. Ablation flag `--ablate timescope` produces byte-identical retrieval to
   `full_ice` for a non-temporal probe and different retrieval for a temporal one.
8. lme adapter: one instance end-to-end (replay → question → official-protocol score).
9. Manifest completeness: every artifact dir carries commit/models/settings/dataset
   hashes.
10. `run_report.py` regenerates all paper tables from artifacts alone (no live DB).

## 6. Look-ahead constraints

- **H1/H2/H4** are partially discharged here (cross-conversation probes, a second
  "user" via the synthetic persona, realism quotas) — update Track H notes on
  completion. **H3** (year-scale) explicitly NOT covered: the synthetic set is
  months-scale; note it as future work in the paper.
- **B3/F15:** the MoE pilot consumes their pool; keep `find_best_model` untouched by
  experiment code.
- **F10:** SHIPPED first (2026-07-20, commit `316e11f`) — the ingestion engine is
  live at `src/ingestion/importer.py::import_conversations(db, runtime, source,
  policy, *, run_id, classifier, embedder, llm, deadline, progress_cb)`, consuming
  an iterable of `NormalizedConversation`. FINAL's lme/synth adapters are consumers
  #2/#3: write them as new functions in `src/ingestion/` that yield
  `NormalizedConversation`s (dataset session timestamps as the turn `ts`, so the
  replay's sessions come out gap-derived), then call `import_conversations` — do
  NOT rebuild the replay loop, decay handling, or idempotency here. `fast_forward`
  is a settled policy string; `formats.py` is the pattern for a new adapter.
  Original note: the lme/synth replay adapters are F10's ingestion path exercised
  at scale — anything built here lands as reusable F10 machinery, not
  experiment-only code.
- **G23:** SHIPPED first (2026-07-19) — call `src/memory/backup.py::snapshot_db(out_path)` for per-checkpoint snapshots (pg_dump -Fc, host-else-container); do NOT rebuild the wrapper here. Original note: pg_dump snapshotting built here is G23's backup mechanism in embryo —
  share the wrapper.
- **Paper corrections block (roadmap intro):** the NER and corroboration-trap
  corrections must land in the paper text alongside the new results.

## 7. Traps

- **Don't pool datasets for headlines** — the single most damaging distortion last
  time (ice_dev's 1.23 baseline made 0.01 look like 0.4).
- **Don't re-run replay per condition** — snapshot/restore is the whole efficiency
  design; a condition that "just replays quickly" will eat the week.
- **Don't let the judge see condition labels** or fragment provenance — answers are
  judged blind, shuffled, with probe_id only.
- **Don't hand-fix GT in place** — every GT edit goes through the citation-verified
  regeneration path or it reintroduces the silent-error class.
- **Don't implement any of this before Z1** — FINAL measures the finished system;
  measuring mid-rework produces numbers that are stale before they're written.
- **Don't quietly swap judge/answer models mid-run** — manifest mismatch = new
  condition dir, full stop.
- **Don't expand the condition matrix** ("just one more variant") — the drop-order
  in §2.10 exists because the week WILL overrun; additions come out of that budget.
