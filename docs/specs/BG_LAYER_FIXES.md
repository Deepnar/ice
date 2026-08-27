# The background layer — what ships, what gets measured, what waits

**Written 2026-08-26/27, during the background-model session. Executes at the
END of this session, in one batch.**

> **⚑ THE MODEL IS DECIDED: `gemma4:e4b` is the general background model.**
> `NuExtract3-Q8_0` remains the codex extractor ([G63](../ROADMAP.md#g63)).
> Everything below is written **for gemma4:e4b specifically** — see §0.

---

## 0. ⚑ THE COUPLING, STATED BEFORE IT IS CREATED

Tuning the background prompts for `gemma4:e4b` is the same move
[G63](../ROADMAP.md#g63) made for extraction, and it is justified by the same
kind of evidence. **It is also the same trap.**

Measured 2026-08-26, n=70/arm, one changed substring:

| model | hard MUST-PRESERVE | softened |
|---|---|---|
| `gemma4:e4b` | 83% faithful | **96% faithful, 3% fabricated** |
| `qwen3:4b-instruct` | 54% faithful | **34% faithful**, 56% incomplete |

**The same prompt change is a large win for one model and a large loss for the
other.** So after this lands, the background prompts are **coupled to
gemma4:e4b**, exactly as the extraction prompt is coupled to NuExtract3.

⚠ **This is how the 15%-correct graph happened.** `qwen3:4b-instruct` ran for
months on a prompt shape that did not suit it, and nothing said so. A future
session that swaps the background model **without** re-measuring these prompts
will silently degrade every summary, and no metric currently in production will
notice — `summary_coverage` cannot see invention (§2.1).

⇒ **Every prompt this spec changes carries an inline comment naming the model it
was tuned for and the measurement that justified it.** A model swap is a
re-measure, not a config edit.

---

## 1. SHIP — evidence in hand

### 1.1 Soften the must-preserve instruction · `src/workers/post_flight.py`

**What it is now.** `_summary_llm_call` orders:
`MUST-PRESERVE TERMS (every one of these must appear verbatim in your summary)`,
and on a coverage miss `retry_on_coverage_miss` re-asks with the harder form:
`Your previous summary DROPPED these required terms — include each of them
verbatim this time`.

**Why it changes.** A model that cannot ground a term is being ordered to
include it anyway, so it invents context to carry it. Measured on gemma4:e4b,
n=70: fabrication **15.7% → 2.9%**, faithful **82.9% → 95.7%**, incomplete
unchanged (1 → 1), z=2.61.

**⚠ BOTH strings change or neither does.** The retry fired on **38 of 70**
turns; softening only the first instruction leaves the strongest form of the
order in place on more than half of them.

**Expected side effect, and it is not a regression:** coverage falls
0.784 → 0.640, and gate clearances 52 → 32 of 70. That is the metric noticing
the model stopped padding — see §2.1 before reading it as a loss.

### 1.2 Log the silent procedural return · `src/workers/procedural_extractor.py:118`

```python
idempotency_key = job_key("procedural", f"{turn.session_id}:{bucket}")
if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
    return                      # ← no log, at any level
```

Diagnosed only by running at DEBUG and seeing **no output whatsoever**.
Violates CLAUDE.md's standing rule (*a fallback emits at WARNING with the
reason, every time*). **Behaviour does not change — one log line.** Folds into
[G61](../ROADMAP.md#g61).

### 1.3 `keep_alive` / the native endpoint · [G32(a)](../ROADMAP.md#g32)

**The maintainer's standing requirement:** *"WE need to work and manage our own
ai models memory, dont let default ollama do it."*

**Today's evidence.** `ollama ps` during the bake-off: three background models
resident at once, ~14 GB of a 24 GB card, every one `UNTIL: Forever`. ⚠ **Not a
leak** — the host sets `OLLAMA_KEEP_ALIVE=-1` deliberately. The defect is that
**ICE cannot express a different policy**: `/v1/chat/completions` silently drops
`keep_alive`, so ICE inherits whatever the host decided, and a stranger's host
will have decided something else.

**Shape.** Background calls move to Ollama's native endpoint at the one seam
already named for it — `bg_client_factory._NoReasoningCompletions`, whose own
docstring says *"when background calls move to Ollama's native endpoint, this
class becomes the place that translates, and no caller changes."* ICE then sets
`keep_alive` per request, and the maintenance runtime can release after a drain
(`POST /api/generate {"keep_alive": 0}` → `done_reason:"unload"`, already
verified in the G32 audit). ~17 call sites change nothing.

⚠ **Scope check before writing:** the audit also found `response_format:
{"type":"json_object"}` is a **no-op** through the shim (0/8 conformance) while
`json_schema` works (8/8) — `maintenance_agent.py` still sends the former. Fix
it in the same pass or leave it explicitly; do not discover it twice.

---

## 2. MEASURE THIS SESSION — the harness exists

### 2.1 ⚑ The summary trust gate is inverted — decide what replaces it

**Not a tuning question.** `summary_coverage` gates whether a summary REPLACES
the raw turn (`inject_raw`, threshold 0.7). Graded by a judge that passed its
own controls:

| verdict | n | mean coverage | clears the gate |
|---|---|---|---|
| faithful | 29 | 0.803 | 23/29 = 79% |
| **fabricated** | 13 | **0.914** | **13/13 = 100%** |

**Every invented summary cleared; a fifth of the honest ones did not.** The gate
admits invention *preferentially*. ⚠ **Moving the threshold cannot fix it** —
raising it admits more invention, since fabricated summaries score higher.

Coverage is also gameable by length: `lfm2.5:8b` emitted prose at **95% of
source length** and took the highest coverage of all 11 models (0.973).

**✅ RAW TEXT IS NEVER LOST — verified in code 2026-08-27.** `raw_text` is
`nullable=False` and `summary_text` is a separate column; `_choose_representation`
picks per query and says so — *"both forms are stored; NOTHING is permanently
raw or permanently summary."* ⇒ a fabricated summary corrupts **one assembled
prompt**, not the store. That makes this much less severe than first framed.

**⚑ DEFERRED TO THE RESEED — maintainer's call, 2026-08-27, and it is correct.**
Everything measured today is **write-side**: summaries generated and judged.
Nothing measures **how often a summary actually substitutes in a real assembled
prompt**, which is what decides whether this matters at all. On a 180-turn store
that rate cannot be observed.

**The two-question test the reseed answers:**
1. **How often does substitution happen?** A counter at
   `_choose_representation` — summary returned vs raw. This is already
   [G70](../ROADMAP.md#g70)'s read-side instrumentation.
2. **Does it hamper the answer?** The [G66](../ROADMAP.md#g66) ablation, with
   substitution on vs off.

**Then, and only then:**
- **rate LOW** ⇒ delete the substitution. No gate, no recheck, no model call —
  `inject_raw` always true. Costs prompt space and nothing else.
- **rate HIGH and it hurts answers** ⇒ build the safe version: a faithfulness
  recheck (maintainer's suggestion — a scheduled pass, or an async check at
  write time that clears `inject_raw` when the summary fails) plus a better
  substitution rule.
- **rate HIGH and it does NOT hurt** ⇒ leave it, and record that coverage is
  decorative rather than load-bearing.

⚠ **Do not "fix" the gate before those two numbers exist.** A compression term
or a length bound would be a change with no measured frequency behind it — the
[G51](../ROADMAP.md#g51) shape, which destroyed 667 true facts by committing
without measuring.

### 2.2 The conversation fold — the worst number on the board

**n=12 folds/model, judge gate passed (10/12 qwen, 12/12 gemma):**

| model | faithful | incomplete | fabricated |
|---|---|---|---|
| `qwen3:4b-instruct` | **0 · 0%** | 4 | **8 · 67%** |
| `gemma4:e4b` | 3 · **25%** | 5 | 4 · 33% |

**Not one of qwen's twelve folds survived judging.** gemma is twice as good and
still fails three times in four.

⚠ **It compounds by design** — each step re-summarises the previous summary, so
an invention in fold 1 becomes established fact by fold 4. Runs every 2 h.

**Candidate mechanisms, to A/B one at a time:**
(a) fold against the ORIGINAL turns rather than the running summary ·
(b) cap fold depth and restart · (c) apply §1.1's softening here too.
**No evidence yet on which helps. Do not ship a rewrite on plausibility.**

### 2.4 ⛔ RETRACTED — the cluster-naming numbers measured the harness

**Everything in this section was produced with the wrong input** and is kept
only so the mistake is legible. `produce_cluster_name` fed
`_generate_cluster_name` **five arbitrary consecutive turns** with
`recurring_entities=None`. Production calls it from
`_create_cluster_from_members` with **similarity-grouped members** and
`_recurring_member_entities(turns)`, which injects a *"Recurring names detected
across these turns"* hint.

⇒ **Void: the 62-76% wrong rate, the model ordering, and the contradiction A/B
null.** A namer given turns with no shared subject, and no entity grounding, has
no right answer available — so "wrong" was partly guaranteed by construction.

⚠ **This is the first question CLAUDE.md asks** — *did the harness call what the
real path calls?* — and the harness passed a shim object built for the bake-off,
which nobody re-checked when the job graduated from a format test to a quality
test.

*(Original section, now void, follows.)*

### 2.4-void Cluster naming is broken for BOTH models — and n=5 lied about it

**n=21 groups/model, judge gate 0.905 (qwen) / 0.714 (gemma):**

| model | apt | generic | **wrong** |
|---|---|---|---|
| `qwen3:4b-instruct` | 5 · **23.8%** | 0 | **16 · 76%** |
| `gemma4:e4b` | 6 · **28.6%** | 2 | 13 · 62% |

**Roughly two names in three describe subject matter the excerpts are not
about.** Cluster names feed cluster-scoped retrieval, so a wrong name is not
cosmetic — it mis-files the turns under it.

⛔ **RETRACTED: at n=5 this read as `qwen 0.60 vs gemma 0.40`, and it was
reported as "the one job where qwen leads on quality."** At n=21 both collapse
and the ordering reverses. The n=5 figure was noise, and quoting a direction
from it — even hedged — was wrong. See [TRAPS #51](../TRAPS.md) for the sibling
failure (a threshold invented by the tester).

⛔ **THE PROMPT WAS THE OBVIOUS SUSPECT AND IT IS NOT THE CAUSE.** A/B'd
2026-08-27, one substring removed (the prompt says *"NEVER be too general"* and
four lines later *"must be generic enough"*): **0.381 → 0.286 apt, z = 0.65 —
null**, and the identical arm varied by the same 9.5 points between two runs.
⇒ Clean the contradiction up on style grounds, but **do not expect it to move
the number**. The next hypothesis worth testing is that the JOB is mis-specified:
a cluster of 5 arbitrary consecutive turns may have no shared subject to name.

### 2.3 Deferred with reasons — not forgotten

| item | why it waits |
|---|---|
| **reconciler `reject_new`** | 7 of 10 models never emit it (4 uses in 30 chances) ⇒ prompt, not model. But the gold set is **n=9, hand-written** — too thin to ship a change against. Build a real set first. |
| **900-token summary cap** | qwen truncated 17/70; **gemma truncated 0/70**. May not exist for the chosen model. Re-check on gemma before touching. |
| **doc-kind reads documents as transcripts** | gemma scored **1.000** on this job. Likely moot; confirm. |
| **batch summary quality** | store-limited — 180 turns yields 2 batches. Manufacturing more by shrinking the token budget would measure an easier task. **Unblocked by the reseed.** |
| **procedural quality** | ⛔ **not a reseed problem.** Planted swapped patterns were called faithful **6/6**; behavioural patterns are generic enough that one from another session still reads as plausible. Needs a different instrument — check the pattern's own cited message numbers, which the job already records. |
| **reflection quality** | never had a judge built at all. |

---

## 3. Order

1. Finish the in-flight `conv_fold` / `cluster_name` re-run at n≈12/21.
2. A/B one fold mechanism (§2.2) if GPU time allows.
3. Put §2.1 to the maintainer — it is a decision, not a measurement.
4. **Then, in one batch: §1.1, §1.2, §1.3.** Smoke + settings freeze + dynamics
   invariants before and after.
5. Propagate: FEATURE_INVENTORY rows, MODELS.md (the gemma decision + the
   NuExtract3 pull requirement), ICE_Architecture, CLEANUP for the new scripts,
   new ROADMAP items for §2.2 and §2.3.
