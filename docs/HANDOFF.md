# Handoff — 2026-08-17, ~21:00 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Overwritten every session,
committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

> **⚑ THE SESSION'S STRUCTURAL FINDING, AND IT COST FOUR FIXES TO LEARN:**
> **"the subsystem produces output" and "the output reaches the prompt" are
> different claims.** Confirmed twice, independently, in one day. Read
> [TRAPS #39](TRAPS.md) before crediting or blaming any retrieval leg.

---

## TOLD → DID

**Told:** run the full 372-probe re-count, then G50's auto-apply policy, then
regenerate probes, then re-seed.

**Did:** all of it except the re-seed, plus a great deal that was not asked for
because the re-count kept exposing broken instruments. **G50 is implemented and
merged**, not just specced. The probe set is rebuilt and larger than before the
loss. Four candidate fixes to the NER/grounding path were tested and **all four
rejected on evidence** — that thread closed with nothing shipped, correctly. The
re-seed did **not** run; it is the next substantial piece and is unblocked.

## ⚑ START HERE — THE STATE OF THE WORLD

| | |
|---|---|
| **Store** | arm 1 (`fixed-qwen3-4b-instruct`) live — 8,280 entities / 9,662 edges / 293 turns / 61 procedural / **3 batch summaries (new)**. All 8,280 entities now carry `properties.merge_key`. |
| **Git** | Clean. **Local only from `07fc689` onward** — the Z1 experiment-phase freeze is being observed. 12 commits this session. |
| **Tests** | smoke 183/183 · `test_codex_write_path` 32/32 · `test_maintenance_agent` 45/45 · `test_retrieval` 3/3 · **new:** `test_g50_difference_kind` 27/27, `test_g50_write_time_tier` 8/8. |
| **Cloud API** | Quota **reset**, usable. Judge stays pinned to `deepseek-v4-flash`. |
| **New serving path** | **CoE campus gateway** — Qwen3.6-35B-A3B, free, `COE_*` in `.env`. See [MODELS.md](MODELS.md). |
| **Probe set** | **444** (was 372). `codex_multihop` 41 → **104**, anchored **12 → 75**; `temporal` 19 → **28**. |

## 1. WHAT SHIPPED

| item | commit | what it does now |
|---|---|---|
| **[G50](ROADMAP.md#g50)** — 3 parts | `e951a4d` `77b129f` `89c1b57` | `difference_kind()` types every pair · write-time `merge_key` tier stops duplicates being minted · typed rejection drops referent-changing pairs at detection with journalled memos |
| migration | `a1c4e7b90d22` | 8,280 rows backfilled, partial index, **0 key mismatches** in 3,000 sampled |
| scorer honesty | `094b989` | an **unanchored** codex probe scores on turn coverage, not as an anchor failure; populations reported separately |
| batch summaries | `764683b` | `max_tokens` 500 → `batch_summary_max_tokens` 1200; 2 of 3 summaries had been truncated **mid-sentence** |
| [MODELS.md](MODELS.md) | `094b989` `6e7f8e4` | every model in one place — 28 local, the cloud endpoint, the CoE gateway, and the vLLM/AWQ record |
| TRAPS **#37 #38 #39** | `d9eeeb8` `e951a4d` | presence metrics · undeclared `.env` keys · produce-vs-arrive |

## 2. ⚑ THE NUMBERS, AND WHICH ONES MEAN ANYTHING

Typed score, 444 probes, arm 1 store:

| type | early | **late** | reading |
|---|---|---|---|
| `codex_multihop` | 0.317 | **0.532** | **real** — anchor via graph **53/75 = 70.7%** |
| `episodic_lookup` | 0.767 | 0.772 | stable; 0.591 → 0.767 earlier was the **ruler**, not ICE |
| `procedural` | 1.000 | 1.000 | **a tautology** — pool of one, see TRAPS #37 |
| `summary_synthesis` | 0.324 | **0.322** | **did not move after fixing its producer** — see §3 |
| `temporal` | 0.211 | 0.214 | timeline provenance still unwired |

⚠ **No number here reflects G50 or the six older extraction fixes.** They are all
invisible until the re-seed. Do not read this table as their effect.

## 3. ⚑ THE FINDING THAT SHOULD DRIVE THE NEXT SESSION

`batch_summarize()` was never broken. `seed_store` calls it; the call failed
**once, transiently**, inside a `try/except`, and `batch_summaries` stayed 0.
Running it drained the backlog — 89 turns summarised, 164 `lossless_flag=True`
and never summarised **by design**.

**And `summary_synthesis` still scores 0.322.** The leg works — called directly
it returns **2 fragments at cosine 0.52 / 0.48**. They are eliminated downstream,
where `leg_budget_share` gives **episodic 95–99% of every request**.

⇒ **Producing, retrieving, and arriving are three different things.** The
procedural leg is the same shape from the other side. This is a **budget/leg-weight**
question, currently frozen under [G48](ROADMAP.md#g48), and it is why
`summary_synthesis` cannot be improved by fixing anything upstream.

## 4. WHAT WAS TESTED AND REJECTED — do not re-walk these

| candidate | verdict |
|---|---|
| drop `_ground_triplets`' superset arm | demotes **24.6%** of grounded entities incl. `project timeline`, `9.65 cgpa` |
| require the span to occur in the source | rejects **0 of 6** fragments — the model *copied* them from the source |
| extraction shape-rule prompt | superset 0.0858 → 0.1166, triplets **−16%**. Flag `codex_extraction_entity_shape_rule` left **False** as the record |
| swap MicroNER → NuNER | junk does not propagate: **8** indefensible NER entries vs **1,390** produced terms, exactly **1** reached a triplet. A9b stands, **A9c stays parked** |

**Fragments are ~0.4% of extracted terms**, not the epidemic the store's 5,616
containment pairs suggested — they are *unique* while real entities *repeat*.

## 5. WHAT A SEED RUN DOES NOT EXERCISE

Real: `evaluate_turn` (density, grounded summary, chunking, codex, procedural),
real `resolve_session_id`, real bg model, turns in **sittings**.

Absent or thin: `context_clusters` **2** · `conversation_summaries` **0** ·
`session_summaries` **0** · `decisions` **0**. Decay, reflection and the
maintenance agent **never run during a seed**.

⇒ **A seeded store is not a lived-in store.** A zero from one of those tables is
**unmeasured, not negative**.

## 6. THREE TIMES A HARNESS MEASURED SOMETHING ADJACENT TO THE SYSTEM

Same shape, three times in one session, each caught only after the run it ruined:

1. **Probe sampling** — `ORDER BY id` drew 25 of 30 turns from one conversation;
   a `length BETWEEN 300 AND 2000` cap then excluded **238 of 293 turns (81%)**
   unevenly, keeping 2 of 87 from one conversation.
2. **G50's spec numbers** — a hand-written query omitted `source='conversation'`
   and `entity_type` equality; the real candidate set is **1,793**, not 2,247.
3. **Batch-summary eligibility** — "130 turns eligible" omitted `lossless_flag`;
   the true remainder was **3**.

⇒ **Call what production calls, or you are measuring an adjacent system and
reporting it under this one's name.**

## 7. HOUSEKEEPING

* **`docs/SESSION.md`** — new, **gitignored**, a live per-session task+findings
  file. Mined into this handoff and **emptied**; it never carries state between
  sessions. Standing-rule wording for CLAUDE.md is drafted but **NOT applied** —
  CLAUDE.md edits are user-gated and this one was never approved.
* **A `.env` key without a `Settings` declaration takes the whole app down**
  (`extra_forbidden`). It happened again this session — TRAPS #38.
* ⚠ **`PROBE_API_KEY` was printed into a session transcript** by a bad redaction
  pattern. Rotate it.
* Six one-off measurement probes kept under `scripts/oneoff/` — they are the
  evidence for TRAPS #37 and the G50/G51 numbers. Logged in [CLEANUP.md](CLEANUP.md).

## NEXT — IN THIS ORDER

1. **Two-arm re-seed.** qwen3 + the six older extraction fixes + G50's write-time
   tier in **both** arms; the only variable is the NER — **arm A MicroNER, arm B
   NuNER**. Settles A9b at 293 turns instead of the 10 it rests on. ~50 min/arm.
   Needs one small gated change: tier selection in `extract_entities`.
   ⚠ `_background_labels()` drops `concept`/`object` — tuned for clustering,
   never for codex grounding, so arm B tests NuNER **as configured**.
   ⚠ Run `batch_summarize()` afterwards **and assert the row count**, or
   `summary_synthesis` is unscoreable in both arms again.
2. **Answer the probes** — `answer_probes.py`, per arm. Answering model pinned to
   `gemma4:26b-a4b-it-q4_K_M`: A12's top-ranked and **neither arm under test**.
3. **Re-score and re-judge.** Judge stays `deepseek-v4-flash` for comparability.
   The honest comparison is **new-pipeline vs old-pipeline qwen3** on the same
   probes; arm 1's snapshot is the "before".
4. **[G51](ROADMAP.md#g51)** — containment linking, 5,487 missing edges. Re-count
   on the re-seeded store first; its `junk` verdict is a minor sub-case, not the
   load-bearing part.
5. **The summary-budget question** (§3) — needs G48's leg-weight freeze lifted.
6. **Wire timeline provenance** — 207 fragments still uncredited.
7. **Finish the ledger** — only `evidence` and `recent_turns` are evictable.

**⚑ Do not write this file, or close a session, without the user saying so.**
