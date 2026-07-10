# Implementation Specs (Phase S)

Decision-complete specs for still-open roadmap items, written during the
Fable-5 window (see ROADMAP Phase S / S1) so a weaker executor can implement
them **mechanically** — the thinking pre-paid, only the typing left.

## Why these exist

The roadmap's entries are deliberately *intent-not-spec*: the concrete design
was meant to be discussed in-session at implementation time. That assumed an
equally strong designer would be present. These specs replace that assumption
for items implemented after the window closes.

## The seven-part template (every spec has ALL seven)

```markdown
# <ITEM-ID> — <title>
Assumes decided specs: <list of earlier specs whose decisions this one builds on, or "none">

## 1. Decisions
Settled calls WITH the reasoning. Nothing left as "option A or B" unless it is
genuinely the user's call — and then state the exact question to ask, the
options, and the recommended default.

## 2. Algorithm & data model
Concrete. Pseudocode, SQL, schema DDL, prompt text where it matters. Name the
thresholds and their values.

## 3. Files & integration points
Every file to touch and what the change looks like there. Name the exact
functions/seams (e.g. "extend `_codex_scope_sets`, do NOT add a parallel path").

## 4. Edge cases & failure modes
Enumerated, each with its intended handling. Include the concurrency, legacy-
data, and empty/None cases.

## 5. Validation checklist
The behavioral live-DB test pattern used throughout this cycle: a standalone
`tests/test_<item>.py` that inserts its own rows and deletes them (NEVER
truncate — the dev DB holds real data), stubs LLM calls (real bg-model
behavior pends Z1), and checks named, specific behaviors. List the checks.

## 6. Look-ahead constraints
What later roadmap items need this implementation to preserve (cite them).

## 7. Traps
What NOT to do and why it will be tempting. Include the "obvious simpler
version" and why it was rejected.
```

## Working rules for the spec-writing session(s)

0. **Set the thinking/effort slider to MAXIMUM for the whole day.** This is a
   pure-reasoning workload — exactly where maximum effort pays; there is no
   implementation churn to save budget for. (Not to be confused with
   `/code-review ultra`, which is unrelated.)
1. **Verify the model first.** These specs are the point of the Fable window —
   if the session is not running Fable 5, stop and switch before writing any.
2. **DECIDE EVERYTHING — a spec that defers a decision is a FAILED spec.**
   Every "decide when built", "open question", "revisit later", and
   "options: A or B" inside an item's scope gets decided *in the spec*, with
   reasoning. The only permitted exceptions: (a) decisions that genuinely
   belong to the user (see rule 4), and (b) decisions that *require empirical
   data that does not exist yet* — and then the spec must name the exact
   measurement, where it comes from (usually Z1/FINAL), and the decision rule
   to apply to it ("if over-rejection > X% → do Y, else Z"), so the executor
   still never has to design anything.
3. **World-state grounding — code for the first spec, code+decisions after.**
   Specs are written strictly in priority order, and each spec is grounded in
   (a) the actual current source of everything it touches (read it — the
   audit found stale claims in the roadmap itself more than once, G2) **and**
   (b) the *decided state* of every earlier spec. Code that exists today but
   is scheduled to change by an earlier spec must NOT be treated as ground
   truth (e.g. once the C7 spec settles the Celery question, every later spec
   that touches workers is written against C7's decided world, not today's
   celery_app.py). Declare this in the spec's `Assumes decided specs:` header.
   If a later spec turns out to need an earlier decision changed: STOP, update
   the earlier spec explicitly, note the revision in both — never fork
   reality between specs.
4. **Interaction model: autonomous with BATCHED user checkpoints — not
   continuous hovering.** Settle everything settleable alone. When an item
   contains a genuinely user-owned fork (product-flavored calls — scope
   semantics, UX behavior, what the product should *feel* like — the kind the
   user has always decided: "store both choose at read time", "30-min gap",
   "incognito = store private + read nothing"), ask those as ONE batched
   question set at the START of that spec, then write it fully settled. The
   user should be reachable during the day but expect only a handful of short
   question batches, not a running dialogue.
5. **Priority order** (from S1): FINAL → Track T (T2–T4 — added 2026-07-10, written first in its origin session) → C7 → D1/D2 → E0+E7 → E1/E1b/E8/E9/E10
   → B1 → C4/C9/C10/C11 → G23+C17 → B3 → F10/F14 → C13/C14 → Z1-prep (tuning
   protocol + coverage matrix, added 2026-07-10) → F-track design brief.
   Mechanical G fixes: one line each in `G_mechanical.md`.
6. **One spec, one commit** — a partial pass that runs out of time still
   leaves whole usable specs.
7. **Budget by regret:** FINAL (the experiment redesign) is the single most
   reasoning-heavy item — give it the largest share of the day; its raw
   material is the FINAL section of the roadmap (the reviewer criticisms),
   `experiments/*/results*`, and the H-track items.
8. **Link back:** when a spec lands, edit its roadmap entry to link it
   (`→ spec: docs/specs/<item>.md`).
9. **S1 is done when** every item in the priority list has either a spec or an
   explicit one-line "no spec needed because …" note in the roadmap — AND the
   close-out deliverables have run (see the S1 roadmap entry): roadmap cleanup +
   top-of-file execution flow, USER-REQUIRED consolidation, recovery note.
10. **Specs rot.** If an implemented item changes something a spec assumed,
    updating the spec is part of that item's propagate-on-completion pass —
    and that includes earlier specs revised by later ones (rule 3).
11. **USER-REQUIRED steps are first-class.** Any step only the human can do
    (labeling data, exporting chat logs, creating accounts/API keys, reviewing
    proposals, keeping the machine on for a long run) is written inside the spec
    as a `**USER-REQUIRED:**` block with exact instructions: what to do, roughly
    how long it takes, and what "done" looks like. The S1 close-out consolidates
    every such block into the roadmap-top execution flow (the user's to-do list).
12. **Divergence protocol (for the implementing session).** If the code disagrees
    with a spec's assumptions: STOP implementing that item; re-ground in the
    current source; update the spec (and any later specs whose `Assumes` header
    names it) FIRST; record the divergence + resolution in the roadmap entry;
    only then code. Never implement against a spec known to be stale, and never
    improvise past a mismatch — improvisation by a weaker executor is exactly
    what these specs exist to prevent.
