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

1. **Verify the model first.** These specs are the point of the Fable window —
   if the session is not running Fable 5, stop and switch before writing any.
2. **Ground every spec in the code, not memory.** Read the item's roadmap
   entry, the completion notes of everything it builds on, and the actual
   current source of every file it touches — the audit found stale claims in
   the roadmap itself more than once (G2); code is authoritative.
3. **Priority order** (from S1): FINAL → C7 → D1/D2 → E0+E7 → E1/E1b/E8/E9/E10
   → B1 → C4/C9/C10/C11 → G23+C17 → B3 → F10/F14 → C13/C14 → F-track design
   brief. Mechanical G fixes: one line each in `G_mechanical.md`.
4. **One spec, one commit** — a partial pass that runs out of time still
   leaves whole usable specs.
5. **Budget by regret:** FINAL (the experiment redesign) is the single most
   reasoning-heavy item — give it the largest share of the day; its raw
   material is the FINAL section of the roadmap (the reviewer criticisms),
   `experiments/*/results*`, and the H-track items.
6. **Link back:** when a spec lands, edit its roadmap entry to link it
   (`→ spec: docs/specs/<item>.md`).
7. **S1 is done when** every item in the priority list has either a spec or an
   explicit one-line "no spec needed because …" note in the roadmap.
8. **Specs rot.** If an implemented item changes something a spec assumed,
   updating the spec is part of that item's propagate-on-completion pass.
