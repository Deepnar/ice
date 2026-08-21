---
name: ice-session-closeout
description: How to END an ICE session — when it is (and is not) time to write docs/HANDOFF.md, and the full propagation checklist that must be done with it (ROADMAP boxes, PROVENANCE entries, TRAPS entries, FEATURE_INVENTORY rows, ICE_Architecture sections, MODELS.md, CLEANUP ledger, README). Load when asked to close out, wrap up, hand off, switch sessions, or when context is running out. Also load before deciding to STOP work early — a convenient stopping point is not a reason to write a handoff.
---

# ICE — ending a session

## Write docs/HANDOFF.md LAST, and only for two reasons

One file, overwritten in place, committed as the session's **last** commit. It
carries the datetime, what the *previous* session was told to do next (so the
next session can compare intent against the git log), current position, the
decisions made, and anything that would otherwise be lost.

**It is state, never a queue** — the roadmap is the queue, and a handoff that
starts listing work becomes a second source of truth that drifts. Format and
rules are in the file itself.

**⚑ There are exactly two things that make it time:**

1. **The work the session was given is DONE**, or
2. **The context is genuinely running out** — and in that second case it is not a
   shortcut: write the handoff *and* do the full end-of-session job with it,
   propagation included.

**A convenient stopping point is NOT one of the two.** Writing it early turns an
unfinished session into a finished-looking one: the next session reads the
handoff, trusts the position it states, and the outstanding items silently
become nobody's.

**If work is outstanding and context is not the reason, say so in chat and let
the user decide** — never narrate a stop into HANDOFF.md as though it were an
ending. Stating what is unfinished, plainly and specifically, is worth more to
the next session than a tidy summary of what is not.

## The propagation checklist — every row, before the handoff commits

An item with no home means **the propagation is unfinished**, not that it can
stay in the session file. Tick or write "n/a".

- [ ] **`docs/HANDOFF.md`** — rewritten: told→did, state of the world, NEXT
- [ ] **`docs/ROADMAP.md`** — items checked off; new items opened; superseded
      claims corrected **in place, above the stale text**
- [ ] **`docs/PROVENANCE.md`** — every run that produced an artifact, with its
      numbers **and what they do NOT show**
- [ ] **`docs/TRAPS.md`** — each mistake that is a *shape*, not a one-off
- [ ] **`docs/FEATURE_INVENTORY.md`** — anything added, removed or re-gated,
      incl. its default and whether it is ON
- [ ] **`docs/ICE_Architecture.md`** — the section for any subsystem whose
      behaviour changed
- [ ] **`docs/MODELS.md`** — any model added, retired or re-assigned to a job
- [ ] **`docs/CLEANUP.md`** — every move/rename; one-off scripts kept, never
      deleted
- [ ] **`README.md`** — only if what the project *is* or how it runs changed
- [ ] **`CLAUDE.md`** — ⚑ **USER-GATED.** Propose, show the exact lines, wait.
- [ ] `git status` clean · smoke tests green · handoff committed **last**

## docs/SESSION.md

Gitignored, one session, emptied back to its skeleton at the end. Mined into the
handoff first. It carries nothing between sessions — **if it matters tomorrow it
belongs in a tracked doc.**

⚠ **A number that lives only in `docs/SESSION.md`, `logs/`, or
`experiments/curation_files/` is LOST** — all three are gitignored, and `/tmp` is
cleared on reboot. Copy any number anyone might cite into a tracked doc, normally
`PROVENANCE.md`, in the same session. See **TRAPS #40**.
