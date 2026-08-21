---
name: ice-git
description: ICE's git rules — commit message format and granularity, the public-repo push policy, the experiment-phase push freeze, never committing personal corpus content, the private-detail-in-a-tracked-doc gate, history-rewrite verification, and the README/paper-link contract. Load BEFORE writing any commit message, before pushing, before removing anything from git history, and before putting a specific personal detail into a tracked doc. The repo is PUBLIC — every push is a publication.
---

# ICE — git, commits, and the public repo

**⚑ THE REPO IS PUBLIC** (`origin` → `github.com/Deepnar/ice`) and has been for a
while. It is not a private backup: **every push is a publication.** Pushing
during normal development is pre-authorized — push at natural points without
asking — but write every commit, message and file knowing it is read as soon as
it lands, with no window to take it back.

## Commit format and granularity are ONE rule

**⚑ SMALL COMMITS *IN THIS FORMAT*.** Many small commits split by *concern* —
never one end-of-session commit; if a message needs bullets for unrelated
changes it should have been several.

```
area: what changed (ITEM)

ITEM — what it does now:
- concrete change: function/file/migration names, measured numbers

Validated N/N (tests/test_x.py: what was checked); regressions green.
Architecture §6.10 updated; roadmap C6 checked.
```

**Subject:** lowercase area prefix (`retrieval:`, `codex:`, `memory:`,
`workers:`, `classifier:`, `budget:`, `roadmap:`, `docs:`, `tooling:`) · what
changed · roadmap id in parens · ≤ ~70 chars.

**Body:** organised by item, dense and concrete, closing with **`Validated`** and
a **docs/roadmap** line.

**The log is the reference — read it before writing one:**

```bash
git log 6bac35d 1a30484 051cea9 5124efb
```

⚠ **A record, not an explanation.** No essays, no rhetorical beats, no argument
structure. Impersonal, in the repository's voice; **never narrate the session**
("the user asked…", "we decided…"). **No AI attribution.**

## Push freeze at the experiment phase

Once **SEMIFINAL (Z1)** or **FINAL** begins, **stop pushing** until the user says
otherwise.

## Never commit personal content

Planning, career notes, conversation corpora, third-party email, credentials.
**Git history is forever the moment you push**, and a rewritten history still
serves old objects by SHA until the host garbage-collects (TRAPS #12 — a support
ticket last time). **Check before the commit, not after the push.**

⚠ A cleanup commit *message* that describes what was removed is itself a
signpost; rewrite messages too, not just diffs.

## ⚑ PRIVATE DETAIL IN A TRACKED DOC — USER-GATED, every time

The tracked doc states the technical claim and stands alone; the specifics go in
`docs/PRIVATE_CONTEXT.md` (gitignored — verify with `git check-ignore` **before**
writing it) under a marker like `[PRIVATE:g43-example]`; the tracked doc points
at the marker.

**Then ask the user to eyeball the exact lines before committing** — not a
summary, the lines. **What belongs where is the user's call, not a judgement to
make for them.**

## History rewrites must rewrite TAGS

`git log` on `main` cannot verify a rewrite. Verification is:

```bash
scripts/git/check_history_clean.sh --clone
```

`--clone` checks what the public actually receives. Wired to `pre-push` by
`install_hooks.sh` — run once per clone. Full story: **TRAPS #12**.

## README and the paper link

- **README.md is a first-class deliverable** — the first thing a visitor reads.
  Update it in the same session as anything that changes what the project *is*
  or how it is run.
- **⚑ README links the VENUE-AGNOSTIC paper, never a venue submission.**
  `experiments/paper/` holds one canonical paper (`ICE_paper_v2.tex`) plus venue
  twins. A twin carries that venue's branding and a dummy DOI, which on a public
  repo reads as *"published there"* — a false claim about work merely submitted.
  When a twin's content is better, **back-port it**; never repoint the link.
- **README and ICE_Architecture follow the same contract** — both describe `main`
  as it is now, both updated in the same session as the change that invalidates
  them. README is the outside view and stays short. A number in both must change
  in both or neither, and must say which snapshot it is (the paper's numbers are
  the `v2-paper-eval` tag, not `main`).
