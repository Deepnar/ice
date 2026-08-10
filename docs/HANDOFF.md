# Handoff — 2026-08-10, 18:20 IST

**State, not a queue.** [ROADMAP.md](ROADMAP.md) is the queue and the only one.
This file exists for the one thing no other doc holds: **what the last session
was told to do, against what it actually did.** Everything else here is a
pointer — if a doc already records it, it does not get repeated. Overwritten
every session, committed last; earlier ones are in `git log -p docs/HANDOFF.md`.

---

## TOLD → DID

**Told:** `NEXT: G36, THEN G30, then G28.` G36 first because G30's job is
trustworthy retrieval tests, and a leg that fails silently makes a passing test
meaningless. G28 stays inside Z1.

**Did:** G36 done · G37 opened and closed on top of it · then a documentation
restructure. **G30 not started, G28 not touched — the order held.** Scope grew
three times, each with an explicit go-ahead: fix the fail-open direction rather
than only log it; close G37; restructure the docs.

## WHERE THINGS STAND

→ [ROADMAP.md](ROADMAP.md)'s **CURRENT POSITION** block. It is the live one; this
file does not keep a second copy.

Two facts it does not carry:

- **Alembic head `d5c81a37e9b2`, unchanged** — no migration this cycle.
- **Store: 1 row** — the deterministic `ice://mcp-notes` shell. Production state,
  not residue; do not "clean" it (TRAPS #15).

## NEXT

**G30, then G28.** G30 inherits two things it should not re-derive:
`tests/test_retrieval_failopen.py`, the first suite seeding codex entities with
**real** embeddings instead of the `[0.05]*1024` stub — the exact blind spot
[G30](ROADMAP.md#g30)'s own entry names — and the `retrieval_leg_failed` event,
which is what makes a green retrieval test mean anything.

## WHAT WOULD OTHERWISE BE LOST

Everything this session decided or found is written where it belongs. Read these
rather than re-deriving them:

| What | Where |
|---|---|
| G36/G37 — the decisions, what the entries got wrong, validation | their entries in [ROADMAP.md](ROADMAP.md) |
| Why Experiment 2 pulled in other conversations; Exp 1's void HyDE arm | [PROVENANCE.md](PROVENANCE.md) |
| The two new failure modes (#15, #16) | [TRAPS.md](TRAPS.md) |
| What moved in the docs, and the public-repo verification | [CLEANUP.md](CLEANUP.md) |
| How retrieval reports a failed leg, and the fail-closed scope rules | [ICE_Architecture.md](ICE_Architecture.md) §6.10, §6.12 |

**The one rule worth repeating, because the next scope change will hit it:**
failing closed whenever a batch set is empty would take the codex leg out of
**every `auto` request**, and nothing in the suite would notice. The guard asks
whether the scope *named* conversations, not whether the result is non-empty.
Three CONTROL checks pin it; they stay green when the fix is reverted.

## OPEN, AND ONLY HERE

- **The ~12 commits from 2026-08-09/10 are off-format** and are being left that
  way — rewriting would invalidate ~20 hashes cited across five docs on a public
  remote. Fix forward.
- **Four environment facts have no permanent home** and have been re-typed into
  every handoff, which is the inflation this file is supposed to avoid. They
  belong in CLAUDE.md; ask before moving them there.
  - `uv run` re-syncs and **removes** anything installed with `uv pip install` —
    use `.venv/bin/python` for scratch work needing extra deps.
  - `nltk` cannot import from the repo root — run text analysis from `/tmp`.
  - `OLLAMA_KEEP_ALIVE=-1`: nothing leaves VRAM. `ollama stop <model>` before
    GPU-heavy tests.
  - `== None` / `== False` in SQLAlchemy filters are **load-bearing** (IS NULL /
    IS false); ruff's E711/E712 "fixes" would break the query.

## WHEN DONE

Propagate on completion per the roadmap's rules, update the docs the change
invalidates in the **same** session, then rewrite this file — carrying the NEXT
above into TOLD → DID — and commit it last.
