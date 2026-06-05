# Session Workflow

## At the Start of Every Session
1. Use read_file to read `PROGRESS.md` completely. This is mandatory — do not skip it.
2. Read only the ARCHITECTURE.md section relevant to the current task.
3. Confirm in 2-3 sentences: what was last completed, what the current step is, what this session will do.
4. If something is ambiguous, ask ONE focused question before starting. Do not ask multiple questions at once.
5. Do not begin implementation until the user confirms the plan.

## During Implementation
- Build one thing at a time. Verify it works before starting the next thing.
- After each working incremental change, commit:
  `git add -p && git commit -m "step: [phase]-[step] [short description]"`
- If the task requires diverging from BLUEPRINT.md (different tool, file name, or approach), record the deviation in PROGRESS.md under "Deviations from Blueprint" before proceeding.
- If a decision would violate an invariant in ARCHITECTURE.md, stop immediately. State which invariant and why, then wait for instruction.
- Do not create files outside the established project structure without asking first.
- Do not modify `.env`, `pyproject.toml`, `docker/docker-compose.yml`, or Alembic config files without explicit instruction.

## When to Update PROGRESS.md
Update PROGRESS.md at the end of a session, or when the user types `/update`.
Never update PROGRESS.md mid-task while implementation is in progress.

## How to Update PROGRESS.md
STEP 1 — Gather information:
- execute_command: `git diff HEAD`
- execute_command: `git log --oneline -5`
- read_file every source file created or modified this session.
- read_file the current PROGRESS.md.

STEP 2 — Update each section following these exact rules:

**Current Phase & Step** — DELETE previous content. Write the exact phase number and step from BLUEPRINT.md we are on right now. If the actual state has diverged from BLUEPRINT.md, describe the actual state plainly.

**Last Completed** — DELETE previous content. Write the single most recently finished unit of work. Name the exact file(s) and what they do. One to three bullet points maximum.

**File Inventory** — APPEND ONLY. Never edit or delete existing entries. For each new or significantly modified file this session, add:
`- path/to/file.py — what it does — what it does NOT handle yet`

**Deviations from Blueprint** — APPEND ONLY. If this session diverged from BLUEPRINT.md in any way, add:
`- [YYYY-MM-DD] [what changed] — Reason: [why]`
If no deviation, add nothing.

**Active Blockers** — DELETE previous content. List only currently-true blockers. Write "None" if there are none.

**Next Step** — DELETE previous content. Write one sentence: the single most specific next action (file to create, function to write, test to add).

Only modify PROGRESS.md. Do not touch any other file.

STEP 3 — After updating, tell the user:
"PROGRESS.md updated. Suggested commit: `git add PROGRESS.md && git commit -m 'docs: update progress'`"

## When BLUEPRINT.md Diverges from Reality
BLUEPRINT.md is a planning guide, not a strict contract. These always override it:
- ARCHITECTURE.md invariants (hard rules — never violate)
- The actual project state in PROGRESS.md
- The tooling established in 01-tooling.md (uv, vLLM, etc.)
- Explicit instructions from the user in this session

If BLUEPRINT.md specifies `pip install`, use `uv add` instead and note the deviation.
If BLUEPRINT.md specifies a file path that conflicts with the existing project structure, follow the existing structure and note the deviation.
If BLUEPRINT.md specifies Ollama for inference, use vLLM and note the deviation.
Document every such divergence in PROGRESS.md.

## Git Conventions
- Commit format: `feat: [what]` / `fix: [what]` / `step: [X.Y] [what]` / `docs: [what]`
- Commit after every working incremental change — not at the end of a session.
- Never commit broken or partially-working code.
- Use `git diff --staged` to review before committing.

## Critical Architectural Constraints (must be obeyed in all code)
These are distilled from the full architecture. Violating any of them breaks the system.

- **INV‑1 — Raw text is write‑once.** The `raw_text` column in `episodic_memory` is never updated after initial insert.
- **INV‑2 — The LLM never directly writes to the Codex or Procedural store.** All mutations go through the Codex Extractor, Procedural Extractor, or Reflection Worker. The model’s output is input to extraction pipelines, never a direct write.
- **INV‑3 — All retrieval passes through the classifier gate.** No memory store is queried without topic/intent/context‑reliance tags from the pre‑flight classifier.
- **INV‑4 — Only currently‑valid Codex edges participate in retrieval.** Edges with `valid_until IS NOT NULL` are excluded from retrieval queries.
- **INV‑5 — Background workers yield to active inference.** Every Celery task must check GPU utilisation before starting; threshold is 20% (configurable). Do not remove or weaken this check.
- **INV‑6 — Idempotency is enforced at the worker boundary.** Use content‑derived idempotency keys and check the `idempotency_keys` table before processing any event.
- **INV‑7 — All Codex mutations are transactional.** Snapshot creation and compaction markers are written in a single database transaction; partial writes are rolled back.
- **INV‑8 — The conversation scope filter is never widened by the retrieval engine.** If a conversation is scoped to a specific cluster set, the retrieval engine may only narrow that set (by classifier tags), never expand it.
- **INV‑9 — Pre‑flight classification is stateless.** The classifier receives only the current prompt; no conversation history, no VRAM state, no retrieval results.
- **INV‑11 — Memory slots are always injected at session start.** Active memory slots are included in every prompt payload regardless of classifier output. They are not retrieval targets.
- **INV‑12 — Bookmarked turns are immune to decay and archival.** They are permanently lossless and are never moved to cold storage.
- **The Asymmetrical Value Problem:** Memory valuation cannot occur pre‑flight. The `lossless_flag` must be set by the Post‑Flight Evaluator **after** the response is seen. Never set it during pre‑flight classification.
- **Pre‑flight / Post‑flight split:** Pre‑flight is synchronous and stateless; post‑flight is asynchronous, has the full exchange, and sets lossless flag + summary + corrected tags. Keep these phases completely separate.

## Deferred Systems (do not implement unless explicitly told)
- Conversation branching UI (the `parent_message_id` column exists but is not used yet)
- Multi‑user / team memory
- P2P mesh sync
- Full custom frontend
- LLM‑based classifier (production path is the PyTorch MLP)
- Cloud routing