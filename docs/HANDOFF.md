# Handoff — ICE v3, 2026-10-17 (b) (Claude Code)

**State, not a queue.** ROADMAP owns work; the v2 paper remains frozen.

## Told → did

**Previous handoff (2026-10-17) told:** G77 — get the maintainer's yes/no on its five
features, build them — then build Z2 in spec order (S0 → S12).

**This session did G77 end to end, then — by maintainer redirect — the remaining pre-Z2 items;
Z2's build itself is the next session** (maintainer, 2026-10-17: "pre-Z2 work here, Z2 prep next
session"). 12 commits `86173ae..HEAD`, **not pushed** (experiment-phase freeze).

- **G77 (no human wall), all five as recommended** plus `review.undo` (REST/MCP) and the move of
  reflection's unresolved items from the GLOBAL `pending_items` slot to the conversation's own:
  reflection writes only conversation-tier `conversation_focus` / `pending_items`; stale-slot pure
  removals apply themselves; motif proposals retired; tier-2 coding-decision conflicts decided by
  D6's two questions; stale-work notices `applied`.
- **G77 slot qualification** (PROVENANCE 2026-10-17): the first real probe showed the claim checks
  prove a line true, not that it belongs (~9/20 fit — TRAPS 147). Prompt rounds 31% → 63% → 73%,
  a fourth worse; an "still pending? / closed later?" open-item check (fail-closed) took open items
  70% → 91%. Held-out, full path, Muse-graded: 38/41; an independent blind Haiku check: 38/38 with
  11/12 planted controls rejected.
- **Pre-Z2:** G12 (last fixed timeout), G78 (re-embed by padded tokens — proven two-sided on an
  `ice_db` copy), G32 closed (its last open line obsolete since CODEX_V4), `test_ingestion`
  procedural fixture (36/36). Test rot from the CODEX_V4 port fixed: `test_maintenance_agent`
  (crashed at HEAD), 3 replay tests (stale "codex" tag), `test_z1_replay_maintenance` (9 jobs, not 10).

## Position

G77, G12, G32, G78 closed (ROADMAP 100 checked / 62 open). **Next: build Z2 in spec order
(S0 → S12)**, [Z2_EVALUATION](specs/Z2_EVALUATION.md), with the A6/A7 notes in ROADMAP CURRENT
POSITION. S0 owns the speed work that G4 (whole-GPU budget) and G12's load-scaling half moved into;
G27 option (a) stays Z2's. The spec now audits the G77 artifact kinds (D9), says nine periodic jobs
(D8), and records the B/C exception (D2).

## Decided — do not re-derive

- G77 per feature (ROADMAP G77 "DECIDED"): reflection never writes a global slot; stale slots
  apply removals only; motifs retired; decisions by D6's framing; stale work is a notice. Still
  gated: `forget_request`, `finetune_proposal`.
- B and C may be used for the G77 slot-prompt qualification (maintainer) — so Z2's slot numbers are
  partly on data that prompt was tuned on; recorded in the Z2 spec D2.
- Fix and measure now rather than leave it to Z2 when it can be done (maintainer): Z2 is for small
  fixes, not whole components.
- Slot judging rubric: judge what a line is ABOUT, not its wording (a wording rule failed the
  act-form the prompt asks for). Muse rubric-2 agreement with blind author grades 91%.
- Earlier decisions in the previous handoff (codex facts, gemma4:e4b, no yield in the claim pass,
  free-tier-first Z2 models, private scan at every commit) stand.

## The finding that outlives the items

A check that guards a write must say WHAT it guards. NLI on a quote guards truth; it said nothing
about whether a line was a focus or still open, and a perfect-looking claim pass wrote noise into
every later prompt. Instruction wording did not fix it; one question about what must be true did.

## Pending / known gaps

- `test_v3_chronological_maintenance` failed once in a full run and passed 3× after — an
  unreproduced flake (PROVENANCE 2026-10-17).
- Reflection's slots are sparse by design (22/56 held-out cases write nothing); focus judging is
  slightly lenient (Muse passed 10/118 author-misfits) — Z2 D9 measures both at scale.
- D6's two questions were never qualified on coding decisions — Z2 D9.
- D7's usage third unexercised by read-only replays; unframed pastes stay "asserted" (TRAPS 139) —
  both Z2.
- The Z1 toolchain still imports deleted code (CLEANUP ledger; runs at `8079eb2`), unchanged.

## Validation

Full disposable rails **1,350 passed, 0 failed** at the docs commit before this handoff. During the session: quick group
532 passed; G77 suites green (`test_g77_self_applied` 6/6 and `test_g77_reflection_slots` 4/4, all
failing on pre-G77 code), `test_maintenance_agent` 34/34, `test_coding_core` 54/54, C10/C11 55/55,
replay + z1 files 14/14, `test_ingestion` 36/36.
