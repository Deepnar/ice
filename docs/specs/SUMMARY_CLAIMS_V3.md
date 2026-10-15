# Summary claims v3 — quoted pointers, per-claim acceptance, pieces

**Version: v3.** Approved by the maintainer 2026-10-15 (Z2 S0 pre-replay fixes).
Assumes decided specs: `V3_POST_RESEED_REPAIR.md` (the 2026-10-08 cited-assertion
contract, `source_window_assertions_v2`), `G32_ollama_transport.md`.

## 1. Why (measured 2026-10-15, history C and B)

- **All-or-nothing acceptance discards good claims.** Real verifier on 86 summary
  pieces of C's largest turns: 688/1,074 claims (64%) supported, 4 contradicted —
  but only **2/86 pieces** passed, because one unknown claim rejects the rest.
- **Loose pointers.** 113 claims (10.5%) cited so many windows that the cited
  source exceeded the verifier's 512-token bound; 106 (10%) scored below 0.5,
  largely wrong-window citations. One attribution error ("the assistant noted" a
  user-written fact). Only 85 (8%) sat just under the 0.95 bar.
- **Unenforced JSON.** 8/24 summaries of 12 short B turns were unparseable.
- **Oversized input.** 11/325 C turns exceed 32,768 tokens in the turn summariser;
  the fold never splits a single turn, so one such turn fails the conversation
  every cycle (C's fold would freeze near turn 37).
- **Output cap.** 900 tokens truncated 21/23 ~4K-token pieces; complete outputs
  needed 0.32 (median) to 0.49 (max) output tokens per input token.

## 2. Decisions

**S1 — Claim shape (v3).** Each claim is
`{"sources": [1–2 window ids], "quote": "<exact text copied from a cited window>",
"speaker": "user|assistant|document", "text": "<one complete claim>"}`, in that
field order (pointer first, claim last). The abstract keeps the same shape, at
most one claim.

**S2 — Mechanical checks, per claim.** A claim is dropped (counted, reason kept)
when: a source id is not in the piece; the quote, whitespace-collapsed, is not a
substring of a cited window's decoded text; or the speaker differs from that
window's role. A bad claim never rejects its siblings.

**S3 — Enforced format.** The schema (window ids as an enum, 1–2 sources) is sent
as `response_format` → Ollama `format`. A server that refuses it is logged and
retried unconstrained (existing factory path).

**S4 — Per-claim acceptance.** Every remaining claim is verified as before (NLI
against its cited windows, threshold unchanged at 0.95). The stored record keeps
**only supported claims**; the stored text is exactly those claims; dropped
claims are listed in the record (`dropped`, outside the proof digest). A record
with zero supported claims is `unknown` (reader uses raw). Readers re-check the
kept claims exactly as before. Method `source_window_assertions_v3`; v2/v1
records stay readable under their own policy.

**S5 — Pieces.** Windows are grouped into pieces of at most
`summary_piece_tokens` (4,096) and summarised one call per piece; claims are
merged in order onto the one turn row (maintainer decision (a): no new table).
A multi-piece turn gets no abstract (a one-line abstract of a piece would claim
to describe the whole turn). The fold splits an oversized single-turn group the
same way; entity notes split their source windows the same way.

**S6 — Output cap scales with input.** `cap = clamp(ceil(summary_output_ratio ×
piece input tokens), floor, summary_output_ceiling)`, where the floor is the
job's existing setting (900) so small inputs keep today's cap. The ratio is set
from the qualification run (§3), not guessed.

**S7 — Switch.** `summary_assertion_method` selects v2 (legacy) or v3. Pieces and
the scaled cap apply to both. The default flips to v3 only after §3 passes.

## 3. Qualification (before the default flips)

Same ~30 pieces (C's large turns, mid and small turns, plus short B turns), v2 vs
v3, native door, GPU otherwise idle. Report per method: unparseable outputs;
claims dropped by each mechanical check; claims supported / unknown /
contradicted; pieces with ≥1 kept claim; output tokens per input token
(sets S6's ratio at max × 1.25); wall time. A maintainer-read sample of kept and
dropped v3 claims against their quotes. Pass: fewer unparseable outputs, a higher
share of supported claims, and no rise in contradicted claims.

### 3.1 Result — 2026-10-15, passed; default flipped to v3

31 pieces (C large/mid/small, B short), native door, `gemma4:e4b`, cap 4,096.
Private units: `logs/z2-s0-summary-v3-qualification-2026-10-15-r1/`.

| | v2 | v3 |
|---|---:|---:|
| pieces usable by the reader | 9 (all-or-nothing) | 28 |
| supported claims | 163 / 222 | 290 / 387 |
| contradicted claims | 1 | 1 |
| outputs lost | 3 (1 broken JSON, 2 citation errors) | 2 (output cap) |
| dropped: quote not in source | — | 65 |
| median wall per piece | 5.6 s | 12.5 s |

Output/input ratio (v3, pieces ≥ 2K tokens): median 0.53, max 1.05 — list-dense
turns emit about one claim per source line → `summary_output_ratio` = 1.3.
Quote failures read against their sources: 10 of 12 were the exact words with
Markdown markers dropped (`**…**`, backticks), so the quote comparison ignores
`*`, `` ` ``, `#` and curly quote marks; word order and spelling still must
match. Remaining misses were a quote running past its cited window and a quote
from a different window (correctly dropped). Cost: ~2.2× background time per
summary, from the larger output.

## 4. Traps

- TRAPS 13: hand-made fixtures agree with the author — qualify on real C/B turns.
- TRAPS 52/37: a presence score cannot see invention; the NLI verdict per claim
  and the quote check are the guards, coverage is not.
- A quote that matches proves the text exists, not that the claim follows from
  it — the NLI check stays.
