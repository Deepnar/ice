# Paper — what the next pass must do

**Written 2026-08-29, after the ACM TIST editorial reject and during the LME-v2
run. Nothing here is urgent: the LongMemEval numbers land first, then one editing
pass does all of it together.** Delete this file once the pass is done.

The canonical file is `ICE_paper_v2.tex`. `ICE_paper_tist.tex` is a frozen record
and takes no edits — see [CLEANUP.md](../../docs/CLEANUP.md) 2026-08-29.

---

## 1. ⚑ TRIM. The front matter is now too long, and that is self-inflicted

Every fix this session ADDED prose and nothing removed any:

| | now | should be |
|---|---|---|
| abstract | **446 words** | 150–250 |
| introduction | **1,068 words** | ~600–700 |
| total | 37 pp | ≤ 25 pp main body for most venues |

This partly *causes* the problem the reject was about. An editor who tunes out in
the first two pages is the failure mode; a 446-word abstract invites exactly that.
**Trimming is worth more than adding another paragraph.** Do this pass FIRST, before
adding the LongMemEval section, or the additions land on top of bloat.

## 2. One sentence still reads as the claim the AE called misleading

Intro ¶1 opens: *"A language model holds a conversation only for as long as its
context window holds the conversation; everything older is silently gone."*

The paragraph qualifies it and §2.1 handles commercial memory properly, but the
**first sentence read alone** is the same absolute claim the AE flagged in the
abstract. A skimming editor reads that sentence. Add a five-word qualifier —
something like *"unless explicitly configured with persistent cross-session
memory"* — and it is closed.

## 3. Add the LongMemEval section once the numbers exist

New subsection in Results. It must carry, in the number's own words:

- **System under test: ICE v2 @ tag `v2-paper-eval`**, not `main`.
- **LongMemEval protocol with a LOCAL judge** (`gemma4:12b`), *not* GPT-4o —
  so NOT directly comparable to published GPT-4o-judged figures. Say it beside
  the number, not in a footnote.
- Judge token cap raised 10 → 256 (the Ollama build reasons before answering);
  decision rule unchanged.
- Ollama GGUF build, not the paper's `mattbucci/gemma-4-12B-AWQ` on SGLang.
- Embedder moved to CUDA: cosine(cpu, cuda) = 0.99986, max elementwise 2.7e-03.
- Whether it is a **subset**, and the stratification, stated explicitly.

**⚑ Lead with the CONTROLLED comparison, not the headline percentage.** `full_ice`
vs `vector_rag` ran on identical hardware, base model, judge and prompt — that is
the only genuinely controlled comparison available. Published mem0/Zep numbers are
context and a reviewer will discount them correctly (different hardware, base
model, judge). Report them as context, with the caveat stated.

**⚑ The interesting result is WHERE ICE wins, not by how much overall.** If it wins
specifically on **knowledge-update** (superseded facts) and **abstention**
(declining rather than confabulating), that is "memory is curation, not collection"
demonstrated on third-party public data — a far stronger claim than a marginally
higher aggregate.

## 4. Reframing already done this session — do NOT redo

A review received 2026-08-29 asked for these. **All were already in the file**; it
was reading an older draft. Verify before acting on any similar feedback.

- ✅ Title leads with the protocol; ICE named "system under test".
- ✅ Abstract opens on the measurement gap, not on ICE.
- ✅ Contribution order swapped — LSREP primary, ICE second.
- ✅ Intro ¶1 cites `liu2024lost`, `hsieh2024ruler`.
- ✅ Intro ¶2 bridge cites `xu2022msc`, `jang2023chronicles`, `maharana2024locomo`,
  `wu2025longmemeval`, `salemi2024lamp`.
- ✅ Related Work opens on the stability-of-knowledge axis with per-section labels.
- ✅ §2.1 "Deployed commercial memory, and why it is not a baseline here".
- ✅ New §2.5 "Evaluating Conversational Memory".
- ✅ New §5.3 "The Standing of a Purpose-Built Protocol", four falsifiers.
- ✅ Judge paragraph cites `zheng2023judging`, `liu2023geval`, `wang2024unfair`.
- ✅ References 15 → 31, all cited, all arXiv-verified and DBLP venue-confirmed.

## 5. Venue

TIST is dead (*"decision is final … no appeals"*). `docs/PUBLISHING.md` has the
written order: **IP&M → Information Retrieval Journal**. The maintainer has since
opened this up to **conferences** as well (2026-08-29).

Re-decide once the LongMemEval number exists — it changes which venues are
realistic, and a conference with a public-benchmark anchor is a different pitch
from a Q1 IR journal. Do not submit before the trim in §1.

## 6. Smaller things

- Datasets section still has no date spans, token counts, example turns or probes,
  or a stated anonymisation procedure. The AE's "no details" was exaggerated but
  pointed at something real.
- `experiments/citation_check/verify_citations.py`'s header says it cannot run in
  the sandbox. **That is stale** — arXiv and CrossRef were both reachable this
  session and all 31 references were verified live.
