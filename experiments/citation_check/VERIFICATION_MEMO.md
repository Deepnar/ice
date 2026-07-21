# Verification Memo: Related Work Section (ICE Paper)

**Purpose of this memo:** you (the AI compiling the full paper) are receiving `RELATED_WORK.md` as a pre-verified component. This memo explains exactly what verification was performed, what it covered, and — just as important — what it did *not* cover, so you don't over-trust or under-trust the file when merging it into the rest of the paper.

## What was verified, and how

The related work section went through three rounds of citation auditing before reaching its current state:

1. **Automated metadata verification.** Every citation with an arXiv ID was checked against the live arXiv API (title + author list pulled and diffed against the claimed citation). Every citation with a resolvable URL was checked for live status (HTTP GET) and, where possible, its page `<title>` compared against the claimed title. This caught: three citations with fabricated "Author(s) unknown" placeholders (two turned out to be real papers with real authors; the author names were simply missing from the original draft), one citation where a real arXiv ID had been given an invented, more dramatic title than the paper actually has, and ten dead links (mostly ResearchGate/OpenReview anti-bot 403s) that turned out to duplicate a working citation to the same paper elsewhere in the list.

2. **Manual primary-source verification.** For gaps the automated pass couldn't close — either because a paper had no arXiv ID, or because two foundational algorithms (Reciprocal Rank Fusion and HyDE) had *no citation at all* in any draft despite being discussed at length — the paper's author (not me) went and pulled the actual paper pages, abstracts, and Google Scholar / ACL Anthology / PMLR / ACM DL / NeurIPS proceedings entries and gave them back to me directly. This is how RRF, HyDE, REALM's real venue (ICML 2020, not just arXiv), and Lewis et al.'s real RAG paper (NeurIPS 2020, full 12-author list) got properly sourced.

3. **Structural integrity check.** A script confirmed every in-text `[n]` citation marker has exactly one matching numbered entry in the works-cited list, and vice versa — no orphaned citations, no gaps in numbering, no reused numbers pointing at two different works.

## What changed as a result

- 60 citations → 48. Twelve were cut: dead-link duplicates of papers already cited via a working source elsewhere, one orphan citation never actually referenced in the body text, and one citation that had been double-booked for two unrelated papers (that double-booking is also why the numbering isn't a strict superset of the original — it was split into two separate, correctly sourced entries).
- Two foundational algorithms discussed in the "Algorithmic Primitives" section (RRF, HyDE) previously had zero citations backing them anywhere in any draft. Both now have real, confirmed citations.
- One citation (CALMem, ref [4]) had an invented subtitle; corrected to the real title. Its associated claim in the prose — the "compaction continuity problem" phrase attributed to it — was independently spot-checked against the actual CALMem abstract and is accurate.
- One citation (CRAG's OpenReview link) pointed at a withdrawn ICLR 2025 submission; removed in favor of the arXiv preprint, which stands on its own.
- One in-text citation had been misused to support a claim about *this paper's own* (ICE's) ablation results by citing an external repo (mem0's GitHub) that has nothing to do with that claim. That citation was removed from that sentence; the claim now correctly reads as pointing to this paper's own evaluation section rather than an external source.

## What was *not* verified — please don't assume otherwise

- **Claim-level accuracy beyond spot checks.** Most citations were verified for *existence, title, authors, and venue* — not for whether every descriptive sentence in the prose is a faithful summary of that paper's full methodology and results. Only a handful of specific claims were directly checked against source text (the CALMem "compaction continuity problem" phrase, and the RRF/HyDE/REALM/RAG abstracts, which were read in full). The rest are reasonable-sounding characterizations that have not been independently fact-checked line-by-line.
- **Some citations are intentionally secondary sources** (GitHub READMEs, Emergent Mind topic pages, Medium/GoPenAI explainer posts, blog posts). These were left in deliberately where they support a specific factual claim (e.g. mem0's own GitHub for benchmark names) or where they're the only accessible summary of a real methodology (e.g. the Sewak Medium post for Think-on-Graph). They are not peer-reviewed and should be weighted accordingly if this paper undergoes review.
- **Two arXiv IDs were deliberately removed rather than kept on a guess** during this process (a speculative MemGPT arXiv ID and a speculative REALM arXiv ID) once it became clear they hadn't been independently confirmed — REALM's citation was replaced with a properly confirmed one (ICML 2020 proceedings); MemGPT's citation reverted to its two already-verified sources (Semantic Scholar page, project website) rather than include an unconfirmed arXiv link.

## Instructions for compiling this into the full paper

- Preserve the `[1]`–`[48]` numbering as an internal reference frame while compiling, but **you will need to renumber to match the paper's global citation scheme** once this is merged with other sections' bibliographies — don't assume `[1]`–`[48]` are the paper's final citation numbers.
- If any other section of the paper cites MemGPT, mem0, RAG, REALM, RRF, HyDE, GraphRAG, Think-on-Graph, or KGP, **reuse the corrected citation entries from this file** rather than re-deriving or re-fetching them, to avoid reintroducing the errors that were just fixed here.
- If you (or a later editing pass) add new citations to this section, apply the same standard: confirm via arXiv/CrossRef/publisher metadata before including a title/author, and don't fill in unknown authors with a placeholder — an unverifiable author field is itself a red flag worth surfacing to the paper's author rather than silently completing.

