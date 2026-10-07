# LSREP / ICE v2: final JMLR submission preparation

Prepared **7 October 2026**, on `work/jmlr-v2-submission` only. This package has
**not been submitted** and does not imply acceptance. All mature-system numbers
refer to frozen **ICE v2**, tag `v2-paper-eval`
(`0521df9171b4a7d69f82d12d70497138c77b2678`). Historical ICE v1 is explicitly
identified; ICE v3 is outside this preparation pass.

## Package and page counts

| Document | Old pages | Final pages | Bytes | Repository path |
|---|---:|---:|---:|---|
| Main manuscript, including normal appendices and references | 42 | **35** | 327315 | `experiments/paper/ICE_paper_JMLR.pdf` |
| Separate Online Appendix 1 | — | **10** | 134267 | `experiments/paper/ICE_paper_JMLR_online_appendix.pdf` |
| Combined reading package | 42 | **45** | — | Two separate PDFs; not a concatenated manuscript |

Main SHA-256: `e04e7abf58455da5f6c57510a60d696d1263fe1b7b911b5977bdd367c79602f2`.
Online Appendix SHA-256: `e2c6ca11e7af91aacf623c3b5d6b12b56fd34122c35f3e86db114540b211b9c6`.

Sources are `experiments/paper/ICE_paper_JMLR.tex` and
`experiments/paper/ICE_paper_JMLR_online_appendix.tex`.
The cover letter is `experiments/paper/JMLR_COVER_LETTER.txt`.
`experiments/paper/JMLR_MANUSCRIPT_PACKAGE.zip` contains exactly the two PDFs,
with byte-identical members and no source, raw data, or cover letter.

## Exact content relocation

No scientific material was deleted or substantively condensed. All ten original
appendix sections survive across the two documents. All numerical results,
scientific citations and labels, all 26 tables, and all six figures are retained.
The canonical `ICE_paper_v2.tex` / `.pdf` remains unchanged.

| Original appendix | Final location | Material retained there |
|---|---|---|
| A, Component Fidelity Audit | Main Appendix A | Complete component-by-component audit and attribution limits; Table 14 |
| B, Historical ICE v1 Pilot | Online Appendix 1, Section 1 | Instantiation comparison, corrected pilot results, token-accounting audit, longitudinal observations and lessons; Tables S1–S2 |
| C, ICE v2 Full Ablation Results | Main Appendix B | Entire fifteen-row buildup, paired contrasts, mechanism caveats, bar chart, recency table and prose; Tables 15–16, Figure 5 |
| D, Detailed LSREP Breakdowns | Online Appendix 1, Section 2 | Metric definitions, complete score distribution and temporal-quality buckets; Tables S3–S5 |
| E, Frozen ICE v2 Implementation Reference | Online Appendix 1, Section 3 | All five subsections: classification, Codex writes/retrieval, detailed Codex limitations, prompt assembly, workers/infrastructure; Table S6 |
| F, Detailed ICE v2 Retrieval Configuration | Main Appendix C | Entire retrieval-leg/weighting/fusion/curation/budget description, equation and diagram; Figure 6 |
| G, Matched LongMemEval Paired Inference | Main Appendix D | Both complete paired-outcome tables, resampling policy, missingness and degradation-denominator explanation; Tables 17–18 |
| H, Matched LongMemEval Cost Strata | Online Appendix 1, Section 4 | Complete category/outcome cost longtable and caveats; Table S7 |
| I, Historical Local Oracle and Adapter Invalidation | Online Appendix 1, Section 5 | Complete local-oracle outcomes/bounds and excluded flattened-adapter history |
| J, LSREP Sensitivity and Reproduction | Main Appendix E | Complete clustered, scoring-source, missing-score, ordinal and reproduction analyses, including sensitivity Table 19 |

Section 5 and Algorithm 2 still explain the evaluated system and its selection
contract. Sections 6–8 still expose adapter omissions, defective/unexercised
components, data/judging limitations, and the negative matched-public results.
Retained Appendices A/B/D/E carry the central audit and statistical support.
Appendix C preserves the detailed retrieval configuration. Moving the original
implementation appendix therefore relocates lower-level supporting implementation detail without making
the scientific argument depend on a repository visit or supplementary reading.
Its known defects and limitations remain physically present in the supplement.

## Mechanical audit and text changes

- No manual page breaks or negative spacing were present or added. Official
  margins, body size, line spacing, caption sizes, headers/footers, and style
  bytes are unchanged. No figure/table was resized in this pass.
- Removed seven redundant/custom float-scheduling overrides (four fractions,
  three counters), allowing the official style/LaTeX defaults to govern floats.
  Existing table-row spacing remains unchanged for legibility.
- The two short paired-outcome tables now use ordinary `[htbp]` instead of
  `[tbp]`. They fit beside their discussion on page 30; section barriers remain
  so appendix evidence stays before the next appendix. No forced `H` or `!`
  placement was introduced. Sparse normal float/reference pages were accepted
  rather than tightening typography.
- Sections 1–10 have no substantive prose edits. Their appendix destinations
  were updated. One existing location sentence was corrected: the
  per-conversation breakdown is in main Table 4; only score distributions and
  temporal-quality breakdowns moved to Online Appendix 1, Section 2.
- Supplement additions identify the accompanying manuscript, frozen versions,
  source/model artifacts, section/table numbering, and private-data boundary.
  “This appendix” became “This section” in the relocated implementation
  introduction. Its internal references to the main paper are explicitly named.
- The Online Appendix delayed Section 2 label was moved ahead of its tables.
  Its main-paper references import labels through `xr-hyper`; the main paper
  uses ten explicit section destinations verified against supplementary `.aux`
  values, without a circular build dependency.
- The cover letter was shortened, its package description updated, Jiho Kim
  substituted for Noah A. Smith on technical fit, and the unnecessary paragraph
  about relationships between suggested evaluators removed.

An automated source audit compares every original appendix block with its final
location after only these declared reference/format normalizations. It also
compares the entire pre-appendix body after the declared destination edits.
All comparisons pass: **claims, results, numerical values and negative findings
are unchanged**. No experiments, scoring analyses, or model runs were repeated.

## Every main-paper reference to Online Appendix 1

Line numbers refer to the final `ICE_paper_JMLR.tex`. These are all ten literal
references; there are no other references to moved material left unresolved.

| # | Source line | Main location | Online section | Purpose |
|---|---:|---|---|---|
| 1 | `173` | 3.4, validity and scope | 1 | Historical v1 context-accounting/judging deficiencies |
| 2 | `195` | 5, architecture introduction | 3 | Operational implementation reference |
| 3 | `229` | 5.1, request lifecycle | 3.5 | Configured worker list |
| 4 | `253` | Table 2 caption, memory stores | 3 | Schemas and write paths |
| 5 | `257` | 5.2, memory stores | 3.2 | Codex write path and entity resolution |
| 6 | `353` | 6.4, matched LongMemEval setup | 3.4 | Faulty HTTP-wrapper check omitted by the adapter |
| 7 | `355` | 6.4, matched LongMemEval setup | 5 | Historical local oracle and adapter invalidation |
| 8 | `385` | 7.1, longitudinal results | 2 | Score distributions and temporal quality |
| 9 | `658` | 7.3.3, quality/context cost | 4 | Cost strata by category and correctness |
| 10 | `904` | Appendix C.1, retrieval configuration | 3.4 | Prompt assembly, slots, and wrapper check |

The supplement labels resolve to Sections 1, 2, 3, 3.2, 3.4, 3.5, 4 and 5 as
expected. Its references back to main Section 3, Section 5, Section 8, Appendix A
and Table 4 resolve against the final manuscript. The two PDFs must retain their
sibling filenames for external PDF hyperlinks; printed references remain clear
if a submission portal renames downloads.

## Official guidance and page-count interpretation

Current primary sources checked on 2026-10-07:

- [Information for Authors](https://www.jmlr.org/author-info.html), “Submission
  procedure”: official JMLR LaTeX is compulsory for initial submissions;
  articles may have online appendices. The same block warns about manuscripts
  exceeding 35 pages, requires justification above 50, and says page counts
  include appendices. It does **not explicitly exempt separate online appendices**.
- [Formatting instructions](https://www.jmlr.org/format/format.html),
  “Appendices”: normal appendices follow acknowledgments and are lettered;
  Online Appendices are omitted from the final manuscript, published as separate
  files, and referred to as Online Appendix 1, 2, etc. This distinction is
  **explicit**, but its application to initial-review length thresholds is not.
- [JMLR FAQ](https://www.jmlr.org/faq.html), “What if I need to submit multiple
  files?”, and the [submission-system FAQ](https://jmlr.csail.mit.edu/manudb/faq):
  additional documents may be uploaded and reviewer visibility selected.
- [Common formatting errors](https://www.jmlr.org/format/formatting-errors.html):
  acknowledgments, lettered appendices, references ordering and readable captions.
- [Current Action Editors](https://www.jmlr.org/editorial-board.html): all three
  suggestions appear in the Action Editors list, not merely the reviewer board.
- [Official style repository](https://github.com/JmlrOrg/jmlr-style-file): current
  `master` commit `f413f638b407af76074813f8f88a82a7a5a81e9d`; the vendored
  `jmlr2e.sty` is byte-identical to that upstream version, SHA-256
  `a430a875d561235951800e4e21d2631e18ddf0b369646ec276f43ea5080f27c3`.
- [Official cover-letter template](https://github.com/JmlrOrg/jmlr-coverletter):
  the author instructions permit a plain-text letter instead of this template.
- [TCET contact page](https://www.tcetmumbai.in/contact.html): institutional
  postal address; retained title-page/letter contact details.

**Interpretation:** 35 pages is the measured main PDF count, including its five
normal appendices and references. Excluding the separate 10-page Online Appendix
from the review threshold is a reasonable **inference**, not an explicit ruling
in the checked guidance. The combined reading load is **45 pages**. If JMLR counts
both PDFs together for initial review, the over-35 warning applies; the package
is still below 50 and does not trigger the mandatory over-50 justification.
No claim of an explicit online-appendix exclusion is made.

## Intended uploads and remaining UI ambiguity

The prepared package is:

1. `ICE_paper_JMLR.pdf` as the main manuscript.
2. `JMLR_COVER_LETTER.txt` as the separate plain-text cover letter, containing all
   six required elements.
3. `ICE_paper_JMLR_online_appendix.pdf` as an additional document titled
   **Online Appendix 1**, made visible to reviewers.

The author guide also says to archive multi-file submissions in tar/zip format.
`JMLR_MANUSCRIPT_PACKAGE.zip` supplies that route with the two manuscript PDFs;
use it if the current portal requests a multi-file archive. The cover letter is
kept separate as the guide specifies. The FAQ additionally documents individual
extra-document uploads. **The current public documentation does not establish
which upload dropdown/category denotes an Online Appendix.** In particular,
`Other` has not been verified as that category. Confirm the live form's labels
and reviewer-visibility control during the author's final upload; do not infer
that `Other` is correct. No login, submission draft, upload, or Submit action was
performed in this pass. The preparation package is complete despite this
explicitly recorded portal detail.

## Editors, reviewers, and conflicts

The concise letter suggests these three AEs in preference order:

| Candidate | Technical rationale and current primary source |
|---|---|
| Qiaozhu Mei, University of Michigan | Direct IR/NLP and text/behavioral-data expertise; [profile](https://websites.umich.edu/~qmei/) |
| Kai-Wei Chang, UCLA | Strong LLM, trustworthy NLP, evidence-grounded agents and long-term-memory fit; [profile](https://web.cs.ucla.edu/~kwchang/) |
| Samuel Kaski, Aalto / Manchester | Multi-source retrieval, user interaction, probabilistic methodology and experimental design; [profile](https://kaski-lab.com/) |

Kai-Wei Chang remains a strong-fit current AE. No substitution is justified
merely by an evaluator-to-evaluator relationship. The public COI rule concerns
relationships between authors and evaluators; the removed paragraph did not
identify an author-level conflict.

The four reviewer suggestions cover three directly relevant evaluation efforts
and one broader methodological perspective:

| Candidate | Technical rationale and checked primary source |
|---|---|
| Di Wu, UCLA | LongMemEval, memory, information retrieval and RAG; [profile](https://xiaowu0162.github.io/) |
| Adyasha Maharana, Databricks Mosaic Research | LoCoMo and very long-term conversational-memory evaluation/data curation; [profile](https://adymaharana.github.io/) |
| Jiho Kim, KAIST postdoctoral researcher | DialSim and simulation-based long-horizon agent evaluation; [profile](https://jiho283.github.io/), [DialSim paper](https://arxiv.org/abs/2406.13144), [project](https://dialsim.github.io/) |
| Tatsunori Hashimoto, Stanford | Statistics-informed robustness, LLM evaluation and failure under distribution shift; [profile](https://thashim.github.io/) |

Jiho Kim replaces Noah A. Smith because his work is more directly connected to
multi-session dialogue and agent evaluation, not because of prior contact.
His identity was matched to the KAIST researcher linked by DialSim, avoiding
similarly named researchers. Noah A. Smith remains a credible broader NLP
candidate, but the chosen four fit the requested composition more closely.

JMLR defines individual conflicts as family/close friendship, graduate advising
(a lifelong conflict), other collaboration within three years, or any other
reason an author feels conflicted. Ordinary correspondence/endorsement alone
is not expressly listed. Public profiles cannot establish private relationships.
The author explicitly reconfirmed no author-level conflict for all seven final
candidates after reviewing prior contacts, and authorized Jiho Kim's inclusion.
**No contact/COI case remains pending.** No private correspondence is reproduced
in tracked artifacts. Funding/support and financial relationships were separately
confirmed as none. Suggestions remain subject to the author's final review and
the editor's suitability/availability decisions.

## Final validation

| Check | Result |
|---|---|
| Official format | `article`, `twoside,11pt`; unmodified official `jmlr2e` with supported `preprint,abbrvbib` options |
| Email presentation | Normal official `\email` uses small capitals; formatting guide explicitly prescribes them; unchanged |
| Abstract | 187 whitespace-delimited words, below 200; unchanged in this pass |
| Keywords | Exactly five: conversational memory; longitudinal evaluation; state replay; retrieval-augmented generation; reproducibility |
| Running title | LSREP: Evaluating Conversational Memory, 39 characters; supplement: LSREP: Online Appendix 1, 24 characters |
| Author | Deepesh Sonar, Department of Computer Engineering, TCET; full Mumbai postal address and `18deepnar@gmail.com`; unchanged |
| Headers | Sonar on even pages, condensed title on odd pages; first page plain; checked in both PDFs |
| Fonts | Embedded Times-family Type 1 body fonts; all listed fonts embedded, no Type 3 |
| Compile | Both complete local `latexmk` builds succeed; final logs have no undefined references/citations, LaTeX warnings or overfull boxes |
| PDF review | All 35 main and 10 supplementary pages rendered and visually inspected; targeted full-size tables/figures checked; no clipping or blank pages |
| Cross-document references | All ten main destinations and imported supplementary references checked; no stale moved-material references |
| End matter | Acknowledgments/Disclosure → normal Appendices A–E → References; bibliography last |
| Publication metadata | No invented editor, volume, paper number, DOI, assigned publication pages or dates; supported preprint option |
| Source conservation | All ten original appendix blocks and core body match after declared reference/format edits; numerical scientific content unchanged |
| Archive | Exactly two PDFs; ZIP integrity passes and extracted members equal final files |
| Size | Both PDFs and combined ZIP below 5 MB |

Build from the repository root, in order:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_JMLR.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_JMLR_online_appendix.tex
```

The built-in standalone editor compiler was checked for both sources but cannot
find sibling `jmlr2e.sty`; its additional-project-file limitation prevents it
from building this multi-file project. The installed local LaTeX toolchain
successfully builds the delivered PDFs. No style or TeX installation was needed.

Required branch smoke rails were attempted in a newly created disposable
PostgreSQL database: **489 passed, 9 failed, 1 skipped**, five warnings. Six
failures report the absent ignored `data/labeled/label_schema.json`, two report
the absent ignored `data/simulation/simulation_full.jsonl`, and one unknown-intent
weight check returns 9.0 instead of 1.0 with schema validation unavailable.
No production/test files changed. These branch-local fixture/validation limits
are recorded rather than claiming a green regression run or importing the active
v3 data. The disposable database was removed; the working ICE database was unused.
These smoke checks are not paper experiments or evidence for ICE v2 results.

## Frozen ICE v2 reproducibility release

- [Classifier](https://huggingface.co/Deepnar/ice-v2-classifier), upload commit
  `b69b2d4e939b3ac296f3d0d06567fe23f9917fc4`.
- [MicroNER](https://huggingface.co/Deepnar/ice-v2-microner), upload commit
  `a93abc5f568c5a38a265a5dcc8ace1a894b87e0d`.
- [ICE / LSREP Reproducibility collection](https://huggingface.co/collections/Deepnar/ice-lsrep-reproducibility-6ac6210257541dd6f8594e14).

| Artifact | Original path | SHA-256 |
|---|---|---|
| ICE v2 classifier, 384→128→25, 11 topic / 11 intent / 3 context | `models/classifier/ice_classifier_v3_qwen_ft3.pt` | `25c758b6a7e5cf449f3e4c8bb250db759d37cb4f0ab7dd8e0c1acd8afbf05831` |
| ICE v2 MicroNER, 384→128→64→3, B-ENT / I-ENT / O | `models/ner/ner_model.pt` | `23e596654065bde16f822db4b6dacf1830f67ae3613234bb7d808f5ea214c1b2` |

The classifier filename's `v3` is its training generation, not ICE system v3.
Both heads use the 384-coordinate prefix of the frozen Qwen embedding snapshot
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Model cards include exact ordering,
preprocessing, loading examples, limitations and checked Apache-2.0 licenses.
Original files were copied, not re-exported. Every public uploaded file was
downloaded without authentication and compared with its local source; all
checkpoint tensors are equal. The public collection contains exactly these two
models. `huggingface/RELEASE_MANIFEST.json` and `PUBLIC_VERIFICATION.json` retain
machine-readable evidence. Original retained weight bytes are hash-pinned at
release; the historical tag pins source/paths and did not archive weight hashes.

CPU loading checks matched the historical classifier's raw probabilities both
with and without context, and MicroNER's spans on full, capped and empty authored
text. This is inference parity, not a new model-quality measurement. No private
text, training data, caches, secrets or current ICE v3 weights were uploaded.
The README adds only the frozen-v2 model links and keeps its venue-agnostic paper
link. The JMLR twin adds the two verified model links to reproducibility.

## Final author review

- [ ] Review both PDFs, the letter, and the stated page-count interpretation.
- [ ] Reconfirm the author/contact, originality/live-review, funding/financial,
      AE/reviewer suitability and COI declarations when actually submitting.
- [ ] Identify the appropriate additional-document category in the current live
      portal, and make Online Appendix 1 visible to reviewers; `Other` is unverified.
- [ ] Enter the exact manuscript title, 187-word abstract and five keywords.
- [ ] Upload the final package and press Submit personally; verify the receipt.

No JMLR submission has been made. This pass stops after publishing the task
branch so the author can perform final review. `main`, `work/v3-system-repair`,
shared SESSION/HANDOFF files, production code, and v3 experiment data were not
modified. No HF artifact was changed during this final formatting pass.

## Files changed in this final pass

```text
experiments/paper/ARTIFACTS.md
experiments/paper/ICE_paper_JMLR.tex
experiments/paper/ICE_paper_JMLR.pdf
experiments/paper/ICE_paper_JMLR_online_appendix.tex
experiments/paper/ICE_paper_JMLR_online_appendix.pdf
experiments/paper/JMLR_MANUSCRIPT_PACKAGE.zip
experiments/paper/JMLR_COVER_LETTER.txt
experiments/paper/JMLR_SUBMISSION.md
```

The official style, canonical paper, shared bibliography/generated analysis
inputs, README and HF model release files are unchanged. Build/audit/test logs
and rendered review images are local-only under `logs/jmlr-final-pass-2026-10-07/`.
