# LSREP / ICE v2: JMLR submission preparation

Prepared **7 October 2026**. This is an initial-submission package, not a
submission receipt or an accepted article. The evaluated system is frozen
**ICE v2**, `v2-paper-eval` (`0521df9171b4a7d69f82d12d70497138c77b2678`);
current ICE v3 work is separate.

## Official sources checked

All accessed on 2026-10-07:

- [Information for Authors](https://www.jmlr.org/author-info.html): current
  submission, length, author-contact, abstract, keyword, cover-letter,
  originality, preprint, funding and conflict requirements.
- [Instructions for Formatting JMLR Articles](https://www.jmlr.org/format/format.html).
- [Common formatting-errors checklist](https://www.jmlr.org/format/formatting-errors.html).
- [JMLR FAQ](https://www.jmlr.org/faq.html).
- [Current Action Editors](https://www.jmlr.org/editorial-board.html): candidates
  were selected from the Action Editors section, not the reviewer board.
- [Official style repository](https://github.com/JmlrOrg/jmlr-style-file), commit
  `f413f638b407af76074813f8f88a82a7a5a81e9d`; read `sample.tex` and
  `jmlr2e.sty`. The vendored style is unmodified, SHA-256
  `a430a875d561235951800e4e21d2631e18ddf0b369646ec276f43ea5080f27c3`.
- [Official cover-letter template](https://github.com/JmlrOrg/jmlr-coverletter),
  commit `f0a8b554a4373d6de36535da8174e97713660dc1`; both template and
  preamble inspected. JMLR explicitly permits plain-text cover letters.
- [TCET contact page](https://www.tcetmumbai.in/contact.html): verified the
  institutional postal address used on the title page and cover letter.

The older formatting guide says its requirements are compulsory for final
articles. The current Information for Authors explicitly requires official
JMLR LaTeX style for initial submissions and takes precedence here.

## Verified manuscript

| Check | Result |
|---|---|
| Full source | `ICE_paper_JMLR.tex`, derived from full `ICE_paper_v2.tex`; NORA compression was not used |
| PDF | `ICE_paper_JMLR.pdf` |
| Total pages, including appendices/references | **42** |
| PDF size | **355,211 bytes** (0.355211 MB; below 5 MB) |
| PDF SHA-256 | `2db0fa502cc804a5ec1f7372c339e561155fdb4a058cf364e7ce9bba605a4aa1` |
| Abstract | **187 whitespace-delimited words** in the LaTeX abstract body; below 200 |
| Running title | **LSREP: Evaluating Conversational Memory** |
| Running-title length | **39 characters**, including spaces and punctuation |
| Exactly five keywords | conversational memory; longitudinal evaluation; state replay; retrieval-augmented generation; reproducibility |
| Corresponding author | Deepesh Sonar; institutional postal address; `18deepnar@gmail.com` |
| Initial-submission metadata | Official `preprint` option; no invented editor, volume, paper number, pages, submission/acceptance or publication dates |
| End matter | Numbered reproducibility/ethics section, unnumbered acknowledgments/funding disclosure, lettered appendices, References last |
| Fonts | Times-family body; every listed PDF font embedded; no Type 3 fonts |
| Compilation | Complete `latexmk`/BibTeX build; no LaTeX warnings, undefined references/citations, or overfull boxes |
| Visual inspection | All 42 pages rendered and reviewed; no clipped tables/figures, broken equations, bad headers or blank pages; figure/table text reviewed at readable size |
| Source preservation | Canonical source unchanged; all citation/label/input multisets, 32 captions, 10 appendix sections and scientific body numeric tokens preserved; non-float scientific prose matches after venue-only normalization |

The abstract was shortened only to meet the word limit. Other changes are
venue formatting, readable figure fonts, caption placement, heading numbering,
funding disclosure and verified ICE v2 model links. Full appendices and negative
findings remain. Two long tables flow across pages; their final captions are
below the tables. The source retains the canonical generated aggregate inputs.

Rebuild from the repository root:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_JMLR.tex
```

## Cover letter and evaluator suggestions

`JMLR_COVER_LETTER.txt` contains all six required elements: overlap disclosure,
sole-author consent, conflicts, three suggested AEs, four suggested reviewers,
and the five manuscript keywords. It discloses the arXiv preprint and no prior
refereed publication; rejected attempts are not prior publications and are not
recited. The author confirmed **none** for third-party support, relevant
financial relationships and editor/reviewer conflicts.

Recommendations require the author's final approval; these are not assignments.
Affiliations and subject fit were checked against these primary profiles:

| Role | Candidate and current affiliation | Fit and checked source |
|---|---|---|
| AE, preferred | Qiaozhu Mei, University of Michigan | Text/behavioral data, NLP and information retrieval; [profile](https://websites.umich.edu/~qmei/) |
| AE | Kai-Wei Chang, UCLA | Trustworthy language models and evidence-grounded agents; [profile](https://web.cs.ucla.edu/~kwchang/) |
| AE | Samuel Kaski, Aalto University / University of Manchester | Multiple information sources, user interaction and experimental design; [profile](https://kaski-lab.com/) |
| Reviewer | Di Wu, UCLA | LongMemEval, memory and RAG; [profile](https://xiaowu0162.github.io/) |
| Reviewer | Adyasha Maharana, Databricks Mosaic Research | LoCoMo and long-term conversational evaluation; [profile](https://adymaharana.github.io/) |
| Reviewer | Tatsunori Hashimoto, Stanford | Robustness and evaluation under distribution shift; [profile](https://thashim.github.io/) |
| Reviewer | Noah A. Smith, University of Washington / Ai2 | NLP methods and evaluation; [profile](https://homes.cs.washington.edu/~nasmith/) |

No listed candidate shares the author's TCET affiliation, and the sole author
reports no JMLR-defined conflict. Public profiles cannot independently establish
private relationships; the declaration relies on the author's confirmation.
Di Wu identifies Kai-Wei Chang as his advisor. Prefer an independent reviewer
if that AE is assigned; the relationship is disclosed in the draft.

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

## Remaining warnings and final author checklist

- **Length:** 42 pages exceeds JMLR's 35-page review-delay threshold. It remains
  below 50, so the mandatory over-50-page justification is not triggered.
  Content has not been automatically compressed.
- **Editor preview:** the built-in standalone compiler cannot locate sibling
  `jmlr2e.sty` and does not support this multi-file project. The editor remains
  open; the delivered PDF is verified by the installed local LaTeX toolchain.
- **Private replay:** exact private LSREP results cannot be regenerated from
  public artifacts alone; the manuscript and model cards state that limit.
- **Vendor whitespace:** Git's whitespace check flags 18 lines in the upstream
  style. They are retained to preserve the official file byte for byte;
  all other task files pass the check. This is not a PDF formatting defect.
- **Repository regressions:** 618 disposable-database smoke tests passed in
  39.86 seconds; five existing warnings. No working research database was used.

Before pressing Submit:

- [ ] Approve the full PDF and retained length.
- [ ] Approve the three AE / four reviewer suggestions and their COI declaration.
- [ ] Confirm affiliation/address/email, sole-author consent, funding/financial
  declarations and absence of any simultaneous live journal/conference review.
- [ ] Upload `ICE_paper_JMLR.pdf` as the manuscript and
  `JMLR_COVER_LETTER.txt` as the separate plain-text cover letter.
- [ ] Copy the exact title, abstract and five keywords into required portal fields.
- [ ] Perform the final JMLR Submit action personally, then verify the receipt.

No JMLR submission button has been pressed. Publication of the Hub models and
a task-only Git branch is distinct from manuscript submission.

## Exact task files

Twenty files belong to this task:

```text
README.md
experiments/paper/ARTIFACTS.md
experiments/paper/ICE_paper_JMLR.tex
experiments/paper/ICE_paper_JMLR.pdf
experiments/paper/JMLR_COVER_LETTER.txt
experiments/paper/JMLR_SUBMISSION.md
experiments/paper/jmlr2e.sty
experiments/paper/huggingface/README.md
experiments/paper/huggingface/RELEASE_MANIFEST.json
experiments/paper/huggingface/PUBLIC_VERIFICATION.json
experiments/paper/huggingface/classifier/README.md
experiments/paper/huggingface/classifier/config.json
experiments/paper/huggingface/classifier/model.py
experiments/paper/huggingface/classifier/frozen_classifier.py
experiments/paper/huggingface/classifier/classifier_inference.py
experiments/paper/huggingface/microner/README.md
experiments/paper/huggingface/microner/config.json
experiments/paper/huggingface/microner/ner_model.py
experiments/paper/huggingface/microner/frozen_ner_utils.py
experiments/paper/huggingface/microner/microner_inference.py
```

Task notes/build logs are local-only under `logs/jmlr-submission-2026-10-07/`.
The shared, ignored `docs/SESSION.md` was restored from its entire 2026-10-07
archived v3 session at the author's explicit request and was not committed.
No production source, v3 experiment artifact or shared HANDOFF was changed.
