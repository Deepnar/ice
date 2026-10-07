# ICE v2 / LSREP paper artifacts

The canonical archive is `ICE_paper_v2.tex` / `ICE_paper_v2.pdf`.
`ICE_paper_NORA.tex` / `ICE_paper_NORA.pdf` is a separate anonymous ACL-style
review version prepared for the NORA 2026 research track (see `NORA_SUBMISSION.md`). Frozen rejected-submission sources are not build targets.
The evaluated system is ICE **v2**, tag `v2-paper-eval`, not current v3 on main.

`ICE_paper_JMLR.tex` / `ICE_paper_JMLR.pdf` is the 35-page JMLR initial-submission
twin. Sections 1–10 and the central evidence appendices remain in the main paper.
`ICE_paper_JMLR_online_appendix.tex` / `.pdf` is the separate 10-page Online
Appendix 1, preserving the historical pilot, detailed replay distributions,
frozen implementation reference, cost strata, and invalidated-adapter history.
`JMLR_SUBMISSION.md` records the exact relocation map, reference audit, official
page-count interpretation, and upload ambiguity; `JMLR_COVER_LETTER.txt` is the
plain-text cover letter. `JMLR_MANUSCRIPT_PACKAGE.zip` contains the two PDFs only,
for the author guide's multi-file archive route; the cover letter remains separate.
Preparation does not submit the manuscript or imply acceptance.

The original paper-specific [classifier](https://huggingface.co/Deepnar/ice-v2-classifier)
and [MicroNER](https://huggingface.co/Deepnar/ice-v2-microner) are public in the
[ICE / LSREP Reproducibility collection](https://huggingface.co/collections/Deepnar/ice-lsrep-reproducibility-6ac6210257541dd6f8594e14).
`huggingface/` holds model cards, original-file hashes, immutable upload commits,
frozen preprocessing and public-fetch verification. These are ICE v2 artifacts;
no current ICE v3 checkpoint is part of this release.

## Reproduce the existing-evidence analyses

Run from the repository root with the local evidence available:

```sh
uv run python experiments/lme/analyze_matched.py
uv run python experiments/lme/test_matched_analysis.py
uv run python experiments/mature/clustered_sensitivity.py
uv run python experiments/mature/scoring_sensitivity.py
uv run python experiments/mature/test_scoring_sensitivity.py
uv run python experiments/flaw_ablation/buildup/exp3_bootstrap.py
uv run python experiments/paper/generate_analysis_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_v2.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_NORA.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_JMLR.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd experiments/paper/ICE_paper_JMLR_online_appendix.tex
```

Build the JMLR main manuscript before its Online Appendix: the latter imports
main-paper labels with `xr-hyper` and links to the sibling main PDF. Rebuild the
Online Appendix after any main-paper label or page change. The main source does
not depend on supplementary auxiliary files. Its ten explicit Online Appendix
section references have been checked against the supplementary labels.

These analyses do not call models, modify databases, or repair production code.
The LongMemEval script reads the locally retained matched-cloud answers and
judgements. The LSREP scripts read private mature-run intermediates; exact
private scores cannot be independently regenerated from public artifacts alone.
The tests check pair preservation, frozen arithmetic, missingness, score-policy
reproduction, and invariance of ordinal comparisons to monotone recoding.

## Public release scope

Release candidates are the manuscripts, shared bibliography, generated aggregate
TeX, analysis scripts/tests, `CLAIM_AUDIT.md`, this artifact description, existing
adapter/harness code, and these aggregate reports:

- `experiments/lme/results/matched_cloud_v2.md`
- `experiments/lme/results/matched_cloud_analysis.json`
- `experiments/mature/results/clustered_sensitivity.json`
- `experiments/mature/results/scoring_sensitivity.json`
- `experiments/flaw_ablation/buildup/results/exp3_bootstrap_report.{json,md}`

Some result directories are gitignored. Inclusion in this list is a release
scope, not evidence that files have been committed. The final release stages only explicitly reviewed paths. Do not use a directory-wide add: local run
folders contain raw private and benchmark material. Any later commit/push must
follow the repository's git skills and experiment-phase rules.

Exclude raw answers, judgements, databases, logs, downloaded LongMemEval data,
personal histories/probes, credentials, scratch scripts, and planning notes.
The public benchmark inputs must be obtained from their original release.
Anonymous review supplements must additionally remove identifying repository
links and metadata; the public canonical package is not itself an anonymous
supplement. No submission or external publication is performed by these builds.

## Analysis interpretation

Questions remain paired across ICE/vector and, for degradation, across phases.
LSREP cluster intervals retain all observations of each selected probe and
condition on the recorded four conversations. Complete-case sensitivity removes
many failed answers and must accompany, not replace, operational reliability.
Ordinal net superiority uses score order; no ratio of rubric levels is treated
as an established efficiency metric. Category LongMemEval intervals are
exploratory; full-S abstention remains descriptive despite a positive bootstrap
interval (eight discordant pairs; exact McNemar p=0.0703125).

## Style provenance

`acl.sty` and `acl_natbib.bst` are unmodified downloads of the official
[ACL style files](https://github.com/acl-org/acl-style-files), retrieved
2026-09-12 from the `master` branch. Embedded notices are retained.
The [ARR call](https://aclrollingreview.org/cfp) permits eight content pages;
limitations, ethics, references and appendices follow the content.
The NORA twin was prepared for NORA at AACL-IJCNLP 2026; preparing or publishing
a venue twin does not submit it or imply acceptance. The JMLR twin uses the
unmodified official `jmlr2e.sty`; its pinned style provenance and current
submission requirements are recorded in `JMLR_SUBMISSION.md`.
