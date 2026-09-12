# Canonical ICE v2 claim and evidence audit — 2026-09-11

Scope: `ICE_paper_v2.tex`; frozen submissions and v3 development documents are
unchanged. This is an evidence record, not a project roadmap. No raw private
text or raw LongMemEval output is included.

## System and numerical evidence

| Claim | Evidence inspected | Disposition |
|---|---|---|
| Frozen v2 architecture | `git show v2-paper-eval:src/{api,classifier,retrieval,workers}/...`; `docs/ICE_Architecture[real_v2].md` sections 1–8 | Code overrides stale descriptions; frozen reference itself unchanged |
| Shared 384-dimensional encoder and 25-logit head | tagged classifier/model.py and classifier.py | Retained; no v3 1024-dimensional/27-logit description imported |
| Raw storage versus lossless selection | tagged api/main.py inserts raw_text before post-flight; frozen reference §3.1 | Corrected: raw text is stored, flags govern later representation and extraction |
| Secondary complete-prompt cap | tagged api/main.py word check counts system message and query only | Removed the guarantee; documented direct adapter versus HTTP-wrapper boundary |
| Lexical scoring | tagged orchestrator `_bm25_episodic` uses PostgreSQL ts_rank | Historical name “BM25” disclosed; no canonical BM25 formula claim |
| Retrieval diversity | tagged `_enforce_token_budget` groups by source_type | Corrected “one per leg” to source-type priority; both episodic legs share a type |
| Parallel legs | tagged retrieve() constructs a dictionary of synchronous calls | Removed diagram's “parallel” claim |
| Mechanism fidelity | paper notes/FIDELITY_AUDIT.md and tagged implementation | Procedural defective; documents empty and defective; graph utility inconclusive; HyDE uncontrolled; source attribution incomplete |
| NER recall and sub-millisecond classification | previous prose and technical report contain point claims without task-level measurement evidence | Removed precise recall/latency claims; loaded model is not a measured recall rate |
| 1,211 probes / 50 checkpoints | unique keys in mature/intermediates/master_results.json | Corrected: 219 distinct probes, 1,211 observations, 52 checkpoints; dataset C has 10 |
| No manual score substitution | mature/exp2_bootstrap.py load_records() and 72 manual records | Corrected: manual absolute scores and tournament ranks merge into aggregate |
| LSREP uncertainty | published aggregation + clustered_sensitivity.py | Retained historical record intervals; added probe-clustered sensitivity; neither estimates user-population variation |
| Four-dataset scores | mature/results aggregate tables and canonical per-dataset breakdown | Main-text table includes A/B/C/D; stress reported separately from ordinary-density results |
| Budget as sole cause of stress advantage | full-system versus bare top-30 comparison | Narrowed: full-system robustness; no isolated budget intervention on density dataset |
| Score-1 equals overflow exceptions | score rubric versus recorded outcomes | Score-1 column relabelled; no direct overflow telemetry count claimed |
| Fragment correlation proves useful selection | observational correlations, vector r=-0.015 | Removed causal and qualitative ranking interpretation |
| Tournament wins are head-to-head | four-condition ranking and aggregation | Relabelled first-place tournament shares |
| LongMemEval overall/category/phase results | analyze_matched.py, all 2,000 local verdict files, paired controls | Reproduced overall CIs; added category cells, uncertainty, common-denominator phase contrast |
| Extra degradation | 499 complete four-way question records | +4.4 pp [-0.2,+9.2]; not established as nonzero |
| Abstention superiority | full-S discordant counts 7 versus 1, n=30 | Descriptive; bootstrap CI excludes zero but exact two-sided McNemar p=0.0703; exploratory categories |
| Cost | answer artifacts and lme_run.py answer_instance() | tokens_injected is whole-prompt estimate; provider input tokens complete; seconds excludes retrieval/ingestion; selected counts not candidate counts |
| Token efficiency | accuracy and recorded input volume | Context economy/quality–cost trade-off only; no matched-budget or frontier claim |
| Post-snapshot improvements | previous “Programme” section | Removed repair catalogue; future experiments remain unmeasured questions |

The matched worktree is the tag plus documented background-route and embedding
execution changes, not a v3 repair. The adapter calls v2 components directly;
this is explicitly narrower than replaying the entire HTTP lifecycle.

## Citation-to-claim audit

Primary pages were checked on 2026-09-11. Short descriptions below identify the
claim retained, not a blanket endorsement of every statement in a cited paper.
The canonical bibliography remains a shared library, so unused entries are not
silently deleted. Publisher access restrictions do not imply a nonexistent work.

| Citation | Primary source / inspected scope | Canonical treatment |
|---|---|---|
| MemGPT | [paper](https://arxiv.org/abs/2310.08560), abstract | Tiered context management; removed unsupported extended-dialogue degradation and missing-feature catalogue |
| Mem0 | [paper §2–3](https://arxiv.org/html/2504.19413v1) | ADD/UPDATE/DELETE/NOOP and graph variant; LoCoMo scope; corrected Dev Khant author name |
| MemoryBank | [paper](https://arxiv.org/abs/2305.10250), abstract | Time-dependent forgetting/reinforcement; no guaranteed persona-invariance claim |
| Generative Agents | [paper](https://arxiv.org/abs/2304.03442), abstract | Observation, reflection and retrieval architecture |
| Zep | [paper §§3–4, Table 2](https://arxiv.org/html/2501.13956v1) | Temporal graph; 63.8/71.2 contextual S scores; answerers, official GPT-4o judge, 1.6K mean context distinguished from cap |
| Hindsight | [v1 §§7.2–7.4, Table 3](https://arxiv.org/html/2512.12818v1) | Contextual 83.6/91.4 S scores; GPT-OSS-120B judge; distinct writer/answerer; v1 retrieval-budget placeholder recorded as NR, not guessed |
| HippoRAG | [paper](https://arxiv.org/abs/2405.14831), abstract | Graph and personalised PageRank; no universal inability to accept updates |
| GraphRAG | [paper](https://arxiv.org/abs/2404.16130), abstract | Graph/community summarisation; removed universal static-corpus limitation |
| KGP | [paper](https://arxiv.org/abs/2308.11730), abstract | Passage graph navigation for multi-document QA |
| Think-on-Graph | [paper](https://arxiv.org/abs/2307.07697), abstract; OpenReview challenge blocked | Graph reasoning paths; no claims about all missing lifecycle features |
| RAG | [paper](https://arxiv.org/abs/2005.11401), abstract | Retrieval/generation coupling; no intrinsic incompatibility with mutable corpora |
| REALM | [PMLR](https://proceedings.mlr.press/v119/guu20a.html) | Retriever in language-model pretraining |
| Fusion-in-Decoder | [ACL](https://aclanthology.org/2021.eacl-main.74/) | Passage aggregation |
| REPLUG | [ACL](https://aclanthology.org/2024.naacl-long.463/) | Retrieval with black-box language models |
| CRAG | [paper](https://arxiv.org/abs/2401.15884), abstract | Retrieved-evidence evaluation |
| Self-RAG | [paper](https://arxiv.org/abs/2310.11511), abstract | Learned retrieval/critique; no closest-system ranking |
| BM25 | existing verified DOI 10.1561/1500000019; publisher access failed this pass | Foundation cited; ICE implementation explicitly distinguished |
| RRF | [author-affiliated publication record](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/), tagged formula | Rank fusion; no guarantee a leg cannot dominate |
| HyDE | [ACL](https://aclanthology.org/2023.acl-long.99/) | Hypothetical-document embeddings; v2 experimental status separately audited |
| Multi-Session Chat | [paper](https://arxiv.org/abs/2107.07567), abstract | Uses previous sessions; no blanket endpoint-only categorisation |
| Conversation Chronicles | [paper](https://arxiv.org/abs/2310.13420), abstract | Temporal intervals and speaker relationships |
| LoCoMo | [paper](https://arxiv.org/abs/2402.17753), abstract | Long-history QA and other tasks |
| LongMemEval | [paper](https://arxiv.org/html/2410.10813v2), benchmark and judge sections | Five abilities; official judge prompt; complementary endpoint regime, not incapable of evaluating updates |
| LaMP | [paper](https://arxiv.org/abs/2304.11406), abstract | Profile-based personalisation |
| Lost in the Middle | [paper](https://arxiv.org/abs/2307.03172), abstract | Context position affects use; no universal capacity threshold |
| RULER | [paper](https://arxiv.org/abs/2404.06654), abstract | Effective context can differ from advertised context |
| MT-Bench judging | [paper](https://arxiv.org/abs/2306.05685), abstract | LLM judging and known biases |
| G-Eval | [paper](https://arxiv.org/abs/2303.16634), abstract | Model-based evaluation; not validation of this paper's judges |
| Unfair Evaluators | [paper](https://arxiv.org/abs/2305.17926), abstract | Position bias; shuffled presentation does not prove unbiased judging |

Removed the SCMoE digression from the canonical argument: internal expert
contrast is not evidence about external memory evolution. Unused foundational
references remain in the shared bibliography. The stale network prohibition in
verify_citations.py was corrected; metadata checking is distinguished from
claim checking.

## Final scoring and consistency pass (2026-09-12)

- Reproduced all 1,211 merged LSREP score dictionaries before running alternatives.
  `scoring_sensitivity.py` crosses manual merge/no merge and archived fallback/
  paired explicit-score-only selection. Ordinary-density differences remain near
  zero; complete-case selection removes most density failures and does not
  replace the reliability result.
- Explicit generalist score origins: ICE 1,211; vector 1,079 explicit, 130
  failed-answer assignments, two sibling-routing substitutions. With no manual
  merge: vector 1,067, 141, three. Rounded-average/default-3 fallbacks are unused.
- Added paired ordinal win/tie/loss and net superiority with whole-probe
  bootstrap uncertainty. The direction-only principle is supported by
  [NIST's sign-comparison documentation](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm).
  We do not apply an independent-observation sign test to repeated probes.
- The ablation script previously resampled identical contrasts independently
  in different report views. Shared cached contrasts and generated TeX macros
  now give one reproducible interval per contrast; raw labels are unchanged.
- Corrected ordinary-density ICE-MoE minus generalist from +0.01 to +0.03
  (4.283822 minus 4.255440); this is a small observed difference, not equivalence.
- Added future requirements for answerer reruns, crossed judge families, and
  strong simple baselines at matched budgets. No model-family bias or superiority
  over mismatched published systems is claimed.
- Diagram connectors now separate state reads and background writes. Retrieval
  schematic names source-type budgeting, sequential curation, and six *defined*
  legs without implying all six were effective.
