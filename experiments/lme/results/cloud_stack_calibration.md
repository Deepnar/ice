# ICE-v2 LongMemEval cloud-stack calibration

**Status:** bounded model selection and isolated smoke only. Neither final
500-question phase has started.

## Selected configuration

- Answerer: `gpt-5.6-luna` through OpenCode Go `/responses`.
- Judge: `muse-spark-1.3-contributor` through OpenCode Go `/responses`.
- Background: exact evaluated `qwen3:4b-instruct-bg` through Ollama, kept
  resident while the answerer runs remotely.
- System: frozen ICE v2 at `v2-paper-eval`, adapter
  `ice-v2-lme-sessions-v2`, isolated v2-schema database.

Muse is the judge rather than the answerer because its prior 15-item
human-labelled judge calibration was the strongest available (73%). It also
passed LongMemEval's three-case yes/no discrimination self-test through the
Responses endpoint. Luna had no corresponding judge evidence and was evaluated
as an answerer instead.

## Answerer calibration

Fourteen public LongMemEval oracle questions: two from each of the six question
types plus two abstention cases. Every model received the same complete
evidence-only history and question. Muse applied LongMemEval's official prompt
and yes/no decision rule. These small-n point estimates select a run
configuration; they are not paper results.

| Answerer | Muse-correct | Mean answer seconds | Mean output tokens |
|---|---:|---:|---:|
| **GPT-5.6 Luna** | **13/14** | **3.2** | **84** |
| Omen Alpha | 13/14 | 3.2 | 324 |
| DeepSeek V4 Flash | 13/14 | 4.9 | 206 |
| Qwen3.8 Flash | 13/14 | 7.3 | 390 |
| LongCat 2.0 | 13/14 | 14.6 | 404 |
| MiMo V2.5 | 12/14 | 15.2 | 228 |

Luna, Omen, DeepSeek, Qwen, and LongCat all failed the same adversarial
abstention question. Omen noticed that the queried role title was absent but
still supplied counts from a different role; Luna supplied one of those counts
without the qualification. MiMo correctly abstained there but gave wrong counts
on one multi-session and one knowledge-update question. Muse's labels agree with
manual inspection on these disagreements.

Luna and Omen both passed the same long-context needle check. Luna recovered a
beginning-of-prompt identifier from 88,025 input tokens in 4.1 seconds with 18
output tokens. Omen recovered its identifier from 88,057 input tokens in 6.6
seconds with 90 output tokens. Luna is selected because it ties the best point
accuracy and short-prompt speed while being materially less verbose and faster
on the long-context control.

## Endpoint compatibility

- Muse: `/responses` works; `/chat/completions` returns HTTP 500.
- Luna: `/responses` works but rejects the `temperature` parameter; the adapter
  omits it explicitly and records that fact.
- Omen Alpha: `/chat/completions` works; `/responses` returns HTTP 500.
- MiMo and LongCat: Chat Completions works; Responses returned HTTP 500.
- Qwen3.8 Flash and DeepSeek V4 Flash: Chat Completions works.

The harness uses explicit endpoint profiles; it does not silently fall back
between API families. Following OpenCode's 2026-09-06 requirement, every cloud
request carries `x-opencode-session`; the harness derives a stable UUID from the
phase, question, condition, and provider profile rather than generating a new id
on retry.

## Background serving decision

The attempted `Qwen/Qwen3-4B-AWQ` vLLM sidecar is rejected for this experiment.
Under the non-thinking Hugging Face template it returned one reversed control
fact (`vertex labs --works_at--> berlin`) instead of both expected user facts.
Under the literal Ollama-style template, reasoning leaked into the output and a
long-turn extraction yielded no usable triplets. The exact Ollama model returned
both directed control facts, a non-empty 63-character summary, and 11 triplets
from a real 6,731-character LongMemEval turn in 3.3 seconds.

Cloud answering removes the large local model and its swap, so Ollama can keep
the 2.5 GB background model resident. Preserving memory construction is worth
more than the unearned vLLM speedup.

## End-to-end and interruption validation

- Isolated oracle instance under `/tmp`: 6 pairs/12 raw turns ingested in 28 s;
  both Luna answer arms written atomically; Muse wrote both judgements with no
  mute. Full ICE lacked the answer and scored 0; vector-RAG answered “over a
  year” and scored 1.
- Re-running generation reported `done 1, remaining 0`; re-running scoring
  reported two existing judgements and zero pending. Neither loaded a model or
  called a provider.
- Disposable full-S SIGINT: stopped after 5/259 pairs, wrote no answer and no
  `.partial`. Restart discarded the partial store and began from pair one; a
  second stop after 4 pairs again wrote no answer.
- Deliberately planted wrong answer/judge profile artifacts are rejected with
  exit code 5 rather than overwritten or mixed.

Raw calibration answers and judgements remain local under
`experiments/lme/runs/model-calibration-v1/` and are not part of this aggregate
report.
