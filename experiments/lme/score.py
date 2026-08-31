#!/usr/bin/env python3
"""Score a LongMemEval run using the benchmark's OWN judge protocol.

The five prompt templates and the decision rule below are reproduced verbatim
from the official evaluator:

    https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py
    (fetched 2026-08-29; MIT)

Reproduced rather than paraphrased on purpose: a rewritten rubric would make the
number incomparable to every published LongMemEval result, which is the entire
reason for running this benchmark.

⚑ ONE DELIBERATE DEVIATION, AND IT MUST BE STATED WITH THE NUMBER. The official
evaluator judges with GPT-4o. This runs a LOCAL judge, so results are
"LongMemEval protocol, local judge" and NOT directly comparable to published
GPT-4o-judged figures. The judge is also held disjoint from the answerer
(`gemma4:26b-a4b-it-q4_K_M`) so a model never grades its own output.

Runs separately from lme_run.py, so the answerer and the judge are never resident
at once -- both are ~17 GB against a 23.5 GB ceiling.

    uv run python experiments/lme/score.py --phase oracle
    uv run python experiments/lme/score.py --phase oracle --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = HARNESS_DIR / "runs" / "v2-paper-eval"

OLLAMA_URL = "http://localhost:11434/v1"
# ⚑ JUDGE CHOICE IS PINNED TO THE PAPER, NOT TO CONVENIENCE.
# The paper judged every Exp-1, Exp-2 and Exp-3 probe with a Gemma-4 12B --
# `mattbucci/gemma-4-12B-AWQ` on SGLang at 150k context, temperature 0.0. Using a
# different judge here would make an LME-v2 number incomparable to the paper's own
# numbers, which is half the point of running it at the tag.
#
# ⚠ RECORDED DEVIATION: this is the OLLAMA GGUF build of Gemma-4 12B, not the AWQ
# build on SGLang. Same family and parameter count, different quantization and
# serving stack. For exact parity, serve mattbucci/gemma-4-12B-AWQ on SGLang and
# pass --judge with its name.
#
# An earlier revision defaulted to mistral-nemo purely because it emitted text at
# LongMemEval's max_tokens=10 judge cap. That optimised for the cap instead of for
# judgement quality, and picked a weaker model unrelated to the paper. Corrected.
DEFAULT_JUDGE = "gemma4:12b"

# ⚑ DEVIATION FROM LongMemEval's max_tokens=10 judge cap, and why it is safe.
# Gemma-4 12B reasons before answering, so at a 10-token cap it hits `length` and
# returns an EMPTY content string -- and the official rule `'yes' in
# response.lower()` reads empty as "no", scoring correct answers wrong. Ollama
# exposes no way to disable thinking on this build (`chat_template_kwargs` and
# `think` were both tried and ignored). Raising the cap lets it finish and emit a
# verdict.
#
# ⚑ 256 WAS NOT ENOUGH AND THE FIRST FULL RUN WAS VOIDED BY IT. On real answers
# the judge returned EMPTY for 413 of 1000 judgements (41%), all silently scored
# 'no'. Worse, the failure was BIASED: ICE's answers are ~1.8x longer than the
# baseline's (337 vs 188 median chars), so ICE hit the cap more often -- 44.0% vs
# 38.6% -- and the arm under test was penalised for verbosity. Measured on the
# actual failing cases: 256 -> 1/6 verdicts, 1024 -> 6/6, 2048 -> 6/6 and slower.
#
# The DECISION RULE is untouched, and reasoning tokens never reach it: the API
# returns reasoning separately from `message.content`, so the rule still reads a
# bare verdict. This changes only whether the judge gets to speak, not how it is
# scored.
JUDGE_MAX_TOKENS = 1024

_stop = False


def _on_signal(signum, _frame):
    global _stop
    _stop = True
    print(f"\n  [signal {signum}] stopping after the current judgement.", flush=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --- official prompts, verbatim ------------------------------------------------
_T_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_T_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_T_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_T_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_T_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
    "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def get_anscheck_prompt(task, question, answer, response, abstention=False) -> str:
    if abstention:
        return _T_ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _T_DEFAULT.format(question, answer, response)
    if task == "temporal-reasoning":
        return _T_TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _T_KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _T_PREFERENCE.format(question, answer, response)
    raise NotImplementedError(f"unknown question_type: {task}")


def judge_selftest(client, model: str, max_tokens: int = JUDGE_MAX_TOKENS) -> bool:
    """Refuse to score with a judge that cannot actually judge.

    ⚑ THIS EXISTS BECAUSE THE FAILURE IS INVISIBLE IN THE OUTPUT. The official
    decision rule is `'yes' in response.lower()`, so a judge returning an empty
    string labels everything WRONG and the run reports a clean, plausible 0.0%.
    That happened here: qwen3.8:27b returned '' on every call and scored 0/4 on
    four answers that were all correct on inspection.

    Two cases, not one -- a judge that always says "yes" is as useless as a mute
    one, so the test checks that it DISCRIMINATES.
    """
    # ⚑ THE LONG CASE IS THE POINT. A short toy case passes at any cap and is what
    # let a 41%-empty run through: the self-test did not resemble the workload. The
    # third case below is the length of a real ICE answer, which is exactly what
    # exhausts a thinking judge's budget.
    _long_ok = (
        "Based on what you told me earlier, you had **two** doctor's appointments "
        "in March. The first was on **March 3rd** with your primary care physician, "
        "Dr. Smith, where you were diagnosed with bronchitis and prescribed a course "
        "of antibiotics. The second was a follow-up on **March 20th** with your "
        "orthopedic surgeon, Dr. Thompson, regarding the knee you injured while "
        "running in February. You also mentioned rescheduling a dermatology "
        "appointment out of March entirely, so it does not count toward the total."
    )
    cases = [
        ("How long did I wait?", "over a year", "The process took over a year.", True),
        ("How long did I wait?", "over a year", "It took about three days.", False),
        ("How many doctor's appointments did I go to in March?", "2", _long_ok, True),
    ]
    for question, answer, response, expected in cases:
        prompt = get_anscheck_prompt("multi-session", question, answer, response)
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                n=1, temperature=0, max_tokens=max_tokens)
            raw = (r.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"⛔ JUDGE SELF-TEST: {model} call failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            return False
        if not raw:
            print(f"⛔ JUDGE SELF-TEST: {model} returned EMPTY even at\n   max_tokens={max_tokens}.\n"
                  f"   Thinking models do this -- they spend the budget reasoning.\n"
                  f"   Every label would default to False and the run would report\n"
                  f"   a plausible 0.0%. Pick a non-thinking judge.", file=sys.stderr)
            return False
        if ("yes" in raw.lower()) != expected:
            print(f"⛔ JUDGE SELF-TEST: {model} got a trivial case wrong.\n"
                  f"   expected {'yes' if expected else 'no'}, got {raw!r}",
                  file=sys.stderr)
            return False
    print(f"  judge self-test OK: {model} discriminates at max_tokens={max_tokens}")
    return True


def _judgement_spoke(entry: dict) -> bool:
    """Read both current and pre-`spoke` judgement records consistently."""
    return entry.get("spoke", bool((entry.get("judge_raw") or "").strip()))


def partition_judgements(records, judge_dir: Path, *, retry_mutes: bool = False,
                         retry_condition: str | None = None):
    """Return existing usable judgements and the exact conditions to judge.

    Normal mode resumes missing/unreadable files. ``retry_mutes`` additionally
    puts existing mute files back into the pending set, optionally for one arm
    only. Spoken files and non-selected arms are never overwritten.
    """
    judged: list[dict] = []
    pending: list[tuple[dict, str]] = []
    for rec in records:
        for cond in (rec.get("answers") or {}):
            jpath = judge_dir / f"{rec['question_id']}__{cond}.json"
            existing = None
            if jpath.exists():
                try:
                    existing = json.loads(jpath.read_text())
                except (json.JSONDecodeError, OSError):
                    existing = None

            selected = retry_condition is None or cond == retry_condition
            if (existing is not None and retry_mutes and selected
                    and not _judgement_spoke(existing)):
                pending.append((rec, cond))
            elif existing is not None:
                judged.append(existing)
            elif not retry_mutes or selected:
                pending.append((rec, cond))
    return judged, pending


# Above this share of mute judgements the table is not reportable. Set low on
# purpose: mute judgements are not noise, they are one-directional false negatives,
# and they fall hardest on whichever arm writes longer answers.
MAX_MUTE_RATE = 0.02


def report(judged: list[dict]) -> None:
    """Per-question-type accuracy, per condition -- the table the paper needs."""
    by_cond: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    mute: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    for j in judged:
        bucket = "abstention" if j["is_abstention"] else j["question_type"]
        tot[j["condition"]] += 1
        # ⚑ A mute judgement is MISSING DATA, not a wrong answer. Counting it as a
        # failure is precisely the bug that voided the first run. It is excluded
        # from the denominator and reported separately, so the reader can bound its
        # effect instead of silently absorbing it as a loss for the wordier arm.
        # Older judgement files predate the `spoke` field; fall back to judge_raw.
        if not j.get("spoke", bool((j.get("judge_raw") or "").strip())):
            mute[j["condition"]] += 1
        else:
            by_cond[j["condition"]][bucket].append(j["label"])

    worst = max((mute[c] / tot[c] for c in tot if tot[c]), default=0.0)
    print("\njudge health — judgements EXCLUDED because the judge returned nothing")
    print("(excluded from the denominator, never counted as wrong):")
    for c in sorted(tot):
        r = mute[c] / tot[c] if tot[c] else 0
        print(f"  {c:12} {mute[c]:5d} / {tot[c]:<5d} ({100*r:5.1f}%)")

    # ⚑ Bound the effect instead of merely flagging it. Excluded judgements are
    # missing, not wrong, so the true accuracy lies between "all excluded were
    # wrong" and "all excluded were right". If the ordering between conditions
    # survives both bounds, the missing data cannot have caused it.
    print("\nworst/best case if every excluded judgement went against / for each arm:")
    bounds = {}
    for c in sorted(tot):
        got = [v for vs in by_cond[c].values() for v in vs]
        k = sum(got)
        lo = k / tot[c] if tot[c] else 0                    # all excluded wrong
        hi = (k + mute[c]) / tot[c] if tot[c] else 0        # all excluded right
        bounds[c] = (lo, hi)
        print(f"  {c:12} {100*lo:5.1f}%  ..  {100*hi:5.1f}%   (n={tot[c]})")
    if len(bounds) == 2:
        (a, (alo, ahi)), (b, (blo, bhi)) = sorted(bounds.items())
        if alo > bhi or blo > ahi:
            hi_arm = a if alo > bhi else b
            print(f"  → ordering is ROBUST: {hi_arm} leads under every imputation.")
        else:
            print("  → ordering is NOT robust to the excluded judgements; "
                  "the gap is within the missing data.")

    all_types = sorted({b for c in by_cond.values() for b in c})
    conds = sorted(by_cond)
    width = max((len(t) for t in all_types), default=10) + 2

    print(f"\n{'question type':<{width}}" + "".join(f"{c:>16}" for c in conds))
    print("-" * (width + 16 * len(conds)))
    for t in all_types:
        row = f"{t:<{width}}"
        for c in conds:
            xs = by_cond[c].get(t, [])
            row += f"{(f'{100*sum(xs)/len(xs):.1f}% ({len(xs)})' if xs else '—'):>16}"
        print(row)
    print("-" * (width + 16 * len(conds)))
    row = f"{'OVERALL':<{width}}"
    for c in conds:
        xs = [v for vs in by_cond[c].values() for v in vs]
        row += f"{(f'{100*sum(xs)/len(xs):.1f}% ({len(xs)})' if xs else '—'):>16}"
    print(row)
    if worst > MAX_MUTE_RATE:
        print(f"\n⚠ {100*worst:.1f}% of judgements were excluded as unobtainable "
              f"(>{100*MAX_MUTE_RATE:.0f}%). Quote the bounds above, not just the point "
              f"estimates.")
    print("\n⚑ LongMemEval protocol with a LOCAL judge — not directly comparable to")
    print("  published GPT-4o-judged numbers. System under test: ICE v2 @ v2-paper-eval.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS,
                    help="judge token cap; raise for the tail of long answers "
                         "that still come back mute at the default")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent judge requests; the server batches them")
    ap.add_argument("--retry-mutes", action="store_true",
                    help="rejudge only existing mute files (plus missing files); "
                         "spoken judgements are never overwritten")
    ap.add_argument("--condition", choices=("full_ice", "vector_rag"),
                    help="limit --retry-mutes/missing work to one condition")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the table from existing judgements; judges nothing")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    phase_dir = args.out / args.phase
    answers_dir = phase_dir / "answers"
    judge_dir = phase_dir / "judgements"
    if not answers_dir.is_dir():
        print(f"⛔ no answers at {answers_dir}", file=sys.stderr)
        return 1

    records = []
    for p in sorted(answers_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("status") == "complete":
            records.append(rec)

    judged, pending = partition_judgements(
        records, judge_dir,
        retry_mutes=args.retry_mutes,
        retry_condition=args.condition,
    )

    print(f"phase={args.phase}  complete instances={len(records)}")
    print(f"  judged already {len(judged)}  pending {len(pending)}  judge={args.judge}")

    if args.report_only or not pending:
        if judged:
            report(judged)
        else:
            print("  nothing judged yet.")
        return 0

    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")
    if not judge_selftest(client, args.judge, args.max_tokens):
        print("\n   Nothing judged. Fix the above and re-run.", file=sys.stderr)
        return 3
    started = time.time()

    def judge_one(item):
        """One judgement, written atomically by its own worker.

        Judging is I/O-bound on the HTTP call and the server batches internally, so
        a small pool overlaps requests. The resume guarantee is unchanged: a
        judgement file exists only once that judgement is complete, so killing this
        mid-flight loses at most the in-flight few.
        """
        rec, cond = item
        if _stop:
            return None
        qid = rec["question_id"]
        hyp = (rec["answers"][cond] or {}).get("answer") or ""
        prompt = get_anscheck_prompt(
            rec["question_type"], rec["question"], rec["reference_answer"], hyp,
            abstention=rec.get("is_abstention", False),
        )
        try:
            completion = client.chat.completions.create(
                model=args.judge,
                messages=[{"role": "user", "content": prompt}],
                n=1, temperature=0, max_tokens=args.max_tokens,
            )
            raw = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⛔ {qid} [{cond}] judge failed: {exc}", flush=True)
            return None

        entry = {
            "question_id": qid,
            "condition": cond,
            "question_type": rec["question_type"],
            "is_abstention": rec.get("is_abstention", False),
            # ⚑ `spoke` is the guard the first run lacked. The official rule maps an
            # empty string to False, indistinguishable from a real "no" once
            # written. Recording it lets report() refuse a contaminated table.
            "spoke": bool(raw),
            "label": "yes" in raw.lower(),   # official decision rule
            "judge_raw": raw,
            "judge_model": args.judge,
            "judged_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(judge_dir / f"{qid}__{cond}.json", entry)
        return entry

    done_n = 0
    mute_n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for entry in pool.map(judge_one, pending):
            done_n += 1
            if entry is not None:
                judged.append(entry)
                if not entry["spoke"]:
                    mute_n += 1
            if done_n % 25 == 0 or done_n == len(pending):
                rate = done_n / max(1e-9, time.time() - started)
                eta = (len(pending) - done_n) / rate if rate else 0
                print(f"  [{done_n}/{len(pending)}] {time.time()-started:.0f}s "
                      f"({rate:.2f}/s, eta {eta/60:.0f}m)  mute so far: {mute_n}",
                      flush=True)

    report(judged)
    print(f"\njudgements: {judge_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
