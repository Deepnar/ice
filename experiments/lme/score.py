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
# ⚑ Disjoint from BOTH the answerer (gemma4:26b) and the background model
# (qwen3:4b-instruct-bg) -- a different family from each, so nothing grades work it
# helped produce.
#
# It must also emit content within LongMemEval's judge token cap of
# max_tokens=10 (their evaluate_qa.py, not ours -- and unrelated to ICE's own
# per-query token budget). That cap is not a given:
# thinking models spend the budget reasoning and return an EMPTY string, which the
# official decision rule (`'yes' in response.lower()`) silently reads as "no". The
# first judge tried here, qwen3.8:27b, did exactly that and scored 0.0% on four
# answers that were all CORRECT. Measured at that cap:
#     gemma4:12b        -> ''      (thinking, hits length)
#     granite4.2:8b     -> ''      (thinking, hits length)
#     qwen3:4b-instruct -> 'yes'
#     mistral-nemo      -> 'Yes'
DEFAULT_JUDGE = "mistral-nemo:latest"

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


def judge_selftest(client, model: str) -> bool:
    """Refuse to score with a judge that cannot actually judge.

    ⚑ THIS EXISTS BECAUSE THE FAILURE IS INVISIBLE IN THE OUTPUT. The official
    decision rule is `'yes' in response.lower()`, so a judge returning an empty
    string labels everything WRONG and the run reports a clean, plausible 0.0%.
    That happened here: qwen3.8:27b returned '' on every call and scored 0/4 on
    four answers that were all correct on inspection.

    Two cases, not one -- a judge that always says "yes" is as useless as a mute
    one, so the test checks that it DISCRIMINATES.
    """
    cases = [
        ("How long did I wait?", "over a year", "The process took over a year.", True),
        ("How long did I wait?", "over a year", "It took about three days.", False),
    ]
    for question, answer, response, expected in cases:
        prompt = get_anscheck_prompt("multi-session", question, answer, response)
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                n=1, temperature=0, max_tokens=10)
            raw = (r.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"⛔ JUDGE SELF-TEST: {model} call failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            return False
        if not raw:
            print(f"⛔ JUDGE SELF-TEST: {model} returned EMPTY at LongMemEval's\n   judge cap of max_tokens=10.\n"
                  f"   Thinking models do this -- they spend the budget reasoning.\n"
                  f"   Every label would default to False and the run would report\n"
                  f"   a plausible 0.0%. Pick a non-thinking judge.", file=sys.stderr)
            return False
        if ("yes" in raw.lower()) != expected:
            print(f"⛔ JUDGE SELF-TEST: {model} got a trivial case wrong.\n"
                  f"   expected {'yes' if expected else 'no'}, got {raw!r}",
                  file=sys.stderr)
            return False
    print(f"  judge self-test OK: {model} discriminates at LongMemEval's max_tokens=10 judge cap")
    return True


def report(judged: list[dict]) -> None:
    """Per-question-type accuracy, per condition -- the table the paper needs."""
    by_cond: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for j in judged:
        bucket = "abstention" if j["is_abstention"] else j["question_type"]
        by_cond[j["condition"]][bucket].append(j["label"])

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
    print("\n⚑ LongMemEval protocol with a LOCAL judge — not directly comparable to")
    print("  published GPT-4o-judged numbers. System under test: ICE v2 @ v2-paper-eval.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
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

    judged: list[dict] = []
    pending: list[tuple[dict, str]] = []
    for rec in records:
        for cond, ans in (rec.get("answers") or {}).items():
            jpath = judge_dir / f"{rec['question_id']}__{cond}.json"
            if jpath.exists():
                try:
                    judged.append(json.loads(jpath.read_text()))
                    continue
                except (json.JSONDecodeError, OSError):
                    pass
            pending.append((rec, cond))

    print(f"phase={args.phase}  complete instances={len(records)}")
    print(f"  judged already {len(judged)}  pending {len(pending)}  judge={args.judge}")

    if args.report_only or not pending:
        if judged:
            report(judged)
        else:
            print("  nothing judged yet.")
        return 0

    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_URL, api_key="dummy")
    if not judge_selftest(client, args.judge):
        print("\n   Nothing judged. Fix the above and re-run.", file=sys.stderr)
        return 3
    started = time.time()

    for i, (rec, cond) in enumerate(pending, 1):
        if _stop:
            break
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
                n=1, temperature=0, max_tokens=10,
            )
            raw = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⛔ {qid} [{cond}] judge failed: {exc}", flush=True)
            continue

        entry = {
            "question_id": qid,
            "condition": cond,
            "question_type": rec["question_type"],
            "is_abstention": rec.get("is_abstention", False),
            "label": "yes" in raw.lower(),   # official decision rule
            "judge_raw": raw,
            "judge_model": args.judge,
            "judged_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(judge_dir / f"{qid}__{cond}.json", entry)
        judged.append(entry)
        if i % 20 == 0 or i == len(pending):
            print(f"  [{i}/{len(pending)}] {time.time()-started:.0f}s", flush=True)

    report(judged)
    print(f"\njudgements: {judge_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
