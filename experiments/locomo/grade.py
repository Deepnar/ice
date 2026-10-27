"""FINAL's LoCoMo grading, the way memory papers grade it (maintainer, 2026-10-28) — Mem0's
evaluation (github.com/mem0ai/mem0, Apache-2.0, evaluation/metrics at commit aae5989), the most
reproduced LoCoMo protocol:

- **J** (LLM-as-a-Judge): its prompt verbatim (J_PROMPT below), one user message, temperature 0,
  a JSON label CORRECT / WRONG;
- **F1** on lower-cased word sets (its `simple_tokenize`);
- **BLEU-1** with NLTK's word tokenizer and smoothing method 1 (`preserve_line=True`: the
  sentence splitter needs a data download and changes nothing for short answers);
- categories 1–4 only — category 5 (adversarial) is excluded, as papers do (adapter.questions).

The judge is FINAL's (Muse 1.3), not gpt-4o-mini, so J is labelled non-official. J is generous
by design ("touches on the same topic"); an audit found the common judge accepts ~63% of
deliberately wrong answers — stated beside the table, not corrected.

Durable: one row per context in <out>/locomo-grades.jsonl and one cached judge call per answer in
<out>/locomo-raw.jsonl; a rerun scores only what is missing. Writes <out>/locomo-report.json.

    uv run --no-sync python experiments/locomo/grade.py --units <read dir> --answers <answers dir> \
        --occurrences logs/final-locomo/conv-26/occurrences.json --out <dir> [--judge muse]
"""

import argparse
import collections
import contextlib
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-final-locomo-grade-1"
# Mem0's ACCURACY_PROMPT, verbatim (Apache-2.0; mem0ai/mem0 evaluation/metrics/llm_judge.py @ aae5989).
J_PROMPT = '\nYour task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:\n    (1) a question (posed by one user to another user), \n    (2) a ’gold’ (ground truth) answer, \n    (3) a generated answer\nwhich you will score as CORRECT/WRONG.\n\nThe point of the question is to ask about something one user should know about the other user based on their prior conversations.\nThe gold answer will usually be a concise and short answer that includes the referenced topic, for example:\nQuestion: Do you remember what I got the last time I went to Hawaii?\nGold answer: A shell necklace\nThe generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. \n\nFor time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n\nNow it\'s time for the real question:\nQuestion: {question}\nGold answer: {gold_answer}\nGenerated answer: {generated_answer}\n\nFirst, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. \nDo NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\nJust return the label CORRECT or WRONG in a json format with the key as "label".\n'


def simple_tokenize(text):
    """Mem0's tokenizer for F1."""
    return str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def f1(prediction, reference):
    """Mem0's F1: overlap of the two answers' word sets."""
    pred, ref = set(simple_tokenize(prediction)), set(simple_tokenize(reference))
    common = pred & ref
    if not pred or not ref or not common:
        return 0.0
    precision, recall = len(common) / len(pred), len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction, reference):
    """Mem0's BLEU-1. NLTK refuses to import packages under the working directory, and ICE's
    environment lives inside the repository, so NLTK runs from a neutral one."""
    with contextlib.chdir(tempfile.gettempdir()):
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        tok = lambda s: nltk.word_tokenize(str(s).lower(), preserve_line=True)
        try:
            return sentence_bleu([tok(reference)], tok(prediction), weights=(1, 0, 0, 0),
                                 smoothing_function=SmoothingFunction().method1)
        except Exception:                                   # noqa: BLE001 — Mem0 scores these 0
            return 0.0


def label(text):
    """CORRECT -> 1, WRONG -> 0, None when the reply carries no label."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        value = str(json.loads(body).get("label", "")).upper()
    except (ValueError, AttributeError):
        value = "CORRECT" if "CORRECT" in body.upper() and "WRONG" not in body.upper() else \
            "WRONG" if "WRONG" in body.upper() and "CORRECT" not in body.upper() else ""
    return 1 if value == "CORRECT" else 0 if value == "WRONG" else None


def contexts(units_dir, answers_dir):
    index = json.loads((Path(answers_dir) / "index.json").read_text())["index"]
    rows = {}
    for line in (Path(answers_dir) / "answers.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["prompt_sha256"]] = r
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        answer = rows.get(index.get(f"{unit['condition']}|{unit['ask']['id']}"))
        if answer is not None:
            yield unit, answer


def run(units_dir, answers_dir, occurrences_path, out, judge, *, say=print):
    """judge(prompt) -> reply text (a durable call)."""
    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads(Path(occurrences_path).read_text())}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store, done = out / "locomo-grades.jsonl", {}
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    unlabelled = 0
    for unit, answer in contexts(units_dir, answers_dir):
        key = f"{unit['condition']}|{unit['ask']['id']}"
        o = occ.get(unit["ask"]["id"])
        if key in done or o is None:
            continue
        gold = str(o["reference"]["answer"])
        if answer.get("finish") != "complete":
            row = dict(key=key, condition=unit["condition"], category=o["type"], status="ungraded",
                       reason=answer.get("finish"))
        else:
            j = label(judge(J_PROMPT.format(question=o["question"], gold_answer=gold,
                                            generated_answer=answer["text"])))
            if j is None:
                unlabelled += 1
                say(f"WARNING: no CORRECT/WRONG label for {key}; left for the next run")
                continue
            row = dict(key=key, condition=unit["condition"], category=o["type"], status="graded", j=j,
                       f1=round(f1(answer["text"], gold), 4), bleu1=round(bleu1(answer["text"], gold), 4))
        with store.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[key] = row
    (out / "locomo-report.json").write_text(json.dumps(report(done.values()), indent=1) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending=unlabelled, finished=not unlabelled)) + "\n")
    return unlabelled


def report(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        if r["status"] == "graded":
            cells[(r["condition"], "all")].append(r)
            cells[(r["condition"], r["category"])].append(r)
    out = collections.defaultdict(dict)
    for (cond, cat), rs in sorted(cells.items()):
        out[cond][cat] = dict(n=len(rs), j=round(statistics.fmean(r["j"] for r in rs), 4),
                              f1=round(statistics.fmean(r["f1"] for r in rs), 4),
                              bleu1=round(statistics.fmean(r["bleu1"] for r in rs), 4))
    return dict(policy=POLICY, label="LoCoMo, Mem0's protocol (J non-official: FINAL's judge)", by_condition=dict(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default="muse")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades are private: keep them under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2 import ledger_merge as M
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    generate, meta = generator(args.judge).generate, metadata(args.judge)
    calls = M.Calls(args.out / "locomo-raw.jsonl")
    # one provider session per conversation, as the other judges keep one per history: the prompt's
    # fixed preamble can then be served from the provider's cache (the prompt itself is unchanged)
    session = f"ice-final-locomo-{hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:16]}"

    def judge(prompt):
        identity = hashlib.sha256(json.dumps(dict(policy=POLICY, meta=meta, prompt=prompt), sort_keys=True).encode()).hexdigest()

        def call():
            result = generate([{"role": "user", "content": prompt}], temperature=0, max_output_tokens=2000,
                              session_id=session)
            return result.text, result.usage
        return calls.get(identity, call)["text"]

    pending = run(args.units, args.answers, args.occurrences, args.out, judge)
    print(json.dumps(dict(pending=pending)))
    return 3 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
