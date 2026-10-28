"""Judge the judge (Z2 RETESTS R5): grade-2 on a stratified sample of grade-1 verdicts.

Draws answers already graded under grade-1, mostly from the strata F12 found mis-graded, the
rest controls, half from ICE settings and half from plain search; grades them under grade-2;
re-grades 20 with one true extra detail from elsewhere in the conversation appended
(invariance); reports verdict changes per stratum, grade-2's agreement with the per-point
rescoring of grade-1 (which decides whether Z2's rescored comparisons stand in for a re-grade),
and writes a blind sheet of every changed verdict for the maintainer.

    uv run --no-sync python experiments/z2/judge_check.py --history B --dry-run \
        --grades logs/z2-partial-B/graded.json --run logs/z2-run-main --questions logs/z2-questions-final \
        --out logs/z2-judge-check/B

`--dry-run` draws and builds every request without a model call. Grades are private: --out
stays under logs/.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-judge-check-1"
# stratum: (share of the sample, rule on a grade-1 verdict); issue strata first — an answer
# counts in the first stratum it matches
STRATA = (
    ("partial, every point covered", 60, lambda r, full: r["grade"] == "partial" and full),
    ("incorrect, every point covered", 40, lambda r, full: r["grade"] == "incorrect" and full),
    ("downgraded citing dates/turns", 30, lambda r, full: r["grade"] != "correct"
     and bool(re.search(r"date|timestamp|turn", r.get("reason", ""), re.I))),
    ("control: correct", 35, lambda r, full: r["grade"] == "correct"),
    ("control: incorrect, points missing", 35, lambda r, full: r["grade"] == "incorrect" and not full),
    ("control: partial, points missing", 30, lambda r, full: r["grade"] == "partial" and not full),
)
INVARIANCE = 20


def covered_all(r):
    return bool(r["items_total"]) and len(set(r["items_covered"])) >= r["items_total"]


def per_point(r):
    if not r["items_total"]:
        return r["grade"] == "correct"
    return covered_all(r) and not r["superseded_used"]


def draw(graded, scale=1.0, seed=20261029):
    """[(stratum, condition, ask id)] — deterministic; half ICE settings, half plain search."""
    rng = random.Random(seed)
    pools = {name: ([], []) for name, _, _ in STRATA}
    for key in sorted(graded):
        r = graded[key]
        if r.get("status") != "graded":
            continue
        cond, ask = key.split("|", 1)
        if cond == "recent_only":
            continue
        full = covered_all(r)
        for name, _, rule in STRATA:
            if rule(r, full):
                pools[name][cond.startswith("bare")].append((name, cond, ask))
                break
    sample = []
    for name, size, _ in STRATA:
        n = round(size * scale)
        ice, plain = pools[name]
        take_plain = min(len(plain), n // 2)
        take_ice = min(len(ice), n - take_plain)
        sample += rng.sample(ice, take_ice) + rng.sample(plain, take_plain)
    return sample


def bodies(sample, graded, run, history, questions):
    """The grade-2 request body for each drawn answer, as grade.py builds it."""
    from experiments.z2 import grade as G
    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads((questions / history / "occurrences.json").read_text())}
    index = json.loads((run / f"s10-{history}" / "index.json").read_text())["index"]
    wanted = {index.get(f"{c}|{a}") for _, c, a in sample}
    texts = {}
    with open(run / f"s10-{history}" / "answers.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["prompt_sha256"] in wanted and row.get("finish") == "complete":
                texts[row["prompt_sha256"]] = row["text"]
    dates = G.turn_dates(history)
    out = []
    for stratum, cond, ask in sample:
        o, text = occ.get(ask), texts.get(index.get(f"{cond}|{ask}"))
        if not o or text is None:
            continue
        ref = o["reference"]
        out.append(dict(stratum=stratum, condition=cond, ask=ask, grade1=graded[f"{cond}|{ask}"],
                        body=dict(QUESTION=o["question"], REQUIRED=ref.get("required", []),
                                  SOURCES=G.sources(ref, dates), SUPERSEDED=ref.get("superseded", []),
                                  EXPECT_ABSTENTION=bool(o.get("expected_abstention")), ANSWER=text)))
    return out


def with_true_detail(item, history, seed=20261029):
    """The same answer plus one sentence from a turn the reference does not cite (true, unseen)."""
    from experiments.z2.ledger import load_history, role_text
    originals = load_history(history)
    cited = {s["turn"] for s in item["body"]["SOURCES"]}
    rng = random.Random(f"{seed}|{item['ask']}|{item['condition']}")
    for _ in range(50):
        turn = rng.randrange(1, len(originals) + 1)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", role_text(originals, turn, "user")) if len(s.split()) >= 6]
        if turn not in cited and sentences:
            body = dict(item["body"], ANSWER=item["body"]["ANSWER"].rstrip()
                        + f"\n\n(Earlier you also mentioned: \"{rng.choice(sentences)[:300]}\")")
            return dict(item, stratum="invariance", body=body)
    return None


def report(items, verdicts, invariance):
    """Text report: per-stratum changes, agreement with per-point rescoring, invariance."""
    lines, by = [], {}
    for it in items:
        v = verdicts.get(it["key"])
        if v:
            by.setdefault(it["stratum"], []).append((it, v))
    agree = total = 0
    lines.append(f"{'stratum':38} {'n':>4} {'grade changed':>14} {'grade-2 correct':>16}")
    for name, _, _ in STRATA:
        rows = by.get(name, [])
        changed = sum(v.get("grade") != it["grade1"]["grade"] for it, v in rows)
        correct = sum(v.get("grade") == "correct" for _, v in rows)
        agree += sum((v.get("grade") == "correct") == per_point(it["grade1"]) for it, v in rows)
        total += len(rows)
        lines.append(f"{name:38} {len(rows):4} {changed:14} {correct:16}")
    if total:
        lines.append(f"grade-2 correct agrees with per-point rescoring of grade-1 on {agree}/{total} "
                     f"({100 * agree / total:.0f}%)")
    if invariance:
        same = sum(a.get("grade") == b.get("grade") for a, b in invariance)
        lines.append(f"invariance: grade unchanged after a true extra detail on {same}/{len(invariance)}")
    return "\n".join(lines)


def blind_sheet(items, verdicts):
    """Every changed verdict, condition hidden, for the maintainer to read."""
    rows = ["# Judge check — changed verdicts (condition hidden)", ""]
    for n, it in enumerate((i for i in items if verdicts.get(i["key"]) and
                            verdicts[i["key"]].get("grade") != i["grade1"]["grade"]), 1):
        b, v = it["body"], verdicts[it["key"]]
        rows += [f"## {n}. {b['QUESTION']}", "", "**Required:** " + "; ".join(map(str, b["REQUIRED"])), "",
                 "**Sources:** " + " | ".join(f"[turn {s['turn']}, {s.get('date', '')}] {s['text']}" for s in b["SOURCES"]), "",
                 "**Answer:**", "", b["ANSWER"], "",
                 f"**grade-1:** {it['grade1']['grade']} — {it['grade1'].get('reason', '')}",
                 f"**grade-2:** {v.get('grade')} — {v.get('reason', '')}", "", "**Your call:** ", ""]
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--grades", type=Path, required=True, help="grade-1 graded verdicts (graded.json or grades.json)")
    parser.add_argument("--run", type=Path, required=True, help="the Z2 run folder (s10-<history> answers)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="B 1.0 (230 answers); C 0.22 (~50)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades are private: keep them under logs/")
    graded = json.loads(args.grades.read_text())
    graded = graded.get("graded", graded)
    items = bodies(draw(graded, args.scale), graded, args.run, args.history, args.questions)
    inv = [x for x in (with_true_detail(i, args.history) for i in items if i["stratum"] == "control: correct") if x][:INVARIANCE]
    for it in items + inv:
        it["key"] = hashlib.sha256(json.dumps(it["body"], sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{args.history}: {len(items)} answers drawn, {len(inv)} invariance pairs", flush=True)
    if args.dry_run:
        (args.out / "requests.json").write_text(json.dumps([dict(i, grade1=None) for i in items + inv], indent=1) + "\n")
        print(f"dry run: requests in {args.out / 'requests.json'}; no model call")
        return 0
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2 import grade as G, ledger_merge as M
    from experiments.z2.bg_audit import _parallel
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    ask = G.judge_caller(M.Calls(args.out / "grade2-raw.jsonl"), generator(G.JUDGE).generate, metadata(G.JUDGE), args.history)
    verdicts = {}
    for item, value, exc in _parallel(items + inv, lambda it: ask(G.GRADE_SYSTEM, json.dumps(it["body"], ensure_ascii=False))):
        if exc is None and value:
            verdicts[item["key"]] = value
    pairs = []
    for x in inv:
        base = next((i for i in items if i["ask"] == x["ask"] and i["condition"] == x["condition"]), None)
        if base and base["key"] in verdicts and x["key"] in verdicts:
            pairs.append((verdicts[base["key"]], verdicts[x["key"]]))
    text = report(items, verdicts, pairs)
    (args.out / "report.txt").write_text(text + "\n")
    (args.out / "blind-sheet.md").write_text(blind_sheet(items, verdicts) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
