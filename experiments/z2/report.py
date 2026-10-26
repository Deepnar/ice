"""Z2 S11 report: the numbers from grades, per history and question type (spec D11, D6e).

**Headline (D6e): `scoring`** — FINAL's measures (experiments/z2/scoring.py): M1 accuracy,
M2 supported current accuracy by evidence distance and window side, M3 cost (TUR), update
accuracy, evidence recall, context precision, the primary tests, hallucination
non-inferiority and the fidelity audit. Everything below it is kept as Z2 planned it.

Never one pooled headline. Per condition and question type: mean 1-5 score, grade
counts, superseded-value and confirmed-invented rates, abstention accuracy on
never-stated questions, item coverage (cumulative/aggregation), ungraded counts.
Paired comparisons against bare vector and warm vector: the per-family mean score
difference with a bootstrap interval over QUESTION FAMILIES (occurrences of one
family are not independent). Ranking win rates against bare and warm vector. Token
efficiency beside quality: TUR = score per 1,000 prompt tokens (full prompts on both
sides), SPF = score per retrieved fragment. Repeats as retention transitions. The
no-memory controls: retrieval rate, memory tokens and better/worse/tie vs recent-only.

    uv run --no-sync python experiments/z2/report.py --history B --units logs/z2-run-main/s9-B \
        --grades logs/z2-run-main/s11-B/grades.json --out logs/z2-run-main/s11-B/report.json
"""

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-report-2"
BASELINES = ("bare_vector", "warm_vector")
BOOTSTRAP = 2_000


def family_of(ask_id):
    return ask_id.split("@")[0]


def bootstrap(per_family, n=BOOTSTRAP, seed=0):
    """95% interval of the mean of per-family values, resampling families."""
    values = list(per_family.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)]


def unit_index(units_dir, count_messages=None):
    """Per context: type, tokens, fragments, the turns its fragments come from, its recent
    window and legs. Bare-vector units store raw chunks (`turn`, no `prompt_tokens`, no
    context ledger): their prompt is counted with ICE's own counter, and their recent window
    is recent_only's for the same ask, whose envelope they reuse (read_passes.bare_messages)."""
    if count_messages is None:
        from src.memory.tokens import count_messages
    out = {}
    for path in Path(units_dir).glob("units/*/*.json"):
        u = json.loads(path.read_text())
        stage, selected = u["stage"], u["stage"].get("selected") or []
        ledger = stage.get("context_ledger")
        prompt_tokens = stage.get("prompt_tokens")
        if prompt_tokens is None and stage.get("prompt_messages"):
            prompt_tokens = count_messages(stage["prompt_messages"])
        out[f"{u['condition']}|{u['ask']['id']}"] = dict(
            type=u["ask"].get("type"), kind=u["ask"]["kind"],
            prompt_tokens=prompt_tokens, memory_tokens=stage.get("selected_tokens") or 0,
            fragments=len(selected),
            source_turns=sorted({t for f in selected for t in (f.get("source_turns") or
                                                                ([f["turn"]] if isinstance(f.get("turn"), int) else []))}),
            recent_tokens=((ledger.get("blocks") or {}).get("recent_turns") or 0) if ledger else None,
            legs=sorted({f.get("leg") for f in selected if f.get("leg")}))
    for key, row in out.items():
        if row["recent_tokens"] is None:
            row["recent_tokens"] = out.get(f"recent_only|{key.split('|', 1)[1]}", {}).get("recent_tokens") or 0
    return out


def scoring_inputs(history, questions, answers, *, audit=None, replay=None, originals=None):
    """What scoring.score needs beyond the grades: occurrences, the originals' token prefix,
    the answers' prompt hashes, S8's audit and S7's per-turn wall times (both optional)."""
    from experiments.z2 import scoring
    from experiments.z2.ledger import load_history
    occurrences = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
                   for o in json.loads((Path(questions) / history / "occurrences.json").read_text())}
    index = Path(answers) / "index.json"
    walls = [json.loads(line)["wall"] for line in Path(replay).read_text().splitlines() if line.strip()] \
        if replay and Path(replay).exists() else None
    return dict(occurrences=occurrences, prefix=scoring.turn_prefix(originals or load_history(history)),
                prompts=json.loads(index.read_text())["index"] if index.exists() else None,
                audit=json.loads(Path(audit).read_text()) if audit and Path(audit).exists() else None,
                replay_walls=walls)


def build(history, units, grades, extras=None):
    graded, ranks, ctl = grades["graded"], grades["ranks"], grades["controls"]
    scored = None
    if extras:
        from experiments.z2 import scoring
        scored = scoring.score(history, graded, units, extras["occurrences"], extras["prefix"],
                               semantic=grades.get("semantic_credit"), prompts=extras.get("prompts"),
                               audit=extras.get("audit"), replay_walls=extras.get("replay_walls"))
    rows = collections.defaultdict(list)                     # (condition, type) -> [row]
    for key, g in graded.items():
        cond, ask_id = key.split("|", 1)
        meta = units.get(key, {})
        rows[(cond, meta.get("type"))].append(dict(g, ask=ask_id, family=family_of(ask_id), **meta))
    conditions = sorted({c for c, _ in rows})
    types = sorted({t for _, t in rows if t})
    table = {}
    for (cond, kind), group in sorted(rows.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        done = [r for r in group if r.get("status") == "graded"]
        scores = [r["score"] for r in done if isinstance(r.get("score"), (int, float))]
        neg = [r for r in done if r.get("abstention_expected")]
        cov = [len(r["items_covered"]) / r["items_total"] for r in done
               if kind in ("cumulative", "aggregation") and r.get("items_total")]
        tokens = [(r["score"], r.get("prompt_tokens"), r.get("fragments")) for r in done
                  if isinstance(r.get("score"), (int, float))]
        table.setdefault(cond, {})[kind] = dict(
            n=len(group), graded=len(done), ungraded=len(group) - len(done),
            mean_score=round(statistics.fmean(scores), 3) if scores else None,
            grades=dict(collections.Counter(r["grade"] for r in done)),
            superseded_rate=round(sum(r["superseded_used"] for r in done) / len(done), 3) if done else None,
            invented_rate=round(sum(1 for r in done if r.get("invented")) / len(done), 3) if done else None,
            ledger_miss_claims=sum(1 for r in done for v in (r.get("claims") or {}).values() if v == "ledger_miss"),
            unconfirmed_source_claims=sum(1 for r in done for v in (r.get("claims") or {}).values()
                                          if v == "unconfirmed_source"),
            abstention_accuracy=round(sum(r["abstained"] for r in neg) / len(neg), 3) if neg else None,
            item_coverage=round(statistics.fmean(cov), 3) if cov else None,
            tur=round(statistics.fmean(s / (p / 1000) for s, p, _ in tokens if p), 3)
            if any(p for _, p, _ in tokens) else None,
            spf=round(statistics.fmean(s / f for s, _, f in tokens if f), 3) if any(f for _, _, f in tokens) else None)
    paired = {}
    for cond in conditions:
        for base in BASELINES:
            if cond == base:
                continue
            for kind in types:
                diffs = collections.defaultdict(list)
                for r in rows.get((cond, kind), []):
                    b = graded.get(f"{base}|{r['ask']}")
                    if r.get("status") == "graded" and b and b.get("status") == "graded" \
                            and isinstance(r.get("score"), (int, float)) and isinstance(b.get("score"), (int, float)):
                        diffs[r["family"]].append(r["score"] - b["score"])
                per_family = {f: statistics.fmean(v) for f, v in diffs.items()}
                if per_family:
                    paired.setdefault(cond, {}).setdefault(base, {})[kind] = dict(
                        families=len(per_family), mean_diff=round(statistics.fmean(per_family.values()), 3),
                        ci95_over_families=bootstrap(per_family))
    wins = collections.defaultdict(lambda: collections.Counter())
    for ask_id, r in ranks.items():
        if r.get("status") != "ranked":
            continue
        c = r["conditions"]
        for cond in c:
            for base in BASELINES:
                if cond != base and base in c:
                    wins[(cond, base)]["win" if c[cond] < c[base] else "tie" if c[cond] == c[base] else "loss"] += 1
    retention = collections.defaultdict(collections.Counter)
    for key, g in graded.items():
        cond, ask_id = key.split("|", 1)
        fam, rest = ask_id.split("@")
        if not rest.endswith("#0") or g.get("status") != "graded":
            continue
        for later_key, later in graded.items():
            if later_key.startswith(f"{cond}|{fam}@") and not later_key.endswith("#0") and later.get("status") == "graded":
                retention[cond][f"{g['grade']}->{later['grade']}"] += 1
    controls = collections.defaultdict(collections.Counter)
    memory = collections.defaultdict(list)
    for key, verdict in ctl.items():
        cond = key.split("|", 1)[0]
        controls[cond][verdict] += 1
    for key, meta in units.items():
        if meta["kind"] == "control":
            memory[key.split("|", 1)[0]].append(meta["memory_tokens"])
    semantic = collections.defaultdict(lambda: collections.Counter())
    for key, r in (grades.get("semantic_credit") or {}).items():
        c = semantic[key.split("|", 1)[0]]
        c["contexts"] += 1
        c["with_gold_fragments"] += bool(r.get("gold_fragments"))
        c["items"] += r.get("items", 0)
        c["items_carried"] += r.get("items_carried", 0)
        c["wrong_fragments"] += r.get("wrong", 0)
        for kinds in (r.get("carried_by_kind") or {}).values():
            for kind in kinds:
                c[f"carried_by_{kind}"] += 1
    return dict(policy=POLICY, history=history, scoring=scored, by_condition_and_type=table, paired=paired,
                semantic_credit={c: dict(v, items_carried_rate=round(v["items_carried"] / v["items"], 3)
                                         if v["items"] else None) for c, v in semantic.items()},
                ranking_win_rates={f"{c} vs {b}": dict(v) for (c, b), v in wins.items()},
                retention_transitions={c: dict(v) for c, v in retention.items()},
                controls={c: dict(verdicts=dict(controls.get(c, {})),
                                  retrieved_share=round(sum(1 for m in memory[c] if m) / len(memory[c]), 3)
                                  if memory[c] else None,
                                  mean_memory_tokens=round(statistics.fmean(memory[c]), 1) if memory[c] else None)
                          for c in sorted(set(controls) | set(memory))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--questions", type=Path, help="questions root (occurrences) — enables scoring")
    parser.add_argument("--answers", type=Path, help="S10 folder (prompt hashes for the fidelity audit)")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extras = scoring_inputs(args.history, args.questions, args.answers or args.grades.parent,
                            audit=args.audit, replay=args.replay) if args.questions else None
    result = build(args.history, unit_index(args.units), json.loads(args.grades.read_text()), extras)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
