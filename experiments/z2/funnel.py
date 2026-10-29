"""Where an answer's evidence is lost, and what each retrieval leg put in front of the model.

Every read unit records four stages: candidates found per leg (generated_by_leg), the merged and
reranked list (ranked_candidates), what fit the token budget (budgeted) and what reached the prompt
(selected). For each question's cited turns this traces the last stage each turn reached — never
found / found but not ranked / ranked but over the budget / budgeted but dropped in final assembly /
in the prompt — and, per leg, its share of the prompt's pieces and tokens and of the cited turns it
delivered. Read-only: saved units plus a turn map from the history's store.

    uv run --no-sync python experiments/z2/funnel.py --history B --database ice_z2_main_b \
        --units logs/z2-retest-B/units --conditions retest_cand64,retest_cand256 \
        --occurrences logs/z2-questions-final/B/occurrences.json
"""

import argparse
import collections
import json
import os
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAGES = ("never found", "found, not ranked", "ranked, over the budget", "budgeted, dropped at assembly", "in the prompt")


def turn_map(database):
    """(episodic row id -> turn, batch id -> {turns}) from the store, turns in timestamp order."""
    from sqlalchemy import create_engine, text
    from src.api.config import settings
    url = settings.database_url.rsplit("/", 1)[0] + "/" + database
    with create_engine(url).connect() as c:
        rows = c.execute(text("SELECT id::text, batch_id::text FROM episodic_memory ORDER BY timestamp, id")).all()
    by_row, by_batch = {}, collections.defaultdict(set)
    for n, (rid, bid) in enumerate(rows, 1):
        by_row[rid] = n
        by_batch[bid].add(n)
    return by_row, by_batch


def turns_of(piece, by_row, by_batch):
    t = piece.get("source_turns")
    if t:
        return set(json.loads(t) if isinstance(t, str) else t)
    out = set()
    row = piece.get("source_row")
    if row and row != "None" and row in by_row:
        out.add(by_row[row])
    batches = piece.get("origin_batches") or []
    batches = json.loads(batches.replace("'", '"')) if isinstance(batches, str) else batches
    for b in batches:
        out |= by_batch.get(b, set())
    return out


def analyse(folder, gold, by_row, by_batch):
    lost = collections.Counter()
    legs = collections.defaultdict(lambda: dict(pieces=0, tokens=0, cited=0))
    fill, n_q = [], 0
    for e in os.scandir(folder):
        u = json.loads(Path(e.path).read_text())
        a, st = u["ask"]["id"], u["stage"]
        if u["ask"]["kind"] != "question":
            continue
        reach = lambda pieces: set().union(*[turns_of(p, by_row, by_batch) for p in pieces]) if pieces else set()
        found = reach([p for v in (st.get("generated_by_leg") or {}).values() for p in v])
        ranked, budgeted = reach(st.get("ranked_candidates") or []), reach(st.get("budgeted") or [])
        selected = st.get("selected") or []
        prompt = reach(selected)
        if st.get("retrieval_budget"):
            fill.append((st.get("selected_tokens") or 0) / st["retrieval_budget"])
        for p in selected:
            leg = p.get("leg") or p.get("type")
            legs[leg]["pieces"] += 1
            legs[leg]["tokens"] += int(p.get("tokens") or 0)
            legs[leg]["cited"] += len(turns_of(p, by_row, by_batch) & gold.get(a, set()))
        n_q += 1
        for t in gold.get(a, ()):
            stage = (4 if t in prompt else 3 if t in budgeted else 2 if t in ranked else 1 if t in found else 0)
            lost[STAGES[stage]] += 1
    return lost, legs, fill, n_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    args = parser.parse_args()
    by_row, by_batch = turn_map(args.database)
    gold = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}":
            {int(c["turn"]) for c in o["reference"].get("citations", []) if c.get("turn")}
            for o in json.loads(args.occurrences.read_text())}
    for cond in args.conditions.split(","):
        lost, legs, fill, n_q = analyse(args.units / cond, gold, by_row, by_batch)
        total = sum(lost.values()) or 1
        print(f"\n== {args.history} {cond}: {n_q} questions, {total} cited turns; budget filled median "
              f"{100 * S.median(fill):.0f}%" if fill else f"\n== {cond}")
        for stage in STAGES:
            print(f"   {stage:32} {lost[stage]:4}  ({100 * lost[stage] / total:4.1f}%)")
        tok = sum(v["tokens"] for v in legs.values()) or 1
        pcs = sum(v["pieces"] for v in legs.values()) or 1
        cited = sum(v["cited"] for v in legs.values()) or 1
        print("   per leg — share of prompt pieces / tokens / cited turns delivered:")
        for leg, v in sorted(legs.items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"     {leg:14} {100 * v['pieces'] / pcs:5.1f}% {100 * v['tokens'] / tok:5.1f}% {100 * v['cited'] / cited:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
