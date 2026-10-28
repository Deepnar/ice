"""Z2 retests (results/RETESTS.md R2, R3, R4): per read condition, how much of each question's
cited turns reaches the prompt, before any answering or grading. Reads saved units only.

    uv run --no-sync python experiments/z2/retest_coverage.py --history B \
        --units logs/z2-retest-B/units --reference logs/z2-run-main/s9-B/units \
        --occurrences logs/z2-questions-final/B/occurrences.json

Coverage is presence by source turn: it says the cited turn reached the prompt in some form,
not that the piece carries the required detail (a summary counts) — read it with F6.
"""

import argparse
import json
import os
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def turns_of(piece, bare):
    if bare:
        return {int(piece["turn"])}
    t = piece.get("source_turns") or []
    return set(json.loads(t) if isinstance(t, str) else t)


def read(folder, gold, only=None):
    """{condition: {ask id: (covered share, pieces, memory tokens, prompt tokens)}}"""
    out = {}
    for cond in sorted((e for e in os.scandir(folder) if e.is_dir() and (only is None or e.name in only)),
                       key=lambda e: e.name):
        rows = {}
        for e in os.scandir(cond.path):
            u = json.loads(Path(e.path).read_text())
            a, st = u["ask"]["id"], u["stage"]
            if u["ask"]["kind"] != "question" or not gold.get(a):
                continue
            pieces = st.get("selected") or []
            seen = set().union(*[turns_of(p, "bare" in cond.name) for p in pieces]) if pieces else set()
            rows[a] = (len(gold[a] & seen) / len(gold[a]), len(pieces), st.get("selected_tokens") or 0,
                       st.get("prompt_tokens"))
        out[cond.name] = rows
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--units", type=Path, required=True, help="the retest read's units folder")
    parser.add_argument("--reference", type=Path, help="Z2's units folder (bare_vector, full_default)")
    parser.add_argument("--occurrences", type=Path, required=True)
    args = parser.parse_args()
    occ = json.loads(args.occurrences.read_text())
    gold = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}":
            {int(c["turn"]) for c in o["reference"].get("citations", []) if c.get("turn") is not None} for o in occ}
    rows = read(args.units, gold)
    if args.reference:
        ref = read(args.reference, gold, only={"full_default", "bare_vector"})
        rows.update({f"z2:{c}": r for c, r in ref.items()})
    print(f"{args.history}: cited-turn coverage per condition (questions with cited turns)")
    print(f"{'condition':28} {'n':>4} {'covered':>8} {'any':>6} {'pieces':>7} {'memory tok':>11} {'prompt tok':>11}")
    for c, r in rows.items():
        if not r:
            continue
        v = list(r.values())
        prompt = [x[3] for x in v if x[3]]
        print(f"{c:28} {len(v):4} {100 * S.mean(x[0] for x in v):7.1f}% {100 * S.mean(x[0] > 0 for x in v):5.1f}% "
              f"{S.median(x[1] for x in v):7.0f} {S.median(x[2] for x in v):11.0f} "
              f"{(S.median(prompt) if prompt else float('nan')):11.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
