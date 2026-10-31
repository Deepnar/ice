"""Would blending the reranker's order with the search's own order bring more cited turns into the
prompt? A simulation on saved read units — no model, no GPU (Z2 F22, RETESTS build list item 2).

Per question: the reranker's order is ranked_candidates; the search's own order is rebuilt from the
legs' candidate lists (generated_by_leg) by reciprocal-rank fusion (k = retrieval_rrf_k). Each piece's
place is w × its rerank position + (1 − w) × its fusion position; pieces are then packed greedily into
the question's retrieval budget. w = 1 is the reranker alone under the same simplified packing (no
dedup/collapse), so only differences between weights are meaningful, not the absolute level.

    uv run --no-sync python experiments/z2/blend_sim.py --history B --database ice_z2_main_b \
        --units logs/z2-retest-B/units/retest_cand256 --occurrences logs/z2-questions-final/B/occurrences.json
"""

import argparse
import json
import os
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WEIGHTS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0)


def ident(piece):
    """One piece's identity across the legs' lists and the reranked list (stored as a list or as text)."""
    key = piece.get("stored_artifact_ids") or piece.get("rendered_text_sha256") or piece.get("text", "")[:200]
    return json.dumps(key, sort_keys=True) if not isinstance(key, str) else key.replace("(", "[").replace(")", "]").replace("'", '"')


def fusion_order(by_leg, k):
    score = {}
    for pieces in by_leg.values():
        for rank, p in enumerate(pieces):
            score[ident(p)] = score.get(ident(p), 0.0) + 1.0 / (k + rank + 1)
    return {pid: n for n, pid in enumerate(sorted(score, key=lambda x: -score[x]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    args = parser.parse_args()
    from experiments.z2.funnel import turn_map, turns_of
    from src.api.config import settings
    by_row, by_batch = turn_map(args.database)
    gold = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}":
            {int(c["turn"]) for c in o["reference"].get("citations", []) if c.get("turn")}
            for o in json.loads(args.occurrences.read_text())}
    results = {w: [] for w in WEIGHTS}
    for e in os.scandir(args.units):
        u = json.loads(Path(e.path).read_text())
        a, st = u["ask"]["id"], u["stage"]
        if u["ask"]["kind"] != "question" or not gold.get(a) or not st.get("retrieval_budget"):
            continue
        ranked = st.get("ranked_candidates") or []
        fusion = fusion_order(st.get("generated_by_leg") or {}, settings.retrieval_rrf_k)
        far = len(fusion) + len(ranked)
        for w in WEIGHTS:
            order = sorted(range(len(ranked)), key=lambda i: w * i + (1 - w) * fusion.get(ident(ranked[i]), far))
            used, seen = 0, set()
            for i in order:
                tokens = int(ranked[i].get("tokens") or 0)
                if used + tokens <= st["retrieval_budget"]:
                    used += tokens
                    seen |= turns_of(ranked[i], by_row, by_batch)
            results[w].append(len(gold[a] & seen) / len(gold[a]))
    print(f"{args.history}: cited-turn coverage under the simulated packing, {len(results[1.0])} questions")
    for w in WEIGHTS:
        label = "reranker only" if w == 1.0 else "search order only" if w == 0.0 else f"{int(w * 100)}% reranker"
        print(f"   {label:20} {100 * S.mean(results[w]):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
